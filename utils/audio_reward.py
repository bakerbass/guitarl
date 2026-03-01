"""
Audio-based reward calculation using HarmonicsClassifier.

Wraps the pretrained harmonic classifier to provide reward signals
for reinforcement learning based on audio quality.

Uses FRACTIONAL FRETS for position to encode musical structure.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import sounddevice as sd
import librosa
import logging
from typing import Dict, Optional, Tuple
import time


# Add HarmonicsClassifier to path
HARMONICS_CLASSIFIER_PATH = Path(__file__).parent.parent.parent / "HarmonicsClassifier"
sys.path.insert(0, str(HARMONICS_CLASSIFIER_PATH))

# Import from HarmonicsClassifier
from osc_realtime_classifier import HarmonicsCNN


logger = logging.getLogger(__name__)


# Import shared reward constants and function
from utils.reward import (
    HARMONIC_FRETS,
    TORQUE_OPTIMAL_HARMONIC,
    TORQUE_MAX,
    FRET_TOLERANCE,
    TORQUE_TOLERANCE,
    REWARD_WEIGHT_AUDIO,
    REWARD_WEIGHT_FRET,
    REWARD_WEIGHT_TORQUE,
    CLASS_NAMES as _CLASS_NAMES,
    HARMONIC_CLASS_IDX as _HARMONIC_CLASS_IDX,
    SUCCESS_THRESHOLD,
    REWARD_MODE_FULL,
    REWARD_MODE_NO_FILTRATION,
    REWARD_MODE_NO_AUDIO,
    REWARD_MODE_COSINE_SIM,
    REWARD_MODE_SPECTRAL,
    compute_reward as _compute_reward,
    compute_reward_no_filtration as _compute_reward_no_filtration,
    compute_reward_no_audio as _compute_reward_no_audio,
    compute_reward_cosine_sim as _compute_reward_cosine_sim,
    compute_reward_spectral as _compute_reward_spectral,
    is_success as _is_success,
    D_STRING_OPEN_FREQ,
    FRET_TO_HARMONIC_NUMBER,
    SPECTRAL_BANDWIDTH_RATIO,
    SPECTRAL_HER_WEIGHT,
    SPECTRAL_FUND_SUPPRESS_WEIGHT,
    SPECTRAL_SIGNAL_WEIGHT,
    SPECTRAL_NOISE_FLOOR_HZ,
    SPECTRAL_SUCCESS_THRESHOLD,
)

# ── Fine-tune mode: mel spec parameters matching image_analysis.py ────────────
# Using the same n_fft / fmax as image_analysis so RL spectrograms are directly
# comparable to the reference dataset spectrograms.
_FT_SR         = 22050
_FT_N_FFT      = 4096
_FT_HOP_LENGTH = 512
_FT_N_MELS     = 128
_FT_FMIN       = 80
_FT_FMAX       = 10000
_FT_MAX_ONSET  = 1.0   # ignore onsets detected later than this many seconds


class HarmonicRewardCalculator:
    """
    Calculate rewards based on harmonic quality using pretrained classifier.
    
    Captures audio from VB-CABLE, classifies as harmonic/dead/general,
    and returns reward signal for RL training.
    """
    
    # Class labels from HarmonicsClassifier (must match train_cnn.py label_map)
    CLASS_NAMES = _CLASS_NAMES
    HARMONIC_CLASS_IDX = _HARMONIC_CLASS_IDX
    
    def __init__(self,
                 model_path: Optional[str],
                 device_name: str = "VB-Audio Virtual Cable",
                 capture_duration: float = 1.0,
                 model_sr: int = 22050,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 reward_mode: str = REWARD_MODE_FULL,
                 ref_dir: Optional[Path] = None,
                 fret_to_pitch: Optional[Dict[int, int]] = None):
        """
        Initialize reward calculator.

        Args:
            model_path: Path to trained harmonic classifier model (.pt file).
                        May be None when reward_mode='cosine_sim' (CNN not used).
            device_name: Audio input device name (VB-CABLE)
            capture_duration: Audio capture duration in seconds
            model_sr: Sample rate expected by model
            device: Torch device (cuda/cpu)
            reward_mode: One of 'full', 'no_filtration', 'no_audio', 'cosine_sim', 'spectral'.
                         'full'        — two-layer reward (default)
                         'no_filtration' — bypass physics gate, Layer 2 only
                         'no_audio'    — Layer 1 + fret/torque shaping, no CNN
                         'cosine_sim'  — Layer 1 + onset-aligned mel cosine
                                         similarity vs reference dataset WAVs
                         'spectral'    — Layer 1 + direct spectral content
                                         analysis (harmonic energy ratio)
            ref_dir:     Directory of reference WAVs (GB_NH_*.wav).
                         Required when reward_mode='cosine_sim'.
            fret_to_pitch: Mapping from target fret → MIDI note number used to
                           select reference WAVs by pitch.  E.g. {4:78, 5:74, 7:69}
                           for D string.  Required when reward_mode='cosine_sim'.
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.device_name = device_name
        self.capture_duration = capture_duration
        self.model_sr = model_sr
        self.device = torch.device(device)
        
        # Audio processing parameters
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        # Find audio device
        self.device_id = self._find_audio_device()
        if self.device_id is None:
            if reward_mode == REWARD_MODE_NO_AUDIO:
                # Audio not used in this mode — skip device requirement
                logger.info("reward_mode=no_audio — audio device not required.")
            elif sys.stdin.isatty():
                self.device_id = self._prompt_select_device()
            else:
                raise RuntimeError(
                    f"Audio input device matching '{device_name}' not found and stdin is "
                    "not a TTY (cannot prompt). Pass a different --audio-device substring "
                    "or fix your ALSA/PulseAudio routing."
                )

        # Get device sample rate
        if self.device_id is not None:
            device_info = sd.query_devices(self.device_id, 'input')
            self.device_sr = int(device_info['default_samplerate'])
            logger.info(f"Audio device SR: {self.device_sr} Hz")
        else:
            self.device_sr = 44100  # Default fallback (no_audio mode)
        
        # Load CNN model (not needed in cosine_sim mode)
        self.reward_mode = reward_mode
        if reward_mode in (REWARD_MODE_COSINE_SIM, REWARD_MODE_SPECTRAL) and self.model_path is None:
            self.model = None
            logger.info(f"[{reward_mode}] CNN model not loaded (model_path=None).")
        else:
            self.model = self._load_model()

        # Fine-tune mode: pre-load reference mel spectrograms keyed by target fret
        self._ref_mels: Dict[int, list] = {}
        if reward_mode == REWARD_MODE_COSINE_SIM:
            if ref_dir is None or fret_to_pitch is None:
                raise ValueError(
                    "ref_dir and fret_to_pitch are required when reward_mode='cosine_sim'"
                )
            self._load_ref_mels(Path(ref_dir), fret_to_pitch)

        logger.info(
            f"HarmonicRewardCalculator initialized: model={model_path}, reward_mode={reward_mode}"
        )
    
    def _find_audio_device(self) -> Optional[int]:
        """Find audio input device by name substring (input channels required)."""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if (self.device_name.lower() in device['name'].lower()
                    and device['max_input_channels'] > 0):
                logger.info(f"Found audio device: {device['name']} (ID: {idx})")
                return idx
        # Second pass: warn about any matching output-only devices so the user
        # understands why the substring matched but was rejected.
        for idx, device in enumerate(devices):
            if (self.device_name.lower() in device['name'].lower()
                    and device['max_input_channels'] == 0):
                logger.warning(
                    f"Skipped output-only device '{device['name']}' (ID: {idx}) — "
                    "no input channels. Check ALSA / PulseAudio routing."
                )
        return None

    def _prompt_select_device(self) -> int:
        """Interactively prompt the user to choose an input device or quit."""
        devices = sd.query_devices()
        input_devices = [
            (idx, dev)
            for idx, dev in enumerate(devices)
            if dev['max_input_channels'] > 0
        ]

        print("\n" + "=" * 60)
        print(f"  Audio device '{self.device_name}' not found (or has no input channels).")
        print("  Available INPUT devices:")
        print("=" * 60)
        for i, (idx, dev) in enumerate(input_devices):
            print(f"  [{i}]  ID={idx:2d}  ch={dev['max_input_channels']:2d}  "
                  f"SR={int(dev['default_samplerate'])}  {dev['name']}")
        print("  [q]  Quit and fix --audio-device, then retry")
        print("=" * 60)

        while True:
            try:
                raw = input("  Select device number (or q): ").strip().lower()
            except EOFError:
                raw = "q"

            if raw == "q":
                print("Exiting. Re-run with the correct --audio-device substring.")
                raise SystemExit(1)

            try:
                choice = int(raw)
                if 0 <= choice < len(input_devices):
                    dev_id, dev = input_devices[choice]
                    self.device_name = dev['name']
                    print(f"  Using: {dev['name']} (ID: {dev_id})")
                    print("=" * 60 + "\n")
                    logger.info(f"User selected audio device: {dev['name']} (ID: {dev_id})")
                    return dev_id
                else:
                    print(f"  Please enter a number between 0 and {len(input_devices)-1}.")
            except ValueError:
                print("  Invalid input — enter a number or 'q'.")
    
    def _load_model(self) -> HarmonicsCNN:
        """Load pretrained harmonic classifier."""
        if self.model_path is None:
            raise ValueError("model_path is required (set reward_mode='cosine_sim' to run without CNN)")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = HarmonicsCNN(num_classes=3, dropout=0.5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from {self.model_path}")
        return model
    
    def capture_audio(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Capture audio from VB-CABLE.
        
        Args:
            duration: Capture duration (uses default if None)
            
        Returns:
            Audio array at device sample rate
        """
        if duration is None:
            duration = self.capture_duration
        
        if self.device_id is None:
            logger.warning("No audio device available, returning silence")
            return np.zeros(int(self.device_sr * duration))
        
        try:
            logger.debug(f"Capturing {duration}s from device {self.device_id}")
            frames = int(duration * self.device_sr)
            buf = np.zeros(frames, dtype='float32')
            with sd.InputStream(
                samplerate=self.device_sr,
                channels=1,
                device=self.device_id,
                dtype='float32',
            ) as stream:
                remaining = frames
                offset = 0
                while remaining > 0:
                    chunk, _ = stream.read(min(remaining, 4096))
                    n = len(chunk)
                    buf[offset:offset + n] = chunk[:, 0]
                    offset += n
                    remaining -= n
            return buf
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            return np.zeros(int(self.device_sr * duration))
    
    def calibrate_silence_threshold(
        self,
        duration: float = 2.0,
        multiplier: float = 3.0,
    ) -> float:
        """
        Measure the ambient noise floor and return a suitable silence threshold.

        Opens a short-lived stream, records ``duration`` seconds of quiescent
        audio in 20 ms chunks, and returns ``median_chunk_rms * multiplier``.
        The median is used rather than the mean so that brief transients (a door
        slam, a key press) during calibration don't inflate the threshold.

        Call this once at startup before any robot actions are sent.

        Args:
            duration:   Seconds of ambient audio to sample (default 2.0).
            multiplier: Scale factor above the noise floor (default 3.0).
                        A value of 3.0 means the silence gate fires when the
                        signal has been < 3× the ambient RMS.

        Returns:
            RMS threshold (always >= 1e-6).
        """
        if self.device_id is None:
            logger.warning("calibrate_silence_threshold: no device, returning default 0.005")
            return 0.005

        chunk_frames = int(0.02 * self.device_sr)  # 20 ms per chunk
        n_chunks = max(1, int(duration / 0.02))
        rms_values: list = []

        logger.info(
            f"Calibrating silence threshold: sampling {duration:.1f}s of ambient audio..."
        )
        try:
            with sd.InputStream(
                samplerate=self.device_sr,
                channels=1,
                device=self.device_id,
                dtype="float32",
                blocksize=chunk_frames,
            ) as stream:
                for _ in range(n_chunks):
                    chunk, _ = stream.read(chunk_frames)
                    rms_values.append(float(np.sqrt(np.mean(chunk[:, 0] ** 2))))
        except Exception as exc:
            logger.error(f"calibrate_silence_threshold failed: {exc}, returning default 0.005")
            return 0.005

        noise_floor = float(np.median(rms_values))
        threshold = max(noise_floor * multiplier, 1e-6)
        logger.info(
            f"  Noise floor (median RMS): {noise_floor:.6f}  "
            f"Threshold (×{multiplier}): {threshold:.6f}"
        )
        return threshold

    def wait_for_silence(
        self,
        rms_threshold: float = 0.005,
        hold_duration: float = 0.5,
        timeout: float = 8.0,
    ) -> bool:
        """
        Block until the audio signal has been below ``rms_threshold`` for
        ``hold_duration`` continuous seconds, or until ``timeout`` elapses.

        Opens a dedicated short-lived stream so the main capture stream is
        not affected.  Reads audio in 20 ms chunks; each chunk is one poll
        cycle (~50 Hz), so the hold counter is in units of 20 ms chunks.

        Args:
            rms_threshold: Per-chunk RMS that counts as "silent" (default 0.005).
            hold_duration: Seconds of continuous silence required (default 0.5).
            timeout:       Maximum wait in seconds before proceeding (default 8.0).

        Returns:
            True if silence was confirmed, False if timed out.
        """
        if self.device_id is None:
            return True  # No device — treat as silent

        chunk_frames   = int(0.02 * self.device_sr)           # samples per 20 ms chunk
        chunks_needed  = max(1, int(hold_duration / 0.02))   # consecutive quiet chunks required
        chunks_limit   = max(1, int(timeout / 0.02))         # total budget before timeout
        # WASAPI (and some other drivers) pre-fill the InputStream buffer with
        # zeros before real audio arrives.  Without a warmup, the first
        # chunks_needed reads all return RMS ≈ 0, triggering a false "silence
        # confirmed" before any actual audio has been read.  Discarding the
        # first ~100 ms of reads lets the buffer flush stale zeros.
        warmup_chunks  = 5                                    # 5 × 20 ms = 100 ms warmup

        try:
            with sd.InputStream(
                samplerate=self.device_sr,
                channels=1,
                device=self.device_id,
                dtype="float32",
                blocksize=chunk_frames,
            ) as stream:
                for _ in range(warmup_chunks):
                    stream.read(chunk_frames)

                consecutive_quiet = 0
                for _ in range(max(1, chunks_limit - warmup_chunks)):
                    chunk, _ = stream.read(chunk_frames)
                    rms = float(np.sqrt(np.mean(chunk[:, 0] ** 2)))
                    if rms < rms_threshold:
                        consecutive_quiet += 1
                        if consecutive_quiet >= chunks_needed:
                            return True
                    else:
                        consecutive_quiet = 0
        except Exception as exc:
            logger.warning(f"wait_for_silence: stream error ({exc}), proceeding")
            return False

        logger.warning(
            f"wait_for_silence: timed out after {timeout:.1f}s "
            f"(threshold={rms_threshold:.4f}, hold={hold_duration:.2f}s)"
        )
        return False

    def close(self) -> None:
        """Release any held audio resources. Safe to call multiple times."""
        try:
            sd.stop()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio to mel spectrogram for model input.
        
        Args:
            audio: Raw audio array
            
        Returns:
            Preprocessed tensor (1, 1, n_mels, time)
        """
        # Resample if needed
        if self.device_sr != self.model_sr:
            audio = librosa.resample(audio, orig_sr=self.device_sr, target_sr=self.model_sr)
        
        # Pad or trim to target length
        target_length = int(self.model_sr * self.capture_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.model_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to tensor (add batch and channel dimensions)
        mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
        
        return mel_tensor
    
    def classify_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Classify audio using pretrained model.
        
        Args:
            audio: Raw audio array
            
        Returns:
            Dictionary with class probabilities and predicted class
        """
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        
        # Use original if too short after trimming
        if len(audio_trimmed) < self.model_sr * 0.1:
            audio_trimmed = audio
        
        # Preprocess
        audio_tensor = self.preprocess_audio(audio_trimmed)
        audio_tensor = audio_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        probs = probabilities[0].cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'harmonic_prob': probs[self.HARMONIC_CLASS_IDX],
            'dead_prob': probs[1],
            'general_prob': probs[2],
        }
    
    # ── Fine-tune helpers (cosine_sim reward mode) ───────────────────────────

    def _onset_align(self, y: np.ndarray, sr: int = _FT_SR) -> np.ndarray:
        """Trim audio to the first detected onset (matching image_analysis.py)."""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=_FT_HOP_LENGTH)
        onset_sample = 0
        if len(onset_frames) > 0:
            candidate = int(librosa.frames_to_samples(onset_frames[0], hop_length=_FT_HOP_LENGTH))
            if candidate <= int(_FT_MAX_ONSET * sr):
                onset_sample = candidate
        return y[onset_sample:]

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    def _compute_ft_mel(self, y: np.ndarray) -> np.ndarray:
        """Mel spectrogram in dB using image_analysis.py parameters."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=_FT_SR, n_fft=_FT_N_FFT, hop_length=_FT_HOP_LENGTH,
            n_mels=_FT_N_MELS, fmin=_FT_FMIN, fmax=_FT_FMAX,
        )
        return librosa.power_to_db(mel, ref=np.max)

    def _load_ref_mels(self, ref_dir: Path, fret_to_pitch: Dict[int, int]) -> None:
        """Pre-load and cache onset-aligned normalized mel spectrograms for each fret."""
        total = 0
        for fret, pitch in fret_to_pitch.items():
            pattern = f"GB_NH*pitches{pitch}*.wav"
            ref_wavs = sorted(ref_dir.glob(pattern))
            if not ref_wavs:
                logger.warning(f"[fine-tune] No reference WAVs matched {ref_dir / pattern}")
                continue
            mels = []
            for wav_path in ref_wavs:
                try:
                    y, _ = librosa.load(str(wav_path), sr=_FT_SR, mono=True)
                    y = self._onset_align(y)
                    if len(y) < int(0.5 * _FT_SR):
                        continue  # skip clips that are too short after alignment
                    mels.append(self._normalize_01(self._compute_ft_mel(y)))
                except Exception as exc:
                    logger.warning(f"[fine-tune] Skipped {wav_path.name}: {exc}")
            self._ref_mels[fret] = mels
            logger.info(
                f"[fine-tune] Fret {fret} (pitch {pitch}): "
                f"{len(mels)} reference spectrograms loaded from {ref_dir}"
            )
            total += len(mels)
        if total == 0:
            raise RuntimeError(
                f"[fine-tune] No reference WAVs loaded from {ref_dir}. "
                "Check --ref-dir and fret-to-pitch mapping."
            )

    # ── Spectral content reward mode ────────────────────────────────────────

    def _compute_spectral_score(
        self, audio: np.ndarray, target_fret: int,
    ) -> float:
        """Compute spectral reward score based on harmonic energy ratio.

        Analyses the power spectrum of the captured audio and measures:
        1. Harmonic Energy Ratio (HER): energy at expected partials / total energy
        2. Fundamental suppression: penalises energy at the open-string fundamental
        3. Signal presence: prevents rewarding silence

        Returns a score in [0.0, 1.0].
        """
        from scipy.signal import welch as scipy_welch

        f0 = D_STRING_OPEN_FREQ
        harmonic_n = FRET_TO_HARMONIC_NUMBER.get(target_fret)
        if harmonic_n is None:
            logger.warning(
                f"[spectral] No harmonic number for fret {target_fret}, returning 0.0"
            )
            return 0.0

        # Resample to analysis SR and onset-align
        sr = _FT_SR
        if self.device_sr != sr:
            y = librosa.resample(audio.astype(np.float32), orig_sr=self.device_sr, target_sr=sr)
        else:
            y = audio.astype(np.float32)
        y = self._onset_align(y, sr=sr)

        # Skip the initial transient (300 ms) and take up to 2 s of steady-state
        skip_samples = int(0.3 * sr)
        window_samples = int(2.0 * sr)
        y = y[skip_samples: skip_samples + window_samples]

        if len(y) < int(0.25 * sr):
            return 0.0  # too short to analyse

        # Power spectrum via Welch's method (robust averaging)
        nperseg = min(4096, len(y))
        freqs, psd = scipy_welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

        # -- Desired partials: n*f0, 2n*f0, 3n*f0, ... up to Nyquist --
        harmonic_fundamental = harmonic_n * f0
        desired_energy = 0.0
        partial = harmonic_fundamental
        bw = SPECTRAL_BANDWIDTH_RATIO
        while partial < sr / 2:
            lo = partial * (1.0 - bw)
            hi = partial * (1.0 + bw)
            mask = (freqs >= lo) & (freqs <= hi)
            desired_energy += float(psd[mask].sum())
            partial += harmonic_fundamental

        # -- Fundamental energy (should be suppressed for a clean harmonic) --
        f0_lo = f0 * (1.0 - bw)
        f0_hi = f0 * (1.0 + bw)
        f0_mask = (freqs >= f0_lo) & (freqs <= f0_hi)
        f0_energy = float(psd[f0_mask].sum())

        # -- Total energy above noise floor --
        signal_mask = freqs >= SPECTRAL_NOISE_FLOOR_HZ
        total_energy = float(psd[signal_mask].sum()) + 1e-12

        # Sub-scores
        her = desired_energy / total_energy
        f0_ratio = f0_energy / total_energy
        fund_suppression = 1.0 - float(np.clip(f0_ratio * 5.0, 0.0, 1.0))

        # Signal presence: ramp up from 0 at very low energy to 1.0
        # Use a threshold relative to a quiet but audible signal (~1e-5 avg power)
        signal_presence = float(np.clip(total_energy / 1e-4, 0.0, 1.0))

        score = (
            SPECTRAL_HER_WEIGHT * her
            + SPECTRAL_FUND_SUPPRESS_WEIGHT * fund_suppression
            + SPECTRAL_SIGNAL_WEIGHT * signal_presence
        )
        score = float(np.clip(score, 0.0, 1.0))

        logger.debug(
            f"[spectral] fret={target_fret} n={harmonic_n} "
            f"HER={her:.3f} fund_sup={fund_suppression:.3f} "
            f"sig={signal_presence:.3f} → score={score:.3f}"
        )
        return score

    def _cosine_sim_vs_refs(self, audio: np.ndarray, target_fret: int) -> float:
        """Return best cosine similarity between captured audio and cached reference mels.

        The captured audio is resampled to _FT_SR, onset-aligned, and converted
        to a normalized mel spectrogram before comparison, matching the same
        processing applied to reference WAVs in _load_ref_mels().
        """
        ref_mels = self._ref_mels.get(target_fret)
        if not ref_mels:
            logger.warning(f"[fine-tune] No reference mels for fret {target_fret}, returning 0.0")
            return 0.0

        # Resample to fine-tune SR and onset-align
        if self.device_sr != _FT_SR:
            audio_ft = librosa.resample(audio, orig_sr=self.device_sr, target_sr=_FT_SR)
        else:
            audio_ft = audio.copy()
        audio_ft = self._onset_align(audio_ft)
        if len(audio_ft) < int(0.5 * _FT_SR):
            return 0.0  # silence or near-silent capture

        mel_rl = self._normalize_01(self._compute_ft_mel(audio_ft))

        best_sim = 0.0
        for mel_ref in ref_mels:
            # Crop both to the shorter time axis before comparing
            min_t = min(mel_rl.shape[1], mel_ref.shape[1])
            v_rl  = mel_rl[:, :min_t].ravel()
            v_ref = mel_ref[:, :min_t].ravel()
            dot      = float(np.dot(v_rl, v_ref))
            norm_rl  = float(np.linalg.norm(v_rl))
            norm_ref = float(np.linalg.norm(v_ref))
            sim = dot / (norm_rl * norm_ref + 1e-8)
            if sim > best_sim:
                best_sim = sim

        return float(np.clip(best_sim, 0.0, 1.0))

    def compute_reward(self,
                       fret_position: float,
                       torque: float,
                       target_fret: int,
                       audio: Optional[np.ndarray] = None,
                       capture_audio: bool = True) -> Dict[str, float]:
        """
        Compute reward for RL based on harmonic quality and action correctness.
        
        Uses FRACTIONAL FRETS for position - the agent learns that integer
        frets 4, 5, 7 are harmonic nodes.
        
        Args:
            fret_position: Fractional fret position (0.0 - 9.0)
            torque: Fretter torque (0 - 1000)
            target_fret: Target harmonic fret (4, 5, or 7)
            audio: Pre-captured audio (if None and capture_audio=True, will capture)
            capture_audio: Whether to capture audio if not provided
            
        Returns:
            Dictionary with reward components and total reward
        """
        # Audio-based classification (skipped entirely in no_audio mode)
        classification = None
        harmonic_prob = 0.0
        audio_rms = None

        needs_audio = self.reward_mode != REWARD_MODE_NO_AUDIO

        if audio is None and capture_audio and needs_audio:
            audio = self.capture_audio()

        if audio is not None and needs_audio:
            audio_rms = float(np.sqrt(np.mean(audio ** 2)))
            if self.reward_mode == REWARD_MODE_COSINE_SIM:
                # Fine-tune mode: replace CNN with onset-aligned mel cosine similarity
                cosine_sim = self._cosine_sim_vs_refs(audio, target_fret)
                # Build a pseudo-classification dict so all downstream success/recording
                # logic (which checks 'harmonic_prob') continues to work unchanged.
                classification = {
                    'predicted_class':  0 if cosine_sim >= 0.8 else 1,
                    'predicted_label':  'harmonic' if cosine_sim >= 0.8 else 'dead_note',
                    'confidence':       cosine_sim,
                    'harmonic_prob':    cosine_sim,
                    'dead_prob':        1.0 - cosine_sim,
                    'general_prob':     0.0,
                    'cosine_sim':       cosine_sim,
                }
                harmonic_prob = cosine_sim
            elif self.reward_mode == REWARD_MODE_SPECTRAL:
                spectral_score = self._compute_spectral_score(audio, target_fret)
                classification = {
                    'predicted_class':  0 if spectral_score >= SPECTRAL_SUCCESS_THRESHOLD else 1,
                    'predicted_label':  'harmonic' if spectral_score >= SPECTRAL_SUCCESS_THRESHOLD else 'dead_note',
                    'confidence':       spectral_score,
                    'harmonic_prob':    spectral_score,
                    'dead_prob':        1.0 - spectral_score,
                    'general_prob':     0.0,
                    'spectral_score':   spectral_score,
                }
                harmonic_prob = spectral_score
            else:
                classification = self.classify_audio(audio)
                harmonic_prob = classification['harmonic_prob']
        elif audio is not None and not needs_audio:
            # Still compute RMS for the silence filtration check
            audio_rms = float(np.sqrt(np.mean(audio ** 2)))

        # Dispatch to the appropriate reward function
        if self.reward_mode == REWARD_MODE_NO_FILTRATION:
            reward_info = _compute_reward_no_filtration(
                fret_position=fret_position,
                torque=torque,
                target_fret=target_fret,
                harmonic_prob=harmonic_prob,
                audio_rms=audio_rms,
            )
        elif self.reward_mode == REWARD_MODE_NO_AUDIO:
            reward_info = _compute_reward_no_audio(
                fret_position=fret_position,
                torque=torque,
                target_fret=target_fret,
                audio_rms=audio_rms,
            )
        elif self.reward_mode == REWARD_MODE_COSINE_SIM:
            cosine_sim_val = classification['cosine_sim'] if classification else 0.0
            reward_info = _compute_reward_cosine_sim(
                fret_position=fret_position,
                torque=torque,
                target_fret=target_fret,
                cosine_sim=cosine_sim_val,
                audio_rms=audio_rms,
            )
        elif self.reward_mode == REWARD_MODE_SPECTRAL:
            spectral_val = classification['spectral_score'] if classification else 0.0
            reward_info = _compute_reward_spectral(
                fret_position=fret_position,
                torque=torque,
                target_fret=target_fret,
                spectral_score=spectral_val,
                audio_rms=audio_rms,
            )
        else:  # REWARD_MODE_FULL
            reward_info = _compute_reward(
                fret_position=fret_position,
                torque=torque,
                target_fret=target_fret,
                harmonic_prob=harmonic_prob,
                audio_rms=audio_rms,
            )
        
        reward_info['classification'] = classification
        reward_info['audio_rms'] = audio_rms
        return reward_info
    
    def get_success_threshold(self) -> float:
        """Get threshold for successful harmonic (harmonic_prob)."""
        return SUCCESS_THRESHOLD
    
    def is_success(self, classification: Dict[str, float]) -> bool:
        """Check if classification indicates successful harmonic."""
        return _is_success(classification['harmonic_prob'])
