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

from utils.audio_buffer import RollingAudioBuffer


# ---------------------------------------------------------------------------
# Rolling-buffer / onset-detection constants
# ---------------------------------------------------------------------------
ROLLING_BUFFER_DURATION = 60.0   # seconds of rolling audio history
ONSET_PRE_ROLL          = 0.05   # seconds before detected onset to include
ONSET_POST_ROLL         = 1.5    # seconds after onset (CNN window duration)
ONSET_THRESHOLD_FACTOR  = 8.0    # energy multiple above background → onset
ONSET_SEARCH_WINDOW     = 3.0    # seconds to search after action timestamp
ONSET_TIMEOUT           = 5.0    # fallback after this many seconds
LATENCY_EMA_ALPHA       = 0.1    # EMA smoothing factor for latency estimate


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
    compute_reward as _compute_reward,
    compute_reward_no_filtration as _compute_reward_no_filtration,
    compute_reward_no_audio as _compute_reward_no_audio,
    is_success as _is_success,
)


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
                 model_path: str,
                 device_name: str = "VB-Audio Virtual Cable",
                 capture_duration: float = 1.0,
                 model_sr: int = 22050,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 reward_mode: str = REWARD_MODE_FULL,
                 temperature: float = 1.5):
        """
        Initialize reward calculator.
        
        Args:
            model_path: Path to trained harmonic classifier model (.pt file)
            device_name: Audio input device name (VB-CABLE)
            capture_duration: Audio capture duration in seconds
            model_sr: Sample rate expected by model
            device: Torch device (cuda/cpu)
            reward_mode: One of 'full', 'no_filtration', 'no_audio'.
                         'full'           — two-layer reward (default)
                         'no_filtration'  — bypass physics gate, Layer 2 only
                         'no_audio'       — Layer 1 + fret/torque shaping, no CNN
            temperature: Temperature for logit scaling before softmax (default 1.5).
                         Values > 1 produce softer, less overconfident probabilities.
        """
        self.model_path = Path(model_path)
        self.device_name = device_name
        self.capture_duration = capture_duration
        self.model_sr = model_sr
        self.device = torch.device(device)
        self.temperature = temperature
        
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
        
        # Load model
        self.model = self._load_model()
        self.reward_mode = reward_mode
        logger.info(f"HarmonicRewardCalculator initialized with model: {model_path}, reward_mode={reward_mode}, temperature={temperature}")

        # Rolling audio buffer (always-on background recording)
        self._rolling_buffer = RollingAudioBuffer(
            device_id=self.device_id,
            device_sr=self.device_sr,
            buffer_duration=ROLLING_BUFFER_DURATION,
        )
        self._rolling_buffer.start()

        # Latency estimate: EMA of (onset_time - action_time)
        self._latency_ema: float = 0.5  # Sensible initial guess (seconds)

        logger.info(f"HarmonicRewardCalculator initialized with model: {model_path}, reward_mode={reward_mode}")

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
    
    @property
    def latency_ema(self) -> float:
        """Current exponential moving average of action-to-onset latency (seconds)."""
        return self._latency_ema

    def capture_audio_from_buffer(self, t_action: float) -> np.ndarray:
        """
        Onset-aligned audio capture using the rolling buffer.

        Waits for a note onset after ``t_action``, then extracts a window
        centred on the onset for CNN classification.  Updates the latency EMA.

        Parameters
        ----------
        t_action : float
            Monotonic wall-clock timestamp (``time.monotonic()``) of the OSC
            send that triggered the robot pluck.

        Returns
        -------
        np.ndarray
            Audio at device sample rate, length ≈ ONSET_PRE_ROLL + ONSET_POST_ROLL.
        """
        t_onset = self._rolling_buffer.wait_for_onset(
            after_time=t_action,
            search_window=ONSET_SEARCH_WINDOW,
            timeout=ONSET_TIMEOUT,
            threshold_factor=ONSET_THRESHOLD_FACTOR,
            fallback_delay=self._latency_ema,
        )

        # Update latency EMA
        measured_latency = t_onset - t_action
        self._latency_ema = (
            LATENCY_EMA_ALPHA * measured_latency
            + (1.0 - LATENCY_EMA_ALPHA) * self._latency_ema
        )
        logger.debug(
            f"Onset latency: {measured_latency:.3f}s  EMA: {self._latency_ema:.3f}s"
        )

        audio = self._rolling_buffer.get_audio_range(
            t_onset - ONSET_PRE_ROLL,
            t_onset + ONSET_POST_ROLL,
        )
        return audio

    def calibrate_silence_threshold(
        self,
        duration: float = 2.0,
        multiplier: float = 3.0,
    ) -> float:
        """
        Measure ambient noise floor and return a suitable silence RMS threshold.

        Delegates to ``RollingAudioBuffer.calibrate_silence_threshold()``.
        Call this once at startup (before any robot actions) when using
        ``--silence-rms auto``.

        Parameters
        ----------
        duration : float
            Seconds of quiescent audio to sample.
        multiplier : float
            Scale factor above the noise floor (default 3.0).

        Returns
        -------
        float
            Recommended RMS threshold for ``wait_for_silence()``.
        """
        return self._rolling_buffer.calibrate_silence_threshold(
            duration=duration,
            multiplier=multiplier,
        )

    def wait_for_silence(
        self,
        rms_threshold: float = 0.005,
        hold_duration: float = 0.5,
        timeout: float = 8.0,
    ) -> bool:
        """
        Block until the audio signal has been below ``rms_threshold`` for
        ``hold_duration`` continuous seconds (delegates to rolling buffer).

        Returns True when silence confirmed, False on timeout.
        """
        return self._rolling_buffer.wait_for_silence(
            rms_threshold=rms_threshold,
            hold_duration=hold_duration,
            timeout=timeout,
        )

    def close(self) -> None:
        """Release any held audio resources. Safe to call multiple times."""
        try:
            self._rolling_buffer.stop()
        except Exception:
            pass
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
        
        # Inference with temperature scaling
        with torch.no_grad():
            logits = self.model(audio_tensor)
            probabilities = torch.softmax(logits / self.temperature, dim=1)
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
        else:  # REWARD_MODE_FULL (default)
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
