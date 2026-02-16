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
    compute_reward as _compute_reward,
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
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize reward calculator.
        
        Args:
            model_path: Path to trained harmonic classifier model (.pt file)
            device_name: Audio input device name (VB-CABLE)
            capture_duration: Audio capture duration in seconds
            model_sr: Sample rate expected by model
            device: Torch device (cuda/cpu)
        """
        self.model_path = Path(model_path)
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
            logger.warning(f"Audio device '{device_name}' not found. Audio capture may fail.")
        
        # Get device sample rate
        if self.device_id is not None:
            device_info = sd.query_devices(self.device_id, 'input')
            self.device_sr = int(device_info['default_samplerate'])
            logger.info(f"Audio device SR: {self.device_sr} Hz")
        else:
            self.device_sr = 44100  # Default fallback
        
        # Load model
        self.model = self._load_model()
        logger.info(f"HarmonicRewardCalculator initialized with model: {model_path}")
    
    def _find_audio_device(self) -> Optional[int]:
        """Find audio input device by name."""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                logger.info(f"Found audio device: {device['name']} (ID: {idx})")
                return idx
        return None
    
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
            audio = sd.rec(
                int(duration * self.device_sr),
                samplerate=self.device_sr,
                channels=1,
                device=self.device_id,
                dtype='float32'
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            return np.zeros(int(self.device_sr * duration))
    
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
        # Audio-based classification
        classification = None
        harmonic_prob = 0.0
        
        if audio is None and capture_audio:
            audio = self.capture_audio()
        
        if audio is not None:
            classification = self.classify_audio(audio)
            harmonic_prob = classification['harmonic_prob']
        
        # Delegate to shared reward function
        reward_info = _compute_reward(
            fret_position=fret_position,
            torque=torque,
            target_fret=target_fret,
            harmonic_prob=harmonic_prob,
        )
        reward_info['classification'] = classification
        return reward_info
    
    def get_success_threshold(self) -> float:
        """Get threshold for successful harmonic (harmonic_prob)."""
        return SUCCESS_THRESHOLD
    
    def is_success(self, classification: Dict[str, float]) -> bool:
        """Check if classification indicates successful harmonic."""
        return _is_success(classification['harmonic_prob'])
