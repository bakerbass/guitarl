"""
Test script for RL action -> audio recording -> classification -> reward loop.

This script demonstrates the complete RL feedback loop:
1. Generate an RL action (fret position + torque)
2. Send OSC command to robot/simulator
3. Record audio during playback
4. Classify audio as harmonic/dead/general
5. Compute reward based on classification

Usage:
    python test_rl_loop.py --model ../HarmonicsClassifier/models/best_model.pt
    python test_rl_loop.py --model ../HarmonicsClassifier/models/best_model.pt --num-tests 10
    python test_rl_loop.py --model ../HarmonicsClassifier/models/best_model.pt --target-harmonic
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import sounddevice as sd
import librosa
import time
from pythonosc import udp_client
import matplotlib.pyplot as plt

# Add HarmonicsClassifier to path
HARMONICS_CLASSIFIER_PATH = Path(__file__).parent.parent / "HarmonicsClassifier"
sys.path.insert(0, str(HARMONICS_CLASSIFIER_PATH))

from osc_realtime_classifier import HarmonicsCNN
from env.action_space import (
    RLFretAction,
    PresserAction,
    GuitarBotActionSpace,
    PLAYABLE_STRINGS,
    HARMONIC_FRETS_IN_RANGE,
    TORQUE_LIGHT,
    TORQUE_NORMAL,
    TORQUE_SAFE_MIN,
)

# Shared reward function — single source of truth for both test & training
from utils.reward import (
    compute_reward_nearest_fret,
    CLASS_NAMES as REWARD_CLASS_NAMES,
    HARMONIC_FRETS,
)


class RLTestLoop:
    """Test loop for RL action -> audio -> reward pipeline."""
    
    def __init__(self, model_path, device, audio_device_id, 
                 osc_host='127.0.0.1', osc_port=12000,
                 record_duration=3.0, plot_enabled=False):
        """
        Initialize test loop.
        
        Args:
            model_path: Path to trained harmonic classifier
            device: Torch device (cuda/cpu)
            audio_device_id: Audio input device ID
            osc_host: OSC server host (GuitarBot receiver)
            osc_port: OSC server port
            record_duration: Audio recording duration in seconds
            plot_enabled: Whether to plot audio waveforms and spectrograms
        """
        self.device = device
        self.audio_device_id = audio_device_id
        self.record_duration = record_duration
        self.plot_enabled = plot_enabled
        self.model_sr = 22050  # Model's expected sample rate
        
        # Get audio device info
        device_info = sd.query_devices(audio_device_id)
        self.record_sr = int(device_info['default_samplerate'])
        print(f"Audio device: {device_info['name']}")
        print(f"  Sample rate: {self.record_sr} Hz")
        print(f"  Model expects: {self.model_sr} Hz")
        
        # Audio preprocessing parameters
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        # Load classifier model
        print(f"\nLoading classifier model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = HarmonicsCNN(num_classes=3, dropout=0.5).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded (epoch {checkpoint['epoch'] + 1})")
        
        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)
        print(f"✓ OSC client: {osc_host}:{osc_port}")
        
        # Action space
        self.action_space = GuitarBotActionSpace(use_normalized=True)
        
        # Class names (from shared reward module)
        self.class_names = list(REWARD_CLASS_NAMES)
        
        # Statistics
        self.test_results = []
    
    def send_action(self, action: RLFretAction, pluck_velocity=None):
        """
        Send RL action via OSC.
        
        Args:
            action: RLFretAction to execute
            pluck_velocity: Optional pluck velocity (0-127)
        """
        # Build OSC message: /RLFret <string_idx> <fret_position> <torque> [velocity]
        osc_args = list(action.to_osc_args())
        if pluck_velocity is not None:
            osc_args.append(pluck_velocity)
        
        self.osc_client.send_message("/RLFret", osc_args)
        print(f"→ OSC: /RLFret {osc_args}")
    
    def record_audio(self, duration=None):
        """Record audio from input device."""
        if duration is None:
            duration = self.record_duration
        
        print(f"Recording {duration}s...")
        recording = sd.rec(
            int(duration * self.record_sr),
            samplerate=self.record_sr,
            channels=1,
            device=self.audio_device_id,
            dtype='float32'
        )
        sd.wait()
        
        # Get as 1D array
        audio = recording[:, 0]
        
        # Resample to model sample rate if needed
        if self.record_sr != self.model_sr:
            audio = librosa.resample(audio, orig_sr=self.record_sr, target_sr=self.model_sr)
        
        return audio
    
    def preprocess_audio(self, audio):
        """Preprocess audio to mel spectrogram."""
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        
        # If too short, use original
        if len(audio_trimmed) < self.model_sr * 0.1:
            audio_trimmed = audio
        
        # Pad or trim to exact duration
        target_length = int(self.model_sr * self.record_duration)
        if len(audio_trimmed) < target_length:
            audio_trimmed = np.pad(audio_trimmed, (0, target_length - len(audio_trimmed)))
        else:
            audio_trimmed = audio_trimmed[:target_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_trimmed,
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
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
        return mel_tensor
    
    def classify_audio(self, audio_tensor):
        """Run classifier inference."""
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def plot_audio_and_spectrogram(self, audio, mel_spec_db, predicted_label, confidence,
                                     reward=None, reward_components=None, action=None,
                                     test_num=None):
        """Plot audio waveform and mel spectrogram with classification + reward in title."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Build suptitle with classification and reward
        title_parts = []
        if test_num is not None:
            title_parts.append(f'Test #{test_num}')
        if action is not None:
            title_parts.append(f'String {action.string_idx}  Fret {action.fret_position:.2f}  Torque {action.torque:.0f}')
        suptitle = ' | '.join(title_parts) if title_parts else ''
        
        reward_str = ''
        if reward is not None:
            reward_str = f'  |  Reward: {reward:+.3f}'
            if reward_components:
                reward_str += (f'  (audio={reward_components["audio"]:.2f}'
                               f'  fret={reward_components["fret"]:.2f}'
                               f'  torque={reward_components["torque"]:.2f})')
        
        if suptitle:
            fig.suptitle(suptitle, fontsize=11, fontweight='bold')
        
        # Plot waveform
        time_axis = np.arange(len(audio)) / self.model_sr
        axes[0].plot(time_axis, audio, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Classified as: {predicted_label.upper()} ({confidence*100:.1f}%){reward_str}')
        axes[0].grid(True, alpha=0.3)
        
        # Plot mel spectrogram
        img = axes[1].imshow(
            mel_spec_db,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, self.record_duration, 0, self.n_mels]
        )
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mel Frequency Bin')
        axes[1].set_title('Mel Spectrogram (dB)')
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def compute_reward(self, action: RLFretAction, predicted_class, confidence, probabilities,
                       audio: Optional[np.ndarray] = None):
        """
        Compute reward using the shared two-layer reward function (utils/reward.py).
        """
        harmonic_prob = float(probabilities[0])  # index 0 = harmonic
        audio_rms = float(np.sqrt(np.mean(audio ** 2))) if audio is not None else None
        reward_info = compute_reward_nearest_fret(
            fret_position=action.fret_position,
            torque=action.torque,
            harmonic_prob=harmonic_prob,
            audio_rms=audio_rms,
        )
        
        # Store components for display / plotting
        self._last_reward_components = {
            'audio':  reward_info['audio_reward'],
            'fret':   reward_info['fret_reward'],
            'torque': reward_info['torque_reward'],
            'fret_error':  reward_info['fret_error'],
            'torque_error': reward_info['torque_error'],
            'nearest_harmonic_fret': reward_info['nearest_harmonic_fret'],
            'filtered': reward_info.get('filtered', False),
            'filter_reason': reward_info.get('filter_reason', ''),
        }
        
        return reward_info['total_reward']
    
    def run_test(self, action: RLFretAction, pluck_velocity=None, pre_delay=0.2):
        """
        Run one complete test iteration.
        
        Args:
            action: RLFretAction to test
            pluck_velocity: Optional pluck velocity
            pre_delay: Delay before recording (to allow motor movement)
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*70)
        print(f"TEST: String {action.string_idx}, Fret {action.fret_position:.2f}, Torque {action.torque:.0f}")
        print("="*70)
        
        # Send action
        self.send_action(action, pluck_velocity)
        
        # Small delay for motor movement and pluck
        time.sleep(pre_delay)
        
        # Record audio
        audio = self.record_audio()
        
        # Preprocess and classify
        audio_tensor = self.preprocess_audio(audio)
        predicted_class, confidence, probabilities = self.classify_audio(audio_tensor)
        
        # Compute reward
        reward = self.compute_reward(action, predicted_class, confidence, probabilities, audio=audio)
        reward_components = getattr(self, '_last_reward_components', None)
        
        # Display results
        predicted_label = self.class_names[predicted_class]
        
        if reward_components and reward_components.get('filtered'):
            print(f"\n→ FILTERED: {reward_components['filter_reason']}")
            print(f"→ Reward: {reward:+.3f} (filtration penalty)")
        else:
            print(f"\n→ Classification: {predicted_label.upper()} ({confidence*100:.1f}%)")
            print(f"→ Probabilities:")
            for i, (name, prob) in enumerate(zip(self.class_names, probabilities)):
                marker = "★" if i == predicted_class else " "
                print(f"   {marker} {name:12s}: {prob*100:5.1f}%")
            print(f"→ Reward: {reward:+.3f}")
            if reward_components:
                print(f"   audio={reward_components['audio']:.3f}  "
                      f"fret={reward_components['fret']:.3f} (err={reward_components['fret_error']:.2f}, "
                      f"nearest={reward_components['nearest_harmonic_fret']})  "
                      f"torque={reward_components['torque']:.3f} (err={reward_components['torque_error']:.1f})")
        
        # Plot individual test note (always unless --no-plot)
        if self.plot_enabled:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            if len(audio_trimmed) < self.model_sr * 0.1:
                audio_trimmed = audio
            target_length = int(self.model_sr * self.record_duration)
            if len(audio_trimmed) < target_length:
                audio_trimmed = np.pad(audio_trimmed, (0, target_length - len(audio_trimmed)))
            else:
                audio_trimmed = audio_trimmed[:target_length]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio_trimmed, sr=self.model_sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels, fmin=80, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            test_num = len(self.test_results) + 1
            self.plot_audio_and_spectrogram(
                audio_trimmed, mel_spec_db, predicted_label, confidence,
                reward=reward, reward_components=reward_components,
                action=action, test_num=test_num
            )
        
        # Store results
        result = {
            'action': action.to_dict(),
            'predicted_class': predicted_label,
            'confidence': float(confidence),
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probabilities)},
            'reward': float(reward),
            'reward_components': reward_components,
            'is_harmonic_fret': action.is_at_harmonic,
        }
        self.test_results.append(result)
        
        return result
    
    def run_random_tests(self, num_tests=5, delay_between=4.0):
        """Run multiple random tests."""
        print(f"\n{'='*70}")
        print(f"RUNNING {num_tests} RANDOM TESTS")
        print(f"{'='*70}")
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            # Sample random action
            action = self.action_space.sample()
            
            # Run test
            self.run_test(action)
            
            # Delay between tests
            if i < num_tests - 1:
                print(f"\nWaiting {delay_between}s before next test...")
                time.sleep(delay_between)
    
    def run_harmonic_tests(self, num_tests=5, delay_between=4.0):
        """Run tests targeting harmonic positions."""
        print(f"\n{'='*70}")
        print(f"RUNNING {num_tests} HARMONIC-TARGETED TESTS")
        print(f"{'='*70}")
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            # Sample harmonic action
            action = self.action_space.sample_harmonic()
            
            # Run test
            self.run_test(action)
            
            # Delay between tests
            if i < num_tests - 1:
                print(f"\nWaiting {delay_between}s before next test...")
                time.sleep(delay_between)
    
    def run_baseline_tests(self, delay_between=4.0):
        """
        Run a fixed set of known-good actions to verify the classifier is working.

        These actions are chosen because they reliably produce clear harmonics
        on the physical robot.  If the classifier scores them poorly, the issue
        is in the audio pipeline or the model — not the RL training.
        """
        # (string_idx, fret_position, torque)  — all known harmonic positions
        BASELINE_ACTIONS = [
            (2, 7.2,  60),   # String 2, fret 7 — perfect 12th, very light touch
            (2, 7.3,  38),   # String 2, fret 7 — slightly lighter
            (2, 7.2,  120),   # String 2, fret 7 — slightly heavier
            (2, 5.3,  68),   # String 2, fret 5 — major 14th
            (2, 4.2,  65),   # String 2, fret 4 — major 17th
        ]

        print(f"\n{'='*70}")
        print(f"RUNNING {len(BASELINE_ACTIONS)} BASELINE (KNOWN-GOOD) TESTS")
        print("These should all classify as HARMONIC if the pipeline is working.")
        print(f"{'='*70}")

        for i, (string_idx, fret, torque) in enumerate(BASELINE_ACTIONS):
            print(f"\n--- Baseline {i+1}/{len(BASELINE_ACTIONS)}: "
                  f"string={string_idx}  fret={fret}  torque={torque} ---")
            action = RLFretAction(
                string_idx=string_idx,
                fret_position=fret,
                press_action=PresserAction.PRESS,
                torque=torque,
            )
            self.run_test(action)

            if i < len(BASELINE_ACTIONS) - 1:
                print(f"\nWaiting {delay_between}s before next test...")
                time.sleep(delay_between)

    def print_summary(self):
        """Print summary statistics."""
        if not self.test_results:
            print("\nNo tests run yet.")
            return
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = len(self.test_results)
        harmonics = sum(1 for r in self.test_results if r['predicted_class'] == 'harmonic')
        dead = sum(1 for r in self.test_results if r['predicted_class'] == 'dead_note')
        general = sum(1 for r in self.test_results if r['predicted_class'] == 'general_note')
        
        avg_reward = np.mean([r['reward'] for r in self.test_results])
        avg_confidence = np.mean([r['confidence'] for r in self.test_results])
        
        print(f"Total tests: {total}")
        print(f"\nClassification distribution:")
        print(f"  Harmonic:     {harmonics:2d} ({harmonics/total*100:5.1f}%)")
        print(f"  Dead note:    {dead:2d} ({dead/total*100:5.1f}%)")
        print(f"  General note: {general:2d} ({general/total*100:5.1f}%)")
        print(f"\nAverage reward:     {avg_reward:+.3f}")
        print(f"Average confidence: {avg_confidence*100:.1f}%")
        
        # Best and worst
        best = max(self.test_results, key=lambda r: r['reward'])
        worst = min(self.test_results, key=lambda r: r['reward'])
        
        print(f"\nBest result:  {best['predicted_class']:12s} (reward: {best['reward']:+.3f})")
        print(f"              Fret {best['action']['fret_position']:.2f}, Torque {best['action']['torque']:.0f}")
        print(f"\nWorst result: {worst['predicted_class']:12s} (reward: {worst['reward']:+.3f})")
        print(f"              Fret {worst['action']['fret_position']:.2f}, Torque {worst['action']['torque']:.0f}")
        
        print("="*70)


def select_audio_device(preferred_substring="Scarlett"):
    """
    Select audio input device. Prefer names containing preferred_substring (e.g., "Scarlett" for Focusrite).
    If none found, list input devices and prompt user for selection.
    
    Args:
        preferred_substring: Substring to search for in device names (default: "Scarlett")
    
    Returns:
        Device ID (int) or None if user cancels
    """
    devices = sd.query_devices()
    preferred_substring_l = preferred_substring.lower()
    
    # Try to pick a Scarlett/Focusrite input device automatically
    for idx, dev in enumerate(devices):
        try:
            name = dev.get('name', '')
            max_in = dev.get('max_input_channels', 0)
        except Exception:
            # Some backends may return tuples; be defensive
            name = dev['name'] if isinstance(dev, dict) else str(dev)
            max_in = dev['max_input_channels'] if isinstance(dev, dict) else 0
        
        if max_in and max_in > 0 and preferred_substring_l in name.lower():
            print(f"\n✓ Auto-selected input device: [{idx}] {name} (inputs: {max_in})")
            return idx
    
    # No preferred device found, list inputs and prompt
    print(f"\nNo '{preferred_substring}' input device found. Available input devices:")
    print("="*70)
    
    input_devices = []
    for idx, dev in enumerate(devices):
        try:
            name = dev.get('name', '')
            max_in = dev.get('max_input_channels', 0)
        except Exception:
            name = dev['name'] if isinstance(dev, dict) else str(dev)
            max_in = dev['max_input_channels'] if isinstance(dev, dict) else 0
        
        if max_in and max_in > 0:
            input_devices.append((idx, name, max_in))
            print(f"  [{idx}] {name} (inputs: {max_in})")
    
    print("="*70)
    
    if not input_devices:
        print("Error: No input-capable devices found.")
        return None
    
    while True:
        sel = input("\nEnter input device index to use (or 'q' to quit): ").strip()
        if sel.lower() == 'q':
            print("Cancelled")
            return None
        
        try:
            sel_idx = int(sel)
            # Check if this index is in our input devices list
            if any(idx == sel_idx for idx, _, _ in input_devices):
                dev_name = next(name for idx, name, _ in input_devices if idx == sel_idx)
                print(f"\n✓ Selected: [{sel_idx}] {dev_name}")
                return sel_idx
            else:
                print("Selected device has no input channels or is invalid. Choose one of the listed indices.")
        except ValueError:
            print("Please enter a valid integer index from the list above.")
        except KeyboardInterrupt:
            print("\nCancelled")
            return None


def main():
    parser = argparse.ArgumentParser(description='Test RL action -> audio -> reward loop')
    parser.add_argument('--model', default='../HarmonicsClassifier/models/best_model.pt',
                        help='Path to trained classifier model')
    parser.add_argument('--osc-host', default='127.0.0.1', help='OSC server host')
    parser.add_argument('--osc-port', type=int, default=12000, help='OSC server port')
    parser.add_argument('--duration', type=float, default=3.0, help='Recording duration (seconds)')
    parser.add_argument('--num-tests', type=int, default=5, help='Number of tests to run')
    parser.add_argument('--target-harmonic', action='store_true', 
                        help='Target harmonic positions instead of random')
    parser.add_argument('--delay', type=float, default=4.0, help='Delay between tests (seconds)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable per-note plotting (plots shown by default)')
    parser.add_argument('--baseline', action='store_true',
                        help='Run fixed known-good actions to verify the classifier pipeline '
                             '(overrides --target-harmonic and --num-tests)')

    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"Expected at: {model_path.absolute()}")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Select audio device
    audio_device_id = select_audio_device()
    if audio_device_id is None:
        print("No audio device selected. Exiting.")
        return
    
    # Create test loop
    print("\nInitializing test loop...")
    test_loop = RLTestLoop(
        model_path=model_path,
        device=device,
        audio_device_id=audio_device_id,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        record_duration=args.duration,
        plot_enabled=not args.no_plot
    )
    
    try:
        # Run tests
        if args.baseline:
            test_loop.run_baseline_tests(args.delay)
        elif args.target_harmonic:
            test_loop.run_harmonic_tests(args.num_tests, args.delay)
        else:
            test_loop.run_random_tests(args.num_tests, args.delay)
        
        # Print summary
        test_loop.print_summary()
        
    except KeyboardInterrupt:
        test_loop.osc_client.send_message("/Reset", [])
        print("\n\nInterrupted by user")
    finally:
        test_loop.osc_client.send_message("/Reset", [])
        print("\nTest complete!")


if __name__ == "__main__":
    main()
