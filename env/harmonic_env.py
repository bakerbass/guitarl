"""
Gymnasium environment for GuitarBot harmonic RL training.

Provides reinforcement learning interface for learning to play natural harmonics.
Uses FRACTIONAL FRETS for position encoding to help the agent learn musical structure.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import time
import sys
from typing import Dict, Tuple, Optional
from pathlib import Path

from .osc_client import GuitarBotOSCClient
from .action_space import (
    GuitarBotActionSpace,
    RLFretAction,
    PresserAction,
    fret_to_mm,
    mm_to_fret,
    PLAYABLE_STRINGS,
    STRING_TO_PLUCKER,
    FRET_MIN,
    FRET_MAX,
    TORQUE_MIN,
    TORQUE_MAX,
    TORQUE_SAFE_MIN,
    TORQUE_LIGHT,
    TORQUE_NORMAL,
    HARMONIC_FRETS_IN_RANGE,
    SLIDER_MAX_MM,
)

# Add parent directory to path for utils import
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from utils.audio_reward import HarmonicRewardCalculator
from utils.reward import (
    REWARD_MODE_FULL, REWARD_MODE_NO_FILTRATION, REWARD_MODE_NO_AUDIO,
    compute_reward_no_audio as _compute_reward_no_audio,
    compute_filtration,
    FILTRATION_PENALTY,
    PRETRAIN_FRET_TOLERANCE,
)
from utils.success_recorder import SuccessRecorder


logger = logging.getLogger(__name__)


class HarmonicEnv(gym.Env):
    """
    Gymnasium environment for learning guitar harmonics.
    
    The agent learns to find the correct fret position and torque to produce
    clear natural harmonics at frets 4, 5, and 7.
    
    Uses FRACTIONAL FRETS for position to encode musical structure, helping
    the agent learn that frets 4, 5, 7 are harmonic nodes.
    
    Observation Space:
        - target_fret_one_hot: Which harmonic to play (4, 5, or 7) [3]
        - string_one_hot: Which string is active this episode (0, 2, 4) [3]
        - current_fret: Last commanded fret position [1]
        - current_torque: Last commanded torque (normalized) [1]
        - fret_history: Last N fret positions [3]
        - torque_history: Last N torque values [3]
        Total: 14 dimensions

        Adding string context lets a single policy learn string-specific
        fret/torque mappings while episodes rotate across strings to
        distribute motor wear.
        
    Action Space (normalized, shape=(5,) with always_press=True):
        - string_logits: String selection [3] (argmax -> 0, 2, 4)
        - fret: Fractional fret position [1] (scaled from [-1,1] to [0,9])
        - torque_magnitude: Fretting torque [1] (scaled from [-1,1] to [16,650])
        When always_press=False, adds press_decision [1] (shape=(6,)).
        
    Reward:
        Combination of:
        - Harmonic classifier confidence (0-1) [weight: 0.2]
        - Fret position accuracy (Gaussian) [weight: 0.3]
        - Torque optimization (shifted Gaussian, [-1,+1]) [weight: 0.5]
    """
    
    metadata = {'render_modes': ['human']}
    
    # Environment constants
    HARMONIC_FRETS = HARMONIC_FRETS_IN_RANGE  # [4, 5, 7]
    MAX_STEPS_PER_EPISODE = 10
    ACTION_DURATION = 5.0       # Total step duration (seconds): 0.5 pre + 4.0 capture + 0.5 pad
    CAPTURE_PRE_DELAY = 0.5     # Wait after send_rlfret before recording (robot arm movement + pluck)
    STRING_SWITCH_WAIT = 4.0    # Seconds to pause after a /Reset between strings

    # Silence gate: wait for harmonic bleed to decay before the next step.
    # silence_rms_threshold is stored as an instance variable so train.py can
    # override it with --silence-rms auto or a numeric value after construction.
    SILENCE_RMS_THRESHOLD = 0.005  # default; overridden by instance var below
    SILENCE_HOLD_DURATION = 0.5    # seconds of continuous quiet required
    SILENCE_TIMEOUT       = 8.0    # max wait before proceeding anyway
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 string_index: int = 2,  # Default to D string — used when string_indices is None
                 string_indices=None,    # List of strings to rotate across (e.g. [0,2,4])
                 osc_host: str = "127.0.0.1",
                 osc_port: int = 12000,
                 audio_device: str = "Scarlett",
                 capture_duration: float = 4.0,
                 max_steps: int = MAX_STEPS_PER_EPISODE,
                 success_threshold: float = 0.8,
                 curriculum_mode: str = "random",
                 fixed_target_fret: int = 7,
                 use_simple_action_space: bool = False,
                 always_press: bool = True,
                 reward_mode: str = REWARD_MODE_FULL,
                 offline: bool = False,
                 success_recorder: Optional['SuccessRecorder'] = None):
        """
        Initialize HarmonicEnv.

        Args:
            model_path: Path to HarmonicsClassifier model. Not required when offline=True.
            string_index: Fallback single string (0, 2, or 4). Ignored when string_indices is set.
            string_indices: List of strings to rotate across each episode (e.g. [0, 2, 4]).
                            Sampled uniformly at each reset(). Distributes motor wear while
                            keeping the policy string-aware via the observation one-hot.
                            Defaults to [string_index] (single-string mode, backward compat).
            osc_host: StringSim OSC host
            osc_port: StringSim OSC port
            audio_device: VB-CABLE device name
            capture_duration: Audio capture duration
            max_steps: Maximum steps per episode
            success_threshold: Harmonic probability threshold for success
            curriculum_mode: "random", "easy_to_hard", or "fixed_fret"
            fixed_target_fret: Fret locked for "fixed_fret" curriculum (default 7).
                               Must be in HARMONIC_FRETS [4, 5, 7].
            use_simple_action_space: If True, use 3D continuous space instead of 5D
            always_press: If True, remove press_decision from action space (always PRESS)
            reward_mode: 'full', 'no_filtration', or 'no_audio' (see reward.py)
            offline: If True, skip all robot/audio hardware. Reward comes from the
                     filtration layer only (fret + torque shaping). Use for fast
                     pre-training before deploying on the physical robot.
        """
        super().__init__()

        self.offline = offline
        self.success_recorder = success_recorder
        # Instance-level silence threshold — can be updated by train.py after
        # construction (e.g. --silence-rms auto calibration).
        self.silence_rms_threshold: float = self.SILENCE_RMS_THRESHOLD
        # Resolve string pool — validate every entry has a plucker
        if string_indices is not None:
            for s in string_indices:
                if s not in PLAYABLE_STRINGS:
                    raise ValueError(f"string_indices contains {s}, must be in {PLAYABLE_STRINGS}")
            self.string_indices = list(string_indices)
        else:
            if string_index not in PLAYABLE_STRINGS:
                raise ValueError(f"string_index must be in {PLAYABLE_STRINGS} (strings with pluckers)")
            self.string_indices = [string_index]
        
        # Active string for the current episode (set at reset)
        self.string_index = self.string_indices[0]
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.curriculum_mode = curriculum_mode
        if fixed_target_fret not in self.HARMONIC_FRETS:
            raise ValueError(
                f"fixed_target_fret={fixed_target_fret} is not in HARMONIC_FRETS {self.HARMONIC_FRETS}"
            )
        self.fixed_target_fret = fixed_target_fret
        self.use_simple_action_space = use_simple_action_space
        self.always_press = always_press
        
        # Initialize action space helper
        self.action_space_helper = GuitarBotActionSpace(use_normalized=True)

        if self.offline:
            # Offline pre-training: no robot, no audio hardware needed.
            # Reward is computed purely from the filtration layer.
            self.osc_client = None
            self.reward_calc = None
            logger.info(
                "HarmonicEnv running in OFFLINE pre-training mode. "
                "No robot or audio connection will be made. "
                "Reward = filtration layer only (fret + torque shaping)."
            )
        else:
            if model_path is None:
                raise ValueError("model_path is required when offline=False")
            # Initialize OSC client
            self.osc_client = GuitarBotOSCClient(host=osc_host, port=osc_port)
            # Initialize reward calculator
            self.reward_calc = HarmonicRewardCalculator(
                model_path=model_path,
                device_name=audio_device,
                capture_duration=capture_duration,
                reward_mode=reward_mode,
            )
        
        # Define action space dimensions
        # When always_press=True, press_decision is removed (always PRESS)
        if use_simple_action_space:
            # [string_continuous, fret, (press_decision), torque_magnitude]
            action_dim = 3 if self.always_press else 4
        else:
            # [string_logits(3), fret, (press_decision), torque_magnitude]
            action_dim = 5 if self.always_press else 6
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Define observation space
        # [target_fret_one_hot(3), string_one_hot(3), current_fret, current_torque_norm, fret_history(3), torque_history(3)]
        obs_dim = 3 + 3 + 1 + 1 + 3 + 3  # = 14
        self.observation_space = spaces.Box(
            low=-1.0,
            high=10.0,  # frets go up to 9
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.target_fret = None
        self.current_fret = 0.0
        self.current_torque = 0.0
        self.fret_history = []
        self.torque_history = []
        self.episode_rewards = []
        
        # Track previous string to detect switches between episodes
        self._prev_string_index = None
        
        # Curriculum learning state
        self.episode_count = 0
        self.curriculum_fret_idx = 0  # Start with easiest (fret 7)

        # Last-step data exposed to callbacks (e.g. --slow plot mode in train.py)
        self.last_audio: Optional[np.ndarray] = None
        self.last_rl_action: Optional[object] = None
        self.last_reward_info: Optional[dict] = None

        # Count of steps where OSC was actually sent (filtered steps excluded).
        # Used by RobotLearningStartCallback to trigger learning after N real
        # robot actions rather than N total timesteps.
        self.robot_step_count: int = 0
        
        logger.info(f"HarmonicEnv initialized: strings={self.string_indices}, max_steps={max_steps}, "
                    f"action_dim={action_dim}, obs_dim={obs_dim}, always_press={always_press}, "
                    f"reward_mode={reward_mode}, offline={offline}")
    
    def _get_target_fret(self) -> int:
        """Select target fret based on curriculum mode."""
        if self.curriculum_mode == "random":
            return int(np.random.choice(self.HARMONIC_FRETS))
        elif self.curriculum_mode == "easy_to_hard":
            # Fret 7 (easiest) -> Fret 5 -> Fret 4 (hardest).
            # Progress is gated on robot_step_count (real OSC actions) rather
            # than episode_count.  episode_count climbs instantly when the
            # random policy generates all-filtered episodes, causing the
            # curriculum to advance long before the agent has learned anything.
            # 500 real robot actions ≈ 25 min at ~3 s/step per fret stage.
            curriculum_order = [7, 5, 4]
            idx = min(self.robot_step_count // 500, len(curriculum_order) - 1)
            return curriculum_order[idx]
        elif self.curriculum_mode == "fixed_fret":
            return self.fixed_target_fret
        else:
            return int(np.random.choice(self.HARMONIC_FRETS))
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector (14-dim)."""
        # One-hot encode target fret (frets 4, 5, 7)
        target_one_hot = np.zeros(3, dtype=np.float32)
        fret_idx = self.HARMONIC_FRETS.index(self.target_fret)
        target_one_hot[fret_idx] = 1.0
        
        # One-hot encode active string (strings 0, 2, 4)
        string_one_hot = np.zeros(3, dtype=np.float32)
        string_one_hot[PLAYABLE_STRINGS.index(self.string_index)] = 1.0
        
        # Current state (fret position and normalized torque)
        torque_normalized = self.current_torque / TORQUE_MAX
        current_state = np.array([self.current_fret, torque_normalized], dtype=np.float32)
        
        # Action history (last 3 actions, pad if necessary)
        fret_history = self.fret_history[-3:] if len(self.fret_history) > 0 else [0.0] * 3
        torque_history = self.torque_history[-3:] if len(self.torque_history) > 0 else [0.0] * 3
        
        # Pad to length 3
        while len(fret_history) < 3:
            fret_history.insert(0, 0.0)
        while len(torque_history) < 3:
            torque_history.insert(0, 0.0)
        
        fret_history = np.array(fret_history, dtype=np.float32)
        torque_history = np.array(torque_history, dtype=np.float32) / TORQUE_MAX
        
        # [target_fret_one_hot(3), string_one_hot(3), current_fret, current_torque_norm, fret_history(3), torque_history(3)]
        obs = np.concatenate([target_one_hot, string_one_hot, current_state, fret_history, torque_history])
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Sample active string for this episode
        new_string_index = int(np.random.choice(self.string_indices))
        
        # If the string changed, reset the robot and wait for it to home safely.
        # In offline mode there is no robot, so this is skipped entirely.
        if (not self.offline
                and self._prev_string_index is not None
                and new_string_index != self._prev_string_index):
            logger.info(
                f"String switch: {self._prev_string_index} -> {new_string_index}. "
                f"Sending /Reset and waiting {self.STRING_SWITCH_WAIT:.0f}s..."
            )
            self.osc_client.reset(wait_time=0.5)
            time.sleep(self.STRING_SWITCH_WAIT)
            logger.info("String switch complete. Starting new episode.")
        
        self.string_index = new_string_index
        self._prev_string_index = new_string_index
        
        # Select target fret
        self.target_fret = self._get_target_fret()
        
        # Reset episode state
        self.current_step = 0
        self.current_fret = 0.0
        self.current_torque = 0.0
        self.fret_history = []
        self.torque_history = []
        self.episode_rewards = []
        
        self.episode_count += 1
        
        obs = self._get_observation()
        info = {
            'target_fret': self.target_fret,
            'episode': self.episode_count,
        }
        
        logger.debug(f"Episode {self.episode_count} reset: target_fret={self.target_fret}, string={self.string_index}")
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in environment."""
        # When always_press=True, inject press_decision=1.0 into the action
        # so from_normalized() always creates a PRESS action with valid torque
        if self.always_press:
            if self.use_simple_action_space:
                # [string, fret, torque] -> [string, fret, press=1.0, torque]
                action = np.insert(action, 2, 1.0)
            else:
                # [logits(3), fret, torque] -> [logits(3), fret, press=1.0, torque]
                action = np.insert(action, 4, 1.0)
        
        # Convert normalized action to RLFretAction
        if self.use_simple_action_space:
            rl_action = self.action_space_helper.from_simple_normalized(action)
        else:
            rl_action = self.action_space_helper.from_normalized(action)
        
        # Override string with environment's fixed string (for single-string training)
        rl_action = RLFretAction(
            string_idx=self.string_index,
            fret_position=rl_action.fret_position,
            press_action=rl_action.press_action,
            torque=rl_action.torque
        )
        
        fret_position = rl_action.fret_position
        torque = rl_action.torque

        step_header = (
            f"Step {self.current_step + 1}/{self.max_steps} | "
            f"str={self.string_index} target={self.target_fret} "
            f"fret={fret_position:.2f} torque={torque:.0f}"
        )

        if self.offline:
            # Offline pre-training: compute reward from filtration layer only.
            # No robot command, no physics wait, no audio capture.
            # Use PRETRAIN_FRET_TOLERANCE (wider Gaussian) so the agent receives
            # a gradient signal across the whole fret range, not just ±0.35 frets.
            reward_info = _compute_reward_no_audio(
                fret_position=fret_position,
                torque=torque,
                target_fret=self.target_fret,
                fret_tolerance=PRETRAIN_FRET_TOLERANCE,
            )
            reward_info['classification'] = None
            reward_info['audio_rms'] = None

            reward = reward_info['total_reward']
            filtered = reward_info.get('filtered', False)
            filter_reason = reward_info.get('filter_reason', '')
            if filtered:
                logger.debug(
                    f"\n  {step_header} [offline]\n"
                    f"    FILTERED: {filter_reason}\n"
                    f"    reward={reward:+.3f}"
                )
            else:
                logger.debug(
                    f"\n  {step_header} [offline]\n"
                    f"    fret={reward_info['fret_reward']:+.3f}  "
                    f"torque={reward_info['torque_reward']:+.3f}  "
                    f"reward={reward:+.3f}"
                )
        else:
            # Online mode: pre-check the action with the filtration layer BEFORE
            # sending any OSC command.  If it fails, return immediately —
            # no robot command, no audio capture, no 3-second wait.  This
            # stops the motors from chasing garbage actions during random
            # exploration and keeps filtered steps instant.
            # Skipped in no_filtration ablation mode where the gate is intentionally bypassed.
            pre_filter_active = (self.reward_calc.reward_mode != REWARD_MODE_NO_FILTRATION)
            filt = compute_filtration(fret_position, torque, self.target_fret) if pre_filter_active else {'passed': True}

            if not filt['passed']:
                filter_reason = filt['reason']
                reward = FILTRATION_PENALTY
                reward_info = {
                    'total_reward':  reward,
                    'audio_reward':  0.0,
                    'fret_reward':   0.0,
                    'torque_reward': 0.0,
                    'fret_error':    abs(fret_position - float(self.target_fret)),
                    'torque_error':  0.0,
                    'filtered':      True,
                    'filter_reason': filter_reason,
                    'classification': None,
                    'audio_rms':     None,
                }
                self.last_rl_action = rl_action
                self.last_reward_info = reward_info
                self.last_audio = None
                logger.debug(
                    f"\n  {step_header}\n"
                    f"    FILTERED (no OSC): {filter_reason}\n"
                    f"    reward={reward:+.3f}"
                )
            else:
                # Action passed filtration — send to robot, capture audio, compute reward.
                self.robot_step_count += 1

                # Wait for any harmonic bleed from the previous step to decay
                # before committing the next action.  Proceeds immediately if
                # already quiet or if no audio device is available.
                self.reward_calc.wait_for_silence(
                    rms_threshold=self.silence_rms_threshold,
                    hold_duration=self.SILENCE_HOLD_DURATION,
                    timeout=self.SILENCE_TIMEOUT,
                )

                self.osc_client.send_rlfret(rl_action)

                # Short pre-delay for the robot arm to physically move and pluck.
                time.sleep(self.CAPTURE_PRE_DELAY)

                # Capture audio now — the note is ringing.
                audio = self.reward_calc.capture_audio()

                # Wait out remaining ACTION_DURATION so total step time is unchanged.
                remaining = self.ACTION_DURATION - self.CAPTURE_PRE_DELAY - self.reward_calc.capture_duration
                if remaining > 0:
                    time.sleep(remaining)

                # Compute reward with the already-captured audio (no second capture).
                reward_info = self.reward_calc.compute_reward(
                    fret_position=fret_position,
                    torque=torque,
                    target_fret=self.target_fret,
                    capture_audio=False,
                    audio=audio,
                )

                # Expose for --slow plot callback
                self.last_audio = audio
                self.last_rl_action = rl_action
                self.last_reward_info = reward_info

                reward = reward_info['total_reward']
                filtered = reward_info.get('filtered', False)
                filter_reason = reward_info.get('filter_reason', '')
                cls = reward_info.get('classification')

                if filtered:
                    logger.info(
                        f"\n  {step_header}\n"
                        f"    FILTERED: {filter_reason}\n"
                        f"    reward={reward:+.3f}"
                    )
                elif cls is not None:
                    label = cls.get('predicted_label', f"class_{cls.get('predicted_class', '?')}")
                    logger.info(
                        f"\n  {step_header}\n"
                        f"    class={label}  H={cls.get('harmonic_prob', 0):.3f}  "
                        f"D={cls.get('dead_prob', 0):.3f}  G={cls.get('general_prob', 0):.3f}\n"
                        f"    reward={reward:+.3f}  "
                        f"(audio={reward_info['audio_reward']:+.3f}  "
                        f"fret={reward_info['fret_reward']:+.3f}  "
                        f"torque={reward_info['torque_reward']:+.3f})"
                    )
                else:
                    logger.info(
                        f"\n  {step_header}\n"
                        f"    no classification | reward={reward:+.3f}"
                    )
        
        # Update state
        self.current_fret = fret_position
        self.current_torque = torque
        self.fret_history.append(fret_position)
        self.torque_history.append(torque)
        self.episode_rewards.append(reward)
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False

        # Success termination: only award bonus for unfiltered actions that actually
        # reached the classifier.  A filtered step (no OSC sent) cannot be a success.
        if not reward_info.get('filtered', False) and reward_info['classification'] is not None:
            is_success = self.reward_calc.is_success(reward_info['classification'])
            if is_success:
                terminated = True
                reward += 1.0  # Bonus for success
                cls = reward_info['classification']
                logger.info(f"Success! Harmonic prob: {cls['harmonic_prob']:.3f}")

                # Non-blocking: hand audio + metadata to the background writer.
                if self.success_recorder is not None and self.last_audio is not None:
                    self.success_recorder.record(
                        audio=self.last_audio,
                        metadata={
                            'episode':        self.episode_count,
                            'step':           self.current_step,
                            'string_index':   self.string_index,
                            'target_fret':    self.target_fret,
                            'fret_position':  fret_position,
                            'torque':         torque,
                            'harmonic_prob':  cls.get('harmonic_prob', 0.0),
                            'dead_prob':      cls.get('dead_prob', 0.0),
                            'general_prob':   cls.get('general_prob', 0.0),
                            'confidence':     cls.get('confidence', 0.0),
                            'predicted_label': cls.get('predicted_label', 'unknown'),
                            'total_reward':   reward_info.get('total_reward', 0.0),
                            'audio_reward':   reward_info.get('audio_reward', 0.0),
                            'fret_reward':    reward_info.get('fret_reward', 0.0),
                            'torque_reward':  reward_info.get('torque_reward', 0.0),
                            'device_sr':      self.reward_calc.device_sr,
                        },
                    )
        
        # Max steps truncation
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Get next observation
        obs = self._get_observation()
        
        # Build info dict
        info = {
            'target_fret': self.target_fret,
            'string_index': self.string_index,
            'fret_position': fret_position,
            'slider_mm': rl_action.slider_mm,
            'torque': torque,
            'audio_reward': reward_info['audio_reward'],
            'fret_reward': reward_info['fret_reward'],
            'torque_reward': reward_info['torque_reward'],
            'fret_error': reward_info['fret_error'],
            'classification': reward_info['classification'],
            'filtered': reward_info.get('filtered', False),
            'filter_reason': reward_info.get('filter_reason', ''),
            'audio_rms': reward_info.get('audio_rms'),
            'step': self.current_step,
            'is_at_harmonic': rl_action.is_at_harmonic,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        if self.reward_calc is not None:
            self.reward_calc.close()
        if self.osc_client is not None:
            self.osc_client.close()
        logger.info(f"HarmonicEnv closed (string_indices={self.string_indices}, offline={self.offline})")
