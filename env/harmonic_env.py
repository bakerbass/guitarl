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
    ACTION_DURATION = 3.0  # Time to wait after each action (seconds)
    
    def __init__(self,
                 model_path: str,
                 string_index: int = 2,  # Default to D string — used when string_indices is None
                 string_indices=None,    # List of strings to rotate across (e.g. [0,2,4])
                 osc_host: str = "127.0.0.1",
                 osc_port: int = 12000,
                 audio_device: str = "Scarlett",
                 capture_duration: float = 0.8,
                 max_steps: int = MAX_STEPS_PER_EPISODE,
                 success_threshold: float = 0.8,
                 curriculum_mode: str = "random",
                 use_simple_action_space: bool = False,
                 always_press: bool = True):
        """
        Initialize HarmonicEnv.
        
        Args:
            model_path: Path to HarmonicsClassifier model
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
            use_simple_action_space: If True, use 3D continuous space instead of 5D
            always_press: If True, remove press_decision from action space (always PRESS)
        """
        super().__init__()
        
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
        self.use_simple_action_space = use_simple_action_space
        self.always_press = always_press
        
        # Initialize action space helper
        self.action_space_helper = GuitarBotActionSpace(use_normalized=True)
        
        # Initialize OSC client
        self.osc_client = GuitarBotOSCClient(host=osc_host, port=osc_port)
        
        # Initialize reward calculator
        self.reward_calc = HarmonicRewardCalculator(
            model_path=model_path,
            device_name=audio_device,
            capture_duration=capture_duration
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
        
        # Curriculum learning state
        self.episode_count = 0
        self.curriculum_fret_idx = 0  # Start with easiest (fret 7)
        
        logger.info(f"HarmonicEnv initialized: strings={self.string_indices}, max_steps={max_steps}, "
                    f"action_dim={action_dim}, obs_dim={obs_dim}, always_press={always_press}")
    
    def _get_target_fret(self) -> int:
        """Select target fret based on curriculum mode."""
        if self.curriculum_mode == "random":
            return np.random.choice(self.HARMONIC_FRETS)
        elif self.curriculum_mode == "easy_to_hard":
            # Fret 7 (easiest) -> Fret 5 -> Fret 4 (hardest)
            curriculum_order = [7, 5, 4]
            # Progress curriculum every 100 episodes
            idx = min(self.episode_count // 100, len(curriculum_order) - 1)
            return curriculum_order[idx]
        elif self.curriculum_mode == "fixed_fret":
            return 7  # Always use fret 7 for initial training
        else:
            return np.random.choice(self.HARMONIC_FRETS)
    
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
        self.string_index = int(np.random.choice(self.string_indices))
        
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
        
        # Send action to StringSim via /rlfret
        self.osc_client.send_rlfret(rl_action)
        
        # Wait for physics to settle and sound to develop
        time.sleep(self.ACTION_DURATION)
        
        # Capture audio and compute reward
        reward_info = self.reward_calc.compute_reward(
            fret_position=fret_position,
            torque=torque,
            target_fret=self.target_fret,
            capture_audio=True
        )
        
        reward = reward_info['total_reward']
        
        # Log per-step classifier results
        filtered = reward_info.get('filtered', False)
        filter_reason = reward_info.get('filter_reason', '')
        cls = reward_info.get('classification')

        step_header = (
            f"Step {self.current_step + 1}/{self.max_steps} | "
            f"str={self.string_index} target={self.target_fret} "
            f"fret={fret_position:.2f} torque={torque:.0f}"
        )
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
                f"D={cls.get('dead_note_prob', 0):.3f}  G={cls.get('general_note_prob', 0):.3f}\n"
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
        
        # Success termination (optional - can disable for fixed-length episodes)
        if reward_info['classification'] is not None:
            is_success = self.reward_calc.is_success(reward_info['classification'])
            if is_success:
                terminated = True
                reward += 1.0  # Bonus for success
                logger.info(f"Success! Harmonic prob: {reward_info['classification']['harmonic_prob']:.3f}")
        
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
        self.osc_client.close()
        logger.info(f"HarmonicEnv closed (string_indices={self.string_indices})")
