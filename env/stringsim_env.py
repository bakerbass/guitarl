"""
Gymnasium environment for StringSim guitar simulator.

Provides reinforcement learning interface for learning to play natural harmonics.
Uses FRACTIONAL FRETS for position encoding to help the agent learn musical structure.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from .osc_client import StringSimOSCClient
from .action_space import (
    GuitarBotActionSpace,
    RLFretAction,
    fret_to_mm,
    mm_to_fret,
    PLAYABLE_STRINGS,
    STRING_TO_PLUCKER,
    FRET_MIN,
    FRET_MAX,
    TORQUE_MIN,
    TORQUE_MAX,
    TORQUE_LIGHT,
    TORQUE_NORMAL,
    HARMONIC_FRETS_IN_RANGE,
    SLIDER_MAX_MM,
)
from ..utils.audio_reward import HarmonicRewardCalculator


logger = logging.getLogger(__name__)


class StringSimEnv(gym.Env):
    """
    Gymnasium environment for learning guitar harmonics on StringSim.
    
    The agent learns to find the correct fret position and torque to produce
    clear natural harmonics at frets 4, 5, and 7.
    
    Uses FRACTIONAL FRETS for position to encode musical structure, helping
    the agent learn that frets 4, 5, 7 are harmonic nodes.
    
    Observation Space:
        - target_fret_one_hot: Which harmonic to play (4, 5, or 7) [3]
        - current_fret: Last commanded fret position [1]
        - current_torque: Last commanded torque (normalized) [1]
        - fret_history: Last N fret positions [3]
        - torque_history: Last N torque values [3]
        Total: 11 dimensions
        
    Action Space (normalized, shape=(5,)):
        - string_logits: String selection [3] (argmax -> 0, 2, 4)
        - fret: Fractional fret position [1] (scaled from [-1,1] to [0,9])
        - torque: Fretting torque [1] (scaled from [-1,1] to [0,1000])
        
    Reward:
        Combination of:
        - Harmonic classifier confidence (0-1) [weight: 0.5]
        - Fret position accuracy (Gaussian) [weight: 0.3]
        - Torque optimization (Gaussian) [weight: 0.2]
    """
    
    metadata = {'render_modes': ['human']}
    
    # Environment constants
    HARMONIC_FRETS = HARMONIC_FRETS_IN_RANGE  # [4, 5, 7]
    MAX_STEPS_PER_EPISODE = 10
    ACTION_DURATION = 0.1  # Time to wait after each action (seconds)
    
    def __init__(self,
                 model_path: str,
                 string_index: int = 2,  # Default to D string (plucker 1 -> string 2)
                 osc_host: str = "127.0.0.1",
                 osc_port: int = 8000,
                 audio_device: str = "VB-Audio Virtual Cable",
                 capture_duration: float = 0.8,
                 max_steps: int = MAX_STEPS_PER_EPISODE,
                 success_threshold: float = 0.8,
                 curriculum_mode: str = "random",
                 use_simple_action_space: bool = False):
        """
        Initialize StringSim environment.
        
        Args:
            model_path: Path to HarmonicsClassifier model
            string_index: Which string to use (0, 2, or 4 - must have plucker)
            osc_host: StringSim OSC host
            osc_port: StringSim OSC port
            audio_device: VB-CABLE device name
            capture_duration: Audio capture duration
            max_steps: Maximum steps per episode
            success_threshold: Harmonic probability threshold for success
            curriculum_mode: "random", "easy_to_hard", or "fixed_fret"
            use_simple_action_space: If True, use 3D continuous space instead of 5D
        """
        super().__init__()
        
        # Validate string has a plucker
        if string_index not in PLAYABLE_STRINGS:
            raise ValueError(f"string_index must be in {PLAYABLE_STRINGS} (strings with pluckers)")
        
        self.string_index = string_index
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.curriculum_mode = curriculum_mode
        self.use_simple_action_space = use_simple_action_space
        
        # Initialize action space helper
        self.action_space_helper = GuitarBotActionSpace(use_normalized=True)
        
        # Initialize OSC client
        self.osc_client = StringSimOSCClient(host=osc_host, port=osc_port)
        
        # Initialize reward calculator
        self.reward_calc = HarmonicRewardCalculator(
            model_path=model_path,
            device_name=audio_device,
            capture_duration=capture_duration
        )
        
        # Define action space: 5D normalized or 3D simple
        if use_simple_action_space:
            # [string_continuous, fret, torque] all in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
        else:
            # [string_logits(3), fret, torque] all in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32
            )
        
        # Define observation space
        # [target_fret_one_hot(3), current_fret, current_torque_norm, fret_history(3), torque_history(3)]
        obs_dim = 3 + 1 + 1 + 3 + 3  # = 11
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
        
        logger.info(f"StringSimEnv initialized: string={string_index}, max_steps={max_steps}, "
                    f"action_space={'3D simple' if use_simple_action_space else '5D full'}")
    
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
        """Construct observation vector."""
        # One-hot encode target fret
        target_one_hot = np.zeros(3, dtype=np.float32)
        fret_idx = self.HARMONIC_FRETS.index(self.target_fret)
        target_one_hot[fret_idx] = 1.0
        
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
        # Normalize torque history
        torque_history = np.array(torque_history, dtype=np.float32) / TORQUE_MAX
        
        # Concatenate all observations
        obs = np.concatenate([target_one_hot, current_state, fret_history, torque_history])
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset StringSim
        self.osc_client.reset(wait_time=0.15)
        
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
        
        logger.debug(f"Episode {self.episode_count} reset: target_fret={self.target_fret}")
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in environment."""
        # Convert normalized action to RLFretAction
        if self.use_simple_action_space:
            rl_action = self.action_space_helper.from_simple_normalized(action)
        else:
            rl_action = self.action_space_helper.from_normalized(action)
        
        # Override string with environment's fixed string (for single-string training)
        # If you want agent to learn string selection, remove this override
        rl_action = RLFretAction(
            string_idx=self.string_index,
            fret_position=rl_action.fret_position,
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
            'fret_position': fret_position,
            'slider_mm': rl_action.slider_mm,
            'torque': torque,
            'audio_reward': reward_info['audio_reward'],
            'fret_reward': reward_info['fret_reward'],
            'torque_reward': reward_info['torque_reward'],
            'fret_error': reward_info['fret_error'],
            'classification': reward_info['classification'],
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
        logger.info("StringSimEnv closed")
