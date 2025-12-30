"""
Gymnasium environment for StringSim guitar simulator.

Provides reinforcement learning interface for learning to play natural harmonics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from .osc_client import StringSimOSCClient
from ..utils.audio_reward import HarmonicRewardCalculator


logger = logging.getLogger(__name__)


class StringSimEnv(gym.Env):
    """
    Gymnasium environment for learning guitar harmonics on StringSim.
    
    The agent learns to find the correct fret position and force to produce
    clear natural harmonics at frets 4, 5, and 7.
    
    Observation Space:
        - target_fret: Which harmonic to play (4, 5, or 7)
        - current_position: Last commanded slider position (mm)
        - current_force: Last commanded force (0-1)
        - action_history: Last N actions
        
    Action Space:
        - position_mm: Continuous (0-234mm)
        - force: Continuous (0-1)
        
    Reward:
        Combination of:
        - Harmonic classifier confidence (0-1)
        - Position accuracy (Gaussian around target)
        - Force optimization (Gaussian around 0.3)
    """
    
    metadata = {'render_modes': ['human']}
    
    # Environment constants
    HARMONIC_FRETS = [4, 5, 7]
    MAX_STEPS_PER_EPISODE = 10
    ACTION_DURATION = 0.1  # Time to wait after each action (seconds)
    
    def __init__(self,
                 model_path: str,
                 string_index: int = 3,  # Default to D string
                 osc_host: str = "127.0.0.1",
                 osc_port: int = 8000,
                 audio_device: str = "VB-Audio Virtual Cable",
                 capture_duration: float = 0.8,
                 max_steps: int = MAX_STEPS_PER_EPISODE,
                 success_threshold: float = 0.8,
                 curriculum_mode: str = "random"):
        """
        Initialize StringSim environment.
        
        Args:
            model_path: Path to HarmonicsClassifier model
            string_index: Which string to use (0-5, default 3=D string)
            osc_host: StringSim OSC host
            osc_port: StringSim OSC port
            audio_device: VB-CABLE device name
            capture_duration: Audio capture duration
            max_steps: Maximum steps per episode
            success_threshold: Harmonic probability threshold for success
            curriculum_mode: "random", "easy_to_hard", or "fixed_fret"
        """
        super().__init__()
        
        self.string_index = string_index
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.curriculum_mode = curriculum_mode
        
        # Initialize OSC client
        self.osc_client = StringSimOSCClient(host=osc_host, port=osc_port)
        
        # Initialize reward calculator
        self.reward_calc = HarmonicRewardCalculator(
            model_path=model_path,
            device_name=audio_device,
            capture_duration=capture_duration
        )
        
        # Define action space: [position_mm, force]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([234.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # [target_fret_one_hot(3), current_position, current_force, last_3_positions, last_3_forces]
        obs_dim = 3 + 1 + 1 + 3 + 3  # = 11
        self.observation_space = spaces.Box(
            low=-1.0,
            high=300.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.target_fret = None
        self.current_position = 0.0
        self.current_force = 0.0
        self.position_history = []
        self.force_history = []
        self.episode_rewards = []
        
        # Curriculum learning state
        self.episode_count = 0
        self.curriculum_fret_idx = 0  # Start with easiest (fret 7)
        
        logger.info(f"StringSimEnv initialized: string={string_index}, max_steps={max_steps}")
    
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
        
        # Current state
        current_state = np.array([self.current_position, self.current_force], dtype=np.float32)
        
        # Action history (last 3 actions, pad if necessary)
        pos_history = self.position_history[-3:] if len(self.position_history) > 0 else [0.0] * 3
        force_history = self.force_history[-3:] if len(self.force_history) > 0 else [0.0] * 3
        
        # Pad to length 3
        while len(pos_history) < 3:
            pos_history.insert(0, 0.0)
        while len(force_history) < 3:
            force_history.insert(0, 0.0)
        
        pos_history = np.array(pos_history, dtype=np.float32)
        force_history = np.array(force_history, dtype=np.float32)
        
        # Concatenate all observations
        obs = np.concatenate([target_one_hot, current_state, pos_history, force_history])
        
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
        self.current_position = 0.0
        self.current_force = 0.0
        self.position_history = []
        self.force_history = []
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
        position_mm, force = action
        position_mm = float(np.clip(position_mm, 0.0, 234.0))
        force = float(np.clip(force, 0.0, 1.0))
        
        # Send action to StringSim
        self.osc_client.send_fret(
            string_index=self.string_index,
            position_mm=position_mm,
            force=force
        )
        
        # Wait for physics to settle and sound to develop
        time.sleep(self.ACTION_DURATION)
        
        # Capture audio and compute reward
        reward_info = self.reward_calc.compute_reward(
            position_mm=position_mm,
            force=force,
            target_fret=self.target_fret,
            capture_audio=True
        )
        
        reward = reward_info['total_reward']
        
        # Update state
        self.current_position = position_mm
        self.current_force = force
        self.position_history.append(position_mm)
        self.force_history.append(force)
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
            'position_mm': position_mm,
            'force': force,
            'audio_reward': reward_info['audio_reward'],
            'position_reward': reward_info['position_reward'],
            'force_reward': reward_info['force_reward'],
            'position_error': reward_info['position_error'],
            'classification': reward_info['classification'],
            'step': self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        self.osc_client.close()
        logger.info("StringSimEnv closed")
