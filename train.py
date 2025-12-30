"""
Training script for harmonic RL agent using Stable-Baselines3.

Usage:
    python train.py --model-path ../HarmonicsClassifier/models/best_model.pt
    python train.py --model-path ../HarmonicsClassifier/models/best_model.pt --curriculum easy_to_hard --total-timesteps 50000
"""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime
import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from env.stringsim_env import StringSimEnv


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HarmonicProgressCallback(CallbackList):
    """Custom callback to log harmonic-specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__([])
        self.verbose = verbose
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_positions = []
        self.episode_forces = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones'):
            dones = self.locals['dones']
            infos = self.locals.get('infos', [])
            
            for idx, done in enumerate(dones):
                if done and idx < len(infos):
                    info = infos[idx]
                    
                    # Log metrics
                    if 'classification' in info and info['classification'] is not None:
                        harmonic_prob = info['classification']['harmonic_prob']
                        success = harmonic_prob > 0.8
                        
                        self.episode_rewards.append(self.locals.get('rewards', [0])[idx])
                        self.episode_successes.append(float(success))
                        self.episode_positions.append(info.get('position_mm', 0))
                        self.episode_forces.append(info.get('force', 0))
                        
                        if self.verbose > 0:
                            logger.info(
                                f"Episode end - Fret: {info['target_fret']}, "
                                f"Harmonic prob: {harmonic_prob:.3f}, "
                                f"Success: {success}, "
                                f"Pos: {info['position_mm']:.1f}mm, "
                                f"Force: {info['force']:.2f}"
                            )
        
        # Log aggregated metrics every 100 episodes
        if len(self.episode_successes) >= 100:
            avg_success = np.mean(self.episode_successes[-100:])
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_pos = np.mean(self.episode_positions[-100:])
            avg_force = np.mean(self.episode_forces[-100:])
            
            logger.info(
                f"\n=== Last 100 Episodes ===\n"
                f"Success rate: {avg_success:.3f}\n"
                f"Avg reward: {avg_reward:.3f}\n"
                f"Avg position: {avg_pos:.1f}mm\n"
                f"Avg force: {avg_force:.3f}\n"
            )
        
        return True


def make_env(model_path: str, curriculum_mode: str, string_index: int = 3):
    """Create and wrap StringSim environment."""
    env = StringSimEnv(
        model_path=model_path,
        string_index=string_index,
        curriculum_mode=curriculum_mode,
        max_steps=10,
        success_threshold=0.8
    )
    env = Monitor(env)
    return env


def train(args):
    """Train harmonic RL agent."""
    logger.info("=" * 60)
    logger.info("Starting harmonic RL training")
    logger.info("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"harmonic_sac_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Curriculum: {args.curriculum}")
    
    # Create environment
    logger.info("Creating environment...")
    env = make_env(
        model_path=args.model_path,
        curriculum_mode=args.curriculum,
        string_index=args.string_index
    )
    
    # Create evaluation environment
    eval_env = make_env(
        model_path=args.model_path,
        curriculum_mode="random",  # Evaluate on all frets
        string_index=args.string_index
    )
    
    # Configure SAC agent
    logger.info("Configuring SAC agent...")
    policy_kwargs = dict(
        net_arch=[256, 256],  # Hidden layers
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=args.device
    )
    
    # Configure logger
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="harmonic_sac"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True
    )
    
    progress_callback = HarmonicProgressCallback(verbose=1)
    
    callbacks = CallbackList([checkpoint_callback, eval_callback, progress_callback])
    
    # Train
    logger.info(f"Starting training for {args.total_timesteps} timesteps...")
    logger.info(f"Device: {args.device}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = output_dir / "final_model"
        model.save(final_model_path)
        logger.info(f"Training complete! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save(output_dir / "interrupted_model")
        logger.info("Model saved")
    
    finally:
        env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Train harmonic RL agent')
    
    # Environment arguments
    parser.add_argument('--model-path', required=True, help='Path to HarmonicsClassifier model')
    parser.add_argument('--string-index', type=int, default=3, help='String to train on (default: 3=D)')
    parser.add_argument('--curriculum', type=str, default='easy_to_hard',
                        choices=['random', 'easy_to_hard', 'fixed_fret'],
                        help='Curriculum learning mode')
    
    # Training arguments
    parser.add_argument('--total-timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000, help='Steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Logging arguments
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=2000, help='Evaluation frequency')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    train(args)


if __name__ == '__main__':
    main()
