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
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from env.harmonic_env import HarmonicEnv


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
        self.episode_frets = []
        self.episode_torques = []
        
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
                        self.episode_frets.append(info.get('fret_position', 0))
                        self.episode_torques.append(info.get('torque', 0))
                        
                        if self.verbose > 0:
                            logger.info(
                                f"Episode end - Fret: {info['target_fret']}, "
                                f"Harmonic prob: {harmonic_prob:.3f}, "
                                f"Success: {success}, "
                                f"Fret pos: {info.get('fret_position', 0):.2f}, "
                                f"Torque: {info.get('torque', 0):.0f}"
                            )
        
        # Log aggregated metrics every 100 episodes
        if len(self.episode_successes) >= 100:
            avg_success = np.mean(self.episode_successes[-100:])
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_fret = np.mean(self.episode_frets[-100:])
            avg_torque = np.mean(self.episode_torques[-100:])
            
            logger.info(
                f"\n=== Last 100 Episodes ===\n"
                f"Success rate: {avg_success:.3f}\n"
                f"Avg reward: {avg_reward:.3f}\n"
                f"Avg fret pos: {avg_fret:.2f}\n"
                f"Avg torque: {avg_torque:.0f}\n"
            )
        
        return True


# ---------------------------------------------------------------------------
# Slow-mode audio plot callback
# ---------------------------------------------------------------------------

def _plot_episode_audio(audio: np.ndarray, reward_info: dict, rl_action,
                        episode_num: int, model_sr: int = 22050,
                        capture_duration: float = 2.0) -> None:
    """Plot waveform + mel spectrogram for one episode step (blocking)."""
    import matplotlib.pyplot as plt
    import librosa

    cls = reward_info.get('classification') or {}
    label = cls.get('predicted_label', 'unknown')
    h_prob = cls.get('harmonic_prob', 0.0)
    d_prob = cls.get('dead_prob', 0.0)
    g_prob = cls.get('general_prob', 0.0)
    confidence = cls.get('confidence', 0.0)
    reward = reward_info.get('total_reward', 0.0)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=model_sr,
        n_mels=128, n_fft=2048, hop_length=512,
        fmin=80, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    action_str = (
        f'String {rl_action.string_idx}  '
        f'Fret {rl_action.fret_position:.2f}  '
        f'Torque {rl_action.torque:.0f}'
    )
    fig.suptitle(f'Episode step {episode_num}  |  {action_str}', fontsize=11, fontweight='bold')

    # Waveform
    t = np.arange(len(audio)) / model_sr
    axes[0].plot(t, audio, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    reward_str = (
        f'Reward: {reward:+.3f}  '
        f'(audio={reward_info.get("audio_reward", 0):+.2f}  '
        f'fret={reward_info.get("fret_reward", 0):+.2f}  '
        f'torque={reward_info.get("torque_reward", 0):+.2f})'
    )
    axes[0].set_title(
        f'Classified: {label.upper()} ({confidence*100:.1f}%)  '
        f'H={h_prob:.3f}  D={d_prob:.3f}  G={g_prob:.3f}\n{reward_str}'
    )
    axes[0].grid(True, alpha=0.3)

    # Mel spectrogram
    img = axes[1].imshow(
        mel_db, aspect='auto', origin='lower', cmap='viridis',
        extent=[0, capture_duration, 0, 128],
    )
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mel Frequency Bin')
    axes[1].set_title('Mel Spectrogram (dB)')
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.show(block=True)   # Blocks — training pauses until the window is closed
    plt.close(fig)


class SlowModeCallback(BaseCallback):
    """
    When --slow is set, plot the waveform + mel spectrogram after every episode
    step and block until the plot window is closed.  Lets you visually verify
    that the classifier is hearing the note correctly before committing a long
    training run.

    Only activates on the *training* env (not the eval env) and only in online
    (non-pretrain) mode.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._step_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if not done:
                continue
            # Unwrap DummyVecEnv → Monitor → HarmonicEnv
            try:
                monitor_env = self.training_env.envs[i]
                base_env = getattr(monitor_env, 'env', monitor_env)
            except (AttributeError, IndexError):
                continue

            audio = getattr(base_env, 'last_audio', None)
            rl_action = getattr(base_env, 'last_rl_action', None)
            reward_info = getattr(base_env, 'last_reward_info', None)

            if audio is None or rl_action is None or reward_info is None:
                logger.warning('[slow] No audio data available for this episode — skipping plot.')
                continue

            self._step_count += 1
            capture_dur = getattr(
                getattr(base_env, 'reward_calc', None), 'capture_duration', 2.0
            )
            model_sr = getattr(
                getattr(base_env, 'reward_calc', None), 'model_sr', 22050
            )
            _plot_episode_audio(
                audio=audio,
                reward_info=reward_info,
                rl_action=rl_action,
                episode_num=self._step_count,
                model_sr=model_sr,
                capture_duration=capture_dur,
            )
        return True


# ---------------------------------------------------------------------------

def make_env(model_path, curriculum_mode: str, string_indices=None, string_index: int = 2,
             osc_port: int = 12000, audio_device: str = "Scarlett",
             reward_mode: str = 'full', offline: bool = False):
    """Create and wrap HarmonicEnv."""
    env = HarmonicEnv(
        model_path=model_path,
        string_index=string_index,
        string_indices=string_indices,
        osc_port=osc_port,
        audio_device=audio_device,
        curriculum_mode=curriculum_mode,
        max_steps=10,
        success_threshold=0.8,
        reward_mode=reward_mode,
        offline=offline,
    )
    env = Monitor(env)
    return env


def train(args):
    """Train harmonic RL agent."""
    logger.info("=" * 60)
    logger.info("Starting harmonic RL training")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {torch.cuda.is_available()}" +
                (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    
    # ── Output directory ──────────────────────────────────────────────
    # When resuming, reuse the existing run directory so logs/checkpoints
    # stay together.  When starting fresh, create a timestamped subdirectory.
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            logger.error(f"Resume directory not found: {resume_dir}")
            sys.exit(1)
        output_dir = resume_dir
        logger.info(f"Resuming from: {output_dir}")
    else:
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
    logger.info(f"Reward mode: {args.reward_mode}")

    if args.pretrain:
        logger.info(
            "*** OFFLINE PRE-TRAINING MODE ***\n"
            "  - No robot or audio hardware required.\n"
            "  - Reward = filtration layer only (fret + torque shaping).\n"
            "  - Fret shaping uses wide Gaussian (σ=1.5 frets) for full-neck gradients.\n"
            "  - Steps are instant (no physics wait).\n"
            "  - TIP: use --ent-coef 0.1 (or higher) to prevent policy entropy collapse.\n"
            "  - Resume on the robot with:  --resume <this run dir>  --clear-buffer  (without --pretrain)"
        )

    # Resolve string pool: --string-indices takes precedence, else fall back to --string-index
    string_indices = args.string_indices if args.string_indices else [args.string_index]
    logger.info(f"String pool: {string_indices}")

    # Create environment
    logger.info("Creating environment...")
    env = make_env(
        model_path=args.model_path,
        curriculum_mode=args.curriculum,
        string_indices=string_indices,
        osc_port=args.osc_port,
        audio_device=args.audio_device,
        reward_mode=args.reward_mode,
        offline=args.pretrain,
    )

    # Create evaluation environment
    eval_env = make_env(
        model_path=args.model_path,
        curriculum_mode="random",  # Evaluate on all frets
        string_indices=string_indices,
        osc_port=args.osc_port,
        audio_device=args.audio_device,
        reward_mode=args.reward_mode,
        offline=args.pretrain,
    )
    
    # Configure SAC agent
    logger.info("Configuring SAC agent...")
    policy_kwargs = dict(
        net_arch=[256, 256],  # Hidden layers
    )
    
    # Parse ent_coef: allow float or the string 'auto' / 'auto_X.X'
    ent_coef = args.ent_coef
    try:
        ent_coef = float(ent_coef)
    except ValueError:
        pass  # keep as string ('auto' or 'auto_X.X')

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
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=args.device
    )
    logger.info(f"Entropy coefficient: {ent_coef}"
                + (" (auto-tuned)" if str(ent_coef).startswith("auto") else " (fixed)"))
    
    # ── Resume: load weights + replay buffer ──────────────────────────
    if args.resume:
        def _latest_checkpoint(ckpt_dir: Path):
            """Return the .zip with the highest timestep number, or None."""
            zips = sorted(ckpt_dir.glob("harmonic_sac_*_steps.zip"))
            return zips[-1].with_suffix('') if zips else None

        if args.resume_checkpoint:
            model_file = Path(args.resume_checkpoint)
        elif (output_dir / "interrupted_model.zip").exists():
            model_file = output_dir / "interrupted_model"
        else:
            model_file = _latest_checkpoint(checkpoint_dir)

        if model_file is None:
            logger.error("No checkpoint found in resume directory. Cannot resume.")
            sys.exit(1)

        logger.info(f"Loading weights from: {model_file}")
        model = SAC.load(model_file, env=env, device=args.device)

        buffer_file = Path(str(model_file) + "_replay_buffer.pkl")
        if args.clear_buffer:
            logger.info(
                "--clear-buffer set: skipping replay buffer load. "
                "SAC will warm up with fresh transitions (useful when transitioning "
                "from offline pre-training to online robot training)."
            )
        elif buffer_file.exists():
            logger.info(f"Loading replay buffer from: {buffer_file}")
            model.load_replay_buffer(buffer_file)
            logger.info(f"  Buffer size restored: {model.replay_buffer.size()} transitions")
        else:
            logger.warning(
                f"Replay buffer not found at {buffer_file}. "
                "Resuming with an empty buffer — learning_starts warmup will apply again."
            )

    # Configure logger
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="harmonic_sac",
        save_replay_buffer=True,   # saves <name>_replay_buffer.pkl alongside each .zip
        save_vecnormalize=False,
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

    cb_list = [checkpoint_callback, eval_callback, progress_callback]
    if args.slow and not args.pretrain:
        cb_list.append(SlowModeCallback())
        logger.info(
            '[slow] Slow-mode enabled: training will pause after every episode to plot audio. '
            'Close the plot window to continue.'
        )
    elif args.slow and args.pretrain:
        logger.warning('[slow] --slow has no effect in --pretrain mode (no audio captured).')

    callbacks = CallbackList(cb_list)
    
    # Train
    logger.info(f"Starting training for {args.total_timesteps} timesteps...")
    logger.info(f"Device: {args.device}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=not args.resume,  # preserve timestep counter when resuming
        )
        
        # Save final model
        final_model_path = output_dir / "final_model"
        model.save(final_model_path)
        logger.info(f"Training complete! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save(output_dir / "interrupted_model")
        model.save_replay_buffer(output_dir / "interrupted_model_replay_buffer")
        logger.info("Model and replay buffer saved")
    
    finally:
        # NOTE: /Reset is sent automatically on exit.
        # Sending /Reset while the robot is executing a trajectory can cause a
        # mechanical malfunction (both threads call RobotController.main() at
        # the same time, corrupting the UDP stream).
        #
        # The GuitarBot's arm_list_recieverNN.py now serialises all robot calls
        # through a robot_lock, so a /Reset sent after Ctrl+C will wait for the
        # current trajectory to finish before executing — but only send it if
        # you are confident the robot has finished its last action.
        
        if args.pretrain:
            logger.info("Offline pre-training complete. No robot reset needed.")
        elif args.reset_on_exit:
            logger.info("Sending /Reset (--reset-on-exit requested)...")
            logger.info("The reset will be queued behind any active trajectory on the robot.")
            # Unwrap Monitor wrapper to access the underlying HarmonicEnv
            base_env = env.env if hasattr(env, 'env') else env
            if hasattr(base_env, 'osc_client') and base_env.osc_client is not None:
                base_env.osc_client.reset(wait_time=0.5)
                logger.info("/Reset sent.")
        else:
            logger.info(
                "Robot holds its last position. "
                "Send /Reset manually from arm_list_recieverNN.py when safe, "
                "or restart training with --reset-on-exit."
            )
        
        env.close()
        eval_env.close()  # eval_env is a separate HarmonicEnv instance — closing both is intentional


def main():
    parser = argparse.ArgumentParser(description='Train harmonic RL agent')
    
    # Environment arguments
    parser.add_argument('--model-path', default=None,
                        help='Path to HarmonicsClassifier model. '
                             'Required for online training (default). '
                             'Not needed when --pretrain is set.')
    parser.add_argument('--string-index', type=int, default=2,
                        help='Single string to train on (0, 2, or 4 — must have plucker; default: 2=D). '
                             'Ignored when --string-indices is provided.')
    parser.add_argument('--string-indices', type=int, nargs='+', default=None,
                        metavar='S',
                        help='One or more strings to rotate across episodes (e.g. --string-indices 0 2 4). '
                             'Each reset() samples uniformly, distributing motor wear while keeping '
                             'the policy string-aware via a 3-element one-hot in the observation. '
                             'Overrides --string-index when provided.')
    parser.add_argument('--osc-port', type=int, default=12000,
                        help='OSC port (default: 12000 for GuitarBot, 8000 for StringSim)')
    parser.add_argument('--audio-device', type=str, default='Scarlett',
                        help='Audio input device name substring (default: Scarlett)')
    parser.add_argument('--curriculum', type=str, default='easy_to_hard',
                        choices=['random', 'easy_to_hard', 'fixed_fret'],
                        help='Curriculum learning mode')
    
    # Training arguments
    parser.add_argument('--total-timesteps', type=int, default=2000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=500, help='Steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--ent-coef', type=str, default='auto',
                        help='SAC entropy coefficient. "auto" lets SAC tune it automatically. '
                             'Set a fixed float (e.g. 0.5) to prevent policy entropy collapse '
                             'during --pretrain — the auto-tuner can drive entropy too low when '
                             'steps are instant, leaving the policy stuck at a point estimate. '
                             'Recommended for --pretrain: 0.1 to 0.5  (default: auto)')
    
    # Logging arguments
    parser.add_argument('--output-dir', type=str, default='./runs', help='Output directory')
    parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=2000, help='Evaluation frequency')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, metavar='RUN_DIR',
                        help='Path to a previous run directory to resume from '
                             '(e.g. ./runs/harmonic_sac_20260218_134500). '
                             'Loads the latest checkpoint (or interrupted_model) '
                             'and its replay buffer, then continues training.')
    parser.add_argument('--resume-checkpoint', type=str, default=None, metavar='CKPT',
                        help='Explicit checkpoint file to resume from (without .zip extension). '
                             'Overrides the automatic latest-checkpoint search within --resume.')
    parser.add_argument('--clear-buffer', action='store_true', default=False,
                        help='When resuming, discard the saved replay buffer and start fresh. '
                             'Use this when transitioning from --pretrain to online robot '
                             'training: the pre-train buffer contains no audio reward signal '
                             'and can bias the policy away from harmonic-seeking behaviour.')
    
    # Pre-training (offline, filtration-only)
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Run in offline pre-training mode. No robot or audio hardware is '
                             'needed. Reward comes from the filtration layer only '
                             '(fret + torque shaping). Steps are instant — no physics wait. '
                             'Save the resulting model and resume on the robot later with '
                             '--resume <run-dir> (without --pretrain).')

    # Robot safety
    parser.add_argument('--reset-on-exit', action='store_true', default=True,
                        help='Send /Reset to GuitarBot when training stops. '
                             'Only use this when you are certain no trajectory is '
                             'actively running on the robot — the GuitarBot '
                             'serialises the reset behind any active trajectory, '
                             'but premature resets can still cause mechanical issues. '
                             'Default: ON (robot resets on exit).')    
    # Reward / ablation
    parser.add_argument('--reward-mode', type=str, default='full',
                        choices=['full', 'no_filtration', 'no_audio'],
                        help='Reward function variant for ablation studies. '
                             '"full" (default): two-layer reward (filtration + CNN). '
                             '"no_filtration": skip physics gate, all actions reach the CNN. '
                             '"no_audio": skip CNN entirely, use fret+torque shaping only (0.5/0.5 weights).')    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')

    # Slow / debug
    parser.add_argument('--slow', action='store_true', default=False,
                        help='After every episode, pause training and display a plot of the '
                             'captured audio waveform + mel spectrogram with classification '
                             'results and reward breakdown. Close the plot window to continue. '
                             'Use this to visually verify the classifier is hearing the note '
                             'before committing to a long run. No-op in --pretrain mode.')

    # Verbosity
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable per-step debug logging. In --pretrain mode steps are '
                             'instant so this output is hidden by default to keep the '
                             'terminal readable.')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate model path (not required for offline pre-training)
    if args.pretrain:
        if args.model_path is not None and not Path(args.model_path).exists():
            logger.error(f"Model not found: {args.model_path}")
            sys.exit(1)
    else:
        if args.model_path is None:
            logger.error("--model-path is required for online training. "
                         "Use --pretrain to run without a robot/audio setup.")
            sys.exit(1)
        if not Path(args.model_path).exists():
            logger.error(f"Model not found: {args.model_path}")
            sys.exit(1)

    train(args)


if __name__ == '__main__':
    main()
