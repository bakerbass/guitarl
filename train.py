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
import select
import threading
from collections import deque
from datetime import datetime
from typing import Optional
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
from utils.success_recorder import SuccessRecorder


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal helper
# ---------------------------------------------------------------------------

def _configure_stdin_single_char():
    """
    Configure stdin for single-character reads without full raw mode.

    Disables ICANON (line-buffered input) and ECHO so keypresses are
    delivered immediately without the user pressing Enter, but deliberately
    leaves OPOST (output post-processing) untouched.  With OPOST on, the
    kernel still inserts CR before every LF, so SB3's progress table and all
    other stdout output prints correctly regardless of when the listener
    thread runs.

    tty.setraw() also clears OPOST, which is what causes the staircase
    output: if any other thread writes to stdout while raw mode is active
    (even briefly), every \\n lands without a preceding \\r.

    Returns (fd, old_settings).  The caller must restore with::

        termios.tcsetattr(fd, termios.TCSANOW, old_settings)

    Raises if stdin is not a tty (pipes, CI runners, etc.).
    """
    import termios
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    # Index 3 is c_lflag — only clear ICANON and ECHO, leave everything else
    # (especially OPOST in c_oflag / index 1) completely alone.
    new[3] &= ~(termios.ICANON | termios.ECHO)
    # c_cc: require 1 byte with no timeout so read() blocks until a byte arrives
    new[6][termios.VMIN]  = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    return fd, old


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


class RobotLearningStartCallback(BaseCallback):
    """
    Triggers SAC gradient updates only after a specified number of *real robot
    actions* (i.e. steps where an OSC command was actually sent and audio was
    captured).  Filtered steps — which return instantly without touching the
    robot — are excluded from the count.

    This replaces the raw `learning_starts` threshold on the SAC model, which
    counts every timestep including filtered ones and is therefore hard to
    reason about when the random policy is generating many out-of-range actions.

    Implementation: the SAC model is initialised with a huge `learning_starts`
    value so SB3 never starts on its own.  Once `robot_step_count` reaches the
    threshold this callback sets `model.learning_starts = 0`, which makes SB3
    begin gradient updates on the very next training check.
    """

    def __init__(self, robot_steps_threshold: int, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = robot_steps_threshold
        self._triggered = False

    def _get_base_env(self, idx: int = 0):
        try:
            monitor_env = self.training_env.envs[idx]
            return getattr(monitor_env, 'env', monitor_env)
        except (AttributeError, IndexError):
            return None

    def _on_step(self) -> bool:
        if self._triggered:
            return True
        base_env = self._get_base_env()
        if base_env is None:
            return True
        robot_steps = getattr(base_env, 'robot_step_count', 0)
        if robot_steps >= self.threshold:
            self.model.learning_starts = 0
            self._triggered = True
            logger.info(
                f'[learning] {robot_steps} real robot actions collected — '
                f'SAC gradient updates now active.'
            )
        return True


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
# Audio history / on-demand dump callback
# ---------------------------------------------------------------------------

class AudioHistoryCallback(BaseCallback):
    """
    Maintains a rolling buffer of the last N real robot audio captures and
    writes them to disk on demand when the user presses a key.

    - Fills only on steps where an OSC command was actually sent (robot_step_count
      increments), so filtered/instant steps are invisible to the buffer.
    - Listens for keypresses on stdin in a non-blocking background thread.
      Default trigger key: 'a'  (mnemonic: audio).
    - Each dump goes into  <output_dir>/audio_dumps/dump_YYYYMMDD_HHMMSS/
      as WAV + JSON sidecar pairs, identical in format to success_recorder.
    """

    def __init__(self,
                 output_dir: Path,
                 history_size: int = 10,
                 trigger_key: str = 'a',
                 reward_override_cb: Optional['RewardOverrideCallback'] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.output_dir        = Path(output_dir)
        self.history_size      = history_size
        self.trigger_key       = trigger_key
        self.reward_override_cb = reward_override_cb  # Optional: dispatch y/n here
        self._buffer: deque = deque(maxlen=history_size)  # (audio, reward_info, rl_action)
        self._dump_event   = threading.Event()
        self._stop_event   = threading.Event()
        self._last_robot_step = 0
        self._listener_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Background stdin listener
    # ------------------------------------------------------------------

    def _stdin_listener(self) -> None:
        """Daemon thread: poll stdin for the trigger key without blocking.

        Single-char mode (ICANON+ECHO disabled, OPOST kept) is entered once
        for the lifetime of this thread and restored in the finally block.
        Keeping OPOST enabled means the kernel still inserts CR before every
        LF, so SB3's progress table prints correctly at all times.
        """
        try:
            import termios
            fd, old = _configure_stdin_single_char()
        except Exception:
            # No real tty (pipes, CI, etc.) — listener is a no-op
            return

        try:
          while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)

            if ch == self.trigger_key:
                if not self._dump_event.is_set():
                    self._dump_event.set()
                    sys.stdout.write(
                        f"\n[audio-history] Dump requested — "
                        f"{len(self._buffer)}/{self.history_size} clips queued.\n"
                    )
                    sys.stdout.flush()
            # Dispatch override/skip keys to RewardOverrideCallback when it is
            # active alongside --audio-history.  Handling them here avoids two
            # threads racing on the same stdin file descriptor.
            elif ch == RewardOverrideCallback.CONFIRM_KEY and self.reward_override_cb is not None:
                self.reward_override_cb.request_override(self.reward_override_cb.confirm_reward)
            elif ch == RewardOverrideCallback.REJECT_KEY and self.reward_override_cb is not None:
                self.reward_override_cb.request_override(self.reward_override_cb.reject_reward)
            elif ch in (' ', '\r', '\n') and self.reward_override_cb is not None:
                self.reward_override_cb.release_step()
            elif ch in ('\x03', 'q'):   # Ctrl+C or q — let SB3 handle it
                self._stop_event.set()
                import signal, os
                os.kill(os.getpid(), signal.SIGINT)
                break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, old)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SB3 callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._listener_thread = threading.Thread(
            target=self._stdin_listener, daemon=True, name="audio-history-listener"
        )
        self._listener_thread.start()
        keys_msg = f"'{self.trigger_key}' to dump audio"
        if self.reward_override_cb is not None:
            keys_msg += (
                f", '{RewardOverrideCallback.CONFIRM_KEY}' to confirm reward "
                f"({self.reward_override_cb.confirm_reward:+.1f}), "
                f"'{RewardOverrideCallback.REJECT_KEY}' to reject "
                f"({self.reward_override_cb.reject_reward:+.1f})"
            )
        logger.info(
            f"[audio-history] Listening on stdin — press {keys_msg}."
        )

    def _on_step(self) -> bool:
        # --- Fill buffer on new real robot steps ---
        try:
            base_env = self.training_env.envs[0]
            base_env = getattr(base_env, 'env', base_env)
        except (AttributeError, IndexError):
            return True

        robot_steps = getattr(base_env, 'robot_step_count', 0)
        if robot_steps > self._last_robot_step:
            self._last_robot_step = robot_steps
            audio       = getattr(base_env, 'last_audio', None)
            reward_info = getattr(base_env, 'last_reward_info', None)
            rl_action   = getattr(base_env, 'last_rl_action', None)
            if audio is not None:
                self._buffer.append((
                    audio.copy(),
                    dict(reward_info) if reward_info else {},
                    rl_action,
                ))

        # --- Drain on request ---
        if self._dump_event.is_set():
            self._dump_event.clear()
            self._save_buffer()

        return True

    def _on_training_end(self) -> None:
        self._stop_event.set()
        # Drain any pending request
        if self._dump_event.is_set():
            self._dump_event.clear()
            self._save_buffer()

    # ------------------------------------------------------------------
    # Disk writer
    # ------------------------------------------------------------------

    def _save_buffer(self) -> None:
        import soundfile as sf

        if not self._buffer:
            logger.warning('[audio-history] Buffer is empty — nothing to dump.')
            return

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_dir  = self.output_dir / f"dump_{ts}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        n = len(self._buffer)
        logger.info(f"[audio-history] Writing {n} clip(s) to {dump_dir}")

        for i, (audio, reward_info, rl_action) in enumerate(self._buffer, start=1):
            cls       = reward_info.get('classification') or {}
            fret      = reward_info.get('fret_position', 0.0)
            torque    = reward_info.get('torque', 0.0)
            str_idx   = reward_info.get('string_index', 0)
            device_sr = reward_info.get('device_sr', 44100)

            stem = (
                f"{i:03d}_str{str_idx}"
                f"_fret{fret:.2f}"
                f"_torque{torque:.0f}"
            )
            wav_path  = dump_dir / f"{stem}.wav"
            json_path = dump_dir / f"{stem}.json"

            sf.write(str(wav_path), audio.astype('float32'), device_sr, subtype='FLOAT')

            meta = {
                'dump_ts':         ts,
                'buffer_position': i,
                'buffer_size':     n,
                'string_index':    str_idx,
                'fret_position':   fret,
                'torque':          torque,
                'harmonic_prob':   cls.get('harmonic_prob'),
                'dead_prob':       cls.get('dead_prob'),
                'general_prob':    cls.get('general_prob'),
                'predicted_label': cls.get('predicted_label'),
                'total_reward':    reward_info.get('total_reward'),
                'audio_reward':    reward_info.get('audio_reward'),
                'fret_reward':     reward_info.get('fret_reward'),
                'torque_reward':   reward_info.get('torque_reward'),
                'device_sr':       device_sr,
                'wav_file':        wav_path.name,
                'filtered':        reward_info.get('filtered', False),
                'rl_action':       str(rl_action) if rl_action is not None else None,
            }
            json_path.write_text(__import__('json').dumps(meta, indent=2, default=str))

        logger.info(f"[audio-history] Dump complete: {dump_dir}")
        print(f"[audio-history] {n} clip(s) saved → {dump_dir}", flush=True)


# ---------------------------------------------------------------------------
# Researcher reward override callback
# ---------------------------------------------------------------------------

# These values deliberately exceed the normal reward ceiling so a researcher's
# override signal is unambiguous to the agent.  The normal per-step reward is
# roughly in [-0.1, +2.0] (filtration penalty → full-success step + bonus).
# Confirmed harmonics get +5.0; classifier false-positives get -3.0.
OVERRIDE_REWARD_CONFIRM: float =  5.0
OVERRIDE_REWARD_REJECT:  float = -3.0


class RewardOverrideCallback(BaseCallback):
    """
    Lets a babysitting researcher inject a strong reward signal directly into
    the SAC replay buffer, overriding whatever the classifier computed.

    Two outcomes:
        y  →  CONFIRM  (+5.0 by default)
               Works for ANY step — not just classifier successes.
               Use when:
                 • The agent played a real harmonic the classifier missed
                   (FALSE NEGATIVE — classifier said "not harmonic").
                 • The classifier correctly flagged a success but you want
                   to reinforce it more strongly.
               On CONFIRM, the replay buffer transition is also marked as
               terminal (done=True) regardless of what the classifier
               decided.  This makes the Q-target purely the override reward
               (+5.0) with no bootstrapping from the next state — giving
               the agent an unambiguous terminal success signal even when
               the episode would otherwise have continued.

        n  →  REJECT   (-3.0 by default)
               Use when the classifier fired a success on noise, a muted
               string, or any false positive.  The existing done=True flag
               in the buffer is preserved (episode correctly terminated),
               only the reward is replaced with a strong negative.

    The override is applied to the *most recently stored* replay buffer
    transition, which is the step the researcher just observed (assuming the
    keypress happens during or immediately after the ~3-second robot action).

    Stdin integration:
        If an AudioHistoryCallback instance is provided at construction,
        'y'/'n' are dispatched through its existing stdin thread so the two
        listeners don't race on the same file descriptor.  When running
        without --audio-history, this callback starts its own stdin thread.
    """

    CONFIRM_KEY = 'y'
    REJECT_KEY  = 'n'

    def __init__(self,
                 confirm_reward: float = OVERRIDE_REWARD_CONFIRM,
                 reject_reward:  float = OVERRIDE_REWARD_REJECT,
                 standalone: bool = True,
                 verbose: int = 0):
        """
        Args:
            confirm_reward: Reward injected on CONFIRM keypress (default +5.0).
            reject_reward:  Reward injected on REJECT  keypress (default -3.0).
            standalone:     True  → spin up own stdin listener thread.
                            False → rely on an external caller (e.g.
                                    AudioHistoryCallback) to call
                                    request_override() directly.
        """
        super().__init__(verbose)
        self.confirm_reward = confirm_reward
        self.reject_reward  = reject_reward
        self.standalone     = standalone
        self._pending: Optional[float] = None
        self._lock            = threading.Lock()
        self._stop_event      = threading.Event()
        # Gate that blocks _on_step after each real robot step until researcher
        # presses y, n, or SPACE/ENTER.  Pre-set so filtered/instant steps
        # that never hit the blocking path are unaffected.
        self._step_gate       = threading.Event()
        self._step_gate.set()
        self._last_robot_step   = 0
        # Buffer position pinned at the moment a robot step is detected.
        # SB3 calls on_step() BEFORE _store_transition(), so the transition
        # hasn't been written yet.  We save buf.pos (the slot it WILL occupy)
        # and apply the override at the TOP of the NEXT on_step() call, by
        # which time buf.add() has already run for that slot.
        self._target_buf_pos: Optional[int] = None
        # Classification / reward context saved at block-time so the apply
        # log is accurate even though it runs a step later.
        self._target_reward_info: dict = {}
        self._listener_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API (called from sibling callbacks or keyboard handler)
    # ------------------------------------------------------------------

    def request_override(self, value: float) -> None:
        """Queue a reward override and release the step gate.

        Only accepted while we are actively waiting (gate is clear).
        Keypresses that arrive between steps are discarded to prevent
        stale presses from being applied to the wrong replay buffer entry.
        """
        if self._step_gate.is_set():
            # Gate is already open — no robot step is waiting for a response.
            sys.stdout.write(
                f'\n[override] Key ignored — no robot step is currently waiting.\n'
                f'           Press y/n/SPACE only after the step prompt appears.\n'
            )
            sys.stdout.flush()
            return
        with self._lock:
            self._pending = value
        label = f'CONFIRM (+{value:.1f})' if value >= 0 else f'REJECT ({value:+.1f})'
        sys.stdout.write(f'\n[override] {label} — unblocking.\n')
        sys.stdout.flush()
        self._step_gate.set()

    def release_step(self) -> None:
        """Pass this step without any override.

        Only accepted while we are actively waiting (gate is clear).
        """
        if self._step_gate.is_set():
            sys.stdout.write(
                f'\n[override] Key ignored — no robot step is currently waiting.\n'
            )
            sys.stdout.flush()
            return
        sys.stdout.write('\n[override] pass — continuing.\n')
        sys.stdout.flush()
        self._step_gate.set()

    # ------------------------------------------------------------------
    # Standalone stdin listener (used when --audio-history is off)
    # ------------------------------------------------------------------

    def _stdin_listener(self) -> None:
        """Daemon thread: poll stdin for 'y' / 'n' without blocking training.

        Single-char mode (ICANON+ECHO disabled, OPOST kept) is entered once
        for the lifetime of this thread and restored in the finally block.
        """
        try:
            import termios
            fd, old = _configure_stdin_single_char()
        except Exception:
            return  # No real tty (pipes, CI, etc.)

        try:
          while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)

            if ch == self.CONFIRM_KEY:
                self.request_override(self.confirm_reward)
            elif ch == self.REJECT_KEY:
                self.request_override(self.reject_reward)
            elif ch in (' ', '\r', '\n'):   # SPACE or ENTER — pass without override
                self.release_step()
            elif ch in ('\x03', 'q'):
                self._stop_event.set()
                import signal as _signal, os
                os.kill(os.getpid(), _signal.SIGINT)
                break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, old)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SB3 callback hooks
    # ------------------------------------------------------------------

    def _get_base_env(self):
        """Unwrap DummyVecEnv → Monitor → HarmonicEnv."""
        try:
            env = self.training_env.envs[0]
            return getattr(env, 'env', env)
        except (AttributeError, IndexError):
            return None

    def _on_training_start(self) -> None:
        if self.standalone:
            self._listener_thread = threading.Thread(
                target=self._stdin_listener, daemon=True, name='reward-override-listener'
            )
            self._listener_thread.start()
        logger.info(
            f'[override] Researcher reward override ACTIVE\n'
            f'           Training PAUSES after every real robot step until you respond.\n'
            f'           "{self.CONFIRM_KEY}"          → CONFIRM as harmonic '
            f'(inject {self.confirm_reward:+.1f}, mark terminal; fixes false negatives)\n'
            f'           "{self.REJECT_KEY}"          → REJECT false-positive '
            f'(inject {self.reject_reward:+.1f})\n'
            f'           SPACE / ENTER  → pass, no override\n'
            f'           Filtered steps (no robot action) are never blocked.'
        )

    def _on_step(self) -> bool:
        base_env    = self._get_base_env()
        reward_info = getattr(base_env, 'last_reward_info', None) or {}
        buf         = getattr(self.model, 'replay_buffer', None)

        # Read the per-step filtered flag directly from env.step()'s info dict
        # (available in self.locals before _store_transition runs).
        # This is the ground-truth signal for whether OSC was actually sent,
        # avoids any robot_step_count counter drift, and works correctly across
        # episode boundaries.
        step_infos  = (self.locals or {}).get('infos') or [{}]
        step_info   = step_infos[0] if step_infos else {}
        is_filtered = step_info.get('filtered', True)   # default True = safe skip

        # ── Apply any pending override from the previous blocked step ─────────
        # SB3's ordering:  on_step()  →  _store_transition()  →  on_step() ...
        # So by the time we arrive here, buf.add() has already written the
        # transition that was pending from the last blocked call, and
        # _target_buf_pos holds the exact slot it landed in.
        target_pos = self._target_buf_pos
        if target_pos is not None:
            with self._lock:
                val           = self._pending
                self._pending = None
            self._target_buf_pos   = None          # clear regardless of val
            saved_info             = self._target_reward_info
            self._target_reward_info = {}

            if val is not None and buf is not None:
                old_reward = float(buf.rewards[target_pos, 0])
                old_done   = bool(buf.dones[target_pos, 0])

                h_prob    = saved_info.get('h_prob')
                cls_label = saved_info.get('cls_label', 'unknown')
                filtered  = saved_info.get('filtered', False)

                is_confirm = val > 0
                success_threshold        = getattr(base_env, 'success_threshold', 0.8)
                classifier_said_harmonic = (h_prob is not None
                                            and h_prob >= success_threshold)
                is_false_negative = (is_confirm
                                     and not classifier_said_harmonic
                                     and not filtered)

                buf.rewards[target_pos, 0] = float(val)
                # CONFIRM: mark terminal so Q-target = override reward only.
                if is_confirm:
                    buf.dones[target_pos, 0] = True

                if h_prob is not None:
                    ctx = f'classifier={cls_label}  H={h_prob:.3f}'
                elif filtered:
                    ctx = 'filtered step'
                else:
                    ctx = 'no classification'

                tag        = 'CONFIRM ✓' if is_confirm else 'REJECT  ✗'
                fn_note    = '  ← FALSE NEGATIVE CORRECTION' if is_false_negative else ''
                done_patch = (f'{old_done} → True'
                              if (is_confirm and not old_done) else str(old_done))

                logger.info(
                    f'\n{"=" * 60}\n'
                    f'  [override] {tag}{fn_note}\n'
                    f'  Buffer pos {target_pos}:\n'
                    f'    reward: {old_reward:+.4f}  →  {val:+.4f}\n'
                    f'    done:   {done_patch}\n'
                    f'    context: {ctx}\n'
                    f'{"=" * 60}\n'
                )
                fn_suffix = ' (false-negative correction)' if is_false_negative else ''
                sys.stdout.write(
                    f'  [override] {tag}{fn_suffix}  '
                    f'reward {old_reward:+.3f} → {val:+.3f}  |  {ctx}\n'
                )
                sys.stdout.flush()

        # ── Gate on new real robot steps ──────────────────────────────────────
        # Block on every step where OSC was actually sent (not filtered).
        # We use the per-step filtered flag from env.step()'s info dict
        # (self.locals['infos'][0]['filtered']) rather than robot_step_count so
        # there is no counter to drift or reset between episodes / resumes.
        if not is_filtered:
            try:

                # Clear any stale pending left over from a previous pass.
                with self._lock:
                    self._pending = None

                # Pin the slot where THIS transition WILL be stored.
                # on_step() fires BEFORE _store_transition(), so buf.pos is the
                # next-write pointer — exactly where this step's data will land.
                if buf is not None:
                    self._target_buf_pos = buf.pos % buf.buffer_size
                else:
                    self._target_buf_pos = None

                # Capture classification context NOW (reward_info is stale in the
                # next on_step() call where we actually apply the override).
                cls       = reward_info.get('classification') or {}
                h_prob    = cls.get('harmonic_prob')
                cls_label = cls.get('predicted_label', 'unknown')
                # Current reward comes from SB3's locals, NOT the buffer
                # (the transition hasn't been stored there yet).
                rewards_arr    = self.locals.get('rewards')
                current_reward = (float(rewards_arr[0])
                                  if rewards_arr is not None
                                  else reward_info.get('total_reward', 0.0))
                self._target_reward_info = {
                    'h_prob'    : h_prob,
                    'cls_label' : cls_label,
                    'filtered'  : is_filtered,
                }

                if h_prob is not None:
                    ctx = (f'classifier={cls_label}  H={h_prob:.3f}  '
                           f'reward={current_reward:+.3f}  buf_pos={self._target_buf_pos}')
                else:
                    ctx = (f'no classification  '
                           f'reward={current_reward:+.3f}  buf_pos={self._target_buf_pos}')

                # Make sure the gate is in a clean state before we wait.
                # Guard: if the previous wait released but the key was pressed
                # in the tiny window before this clear, we clear it here so
                # we don't skip this step.
                self._step_gate.clear()
                sys.stdout.write(
                    f'\n[override] Robot step complete — {ctx}\n'
                    f'           y=CONFIRM({self.confirm_reward:+.0f})  '
                    f'n=REJECT({self.reject_reward:+.0f})  '
                    f'SPACE/ENTER=pass\n'
                )
                sys.stdout.flush()

                # Block until key pressed.  300 s timeout auto-passes so a dead
                # terminal can't deadlock training indefinitely.
                triggered = self._step_gate.wait(timeout=300)
                if not triggered:
                    sys.stdout.write('\n[override] 5-minute timeout — auto-pass.\n')
                    sys.stdout.flush()
                # Do NOT apply here — return now so _store_transition() can run.
                # The apply happens at the top of the next _on_step() call.

            except Exception as exc:  # pragma: no cover
                logger.error(f'[override] Unexpected error in gate section: {exc}', exc_info=True)
                # Ensure gate is open so training doesn't deadlock.
                self._step_gate.set()

        return True

    def _on_training_end(self) -> None:
        self._stop_event.set()
        self._step_gate.set()   # unblock in case training ends while gate is held


# ---------------------------------------------------------------------------

def make_env(model_path, curriculum_mode: str, string_indices=None, string_index: int = 2,
             osc_port: int = 12000, audio_device: str = "Scarlett",
             reward_mode: str = 'full', offline: bool = False,
             success_recorder=None, temperature: float = 1.5):
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
        success_recorder=success_recorder,
        temperature=temperature,
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
    # Always create a fresh timestamped subdirectory so every run has its own
    # isolated logs, checkpoints, and best_model — even when resuming.
    # The source run directory is kept as resume_dir and used only for
    # locating the checkpoint/buffer to load from.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"harmonic_sac_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_dir = None
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            logger.error(f"Resume directory not found: {resume_dir}")
            sys.exit(1)
        # Record lineage so the chain of runs is traceable.
        (output_dir / "resumed_from.txt").write_text(str(resume_dir.resolve()) + "\n")
        logger.info(f"Resuming weights from: {resume_dir}")

    logger.info(f"Output directory: {output_dir}")

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Curriculum: {args.curriculum}")
    logger.info(f"Reward mode: {args.reward_mode}")
    logger.info(f"Classifier temperature: {args.temperature}")

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

    # Create success recorder if requested (online mode only)
    success_recorder = None
    if args.record_successes and not args.pretrain:
        success_recorder = SuccessRecorder(output_dir / "successes")
    elif args.record_successes and args.pretrain:
        logger.warning("[record-successes] No-op in --pretrain mode (no audio captured).")

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
        success_recorder=success_recorder,
        temperature=args.temperature,
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
        temperature=args.temperature,
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
        # In online mode, set learning_starts to a sentinel so SB3 never
        # auto-starts.  RobotLearningStartCallback controls the actual trigger
        # based on robot_step_count (real OSC actions, filtered steps excluded).
        # In pretrain mode every step is real, so pass the arg directly.
        learning_starts=args.learning_starts if args.pretrain else 10_000_000,
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
    if resume_dir is not None:
        def _latest_checkpoint(ckpt_dir: Path):
            """Return the .zip with the highest timestep number, or None."""
            zips = sorted(ckpt_dir.glob("harmonic_sac_*_steps.zip"))
            return zips[-1].with_suffix('') if zips else None

        resume_ckpt_dir = resume_dir / "checkpoints"
        if args.resume_checkpoint:
            model_file = Path(args.resume_checkpoint)
        elif (resume_dir / "interrupted_model.zip").exists():
            model_file = resume_dir / "interrupted_model"
        else:
            model_file = _latest_checkpoint(resume_ckpt_dir)

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
    if not args.pretrain:
        # In online mode, learning_starts on the SAC model is set to a sentinel
        # value (see SAC constructor above).  RobotLearningStartCallback lowers
        # it to 0 once enough real robot actions have been collected.
        cb_list.append(RobotLearningStartCallback(args.learning_starts))
    if args.slow and not args.pretrain:
        cb_list.append(SlowModeCallback())
        logger.info(
            '[slow] Slow-mode enabled: training will pause after every episode to plot audio. '
            'Close the plot window to continue.'
        )
    elif args.slow and args.pretrain:
        logger.warning('[slow] --slow has no effect in --pretrain mode (no audio captured).')

    if args.audio_history and not args.pretrain:
        # Build the override callback first (may be None) so it can be wired
        # into AudioHistoryCallback's stdin dispatcher to avoid stdin races.
        override_cb: Optional[RewardOverrideCallback] = None
        if args.override:
            override_cb = RewardOverrideCallback(
                confirm_reward=args.override_confirm_reward,
                reject_reward=args.override_reject_reward,
                standalone=False,   # AudioHistoryCallback owns the stdin thread
            )
            cb_list.append(override_cb)

        audio_history_cb = AudioHistoryCallback(
            output_dir=output_dir / 'audio_dumps',
            history_size=args.audio_history_size,
            reward_override_cb=override_cb,
        )
        cb_list.append(audio_history_cb)
        logger.info(
            f"[audio-history] Enabled — rolling buffer of last {args.audio_history_size} "
            f"robot audio captures. Press 'a' during training to dump to disk."
        )
    elif args.audio_history and args.pretrain:
        logger.warning('[audio-history] --audio-history has no effect in --pretrain mode (no audio captured).')
        if args.override and args.pretrain:
            logger.warning('[override] --override has no effect in --pretrain mode (no audio captured).')
    else:
        # No audio-history — wire standalone override if requested
        if args.override and not args.pretrain:
            override_cb = RewardOverrideCallback(
                confirm_reward=args.override_confirm_reward,
                reject_reward=args.override_reject_reward,
                standalone=True,    # Owns its own stdin listener thread
            )
            cb_list.append(override_cb)
        elif args.override and args.pretrain:
            logger.warning('[override] --override has no effect in --pretrain mode (no audio captured).')

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
        
        # Drain success recorder before closing env
        if success_recorder is not None:
            logger.info("[record-successes] Flushing pending writes...")
            success_recorder.close()

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
    parser.add_argument('--learning-starts', type=int, default=100,
                        help='Number of *real robot actions* (OSC sent, audio captured) to collect '
                             'before SAC gradient updates begin.  Filtered steps — which return '
                             'instantly without touching the robot — are excluded from this count. '
                             'In --pretrain mode this reverts to the standard SB3 total-timestep '
                             'threshold since every step is real (default: 100).')
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
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Temperature for classifier logit scaling before softmax (default: 1.5). '
                             'Values > 1 produce softer, less overconfident harmonic probabilities. '
                             'T=1.0 is equivalent to standard softmax.')
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')

    parser.add_argument('--record-successes', action='store_true', default=False,
                        help='Save every successful harmonic to disk for classifier retraining. '
                             'Each success produces a WAV (raw captured audio at device SR) and '
                             'a JSON sidecar with full action/reward/classification metadata. '
                             'Files are written in the background so step() timing is unaffected. '
                             'Output: <run-dir>/successes/  — review with --slow or any audio '
                             'player, edit suggested_label in the JSON, then feed back into '
                             'HarmonicsClassifier retraining. No-op in --pretrain mode.')

    # Slow / debug
    parser.add_argument('--slow', action='store_true', default=False,
                        help='After every episode, pause training and display a plot of the '
                             'captured audio waveform + mel spectrogram with classification '
                             'results and reward breakdown. Close the plot window to continue. '
                             'Use this to visually verify the classifier is hearing the note '
                             'before committing to a long run. No-op in --pretrain mode.')

    parser.add_argument('--audio-history', action='store_true', default=True,
                        help='Keep a rolling buffer of the last N real robot audio captures '
                             '(filtered/instant steps are excluded). Press \'a\' at any point '
                             'during training to dump the current buffer to '
                             '<run-dir>/audio_dumps/dump_TIMESTAMP/ as WAV + JSON pairs. '
                             'Useful for spot-checking what the robot is actually hearing '
                             'without interrupting the training run. No-op in --pretrain mode.')
    parser.add_argument('--audio-history-size', type=int, default=5, metavar='N',
                        help='Number of recent robot audio captures to keep in the rolling '
                             'buffer for --audio-history (default: 5).')

    # Researcher reward override
    parser.add_argument('--override', action='store_true', default=False,
                        help='Enable researcher reward override during babysitting. '
                             'Press "y" at any time during a robot action to CONFIRM the '
                             'step as a real harmonic — this works on ANY step, including '
                             'ones the classifier labelled as non-harmonic (false negatives). '
                             'The transition is marked terminal and given '
                             '--override-confirm-reward (default +5.0). '
                             'Press "n" to REJECT a classifier false-positive '
                             '(injects --override-reject-reward, default -3.0; keeps existing '
                             'done flag). '
                             'Both values far exceed the normal reward ceiling (~+2.0), so the '
                             'override unambiguously dominates the agent\'s learning. '
                             'The keypress is captured by the same stdin thread as '
                             '--audio-history when both are active (no race condition). '
                             'No-op in --pretrain mode.')
    parser.add_argument('--override-confirm-reward', type=float, default=OVERRIDE_REWARD_CONFIRM,
                        metavar='R',
                        help=f'Reward injected when the researcher presses "y" (CONFIRM). '
                             f'Should be >> the normal max reward (~+2.0). '
                             f'Default: {OVERRIDE_REWARD_CONFIRM}')
    parser.add_argument('--override-reject-reward', type=float, default=OVERRIDE_REWARD_REJECT,
                        metavar='R',
                        help=f'Reward injected when the researcher presses "n" (REJECT). '
                             f'Should be a strong negative well below 0. '
                             f'Default: {OVERRIDE_REWARD_REJECT}')

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
