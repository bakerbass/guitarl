"""
Evaluation script for trained harmonic RL agent.

Usage:
    python evaluate.py --model runs/harmonic_sac_20250101/best_model/best_model.zip --episodes 20
    python evaluate.py --model runs/harmonic_sac_20250101/best_model/best_model.zip --target-fret 7 --visualize
"""

import argparse
from collections import defaultdict
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import sys

from stable_baselines3 import SAC
from env.harmonic_env import HarmonicEnv
from utils.success_recorder import SuccessRecorder
from utils.reward import REWARD_MODE_FULL, REWARD_MODE_NO_FILTRATION, COSINE_SIM_SUCCESS_THRESHOLD

# D-string (string_index=2) fret → MIDI note number mapping.
# Source: GuitarBot/Recording/HARMONICS_RECORDING_README.md
#   Harmonic #1 (4th fret)  → MNN 78
#   Harmonic #2 (5th fret)  → MNN 74
#   Harmonic #3 (7th fret)  → MNN 69
_D_STRING_FRET_TO_PITCH: Dict[int, int] = {4: 78, 5: 74, 7: 69}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_ref_mels(ref_dir: Path, fret_to_pitch: Dict[int, int]) -> Dict[int, list]:
    """Pre-load and mel-encode reference WAVs, keyed by fret. Returns {} on import error."""
    try:
        import librosa
        from image_analysis import compute_mel_db
    except ImportError as exc:
        logger.warning(f"[cosine-success] Cannot load reference mels: {exc}")
        return {}

    _SR = 22050
    _HOP = 512
    _MAX_ONSET_SAMPLES = _SR

    def _norm01(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    def _onset_align(y):
        frames = librosa.onset.onset_detect(y=y, sr=_SR, hop_length=_HOP)
        onset = 0
        if len(frames):
            cand = int(librosa.frames_to_samples(frames[0], hop_length=_HOP))
            if cand <= _MAX_ONSET_SAMPLES:
                onset = cand
        return y[onset:]

    ref_mels: Dict[int, list] = {}
    for fret, pitch in fret_to_pitch.items():
        wavs = sorted(ref_dir.glob(f"GB_NH*pitches{pitch}*.wav"))
        mels = []
        for wav in wavs:
            try:
                y, _ = librosa.load(str(wav), sr=_SR, mono=True)
                y = _onset_align(y)
                if len(y) >= int(0.5 * _SR):
                    mels.append(_norm01(compute_mel_db(y, sr=_SR)))
            except Exception as exc:
                logger.warning(f"[cosine-success] Skipped ref {wav.name}: {exc}")
        ref_mels[fret] = mels
        logger.info(f"[cosine-success] Fret {fret} (pitch {pitch}): {len(mels)} reference spectrograms loaded")
    return ref_mels


def _cosine_sim_for_audio(audio: np.ndarray, fret: int, ref_mels: Dict[int, list],
                           device_sr: int = 44100) -> float:
    """Compute best cosine similarity between one audio capture and reference mels for a fret."""
    try:
        import librosa
        from image_analysis import compute_mel_db, compute_similarity_metrics
    except ImportError:
        return 0.0

    _SR = 22050
    _HOP = 512
    _MAX_ONSET_SAMPLES = _SR

    def _norm01(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    def _onset_align(y):
        frames = librosa.onset.onset_detect(y=y, sr=_SR, hop_length=_HOP)
        onset = 0
        if len(frames):
            cand = int(librosa.frames_to_samples(frames[0], hop_length=_HOP))
            if cand <= _MAX_ONSET_SAMPLES:
                onset = cand
        return y[onset:]

    refs = ref_mels.get(fret, [])
    if not refs:
        return 0.0

    y = librosa.resample(audio.astype(np.float32), orig_sr=device_sr, target_sr=_SR) \
        if device_sr != _SR else audio.astype(np.float32)
    y = _onset_align(y)
    if len(y) < int(0.5 * _SR):
        return 0.0

    mel_rl = _norm01(compute_mel_db(y, sr=_SR))
    best_cos = 0.0
    for mel_ref in refs:
        t = min(mel_rl.shape[1], mel_ref.shape[1])
        m = compute_similarity_metrics(mel_rl[:, :t], mel_ref[:, :t])
        if m['cosine_sim'] > best_cos:
            best_cos = m['cosine_sim']
    return float(np.clip(best_cos, 0.0, 1.0))


def evaluate_policy(model, env, n_episodes=10, deterministic=True, render=False,
                    ref_mels: Optional[Dict[int, list]] = None, device_sr: int = 44100):
    """
    Evaluate trained policy.

    Filtered steps (actions rejected by the physics gate before reaching the
    robot) do not count toward max_steps in the env, so each episode runs
    until the requested number of real robot actions have been attempted.
    Only unfiltered steps are included in reward / metric accounting.

    Returns:
        Dictionary with evaluation metrics, including per-fret and per-string breakdowns.
    """
    episode_rewards = []
    episode_successes = []
    episode_step_success_rates = []   # per-episode fraction of robot steps that were harmonics
    episode_steps = []          # robot (unfiltered) steps only
    episode_attempts = []       # total step() calls including filtered
    episode_harmonic_probs = []
    episode_target_frets = []
    episode_string_indices = []
    episode_fret_positions = []
    episode_torques = []

    all_trajectories = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        trajectory = {
            'target_fret': info['target_fret'],
            'string_index': getattr(env, 'string_index', None),
            'positions': [],
            'torques': [],
            'rewards': [],
            'harmonic_probs': [],
            'cosine_sims': [],      # populated only in --cosine-success mode
            'step_successes': [],   # 1.0 per robot step where success criterion is met
            'audios': [],           # raw numpy float32 captures at device_sr (for image analysis)
        }

        last_info = info
        robot_steps = 0
        total_attempts = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_attempts += 1

            # Only score and record steps that actually reached the robot
            if info.get('filtered', False):
                last_info = info  # keep target_fret etc. up to date
                continue

            last_info = info
            robot_steps += 1
            episode_reward += reward

            # Record trajectory — use keys that harmonic_env.step() actually returns
            trajectory['positions'].append(info.get('slider_mm', info.get('fret_position', 0)))
            trajectory['torques'].append(info.get('torque', 0))
            trajectory['rewards'].append(reward)

            if info['classification'] is not None:
                harmonic_prob = info['classification']['harmonic_prob']
                trajectory['harmonic_probs'].append(harmonic_prob)
            else:
                harmonic_prob = 0.0

            # Store raw audio for post-hoc image analysis
            step_audio = getattr(env, 'last_audio', None)
            trajectory['audios'].append(step_audio.copy() if step_audio is not None else None)

            if ref_mels is not None:
                fret = info.get('target_fret', trajectory['target_fret'])
                cos_sim = _cosine_sim_for_audio(step_audio, fret, ref_mels, device_sr) \
                    if step_audio is not None else 0.0
                trajectory['cosine_sims'].append(cos_sim)
                step_success = cos_sim >= COSINE_SIM_SUCCESS_THRESHOLD
                trajectory['step_successes'].append(float(step_success))
                logger.info(
                    f"  Step {robot_steps}: pos={info.get('slider_mm', info.get('fret_position', 0)):.1f}  "
                    f"torque={info.get('torque', 0):.0f}  "
                    f"cos_sim={cos_sim:.4f}  "
                    f"h_prob={harmonic_prob:.3f}  "
                    f"{'SUCCESS' if step_success else 'fail'}"
                )
                if step_success:
                    break  # terminate episode on first cosine success
            else:
                trajectory['step_successes'].append(float(harmonic_prob > 0.8))

            if render:
                env.render()

        # Episode metrics
        final_classification = last_info['classification']
        if final_classification is not None:
            final_harmonic_prob = final_classification['harmonic_prob']
        else:
            final_harmonic_prob = 0.0

        if ref_mels is not None:
            # Cosine-success mode: episode succeeds if the final step met the cosine threshold
            episode_succeeded = bool(trajectory['cosine_sims'] and
                                     trajectory['cosine_sims'][-1] >= COSINE_SIM_SUCCESS_THRESHOLD)
        else:
            episode_succeeded = final_harmonic_prob > 0.8

        # Step-level: fraction of robot steps that were harmonics this episode
        step_succs = trajectory['step_successes']
        ep_step_success = float(np.mean(step_succs)) if step_succs else 0.0

        target_fret = last_info.get('target_fret', trajectory['target_fret'])
        string_index = last_info.get('string_index', trajectory['string_index'])

        episode_rewards.append(episode_reward)
        episode_successes.append(float(episode_succeeded))
        episode_steps.append(robot_steps)
        episode_attempts.append(total_attempts)
        episode_harmonic_probs.append(final_harmonic_prob)
        episode_target_frets.append(target_fret)
        episode_string_indices.append(string_index)
        episode_fret_positions.append(np.mean(trajectory['positions']) if trajectory['positions'] else 0.0)
        episode_torques.append(np.mean(trajectory['torques']) if trajectory['torques'] else 0.0)
        episode_step_success_rates.append(ep_step_success)

        all_trajectories.append(trajectory)

        if ref_mels is not None:
            final_cos_sim = trajectory['cosine_sims'][-1] if trajectory['cosine_sims'] else 0.0
            mean_cos_sim = float(np.mean(trajectory['cosine_sims'])) if trajectory['cosine_sims'] else 0.0
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Fret={target_fret}, String={string_index}, "
                f"Reward={episode_reward:.3f}, "
                f"Episode success={episode_succeeded}, "
                f"Step success={ep_step_success:.1%} ({sum(int(s) for s in step_succs)}/{robot_steps} steps), "
                f"cos_sim final={final_cos_sim:.4f}  mean={mean_cos_sim:.4f}, "
                f"H-prob={final_harmonic_prob:.3f}, "
                f"Robot steps={robot_steps}/{total_attempts} attempts"
            )
        else:
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Fret={target_fret}, String={string_index}, "
                f"Reward={episode_reward:.3f}, "
                f"Episode success={episode_succeeded}, "
                f"Step success={ep_step_success:.1%} ({sum(int(s) for s in step_succs)}/{robot_steps} steps), "
                f"H-prob={final_harmonic_prob:.3f}, "
                f"Robot steps={robot_steps}/{total_attempts} attempts"
            )

    # ── Aggregate metrics ────────────────────────────────────────────
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'episode_success_rate': float(np.mean(episode_successes)),
        'step_success_rate': float(np.mean(episode_step_success_rates)),
        'mean_steps': float(np.mean(episode_steps)),
        'mean_attempts': float(np.mean(episode_attempts)),
        'filter_rate': float(1.0 - np.sum(episode_steps) / max(np.sum(episode_attempts), 1)),
        'mean_harmonic_prob': float(np.mean(episode_harmonic_probs)),
        'mean_position': float(np.mean(episode_fret_positions)),
        'mean_torque': float(np.mean(episode_torques)),
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'episode_step_success_rates': episode_step_success_rates,
        'episode_harmonic_probs': episode_harmonic_probs,
        'episode_target_frets': episode_target_frets,
        'episode_string_indices': episode_string_indices,
        'trajectories': all_trajectories,
    }

    # ── Per-fret breakdown ───────────────────────────────────────────
    results_by_fret = {}
    for fret in sorted(set(episode_target_frets)):
        mask = [i for i, f in enumerate(episode_target_frets) if f == fret]
        if not mask:
            continue
        results_by_fret[str(fret)] = {
            'n_episodes': len(mask),
            'episode_success_rate': float(np.mean([episode_successes[i] for i in mask])),
            'step_success_rate': float(np.mean([episode_step_success_rates[i] for i in mask])),
            'mean_harmonic_prob': float(np.mean([episode_harmonic_probs[i] for i in mask])),
            'mean_reward': float(np.mean([episode_rewards[i] for i in mask])),
            'mean_steps': float(np.mean([episode_steps[i] for i in mask])),
        }
    results['results_by_fret'] = results_by_fret

    # ── Per-string breakdown ─────────────────────────────────────────
    results_by_string = {}
    for string in sorted(set(s for s in episode_string_indices if s is not None)):
        mask = [i for i, s in enumerate(episode_string_indices) if s == string]
        if not mask:
            continue
        results_by_string[str(string)] = {
            'n_episodes': len(mask),
            'episode_success_rate': float(np.mean([episode_successes[i] for i in mask])),
            'step_success_rate': float(np.mean([episode_step_success_rates[i] for i in mask])),
            'mean_harmonic_prob': float(np.mean([episode_harmonic_probs[i] for i in mask])),
            'mean_reward': float(np.mean([episode_rewards[i] for i in mask])),
            'mean_steps': float(np.mean([episode_steps[i] for i in mask])),
        }
    results['results_by_string'] = results_by_string

    return results


def visualize_results(results: Dict, output_path: Path = None):
    """Visualize evaluation results including per-fret and per-string breakdowns."""
    has_breakdown = bool(results.get('results_by_fret'))
    ncols = 2
    nrows = 3 if has_breakdown else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))

    # Plot 1: Rewards over episodes
    ax = axes[0, 0]
    ax.plot(results['episode_rewards'], marker='o', markersize=4)
    ax.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate
    ax = axes[0, 1]
    window_size = 5
    successes = np.array(results['episode_successes'])
    if len(successes) >= window_size:
        smoothed = np.convolve(successes, np.ones(window_size) / window_size, mode='valid')
        ax.plot(range(window_size - 1, len(successes)), smoothed, marker='o', markersize=4)
    else:
        ax.plot(successes, marker='o', markersize=4)
    ax.axhline(y=results['success_rate'], color='r', linestyle='--',
               label=f'Mean: {results["success_rate"]:.3f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success (5-ep moving avg)')
    ax.set_title('Success Rate Over Episodes')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Position trajectories (first 3 episodes)
    ax = axes[1, 0]
    target_positions_mm = {4: 112.0, 5: 139.0, 7: 187.0}
    for i, traj in enumerate(results['trajectories'][:3]):
        steps = range(len(traj['positions']))
        ax.plot(steps, traj['positions'], marker='o', markersize=4,
                label=f"Ep {i + 1} (Fret {traj['target_fret']})")
        target_pos = target_positions_mm.get(traj['target_fret'], 0)
        ax.axhline(y=target_pos, linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Position Trajectories (First 3 Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Harmonic probability over steps
    ax = axes[1, 1]
    for i, traj in enumerate(results['trajectories'][:3]):
        if traj['harmonic_probs']:
            steps = range(len(traj['harmonic_probs']))
            ax.plot(steps, traj['harmonic_probs'], marker='o', markersize=4,
                    label=f"Ep {i + 1} (Fret {traj['target_fret']})")
    ax.axhline(y=0.8, color='r', linestyle='--', label='Success threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Harmonic Probability')
    ax.set_title('Harmonic Quality Over Steps (First 3 Episodes)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plots 5 & 6: Per-fret and per-string breakdowns (only if data available)
    if has_breakdown:
        # Per-fret success rate
        ax = axes[2, 0]
        fret_data = results['results_by_fret']
        frets = sorted(int(k) for k in fret_data)
        sr_by_fret = [fret_data[str(f)]['success_rate'] for f in frets]
        n_by_fret = [fret_data[str(f)]['n_episodes'] for f in frets]
        bars = ax.bar([str(f) for f in frets], sr_by_fret, color='steelblue', alpha=0.8)
        for bar, n in zip(bars, n_by_fret):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'n={n}', ha='center', va='bottom', fontsize=9)
        ax.axhline(y=results['success_rate'], color='r', linestyle='--', label='Overall mean')
        ax.set_xlabel('Target Fret')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Target Fret')
        ax.set_ylim([0, 1.15])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Per-string success rate
        ax = axes[2, 1]
        string_data = results.get('results_by_string', {})
        if string_data:
            strings = sorted(int(k) for k in string_data)
            sr_by_string = [string_data[str(s)]['success_rate'] for s in strings]
            n_by_string = [string_data[str(s)]['n_episodes'] for s in strings]
            bars = ax.bar([f"String {s}" for s in strings], sr_by_string, color='darkorange', alpha=0.8)
            for bar, n in zip(bars, n_by_string):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'n={n}', ha='center', va='bottom', fontsize=9)
            ax.axhline(y=results['success_rate'], color='r', linestyle='--', label='Overall mean')
            ax.set_xlabel('String Index')
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rate by String')
            ax.set_ylim([0, 1.15])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            axes[2, 1].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")

    plt.show()


def print_summary(results: Dict):
    """Print evaluation summary with per-fret and per-string breakdowns."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:            {results['n_episodes']}")
    print(f"Mean Reward:         {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Episode Success:     {results['episode_success_rate']:.1%}  (≥1 harmonic step per episode)")
    print(f"Step Success:        {results['step_success_rate']:.1%}  (harmonic steps / total robot steps)")
    print(f"Mean Harmonic Prob:  {results['mean_harmonic_prob']:.3f}")
    print(f"Mean Robot Steps:    {results['mean_steps']:.1f}  "
          f"(attempts: {results.get('mean_attempts', results['mean_steps']):.1f}, "
          f"filter rate: {results.get('filter_rate', 0.0):.1%})")
    print(f"Mean Position:       {results['mean_position']:.1f} mm")
    print(f"Mean Torque:         {results['mean_torque']:.1f}")

    if results.get('results_by_fret'):
        print("\n── Per-fret breakdown ──────────────────────────────────")
        print(f"  {'Fret':>5}  {'N':>5}  {'Ep.Succ':>8}  {'StpSucc':>8}  {'H-prob':>7}  {'Reward':>8}")
        for fret, d in sorted(results['results_by_fret'].items(), key=lambda x: int(x[0])):
            print(f"  {fret:>5}  {d['n_episodes']:>5}  "
                  f"{d['episode_success_rate']:>8.1%}  {d['step_success_rate']:>8.1%}  "
                  f"{d['mean_harmonic_prob']:>7.3f}  {d['mean_reward']:>8.3f}")

    if results.get('results_by_string'):
        print("\n── Per-string breakdown ────────────────────────────────")
        print(f"  {'String':>6}  {'N':>5}  {'Ep.Succ':>8}  {'StpSucc':>8}  {'H-prob':>7}  {'Reward':>8}")
        for string, d in sorted(results['results_by_string'].items(), key=lambda x: int(x[0])):
            print(f"  {string:>6}  {d['n_episodes']:>5}  "
                  f"{d['episode_success_rate']:>8.1%}  {d['step_success_rate']:>8.1%}  "
                  f"{d['mean_harmonic_prob']:>7.3f}  {d['mean_reward']:>8.3f}")

    print("=" * 60 + "\n")


def _get_success_wavs_for_fret(successes_dir: Path, target_fret: int) -> List[Path]:
    """Return WAV paths from successes_dir whose JSON sidecar matches target_fret."""
    import json as _json
    wavs = []
    for json_path in sorted(successes_dir.glob("*.json")):
        try:
            data = _json.loads(json_path.read_text())
            if data.get("target_fret") == target_fret:
                wav_path = json_path.with_suffix(".wav")
                if wav_path.exists():
                    wavs.append(wav_path)
        except Exception:
            pass
    return sorted(wavs)


def run_step_image_analysis(
    trajectories: List[Dict],
    ref_dir: Path,
    fret_to_pitch: Dict[int, int],
    device_sr: int = 44100,
) -> Dict:
    """Compute cosine similarity and MSE between every captured robot-step audio and
    the matching reference WAVs.  Operates fully in-memory — no disk writes needed.

    Uses the same onset-alignment and mel-spectrogram parameters as image_analysis.py
    so results are directly comparable to batch_compare() output.

    Returns a summary dict (also printed to stdout).
    """
    from image_analysis import compute_mel_db, compute_similarity_metrics
    import librosa

    _SR     = 22050   # image_analysis.SR
    _HOP    = 512     # image_analysis.HOP_LENGTH
    _MAX_ONSET_SAMPLES = _SR   # MAX_ONSET_SEC = 1 second

    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    def _onset_align(y: np.ndarray) -> np.ndarray:
        frames = librosa.onset.onset_detect(y=y, sr=_SR, hop_length=_HOP)
        onset = 0
        if len(frames):
            cand = int(librosa.frames_to_samples(frames[0], hop_length=_HOP))
            if cand <= _MAX_ONSET_SAMPLES:
                onset = cand
        return y[onset:]

    # ── Pre-load reference mel spectrograms per fret ─────────────────
    ref_mels: Dict[int, List[np.ndarray]] = {}
    for fret, pitch in fret_to_pitch.items():
        wavs = sorted(ref_dir.glob(f"GB_NH*pitches{pitch}*.wav"))
        mels = []
        for wav in wavs:
            try:
                y, _ = librosa.load(str(wav), sr=_SR, mono=True)
                y = _onset_align(y)
                if len(y) >= int(0.5 * _SR):
                    mels.append(_norm01(compute_mel_db(y, sr=_SR)))
            except Exception as exc:
                logger.warning(f"[image-analysis] Skipped ref {wav.name}: {exc}")
        ref_mels[fret] = mels
        logger.info(
            f"[image-analysis] Fret {fret} (pitch {pitch}): {len(mels)} reference spectrograms"
        )

    # ── Score every recorded robot step ──────────────────────────────
    step_metrics: List[Dict] = []
    for traj in trajectories:
        fret = traj['target_fret']
        refs = ref_mels.get(fret, [])
        for audio in traj.get('audios', []):
            if audio is None:
                continue
            # Resample → onset-align → mel
            y = librosa.resample(
                audio.astype(np.float32), orig_sr=device_sr, target_sr=_SR
            ) if device_sr != _SR else audio.astype(np.float32)
            y = _onset_align(y)
            if len(y) < int(0.5 * _SR):
                step_metrics.append({'fret': fret, 'cosine_sim': 0.0, 'mse': 1.0})
                continue
            if not refs:
                step_metrics.append({'fret': fret, 'cosine_sim': 0.0, 'mse': 1.0})
                continue
            mel_rl = _norm01(compute_mel_db(y, sr=_SR))
            best_cos, best_mse = -1.0, 1.0
            for mel_ref in refs:
                t = min(mel_rl.shape[1], mel_ref.shape[1])
                m = compute_similarity_metrics(mel_rl[:, :t], mel_ref[:, :t])
                if m['cosine_sim'] > best_cos:
                    best_cos, best_mse = m['cosine_sim'], m['mse']
            step_metrics.append({
                'fret': fret,
                'cosine_sim': float(np.clip(best_cos, 0.0, 1.0)),
                'mse': float(best_mse),
            })

    if not step_metrics:
        logger.warning('[image-analysis] No audio captured — nothing to analyze.')
        return {}

    cos_all = np.array([m['cosine_sim'] for m in step_metrics])
    mse_all = np.array([m['mse']        for m in step_metrics])

    per_fret: Dict[int, Dict] = {}
    for fret in sorted({m['fret'] for m in step_metrics}):
        fc = np.array([m['cosine_sim'] for m in step_metrics if m['fret'] == fret])
        fm = np.array([m['mse']        for m in step_metrics if m['fret'] == fret])
        per_fret[fret] = {
            'n':              int(len(fc)),
            'mean_cosine_sim': float(fc.mean()),
            'std_cosine_sim':  float(fc.std(ddof=1) if len(fc) > 1 else 0.0),
            'mean_mse':        float(fm.mean()),
            'std_mse':         float(fm.std(ddof=1) if len(fm) > 1 else 0.0),
        }

    sep = "─" * 60
    print(f"\n{sep}")
    print("IMAGE ANALYSIS  (every robot step vs reference WAVs)")
    print(sep)
    print(f"  Total steps:  {len(step_metrics)}")
    print(f"  Cosine sim:   {cos_all.mean():.4f} \u00b1 {cos_all.std(ddof=1):.4f}  "
          f"(min {cos_all.min():.4f}  max {cos_all.max():.4f})")
    print(f"  MSE:          {mse_all.mean():.4f} \u00b1 {mse_all.std(ddof=1):.4f}  "
          f"(min {mse_all.min():.4f}  max {mse_all.max():.4f})")
    if per_fret:
        print(f"\n  {'Fret':>5}  {'N':>5}  {'CosSim':>8}  {'±':>7}  {'MSE':>8}  {'±':>7}")
        for fret, d in sorted(per_fret.items()):
            print(f"  {fret:>5}  {d['n']:>5}  {d['mean_cosine_sim']:>8.4f}  "
                  f"{d['std_cosine_sim']:>7.4f}  {d['mean_mse']:>8.4f}  {d['std_mse']:>7.4f}")
    print(sep + "\n")

    return {
        'mean_cosine_sim': float(cos_all.mean()),
        'std_cosine_sim':  float(cos_all.std(ddof=1)),
        'mean_mse':        float(mse_all.mean()),
        'std_mse':         float(mse_all.std(ddof=1)),
        'n_steps':         len(step_metrics),
        'per_fret':        {str(k): v for k, v in per_fret.items()},
        'step_metrics':    step_metrics,
    }


def run_image_analysis(
    results: Dict,
    output_dir: Path,
    ref_dir: Path,
    top_n: int = 1,
    rank_by: str = "ssim",
    baseline: bool = False,
    neg_ctrl_dir: Optional[Path] = None,
) -> None:
    """Run image_analysis batch_compare for each fret present in evaluation results.

    For each evaluated target fret, filters the success WAVs by fret (via JSON
    sidecars), resolves the correct reference pitch for the D string, and calls
    batch_compare() from image_analysis.py.

    Results are saved to output_dir/image_analysis_fret{N}/.
    """
    from image_analysis import batch_compare

    successes_dir = output_dir / "successes"
    if not successes_dir.exists():
        logger.warning(
            f"No successes directory at {successes_dir} — "
            "skipping image analysis (no success WAVs were recorded)"
        )
        return

    evaluated_frets = sorted(set(results['episode_target_frets']))

    for fret in evaluated_frets:
        pitch = _D_STRING_FRET_TO_PITCH.get(fret)
        if pitch is None:
            logger.warning(
                f"No D-string pitch mapping for fret {fret} — skipping image analysis for this fret"
            )
            continue

        rl_wavs = _get_success_wavs_for_fret(successes_dir, fret)
        if not rl_wavs:
            logger.info(f"No success WAVs found for fret {fret} — skipping image analysis")
            continue

        pitch_pattern = f"pitches{pitch}"
        img_output_dir = output_dir / f"image_analysis_fret{fret}"
        logger.info(
            f"Image analysis: fret={fret}, pitch={pitch}, "
            f"{len(rl_wavs)} RL WAV(s), ref pattern={pitch_pattern}"
        )

        batch_compare(
            rl_dir=successes_dir,
            ref_dir=ref_dir,
            pitch_pattern=pitch_pattern,
            rl_wavs=rl_wavs,
            top_n=top_n,
            rank_by=rank_by,
            output_dir=img_output_dir,
            play_audio=False,
            baseline=baseline,
            neg_ctrl_dir=neg_ctrl_dir,
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluate harmonic RL agent')
    parser.add_argument('--model', required=True, help='Path to trained model (.zip)')
    parser.add_argument('--model-classifier',
                        default='../HarmonicsClassifier/models/best_model.pt',
                        help='Path to HarmonicsClassifier model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--string-index', type=int, default=2, help='String to evaluate on')
    parser.add_argument('--string-indices', type=int, nargs='+', default=None,
                        help='Multiple strings to rotate across (e.g. --string-indices 0 2 4)')
    parser.add_argument('--target-fret', type=int, choices=[4, 5, 7], default=None,
                        help='Specific fret to test (default: random across all)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--visualize', action='store_true', help='Show visualization plots')
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')

    # Image analysis
    parser.add_argument('--image-analysis', action='store_true',
                        help='Run image_analysis batch comparison after evaluation')
    parser.add_argument('--ref-dir', type=str,
                        default='../HarmonicsClassifier/note_clips/harmonic',
                        help='Reference WAV directory for image analysis (default: %(default)s)')
    parser.add_argument('--top-n', type=int, default=1,
                        help='Number of top pairs to plot in image analysis (default: %(default)s)')
    parser.add_argument('--rank-by', type=str, default='ssim',
                        choices=['cosine_sim', 'mse', 'ssim', 'pearson_r'],
                        help='Ranking metric for image analysis (default: %(default)s)')
    parser.add_argument('--baseline', action='store_true',
                        help='Compute ref-vs-ref similarity ceiling in image analysis')
    parser.add_argument('--neg-ctrl-dir', type=str, default=None, metavar='DIR',
                        help='Dead-note WAVs directory for negative control in image analysis')

    parser.add_argument('--cosine-success', action='store_true', default=False,
                        help='Use cosine similarity >= 0.85 (vs reference WAVs in --ref-dir) as '
                             'the success criterion instead of the CNN classifier harmonic '
                             'probability. Requires --ref-dir. Applied per-step and per-episode.')

    parser.add_argument('--no-filtration', action='store_true', default=False,
                        help='Disable the physics filtration gate (Layer 1) so that every action '
                             'is sent to the robot regardless of torque/fret range.  Useful when '
                             'evaluating a model trained with different filtration thresholds that '
                             'would otherwise block all steps.')

    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    classifier_path = Path(args.model_classifier)
    if not classifier_path.exists():
        logger.error(f"Classifier model not found: {classifier_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    # Resolve string pool
    string_indices = args.string_indices if args.string_indices else [args.string_index]

    # Create success recorder — saves WAV + JSON for each successful step
    success_recorder = SuccessRecorder(output_dir / "successes")

    # Create environment
    curriculum_mode = 'random' if args.target_fret is None else 'fixed_fret'
    fixed_target_fret = args.target_fret if args.target_fret is not None else 7
    reward_mode = REWARD_MODE_NO_FILTRATION if args.no_filtration else REWARD_MODE_FULL
    if args.no_filtration:
        logger.info(
            '[eval] --no-filtration: physics gate disabled — all actions will be sent to the robot.'
        )
    # max_steps=10 means 10 real robot steps per episode — filtered steps
    # do not count (harmonic_env only increments current_step when unfiltered).
    # In cosine-success mode the env must not self-terminate on harmonic_prob —
    # episode termination is controlled entirely by the cosine threshold break
    # in evaluate_policy. Set success_threshold=2.0 (unreachable) to disable it.
    success_threshold = 2.0 if args.cosine_success else 0.8
    env = HarmonicEnv(
        model_path=str(classifier_path),
        string_indices=string_indices,
        curriculum_mode=curriculum_mode,
        fixed_target_fret=fixed_target_fret,
        max_steps=10,
        success_threshold=success_threshold,
        success_recorder=success_recorder,
        reward_mode=reward_mode,
    )

    # Pre-load reference mels for cosine-success mode
    ref_mels_for_eval: Optional[Dict[int, list]] = None
    if args.cosine_success:
        ref_dir_path = Path(args.ref_dir)
        if not ref_dir_path.exists():
            logger.error(f"--cosine-success requires --ref-dir, but {ref_dir_path} does not exist.")
            sys.exit(1)
        ref_mels_for_eval = _load_ref_mels(ref_dir_path, _D_STRING_FRET_TO_PITCH)
        if not any(ref_mels_for_eval.values()):
            logger.error("No reference spectrograms loaded — cannot use --cosine-success.")
            sys.exit(1)
        logger.info("[cosine-success] Success criterion: cosine_sim >= "
                    f"{COSINE_SIM_SUCCESS_THRESHOLD} (vs reference WAVs)")

    device_sr: int = getattr(getattr(env, 'reward_calc', None), 'device_sr', 44100)

    # Evaluate
    logger.info(f"Evaluating for {args.episodes} episodes "
                f"(strings={string_indices}, fret={args.target_fret or 'random'})...")
    results = evaluate_policy(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        ref_mels=ref_mels_for_eval,
        device_sr=device_sr,
    )

    # Print summary
    print_summary(results)

    # Visualize
    if args.visualize:
        viz_path = output_dir / f"evaluation_{model_path.stem}.png"
        visualize_results(results, output_path=viz_path)

    # Save results JSON (non-fatal — image analysis still runs if this fails)
    import json
    results_path = output_dir / f"results_{model_path.stem}.json"

    class _NumpyEncoder(json.JSONEncoder):
        """Recursively coerce numpy scalars/arrays to native Python types."""
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    json_results = {k: v for k, v in results.items() if k != 'trajectories'}

    def _save_json():
        try:
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2, cls=_NumpyEncoder)
            logger.info(f"Results saved to {results_path}")
        except Exception as exc:
            logger.error(f"Failed to save results JSON: {exc}")

    _save_json()

    env.close()
    success_recorder.close()  # drain queue — ensures all WAVs are written before image analysis

    # Image analysis
    if args.image_analysis:
        ref_dir_path = Path(args.ref_dir)

        # ── 1. Per-step inline metrics (cosine sim + MSE for every robot step) ──
        img_stats = run_step_image_analysis(
            trajectories=results['trajectories'],
            ref_dir=ref_dir_path,
            fret_to_pitch=_D_STRING_FRET_TO_PITCH,
            device_sr=device_sr,
        )
        if img_stats:
            results['image_analysis'] = img_stats
            json_results['image_analysis'] = img_stats
            _save_json()

        # ── 2. Visual batch comparison against success WAVs (plots + stats table) ──
        run_image_analysis(
            results=results,
            output_dir=output_dir,
            ref_dir=ref_dir_path,
            top_n=args.top_n,
            rank_by=args.rank_by,
            baseline=args.baseline,
            neg_ctrl_dir=Path(args.neg_ctrl_dir) if args.neg_ctrl_dir else None,
        )


if __name__ == '__main__':
    main()
