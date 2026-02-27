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
from utils.reward import REWARD_MODE_FULL, REWARD_MODE_NO_FILTRATION

# D-string (string_index=2) fret → MIDI note number mapping.
# Source: GuitarBot/Recording/HARMONICS_RECORDING_README.md
#   Harmonic #1 (4th fret)  → MNN 78
#   Harmonic #2 (5th fret)  → MNN 74
#   Harmonic #3 (7th fret)  → MNN 69
_D_STRING_FRET_TO_PITCH: Dict[int, int] = {4: 78, 5: 74, 7: 69}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_policy(model, env, n_episodes=10, deterministic=True, render=False):
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
            'step_successes': [],   # 1.0 per robot step where harmonic_prob > 0.8
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
                trajectory['step_successes'].append(float(harmonic_prob > 0.8))
            else:
                trajectory['step_successes'].append(0.0)

            if render:
                env.render()

        # Episode metrics
        final_classification = last_info['classification']
        if final_classification is not None:
            final_harmonic_prob = final_classification['harmonic_prob']
            episode_succeeded = final_harmonic_prob > 0.8   # ended on a harmonic
        else:
            final_harmonic_prob = 0.0
            episode_succeeded = False

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
    env = HarmonicEnv(
        model_path=str(classifier_path),
        string_indices=string_indices,
        curriculum_mode=curriculum_mode,
        fixed_target_fret=fixed_target_fret,
        max_steps=10,
        success_recorder=success_recorder,
        reward_mode=reward_mode,
    )

    # Evaluate
    logger.info(f"Evaluating for {args.episodes} episodes "
                f"(strings={string_indices}, fret={args.target_fret or 'random'})...")
    results = evaluate_policy(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
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

    try:
        json_results = {k: v for k, v in results.items() if k != 'trajectories'}
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"Results saved to {results_path}")
    except Exception as exc:
        logger.error(f"Failed to save results JSON: {exc}")

    env.close()
    success_recorder.close()  # drain queue — ensures all WAVs are written before image analysis

    # Image analysis: compare success WAVs against dataset reference harmonics
    if args.image_analysis:
        run_image_analysis(
            results=results,
            output_dir=output_dir,
            ref_dir=Path(args.ref_dir),
            top_n=args.top_n,
            rank_by=args.rank_by,
            baseline=args.baseline,
            neg_ctrl_dir=Path(args.neg_ctrl_dir) if args.neg_ctrl_dir else None,
        )


if __name__ == '__main__':
    main()
