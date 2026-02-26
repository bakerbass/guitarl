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
from typing import List, Dict
import sys

from stable_baselines3 import SAC
from env.harmonic_env import HarmonicEnv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_policy(model, env, n_episodes=10, deterministic=True, render=False):
    """
    Evaluate trained policy.

    Returns:
        Dictionary with evaluation metrics, including per-fret and per-string breakdowns.
    """
    episode_rewards = []
    episode_successes = []
    episode_steps = []
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
        step_count = 0

        trajectory = {
            'target_fret': info['target_fret'],
            'string_index': getattr(env, 'string_index', None),
            'positions': [],
            'torques': [],
            'rewards': [],
            'harmonic_probs': [],
        }

        last_info = info
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            last_info = info

            episode_reward += reward
            step_count += 1

            # Record trajectory — use keys that harmonic_env.step() actually returns
            trajectory['positions'].append(info.get('slider_mm', info.get('fret_position', 0)))
            trajectory['torques'].append(info.get('torque', 0))
            trajectory['rewards'].append(reward)

            if info['classification'] is not None:
                harmonic_prob = info['classification']['harmonic_prob']
                trajectory['harmonic_probs'].append(harmonic_prob)

            if render:
                env.render()

        # Episode metrics
        final_classification = last_info['classification']
        if final_classification is not None:
            final_harmonic_prob = final_classification['harmonic_prob']
            success = final_harmonic_prob > 0.8
        else:
            final_harmonic_prob = 0.0
            success = False

        target_fret = last_info.get('target_fret', trajectory['target_fret'])
        string_index = last_info.get('string_index', trajectory['string_index'])

        episode_rewards.append(episode_reward)
        episode_successes.append(float(success))
        episode_steps.append(step_count)
        episode_harmonic_probs.append(final_harmonic_prob)
        episode_target_frets.append(target_fret)
        episode_string_indices.append(string_index)
        episode_fret_positions.append(np.mean(trajectory['positions']) if trajectory['positions'] else 0.0)
        episode_torques.append(np.mean(trajectory['torques']) if trajectory['torques'] else 0.0)

        all_trajectories.append(trajectory)

        logger.info(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Fret={target_fret}, String={string_index}, "
            f"Reward={episode_reward:.3f}, "
            f"Success={success}, "
            f"Harmonic prob={final_harmonic_prob:.3f}, "
            f"Steps={step_count}"
        )

    # ── Aggregate metrics ────────────────────────────────────────────
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'success_rate': float(np.mean(episode_successes)),
        'mean_steps': float(np.mean(episode_steps)),
        'mean_harmonic_prob': float(np.mean(episode_harmonic_probs)),
        'mean_position': float(np.mean(episode_fret_positions)),
        'mean_torque': float(np.mean(episode_torques)),
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
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
            'success_rate': float(np.mean([episode_successes[i] for i in mask])),
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
            'success_rate': float(np.mean([episode_successes[i] for i in mask])),
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
    print(f"Success Rate:        {results['success_rate']:.1%}")
    print(f"Mean Harmonic Prob:  {results['mean_harmonic_prob']:.3f}")
    print(f"Mean Steps/Episode:  {results['mean_steps']:.1f}")
    print(f"Mean Position:       {results['mean_position']:.1f} mm")
    print(f"Mean Torque:         {results['mean_torque']:.1f}")

    if results.get('results_by_fret'):
        print("\n── Per-fret breakdown ──────────────────────────────────")
        print(f"  {'Fret':>5}  {'N':>5}  {'Success':>8}  {'H-prob':>7}  {'Reward':>8}")
        for fret, d in sorted(results['results_by_fret'].items(), key=lambda x: int(x[0])):
            print(f"  {fret:>5}  {d['n_episodes']:>5}  "
                  f"{d['success_rate']:>8.1%}  {d['mean_harmonic_prob']:>7.3f}  "
                  f"{d['mean_reward']:>8.3f}")

    if results.get('results_by_string'):
        print("\n── Per-string breakdown ────────────────────────────────")
        print(f"  {'String':>6}  {'N':>5}  {'Success':>8}  {'H-prob':>7}  {'Reward':>8}")
        for string, d in sorted(results['results_by_string'].items(), key=lambda x: int(x[0])):
            print(f"  {string:>6}  {d['n_episodes']:>5}  "
                  f"{d['success_rate']:>8.1%}  {d['mean_harmonic_prob']:>7.3f}  "
                  f"{d['mean_reward']:>8.3f}")

    print("=" * 60 + "\n")


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

    # Create environment
    curriculum_mode = 'random' if args.target_fret is None else 'fixed_fret'
    env = HarmonicEnv(
        model_path=str(classifier_path),
        string_indices=string_indices,
        curriculum_mode=curriculum_mode,
        max_steps=10,
    )

    # Override fret list if a specific fret was requested
    if args.target_fret is not None:
        env.HARMONIC_FRETS = [args.target_fret]

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

    # Save results JSON
    import json
    results_path = output_dir / f"results_{model_path.stem}.json"

    # Strip non-serialisable fields
    json_results = {
        k: (v.tolist() if isinstance(v, np.ndarray) else
            float(v) if isinstance(v, (np.float32, np.float64)) else
            v)
        for k, v in results.items()
        if k != 'trajectories'  # too large for JSON
    }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    env.close()


if __name__ == '__main__':
    main()
