"""
Offline reward-parameter sensitivity analysis for guitaRL.

Sweeps reward weights, Gaussian sigmas, and filtration thresholds in offline
(--pretrain) mode — no robot or audio hardware required, steps are instant.
Reports how each parameter affects the final policy's mean episode reward and
the fret/torque distribution at convergence.

Usage:
    # Full sweep, 10 000 steps per configuration (takes ~5 min)
    python scripts/sensitivity_analysis.py --timesteps 10000

    # Quick smoke-test
    python scripts/sensitivity_analysis.py --timesteps 1000 --output sensitivity_test.csv

    # Sweep only one group of parameters
    python scripts/sensitivity_analysis.py --timesteps 5000 --sweep audio-weight torque-max
"""

import argparse
import csv
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import utils.reward as reward_module
from env.harmonic_env import HarmonicEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


logging.basicConfig(level=logging.WARNING)  # suppress SB3 spam during sweep
logger = logging.getLogger(__name__)


# ── Parameter grids ──────────────────────────────────────────────────────────

SWEEP_GROUPS = {
    'audio-weight': {
        'constant': 'REWARD_WEIGHT_AUDIO',
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'description': 'CNN audio component weight (online)',
    },
    'fret-sigma': {
        'constant': 'PRETRAIN_FRET_TOLERANCE',
        'values': [0.5, 1.0, 1.5, 2.0, 3.0],
        'description': 'Fret Gaussian sigma used during offline pre-training',
    },
    'torque-max': {
        'constant': 'TORQUE_HARD_MAX',
        'values': [200.0, 250.0, 350.0, 450.0, 550.0],
        'description': 'Filtration: torque hard maximum (encoder units)',
    },
    'fret-max-error': {
        'constant': 'FRET_MAX_ERROR',
        'values': [1.5, 2.0, 3.0, 4.0, 5.0],
        'description': 'Filtration: maximum fret distance before penalty',
    },
    'torque-sigma': {
        'constant': 'TORQUE_TOLERANCE',
        'values': [30.0, 50.0, 75.0, 100.0, 150.0],
        'description': 'Torque Gaussian sigma (TORQUE_TOLERANCE)',
    },
}

DEFAULT_GROUPS = list(SWEEP_GROUPS.keys())


# ── Single-configuration evaluation ─────────────────────────────────────────

def patch_reward_module(constant: str, value: Any):
    """Monkey-patch a constant in utils.reward at runtime."""
    original = getattr(reward_module, constant)
    setattr(reward_module, constant, value)
    return original


def restore_reward_module(constant: str, original: Any):
    setattr(reward_module, constant, original)


def run_config(
    constant: str,
    value: Any,
    timesteps: int,
    curriculum: str = 'easy_to_hard',
    seed: int = 42,
) -> dict:
    """
    Patch one reward constant, run offline SAC for `timesteps` steps,
    and return summary metrics.
    """
    original = patch_reward_module(constant, value)

    # Re-import the patched functions inside harmonic_env so the env picks up
    # the new values (they import from reward_module at call time via function
    # calls, so monkey-patching the module is sufficient).
    try:
        env = HarmonicEnv(
            model_path=None,
            string_index=2,
            curriculum_mode=curriculum,
            max_steps=10,
            reward_mode='no_audio',
            offline=True,
        )
        env = Monitor(env)

        model = SAC(
            'MlpPolicy', env,
            learning_rate=3e-4,
            buffer_size=50_000,
            learning_starts=500,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            ent_coef=0.1,
            verbose=0,
            seed=seed,
        )

        t0 = time.time()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        elapsed = time.time() - t0

        # Evaluate: 50 deterministic rollouts
        eval_rewards = []
        eval_frets = []
        eval_torques = []
        obs, _ = env.reset()
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            eval_rewards.append(reward)
            eval_frets.append(info.get('fret_position', 0.0))
            eval_torques.append(info.get('torque', 0.0))
            if done or truncated:
                obs, _ = env.reset()

        result = {
            'constant': constant,
            'value': value,
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'mean_fret': float(np.mean(eval_frets)),
            'std_fret': float(np.std(eval_frets)),
            'mean_torque': float(np.mean(eval_torques)),
            'std_torque': float(np.std(eval_torques)),
            'elapsed_s': round(elapsed, 1),
        }

        env.close()

    except Exception as e:
        logger.error(f"Config {constant}={value} failed: {e}")
        result = {
            'constant': constant,
            'value': value,
            'error': str(e),
            'mean_reward': float('nan'),
        }

    restore_reward_module(constant, original)
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_sensitivity(all_results: list, output_path: Path = None):
    """One subplot per swept constant showing mean_reward vs parameter value."""
    groups = {}
    for r in all_results:
        key = r['constant']
        groups.setdefault(key, []).append(r)

    ncols = 2
    nrows = (len(groups) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, (constant, results) in enumerate(groups.items()):
        ax = axes[ax_idx]
        values = [r['value'] for r in results]
        means = [r.get('mean_reward', float('nan')) for r in results]
        stds = [r.get('std_reward', 0.0) for r in results]
        default_val = getattr(reward_module, constant, None)

        ax.errorbar(values, means, yerr=stds, marker='o', capsize=4)
        if default_val is not None and default_val in values:
            ax.axvline(x=default_val, color='r', linestyle='--', alpha=0.6, label='Default')
        ax.set_xlabel(constant)
        ax.set_ylabel('Mean reward (50 steps)')
        # Find group description
        desc = next((g['description'] for g in SWEEP_GROUPS.values()
                     if g['constant'] == constant), constant)
        ax.set_title(desc, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[len(groups):]:
        ax.set_visible(False)

    plt.suptitle('Reward Parameter Sensitivity (offline pretrain mode)', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved sensitivity plot to {output_path}")

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Offline reward-parameter sensitivity sweep for guitaRL')
    parser.add_argument('--timesteps', type=int, default=10_000,
                        help='SAC training steps per configuration (default: 10 000)')
    parser.add_argument('--sweep', nargs='+', choices=list(SWEEP_GROUPS.keys()),
                        default=DEFAULT_GROUPS,
                        help='Which parameter groups to sweep (default: all)')
    parser.add_argument('--curriculum', choices=['random', 'easy_to_hard', 'fixed_fret'],
                        default='easy_to_hard',
                        help='Curriculum mode for all configurations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='sensitivity_results.csv',
                        help='Path to save results CSV')
    parser.add_argument('--plot', type=str, default='sensitivity_plot.png',
                        help='Path to save sensitivity plot PNG')
    args = parser.parse_args()

    all_results = []

    for group_name in args.sweep:
        group = SWEEP_GROUPS[group_name]
        constant = group['constant']
        values = group['values']
        print(f"\n── Sweeping {group_name} ({constant}) ──────────────────────────────")
        print(f"   Description: {group['description']}")
        print(f"   Values: {values}")

        for value in values:
            print(f"   {constant} = {value} ...", end=' ', flush=True)
            result = run_config(
                constant=constant,
                value=value,
                timesteps=args.timesteps,
                curriculum=args.curriculum,
                seed=args.seed,
            )
            status = f"reward={result['mean_reward']:.3f}" if 'error' not in result else f"ERROR: {result['error']}"
            print(status)
            all_results.append(result)

    # Print table
    print(f"\n{'=' * 70}")
    print(f"{'Constant':<25} {'Value':>8} {'Mean Reward':>12} {'Std':>8} {'Time(s)':>8}")
    print(f"{'=' * 70}")
    for r in all_results:
        if 'error' in r:
            print(f"{r['constant']:<25} {str(r['value']):>8} {'ERROR':>12}")
        else:
            print(f"{r['constant']:<25} {str(r['value']):>8} "
                  f"{r['mean_reward']:>12.4f} {r['std_reward']:>8.4f} "
                  f"{r['elapsed_s']:>8.1f}")
    print(f"{'=' * 70}\n")

    # Save CSV
    output_path = Path(args.output)
    fieldnames = ['constant', 'value', 'mean_reward', 'std_reward',
                  'mean_fret', 'std_fret', 'mean_torque', 'std_torque', 'elapsed_s', 'error']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to {output_path}")

    # Plot
    plot_path = Path(args.plot)
    plot_sensitivity(all_results, output_path=plot_path)


if __name__ == '__main__':
    main()
