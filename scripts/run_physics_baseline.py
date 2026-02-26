"""
Evaluate physics-only and random-policy baselines on the physical GuitarBot.

Reports the same metrics as evaluate.py so results are directly comparable.

Modes:
    physics  — Hardcoded fret position = target_fret, torque = 70 (known optimal).
               Represents the best a fixed controller can do with no learning.
    random   — Uniform random actions sampled from the action space.
               Establishes the lower bound (chance performance).

Usage:
    # Physics-only controller, 30 episodes
    python scripts/run_physics_baseline.py \\
        --model-path ../HarmonicsClassifier/models/best_model.pt \\
        --mode physics --episodes 30

    # Random policy, 20 episodes, single string
    python scripts/run_physics_baseline.py \\
        --model-path ../HarmonicsClassifier/models/best_model.pt \\
        --mode random --episodes 20 --string-index 2

    # Both modes back-to-back
    python scripts/run_physics_baseline.py \\
        --model-path ../HarmonicsClassifier/models/best_model.pt \\
        --mode all --episodes 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so env/ and utils/ are importable when called from scripts/
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from env.action_space import (
    GuitarBotActionSpace,
    RLFretAction,
    PresserAction,
    HARMONIC_FRETS_IN_RANGE,
    PLAYABLE_STRINGS,
    TORQUE_OPTIMAL_HARMONIC,
)
from env.osc_client import GuitarBotOSCClient
from utils.audio_reward import HarmonicRewardCalculator


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step timing — must match harmonic_env.py
CAPTURE_PRE_DELAY = 0.5   # seconds to wait after OSC send before recording
CAPTURE_DURATION = 2.0    # seconds of audio to capture
ACTION_DURATION = 3.0     # total step budget


def run_episode(
    target_fret: int,
    string_index: int,
    fret_position: float,
    torque: float,
    osc_client: GuitarBotOSCClient,
    reward_calc: HarmonicRewardCalculator,
) -> dict:
    """
    Send one action to the robot, capture audio, classify, return result dict.
    """
    action = RLFretAction(string_index, fret_position, PresserAction.PRESS, torque)
    osc_client.send_rlfret(action)

    time.sleep(CAPTURE_PRE_DELAY)
    audio = reward_calc.capture_audio(duration=CAPTURE_DURATION)

    remaining = ACTION_DURATION - CAPTURE_PRE_DELAY - CAPTURE_DURATION
    if remaining > 0:
        time.sleep(remaining)

    classification = reward_calc.classify_audio(audio)
    harmonic_prob = classification['harmonic_prob']
    success = harmonic_prob > 0.8

    logger.info(
        f"  fret={fret_position:.2f}  torque={torque:.0f}  target={target_fret}  "
        f"H={harmonic_prob:.3f}  D={classification['dead_prob']:.3f}  "
        f"G={classification['general_prob']:.3f}  {'SUCCESS' if success else ''}"
    )

    return {
        'target_fret': target_fret,
        'string_index': string_index,
        'fret_position': fret_position,
        'torque': torque,
        'harmonic_prob': harmonic_prob,
        'dead_prob': classification['dead_prob'],
        'general_prob': classification['general_prob'],
        'predicted_label': classification['predicted_label'],
        'success': success,
    }


def run_baseline(
    mode: str,
    n_episodes: int,
    string_indices: list,
    model_path: str,
    audio_device: str,
    osc_port: int,
) -> dict:
    """
    Run one baseline mode and return aggregated results.

    mode: 'physics' | 'random'
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running baseline: {mode.upper()}  ({n_episodes} episodes)")
    logger.info(f"{'=' * 60}")

    osc_client = GuitarBotOSCClient(host="127.0.0.1", port=osc_port)
    reward_calc = HarmonicRewardCalculator(
        model_path=model_path,
        device_name=audio_device,
        capture_duration=CAPTURE_DURATION,
    )
    action_space = GuitarBotActionSpace(use_normalized=True)

    episode_results = []
    for ep in range(n_episodes):
        target_fret = int(np.random.choice(HARMONIC_FRETS_IN_RANGE))
        string_index = int(np.random.choice(string_indices))

        if mode == 'physics':
            # Hardcoded: command the exact target fret with optimal light-touch torque
            fret_position = float(target_fret)
            torque = float(TORQUE_OPTIMAL_HARMONIC)
        elif mode == 'random':
            # Uniform random over the full action space
            raw_action = action_space.sample()
            fret_position = raw_action.fret_position
            torque = raw_action.torque
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info(f"Episode {ep + 1}/{n_episodes} | string={string_index} target_fret={target_fret}")
        result = run_episode(
            target_fret=target_fret,
            string_index=string_index,
            fret_position=fret_position,
            torque=torque,
            osc_client=osc_client,
            reward_calc=reward_calc,
        )
        episode_results.append(result)

    # Aggregate
    successes = [r['success'] for r in episode_results]
    h_probs = [r['harmonic_prob'] for r in episode_results]

    aggregated = {
        'mode': mode,
        'n_episodes': n_episodes,
        'success_rate': float(np.mean(successes)),
        'mean_harmonic_prob': float(np.mean(h_probs)),
        'std_harmonic_prob': float(np.std(h_probs)),
        'n_successes': int(sum(successes)),
        'episodes': episode_results,
    }

    # Per-fret breakdown
    results_by_fret = {}
    for fret in HARMONIC_FRETS_IN_RANGE:
        mask = [r for r in episode_results if r['target_fret'] == fret]
        if mask:
            results_by_fret[str(fret)] = {
                'n_episodes': len(mask),
                'success_rate': float(np.mean([r['success'] for r in mask])),
                'mean_harmonic_prob': float(np.mean([r['harmonic_prob'] for r in mask])),
            }
    aggregated['results_by_fret'] = results_by_fret

    # Per-string breakdown
    results_by_string = {}
    for string in string_indices:
        mask = [r for r in episode_results if r['string_index'] == string]
        if mask:
            results_by_string[str(string)] = {
                'n_episodes': len(mask),
                'success_rate': float(np.mean([r['success'] for r in mask])),
                'mean_harmonic_prob': float(np.mean([r['harmonic_prob'] for r in mask])),
            }
    aggregated['results_by_string'] = results_by_string

    reward_calc.close()
    return aggregated


def print_result(result: dict):
    print(f"\n── {result['mode'].upper()} baseline ──────────────────────────────")
    print(f"  Episodes:          {result['n_episodes']}")
    print(f"  Success rate:      {result['success_rate']:.1%}  ({result['n_successes']}/{result['n_episodes']})")
    print(f"  Mean H-prob:       {result['mean_harmonic_prob']:.3f} ± {result['std_harmonic_prob']:.3f}")
    if result.get('results_by_fret'):
        print(f"\n  Per-fret:")
        for fret, d in sorted(result['results_by_fret'].items(), key=lambda x: int(x[0])):
            print(f"    Fret {fret}: success={d['success_rate']:.1%}  H-prob={d['mean_harmonic_prob']:.3f}  (n={d['n_episodes']})")


def main():
    parser = argparse.ArgumentParser(description='Run physics / random baselines on GuitarBot')
    parser.add_argument('--model-path', required=True,
                        help='Path to HarmonicsClassifier .pt model')
    parser.add_argument('--mode', choices=['physics', 'random', 'all'], default='physics',
                        help='Baseline mode: physics, random, or all (runs both)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes per baseline mode')
    parser.add_argument('--string-index', type=int, default=2,
                        help='Single string to use (ignored when --string-indices given)')
    parser.add_argument('--string-indices', type=int, nargs='+', default=None,
                        help='Multiple strings to rotate across (e.g. --string-indices 0 2 4)')
    parser.add_argument('--audio-device', type=str, default='Scarlett',
                        help='Audio input device name substring (default: Scarlett)')
    parser.add_argument('--osc-port', type=int, default=12000,
                        help='GuitarBot OSC port (default: 12000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON (e.g. baselines.json)')
    args = parser.parse_args()

    string_indices = args.string_indices if args.string_indices else [args.string_index]
    modes = ['physics', 'random'] if args.mode == 'all' else [args.mode]

    all_results = {}
    for mode in modes:
        result = run_baseline(
            mode=mode,
            n_episodes=args.episodes,
            string_indices=string_indices,
            model_path=args.model_path,
            audio_device=args.audio_device,
            osc_port=args.osc_port,
        )
        all_results[mode] = result
        print_result(result)

    # Print comparison if both modes were run
    if len(all_results) == 2:
        phys = all_results['physics']
        rand = all_results['random']
        print("\n── Comparison ──────────────────────────────────────────────")
        print(f"  {'Metric':<30} {'Physics':>12} {'Random':>12}")
        print(f"  {'Success rate':<30} {phys['success_rate']:>12.1%} {rand['success_rate']:>12.1%}")
        print(f"  {'Mean H-prob':<30} {phys['mean_harmonic_prob']:>12.3f} {rand['mean_harmonic_prob']:>12.3f}")

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        # Strip per-episode data to keep file small if large n_episodes
        for mode, result in all_results.items():
            result.pop('episodes', None)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
