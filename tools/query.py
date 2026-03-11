"""
query.py — Ask a trained SAC policy for a single action without any robot or audio.

Builds a neutral 14-dim observation for the requested target fret and string,
runs the policy deterministically, decodes the output to physical units, and
prints the result as an RLFret command.

Usage:
    python query.py --model runs/harmonic_sac_20260218/best_model/best_model.zip
    python query.py --model best_model.zip --target-fret 5 --string 0
    python query.py --model best_model.zip --target-fret 7 --string 2 --fret-history 6.8 7.1 7.0 --torque-history 45 52 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

# Make env/ and utils/ importable when run from any subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.action_space import (
    GuitarBotActionSpace,
    PresserAction,
    PLAYABLE_STRINGS,
    HARMONIC_FRETS_IN_RANGE,
    TORQUE_MAX,
)


# Observation layout mirrors HarmonicEnv._get_observation()
# [target_fret_one_hot(3), string_one_hot(3), current_fret, current_torque_norm,
#  fret_history(3), torque_history(3)]  →  14 dimensions
OBS_DIM = 14


def build_observation(
    target_fret: int,
    string_idx: int,
    current_fret: float = 0.0,
    current_torque: float = 0.0,
    fret_history: list = None,
    torque_history: list = None,
) -> np.ndarray:
    """Construct a 14-dim observation matching HarmonicEnv._get_observation()."""
    fret_history = (fret_history or [0.0, 0.0, 0.0])[-3:]
    torque_history = (torque_history or [0.0, 0.0, 0.0])[-3:]

    # Pad to exactly 3 entries
    while len(fret_history) < 3:
        fret_history.insert(0, 0.0)
    while len(torque_history) < 3:
        torque_history.insert(0, 0.0)

    target_one_hot = np.zeros(3, dtype=np.float32)
    target_one_hot[HARMONIC_FRETS_IN_RANGE.index(target_fret)] = 1.0

    string_one_hot = np.zeros(3, dtype=np.float32)
    string_one_hot[PLAYABLE_STRINGS.index(string_idx)] = 1.0

    return np.concatenate([
        target_one_hot,
        string_one_hot,
        [current_fret, current_torque / TORQUE_MAX],
        np.array(fret_history, dtype=np.float32),
        np.array(torque_history, dtype=np.float32) / TORQUE_MAX,
    ]).astype(np.float32)


def decode_action(raw_action: np.ndarray) -> "RLFretAction":
    """Convert the raw policy output to an RLFretAction."""
    action_space = GuitarBotActionSpace(use_normalized=True)
    action_dim = len(raw_action)

    if action_dim == 5:
        # always_press=True: [string_logits(3), fret, torque] — inject press=1.0
        action = np.insert(raw_action, 4, 1.0)
    elif action_dim == 6:
        # always_press=False: [string_logits(3), fret, press_decision, torque]
        action = raw_action
    else:
        raise ValueError(f"Unexpected action dimension {action_dim}. Expected 5 or 6.")

    return action_space.from_normalized(action)


def main():
    parser = argparse.ArgumentParser(
        description="Query a trained SAC policy for one or more RLFret actions."
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained model (.zip)")
    parser.add_argument("--target-fret", type=int, default=7,
                        choices=HARMONIC_FRETS_IN_RANGE,
                        help=f"Target harmonic fret {HARMONIC_FRETS_IN_RANGE} (default: 7)")
    parser.add_argument("--string", type=int, default=2,
                        choices=PLAYABLE_STRINGS,
                        dest="string_idx",
                        help=f"Active string {PLAYABLE_STRINGS} (default: 2=D)")
    parser.add_argument("--current-fret", type=float, default=0.0,
                        help="Current fret position in the observation (default: 0.0)")
    parser.add_argument("--current-torque", type=float, default=0.0,
                        help="Current torque in the observation (default: 0)")
    parser.add_argument("--fret-history", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar="F",
                        help="Last 3 fret positions (default: 0 0 0)")
    parser.add_argument("--torque-history", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar="T",
                        help="Last 3 torque values (default: 0 0 0)")
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="Sample stochastically instead of using the deterministic mean")
    parser.add_argument("--num-actions", type=int, default=1, metavar="N",
                        help="Number of actions to sample and display (default: 1). "
                             "Use --stochastic to get varied samples; deterministic mode "
                             "always returns the same action for the same observation.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Load model (CPU is fine for forward passes)
    model = SAC.load(model_path, device="cpu")

    obs = build_observation(
        target_fret=args.target_fret,
        string_idx=args.string_idx,
        current_fret=args.current_fret,
        current_torque=args.current_torque,
        fret_history=args.fret_history,
        torque_history=args.torque_history,
    )

    print(f"\nModel:       {model_path.name}")
    print(f"Target fret: {args.target_fret}  |  String: {args.string_idx}"
          f"  |  n={args.num_actions}"
          f"  |  {'stochastic' if args.stochastic else 'deterministic'}")
    print()

    if args.num_actions == 1:
        raw_action, _ = model.predict(obs, deterministic=not args.stochastic)
        rl_action = decode_action(raw_action)
        print(f"  /RLFret {rl_action.string_idx} {rl_action.fret_position:.3f} {rl_action.torque:.0f}")
        print()
        print(f"  fret:     {rl_action.fret_position:.3f}  ({rl_action.slider_mm:.1f} mm)")
        print(f"  torque:   {rl_action.torque:.0f}")
        print(f"  harmonic: {'YES' if rl_action.is_at_harmonic else 'no'}"
              f"  (nearest fret {rl_action.nearest_fret})")
    else:
        # Header row
        print(f"  {'#':>3}  {'fret':>6}  {'torque':>6}  {'mm':>6}  harmonic")
        print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  --------")
        for i in range(args.num_actions):
            raw_action, _ = model.predict(obs, deterministic=not args.stochastic)
            a = decode_action(raw_action)
            harmonic_tag = "YES" if a.is_at_harmonic else "-"
            print(f"  {i+1:>3}  {a.fret_position:>6.3f}  {a.torque:>6.0f}  "
                  f"{a.slider_mm:>6.1f}  {harmonic_tag}")

    print()


if __name__ == "__main__":
    main()
