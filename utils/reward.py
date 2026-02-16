"""
Shared reward function for RL harmonic training.

Both the training pipeline (audio_reward.py / harmonic_env.py) and the
diagnostic test loop (test_rl_loop.py) import from here so the reward
logic is defined in exactly one place.
"""

import numpy as np
from typing import Dict, Optional


# ── Reward constants ──────────────────────────────────────────────────
HARMONIC_FRETS = [4, 5, 7]
TORQUE_OPTIMAL_HARMONIC = 30.0   # Light touch for harmonics
TORQUE_MAX = 650.0

FRET_TOLERANCE = 0.3             # ~1/3 of a fret is acceptable
TORQUE_TOLERANCE = 75.0          # Within 75 units of optimal

REWARD_WEIGHT_AUDIO  = 0.2
REWARD_WEIGHT_FRET   = 0.3
REWARD_WEIGHT_TORQUE = 0.5

# Classifier label order (must match train_cnn.py label_map)
CLASS_NAMES = ['harmonic', 'dead_note', 'general_note']
HARMONIC_CLASS_IDX = 0

SUCCESS_THRESHOLD = 0.8          # harmonic_prob above this = success


# ── Reward computation ────────────────────────────────────────────────

def compute_reward(
    fret_position: float,
    torque: float,
    target_fret: int,
    harmonic_prob: float,
) -> Dict[str, float]:
    """
    Pure reward computation — no audio capture or model inference.

    Reward = w_audio * harmonic_prob
           + w_fret  * exp(-fret_error² / 2σ_fret²)
           + w_torque* (2·exp(-torque_error² / 2σ_torque²) - 1)

    The torque component is shifted to [-1, +1] so excessive torque
    actively penalises the reward rather than merely contributing zero.

    Args:
        fret_position: Fractional fret position the agent chose (0.0 – 9.0)
        torque:        Fretter torque the agent chose
        target_fret:   Target harmonic fret (4, 5, or 7)
        harmonic_prob: Probability of 'harmonic' class from classifier

    Returns:
        Dict with total_reward and all component values.
    """
    if target_fret not in HARMONIC_FRETS:
        raise ValueError(
            f"Invalid target fret: {target_fret}. Must be in {HARMONIC_FRETS}"
        )

    # Audio reward: harmonic probability straight from classifier
    audio_reward = float(harmonic_prob)

    # Fret reward: Gaussian centred on target fret
    fret_error = abs(fret_position - float(target_fret))
    fret_reward = np.exp(-(fret_error ** 2) / (2 * FRET_TOLERANCE ** 2))

    # Torque reward: shifted Gaussian, range [-1, +1]
    #   optimal (error=0) → +1 (bonus)
    #   moderate error    →  0 (neutral)
    #   extreme torque    → -1 (penalty — harmonics are physically impossible)
    torque_error = abs(torque - TORQUE_OPTIMAL_HARMONIC)
    torque_reward = 2.0 * np.exp(-(torque_error ** 2) / (2 * TORQUE_TOLERANCE ** 2)) - 1.0

    # Weighted combination
    total_reward = (
        REWARD_WEIGHT_AUDIO  * audio_reward
        + REWARD_WEIGHT_FRET * fret_reward
        + REWARD_WEIGHT_TORQUE * torque_reward
    )

    return {
        'total_reward':  total_reward,
        'audio_reward':  audio_reward,
        'fret_reward':   fret_reward,
        'torque_reward': torque_reward,
        'fret_error':    fret_error,
        'torque_error':  torque_error,
        'target_fret':   target_fret,
    }


def compute_reward_nearest_fret(
    fret_position: float,
    torque: float,
    harmonic_prob: float,
) -> Dict[str, float]:
    """
    Convenience wrapper that picks the nearest harmonic fret automatically.

    Useful in the test loop where no explicit target fret is specified.
    """
    nearest_fret = min(HARMONIC_FRETS, key=lambda f: abs(fret_position - f))
    result = compute_reward(fret_position, torque, nearest_fret, harmonic_prob)
    result['nearest_harmonic_fret'] = nearest_fret
    return result


def is_success(harmonic_prob: float) -> bool:
    """Check if harmonic probability exceeds the success threshold."""
    return harmonic_prob > SUCCESS_THRESHOLD
