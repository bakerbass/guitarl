"""
Two-layer reward function for RL harmonic training.

Layer 1 — Filtration (physics):
    Gates the reward with physics-based checks.  If the action is
    physically nonsensical (extreme torque, silent audio, fret nowhere
    near a harmonic node) the agent receives a flat penalty and the
    classifier is never consulted.  This saves the CNN from evaluating
    garbage and gives the agent a strong, immediate signal to avoid
    obviously bad regions of the action space.

Layer 2 — Audio (classifier):
    Only reached when the filtration layer passes.  Uses the CNN
    classifier's harmonic probability as the primary reward, plus a
    small shaping bonus for fret accuracy and torque optimality.

Both the training env (audio_reward.py / harmonic_env.py) and the
diagnostic test loop (test_rl_loop.py) import from here so the reward
logic is defined in exactly one place.
"""

import numpy as np
from typing import Dict, Optional


# ── Constants ─────────────────────────────────────────────────────────
HARMONIC_FRETS = [4, 5, 7, 9]
TORQUE_OPTIMAL_HARMONIC = 30.0   # Light touch for harmonics
TORQUE_MAX = 650.0
TORQUE_SAFE_MIN = 16.0

FRET_TOLERANCE = 0.3             # ~1/3 of a fret is acceptable
TORQUE_TOLERANCE = 75.0          # Within 75 units of optimal

# Classifier label order (must match train_cnn.py label_map)
CLASS_NAMES = ['harmonic', 'dead_note', 'general_note']
HARMONIC_CLASS_IDX = 0

SUCCESS_THRESHOLD = 0.8          # harmonic_prob above this = success

# ── Filtration layer thresholds ───────────────────────────────────────
TORQUE_HARD_MAX   = 500.0   # Anything above this is too aggressive for harmonics
TORQUE_HARD_MIN   = 10.0    # Below this the presser barely touches the string
FRET_MAX_ERROR    = 3.0     # More than 3 frets away? Not even trying
RMS_SILENCE_THRESH = 0.005  # RMS below this = no audible onset

FILTRATION_PENALTY = -1.0   # Flat penalty when filtration rejects an action

# ── Layer 2 weights (audio layer) ─────────────────────────────────────
REWARD_WEIGHT_AUDIO  = 0.6  # Classifier is the primary signal
REWARD_WEIGHT_FRET   = 0.2  # Small shaping bonus for fret accuracy
REWARD_WEIGHT_TORQUE = 0.2  # Small shaping bonus for light torque


# ── Layer 1: Filtration ───────────────────────────────────────────────

def compute_filtration(
    fret_position: float,
    torque: float,
    target_fret: int,
    audio_rms: Optional[float] = None,
) -> Dict[str, object]:
    """
    Physics-based gate.  Returns whether the action is sane enough to
    bother running the classifier on.

    Returns:
        Dict with:
            passed (bool):  True = proceed to Layer 2
            reason (str):   Why it was rejected (empty if passed)
            penalty (float): Reward to use when rejected
    """
    # Check: excessive torque
    if torque > TORQUE_HARD_MAX:
        return {
            'passed': False,
            'reason': f'torque_too_high ({torque:.0f} > {TORQUE_HARD_MAX:.0f})',
            'penalty': FILTRATION_PENALTY,
        }

    # Check: torque too low (presser barely engaged)
    if torque < TORQUE_HARD_MIN:
        return {
            'passed': False,
            'reason': f'torque_too_low ({torque:.0f} < {TORQUE_HARD_MIN:.0f})',
            'penalty': FILTRATION_PENALTY,
        }

    # Check: fret way too far from target
    fret_error = abs(fret_position - float(target_fret))
    if fret_error > FRET_MAX_ERROR:
        return {
            'passed': False,
            'reason': f'fret_too_far ({fret_position:.2f}, err={fret_error:.2f} > {FRET_MAX_ERROR})',
            'penalty': FILTRATION_PENALTY,
        }

    # Check: silence (no onset detected)
    if audio_rms is not None and audio_rms < RMS_SILENCE_THRESH:
        return {
            'passed': False,
            'reason': f'silence (rms={audio_rms:.4f} < {RMS_SILENCE_THRESH})',
            'penalty': FILTRATION_PENALTY,
        }

    return {'passed': True, 'reason': '', 'penalty': 0.0}


# ── Layer 2: Audio reward ────────────────────────────────────────────

def compute_audio_reward(
    fret_position: float,
    torque: float,
    target_fret: int,
    harmonic_prob: float,
) -> Dict[str, float]:
    """
    Classifier-based reward, only called when the filtration layer passes.

    Reward = w_audio * harmonic_prob
           + w_fret  * exp(-fret_error² / 2σ_fret²)
           + w_torque* (2·exp(-torque_error² / 2σ_torque²) - 1)

    Args:
        fret_position: Fractional fret position the agent chose (0.0 – 9.0)
        torque:        Presser torque the agent chose
        target_fret:   Target harmonic fret (4, 5, or 7)
        harmonic_prob: Probability of 'harmonic' class from classifier

    Returns:
        Dict with total_reward and all component values.
    """
    # Audio reward: harmonic probability straight from classifier
    audio_reward = float(harmonic_prob)

    # Fret shaping: Gaussian centred on target fret
    fret_error = abs(fret_position - float(target_fret))
    fret_reward = np.exp(-(fret_error ** 2) / (2 * FRET_TOLERANCE ** 2))

    # Torque shaping: shifted Gaussian, range [-1, +1]
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
    }


# ── Combined entry point ─────────────────────────────────────────────

def compute_reward(
    fret_position: float,
    torque: float,
    target_fret: int,
    harmonic_prob: float,
    audio_rms: Optional[float] = None,
) -> Dict[str, object]:
    """
    Two-layer reward: filtration first, then audio.

    If the filtration layer rejects the action, the classifier-based
    components are zeroed out and the total reward equals the flat
    filtration penalty.  Otherwise, the full audio-layer reward is used.

    Returns:
        Dict with all reward components plus filtration metadata.
    """
    if target_fret not in HARMONIC_FRETS:
        raise ValueError(
            f"Invalid target fret: {target_fret}. Must be in {HARMONIC_FRETS}"
        )

    # Layer 1
    filt = compute_filtration(fret_position, torque, target_fret, audio_rms)

    if not filt['passed']:
        # Short-circuit: return penalty, zero out audio components
        return {
            'total_reward':      filt['penalty'],
            'audio_reward':      0.0,
            'fret_reward':       0.0,
            'torque_reward':     0.0,
            'fret_error':        abs(fret_position - float(target_fret)),
            'torque_error':      abs(torque - TORQUE_OPTIMAL_HARMONIC),
            'target_fret':       target_fret,
            'filtered':          True,
            'filter_reason':     filt['reason'],
        }

    # Layer 2
    reward_info = compute_audio_reward(
        fret_position, torque, target_fret, harmonic_prob
    )
    reward_info['target_fret'] = target_fret
    reward_info['filtered'] = False
    reward_info['filter_reason'] = ''
    return reward_info


def compute_reward_nearest_fret(
    fret_position: float,
    torque: float,
    harmonic_prob: float,
    audio_rms: Optional[float] = None,
) -> Dict[str, object]:
    """
    Convenience wrapper that picks the nearest harmonic fret automatically.
    """
    nearest_fret = min(HARMONIC_FRETS, key=lambda f: abs(fret_position - f))
    result = compute_reward(fret_position, torque, nearest_fret, harmonic_prob, audio_rms)
    result['nearest_harmonic_fret'] = nearest_fret
    return result


def is_success(harmonic_prob: float) -> bool:
    """Check if harmonic probability exceeds the success threshold."""
    return harmonic_prob > SUCCESS_THRESHOLD
