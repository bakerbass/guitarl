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
TORQUE_OPTIMAL_HARMONIC = 70.0   # Light touch for harmonics
TORQUE_MAX = 650.0
TORQUE_SAFE_MIN = 16.0

FRET_TOLERANCE = 0.35             # ~1/3 of a fret is acceptable
TORQUE_TOLERANCE = 75.0          # Within 75 units of optimal

# Classifier label order (must match train_cnn.py label_map)
CLASS_NAMES = ['harmonic', 'dead_note', 'general_note']
HARMONIC_CLASS_IDX = 0

SUCCESS_THRESHOLD = 0.8          # harmonic_prob above this = success

# ── Filtration layer thresholds ───────────────────────────────────────
TORQUE_HARD_MAX   = 300.0   # Anything above this is too aggressive for harmonics
TORQUE_HARD_MIN   = 15.0    # Below this the presser barely touches the string or triggers a weird edge case
FRET_MAX_ERROR    = 0.3     # More than  frets away? Not even trying
RMS_SILENCE_THRESH = 0.005  # RETIRED — silence detection was inconsistent; kept for reference only

FILTRATION_PENALTY = -1.0   # Flat penalty when filtration rejects an action

# ── Layer 2 weights (audio layer) ─────────────────────────────────────
REWARD_WEIGHT_AUDIO  = 0.45  # Classifier is the primary signal
REWARD_WEIGHT_FRET   = 0.2  # Small shaping bonus for fret accuracy
REWARD_WEIGHT_TORQUE = 0.35  # Small shaping bonus for light torque

# Ablation: no-audio mode — fret + torque shaping only, rebalanced
ABLATION_NO_AUDIO_FRET_WEIGHT   = 0.5
ABLATION_NO_AUDIO_TORQUE_WEIGHT = 0.5

# Offline pre-training: wider fret Gaussian so the agent gets useful gradients
# across the whole fret range, not just within ±FRET_TOLERANCE of target.
#
# At σ=0.35 (online):  reward at 1 fret away ≈ 0.01  → effectively zero; no gradient
# At σ=1.5  (pretrain): reward at 1 fret away ≈ 0.80
#                        reward at 2 frets away ≈ 0.41
#                        reward at 3 frets away ≈ 0.14 (filtration boundary)
# This ensures the agent always has a direction signal toward the harmonic node
# even when exploring far from target.
PRETRAIN_FRET_TOLERANCE = 1.5

# Reward mode strings
REWARD_MODE_FULL          = 'full'           # Layer 1 + Layer 2 (default)
REWARD_MODE_NO_FILTRATION = 'no_filtration'  # Layer 2 only — bypass physics gate
REWARD_MODE_NO_AUDIO      = 'no_audio'       # Layer 1 + fret/torque shaping only
REWARD_MODE_COSINE_SIM    = 'cosine_sim'     # Layer 1 + onset-aligned mel cosine sim vs reference WAVs

# ── Fine-tune (cosine_sim) reward curve ───────────────────────────────────────
# cosine_sim >= threshold  →  +5.0 (success bonus)
# cosine_sim <  threshold  →  -5.0 * (1 - exp(-k * (threshold - sim)))
#   At 0.79: ≈ -0.15  (not too bad)
#   At 0.50: ≈ -3.0   (pretty far negative)
#   At 0.00: ≈ -4.55  (saturates near -5)
COSINE_SIM_SUCCESS_THRESHOLD = 0.85
COSINE_SIM_SUCCESS_REWARD    = 5.0
COSINE_SIM_FLOOR             = -5.0   # minimum (asymptotic) reward
COSINE_SIM_DECAY_K           = 3.0    # controls how fast reward drops below threshold


# ── Layer 1: Filtration ───────────────────────────────────────────────

def compute_filtration(
    fret_position: float,
    torque: float,
    target_fret: int,
    audio_rms: Optional[float] = None,  # retained for API compat; no longer used
) -> Dict[str, object]:
    """
    Physics-based gate.  Returns whether the action is sane enough to
    bother running the classifier on.

    Checks: torque range, fret distance from target.
    Silence detection (RMS) was removed — too inconsistent in practice.

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


# ── Ablation variants ────────────────────────────────────────────────

def compute_reward_no_filtration(
    fret_position: float,
    torque: float,
    target_fret: int,
    harmonic_prob: float,
    audio_rms: Optional[float] = None,
) -> Dict[str, object]:
    """
    Ablation: Layer 1 (filtration) is fully bypassed.

    Every action — no matter how extreme — proceeds to the classifier-
    based Layer 2 reward.  Use this to isolate whether the physics gate
    helps or hurts learning.  Expensive actions are still distinguishable
    by Layer 2's torque/fret shaping terms, but there is no hard penalty.
    """
    if target_fret not in HARMONIC_FRETS:
        raise ValueError(f"Invalid target fret: {target_fret}. Must be in {HARMONIC_FRETS}")

    reward_info = compute_audio_reward(fret_position, torque, target_fret, harmonic_prob)
    reward_info['target_fret']   = target_fret
    reward_info['filtered']      = False
    reward_info['filter_reason'] = ''
    return reward_info


def compute_reward_no_audio(
    fret_position: float,
    torque: float,
    target_fret: int,
    audio_rms: Optional[float] = None,
    fret_tolerance: Optional[float] = None,
) -> Dict[str, object]:
    """
    Ablation: Layer 2 audio signal (CNN classifier) is removed.

    Layer 1 (filtration) still runs (torque + fret checks).
    When it passes, reward is fret + torque shaping only (equal 0.5 / 0.5 weights).

    Args:
        fret_tolerance: Width (σ) of the fret Gaussian. Defaults to FRET_TOLERANCE
                        (0.35 frets, tight — good for online fine-tuning).
                        Use PRETRAIN_FRET_TOLERANCE (1.5 frets) for offline
                        pre-training so the agent gets a gradient signal across
                        the whole fret range, not just within ±0.35 of target.
    """
    if target_fret not in HARMONIC_FRETS:
        raise ValueError(f"Invalid target fret: {target_fret}. Must be in {HARMONIC_FRETS}")

    _fret_tol = fret_tolerance if fret_tolerance is not None else FRET_TOLERANCE

    filt = compute_filtration(fret_position, torque, target_fret, audio_rms)
    if not filt['passed']:
        return {
            'total_reward':  filt['penalty'],
            'audio_reward':  0.0,
            'fret_reward':   0.0,
            'torque_reward': 0.0,
            'fret_error':    abs(fret_position - float(target_fret)),
            'torque_error':  abs(torque - TORQUE_OPTIMAL_HARMONIC),
            'target_fret':   target_fret,
            'filtered':      True,
            'filter_reason': filt['reason'],
        }

    fret_error    = abs(fret_position - float(target_fret))
    fret_reward   = np.exp(-(fret_error ** 2) / (2 * _fret_tol ** 2))
    torque_error  = abs(torque - TORQUE_OPTIMAL_HARMONIC)
    torque_reward = 2.0 * np.exp(-(torque_error ** 2) / (2 * TORQUE_TOLERANCE ** 2)) - 1.0

    total_reward = (
        ABLATION_NO_AUDIO_FRET_WEIGHT   * fret_reward
        + ABLATION_NO_AUDIO_TORQUE_WEIGHT * torque_reward
    )

    return {
        'total_reward':  total_reward,
        'audio_reward':  0.0,         # no CNN
        'fret_reward':   fret_reward,
        'torque_reward': torque_reward,
        'fret_error':    fret_error,
        'torque_error':  torque_error,
        'target_fret':   target_fret,
        'filtered':      False,
        'filter_reason': '',
    }


def compute_reward_cosine_sim(
    fret_position: float,
    torque: float,
    target_fret: int,
    cosine_sim: float,
    audio_rms: Optional[float] = None,
) -> Dict[str, object]:
    """
    Fine-tune reward: Layer 1 filtration + exponential cosine-similarity curve.

    Layer 1 still runs first — physically nonsensical actions (bad torque /
    fret way off target) get the flat -1.0 penalty before the audio signal is
    ever consulted.

    Reward curve (when Layer 1 passes):
        cosine_sim >= 0.8  →  +5.0   (success)
        cosine_sim <  0.8  →  -5.0 * (1 − exp(−3.0 × (0.8 − cosine_sim)))
                               ≈ −0.15 at 0.79  (barely below threshold)
                               ≈ −3.0  at 0.50  (fairly negative)
                               ≈ −4.55 at 0.00  (saturates near −5)
    """
    if target_fret not in HARMONIC_FRETS:
        raise ValueError(f"Invalid target fret: {target_fret}. Must be in {HARMONIC_FRETS}")

    # Layer 1 — use a stronger penalty in fine-tune mode to keep the agent
    # near the harmonic node (reward range here is ±5, not ±1).
    _FT_FILTRATION_PENALTY = -10.0

    filt = compute_filtration(fret_position, torque, target_fret, audio_rms)
    if not filt['passed']:
        return {
            'total_reward':  _FT_FILTRATION_PENALTY,
            'audio_reward':  0.0,
            'fret_reward':   0.0,
            'torque_reward': 0.0,
            'fret_error':    abs(fret_position - float(target_fret)),
            'torque_error':  abs(torque - TORQUE_OPTIMAL_HARMONIC),
            'target_fret':   target_fret,
            'filtered':      True,
            'filter_reason': filt['reason'],
            'cosine_sim':    cosine_sim,
        }

    # Layer 2: cosine-sim reward curve
    if cosine_sim >= COSINE_SIM_SUCCESS_THRESHOLD:
        total_reward = COSINE_SIM_SUCCESS_REWARD
    else:
        total_reward = COSINE_SIM_FLOOR * (
            1.0 - np.exp(-COSINE_SIM_DECAY_K * (COSINE_SIM_SUCCESS_THRESHOLD - cosine_sim))
        )

    return {
        'total_reward':  total_reward,
        'audio_reward':  total_reward,  # cosine_sim is the audio signal in fine-tune mode
        'fret_reward':   0.0,
        'torque_reward': 0.0,
        'fret_error':    abs(fret_position - float(target_fret)),
        'torque_error':  abs(torque - TORQUE_OPTIMAL_HARMONIC),
        'target_fret':   target_fret,
        'filtered':      False,
        'filter_reason': '',
        'cosine_sim':    cosine_sim,
    }


def is_success(harmonic_prob: float) -> bool:
    """Check if harmonic probability exceeds the success threshold."""
    return harmonic_prob > SUCCESS_THRESHOLD
