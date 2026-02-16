"""
Gymnasium environment for GuitarBot harmonic RL training.

Uses fractional frets for position encoding to help agents learn
that frets 4, 5, 7 are harmonic nodes.
"""

from .action_space import (
    GuitarBotActionSpace,
    RLFretAction,
    PresserAction,
    fret_to_mm,
    mm_to_fret,
    PLAYABLE_STRINGS,
    STRING_TO_PLUCKER,
    PLUCKER_TO_STRING,
    FRET_MIN,
    FRET_MAX,
    TORQUE_MIN,
    TORQUE_MAX,
    TORQUE_SAFE_MIN,
    TORQUE_UNPRESSED,
    TORQUE_LIGHT,
    TORQUE_NORMAL,
    HARMONIC_FRETS_IN_RANGE,
)
from .osc_client import GuitarBotOSCClient
from .harmonic_env import HarmonicEnv

__all__ = [
    # Environment
    'HarmonicEnv',
    
    # OSC Client
    'GuitarBotOSCClient',
    
    # Action Space
    'GuitarBotActionSpace',
    'RLFretAction',
    'PresserAction',
    'fret_to_mm',
    'mm_to_fret',
    
    # Constants
    'PLAYABLE_STRINGS',
    'STRING_TO_PLUCKER',
    'PLUCKER_TO_STRING',
    'FRET_MIN',
    'FRET_MAX',
    'TORQUE_MIN',
    'TORQUE_MAX',
    'TORQUE_SAFE_MIN',
    'TORQUE_UNPRESSED',
    'TORQUE_LIGHT',
    'TORQUE_NORMAL',
    'HARMONIC_FRETS_IN_RANGE',
]
