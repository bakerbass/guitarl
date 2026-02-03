"""
action_space.py
RL Action Space definitions for GuitarBot low-level control.

Uses FRACTIONAL FRETS instead of raw millimeters to encode musical structure.
This helps the agent learn that frets 4, 5, 7, 12 are harmonic nodes.

The action space for the /rlfret command provides fine-grained control:
  - String selection (discrete: 0, 2, 4 - strings with pluckers)
  - Fret position (continuous: 0.0 to 9.0 fractional frets)
  - Fretter torque (continuous: 0 to 1000)

OSC Message Format: /RLFret <string_idx> <fret_position> <torque> [pluck_velocity]
Example: /RLFret 0 5.0 400  (string 0, fret 5, normal press)
Example: /RLFret 2 4.5 100 80  (string 2, between frets 4-5, light touch, velocity 80)

The agent works in fractional frets, which are sent directly via OSC.
The robot controller converts fret positions to mm internally.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

# ----------------------------------------------------------------------------
# Physical Constraints (from GuitarBot/tune.py)
# ----------------------------------------------------------------------------

# Fret positions in mm from nut (index 0 = fret 1)
SLIDER_MM_PER_FRET = [19.0, 52.0, 85.0, 112.0, 139.0, 164.0, 187.0, 211.0, 234.0]

# Derived: fret 0 (open) = 0mm, frets 1-9 from lookup
FRET_TO_MM = {0: 0.0}
for i, mm in enumerate(SLIDER_MM_PER_FRET):
    FRET_TO_MM[i + 1] = mm

# Fret range for RL
FRET_MIN = 0.0   # Open string
FRET_MAX = 9.0   # Highest fret position

# Slider range in mm
SLIDER_MIN_MM = 0.0
SLIDER_MAX_MM = max(SLIDER_MM_PER_FRET)  # 234.0

# Torque constraints (encoder units, 1000 = 100% rated torque)
TORQUE_MIN = -650      # No press / released
TORQUE_MAX = 1000   # Maximum safe torque
TORQUE_LIGHT = 50  # Light touch for harmonics
TORQUE_NORMAL = 400 # Normal fretting pressure
TORQUE_HEAVY = 650  # Heavy press for bends

# Harmonic nodes (where natural harmonics occur)
HARMONIC_FRETS = [4, 5, 7, 12]  # 12 is octave harmonic (beyond our range but noted)
HARMONIC_FRETS_IN_RANGE = [4, 5, 7]

# Plucker to String mapping (3 pluckers for 6 strings)
PLUCKER_TO_STRING = {
    0: 0,  # Plucker 0 -> String 0 (Low E)
    1: 2,  # Plucker 1 -> String 2 (D)
    2: 4,  # Plucker 2 -> String 4 (B)
}
STRING_TO_PLUCKER = {v: k for k, v in PLUCKER_TO_STRING.items()}
PLAYABLE_STRINGS = list(PLUCKER_TO_STRING.values())  # [0, 2, 4]

# Motor direction multipliers per string
SLIDER_MOTOR_DIRECTION = [-1, 1, 1, -1, -1, 1]
MM_TO_ENCODER_CONVERSION_FACTOR = 9.4
SLIDER_ENCODER_OFFSET = -2000


# ----------------------------------------------------------------------------
# Fret <-> MM Conversion
# ----------------------------------------------------------------------------

def fret_to_mm(fret: float) -> float:
    """
    Convert fractional fret position to mm.
    
    Uses linear interpolation between fret positions.
    
    Args:
        fret: Fractional fret (0.0 = open, 1.0 = fret 1, 4.5 = halfway between 4 and 5)
    
    Returns:
        Slider position in mm
    """
    fret = float(np.clip(fret, FRET_MIN, FRET_MAX))
    
    if fret == 0.0:
        return 0.0
    
    # Get integer fret bounds
    fret_low = int(np.floor(fret))
    fret_high = int(np.ceil(fret))
    
    # Handle exact fret positions
    if fret_low == fret_high or fret_low == 0:
        if fret_low == 0:
            mm_low = 0.0
            mm_high = FRET_TO_MM.get(1, SLIDER_MM_PER_FRET[0])
            return mm_low + (fret - fret_low) * (mm_high - mm_low)
        return FRET_TO_MM.get(fret_low, SLIDER_MAX_MM)
    
    # Linear interpolation
    mm_low = FRET_TO_MM.get(fret_low, SLIDER_MAX_MM)
    mm_high = FRET_TO_MM.get(fret_high, SLIDER_MAX_MM)
    
    fraction = fret - fret_low
    return mm_low + fraction * (mm_high - mm_low)


def mm_to_fret(mm: float) -> float:
    """
    Convert mm position to fractional fret.
    
    Inverse of fret_to_mm using linear interpolation.
    
    Args:
        mm: Slider position in mm
    
    Returns:
        Fractional fret position
    """
    mm = float(np.clip(mm, SLIDER_MIN_MM, SLIDER_MAX_MM))
    
    if mm <= 0.0:
        return 0.0
    
    # Find which fret interval we're in
    prev_mm = 0.0
    for fret_num in range(1, len(SLIDER_MM_PER_FRET) + 1):
        curr_mm = FRET_TO_MM[fret_num]
        if mm <= curr_mm:
            # Linearly interpolate within this fret interval
            interval_size = curr_mm - prev_mm
            if interval_size == 0:
                return float(fret_num)
            fraction = (mm - prev_mm) / interval_size
            return float(fret_num - 1) + fraction
        prev_mm = curr_mm
    
    # Beyond last fret
    return FRET_MAX


# ----------------------------------------------------------------------------
# RLFretAction Dataclass
# ----------------------------------------------------------------------------

@dataclass
class RLFretAction:
    """
    Represents a low-level fretting action for the RL agent.
    
    The agent specifies position in fractional frets, which is converted
    to mm for the actual robot/simulator control.
    
    Attributes:
        string_idx: Which string to fret (0, 2, or 4)
        fret_position: Fractional fret position (0.0 - 9.0)
        torque: Fretting pressure (0-1000)
    """
    string_idx: int
    fret_position: float
    torque: float
    
    def __post_init__(self):
        """Validate and clamp values to physical constraints."""
        # Validate string selection
        if self.string_idx not in PLAYABLE_STRINGS:
            raise ValueError(
                f"Invalid string_idx {self.string_idx}. "
                f"Must be one of {PLAYABLE_STRINGS}"
            )
        
        # Clamp fret position
        self.fret_position = float(np.clip(self.fret_position, FRET_MIN, FRET_MAX))
        
        # Clamp torque
        self.torque = float(np.clip(self.torque, TORQUE_MIN, TORQUE_MAX))
    
    @property
    def plucker_idx(self) -> int:
        """Get the plucker index for this string."""
        return STRING_TO_PLUCKER[self.string_idx]
    
    @property
    def slider_mm(self) -> float:
        """Convert fret position to mm."""
        return fret_to_mm(self.fret_position)
    
    @property
    def slider_direction(self) -> int:
        """Get the motor direction multiplier for this string's slider."""
        return SLIDER_MOTOR_DIRECTION[self.string_idx]
    
    @property
    def nearest_fret(self) -> int:
        """Get the nearest integer fret."""
        return int(round(self.fret_position))
    
    @property
    def is_at_harmonic(self) -> bool:
        """Check if position is close to a harmonic node."""
        return any(abs(self.fret_position - h) < 0.3 for h in HARMONIC_FRETS_IN_RANGE)
    
    def to_encoder_position(self) -> int:
        """Convert slider_mm to encoder ticks."""
        mm = self.slider_mm
        enc = int(mm * MM_TO_ENCODER_CONVERSION_FACTOR)
        enc = enc * self.slider_direction + SLIDER_ENCODER_OFFSET
        return enc
    
    def to_osc_args(self) -> Tuple[int, float, float]:
        """Return args for /RLFret OSC message (string_idx, fret_position, torque)."""
        return (self.string_idx, self.fret_position, self.torque)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'string_idx': self.string_idx,
            'fret_position': self.fret_position,
            'slider_mm': self.slider_mm,
            'torque': self.torque,
            'plucker_idx': self.plucker_idx,
            'encoder_position': self.to_encoder_position(),
            'nearest_fret': self.nearest_fret,
            'is_at_harmonic': self.is_at_harmonic,
        }


# ----------------------------------------------------------------------------
# GuitarBotActionSpace (Gymnasium-compatible)
# ----------------------------------------------------------------------------

class GuitarBotActionSpace:
    """
    Gymnasium-compatible action space for GuitarBot RL control.
    
    Uses FRACTIONAL FRETS for position to encode musical structure.
    
    Action representation (normalized, shape=(5,)):
        - [0:3]: String selection logits (argmax -> string 0, 2, or 4)
        - [3]: Fret position normalized to [-1, 1] -> [0, 9]
        - [4]: Torque normalized to [-1, 1] -> [0, 1000]
    
    For simpler continuous control without discrete string selection,
    use continuous_action_space which is Box(low=[-1,-1,-1], high=[1,1,1]).
    """
    
    def __init__(self, use_normalized: bool = True):
        """
        Initialize the action space.
        
        Args:
            use_normalized: If True, actions are in [-1, 1] range and scaled
                           to physical units. If False, use raw physical units.
        """
        self.use_normalized = use_normalized
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Set up gymnasium spaces."""
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError:
            import gym
            from gym import spaces
        
        # Discrete string selection (0, 1, 2 -> maps to strings 0, 2, 4)
        self.string_space = spaces.Discrete(len(PLAYABLE_STRINGS))
        
        if self.use_normalized:
            # Normalized continuous space [-1, 1] for fret and torque
            self.continuous_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
            
            # Flattened Box space for algorithms that prefer single Box
            # [string_logits(3), fret, torque]
            self.flat_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32
            )
            
            # Simpler 3D continuous space (string as continuous, fret, torque)
            self.simple_continuous_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
        else:
            # Raw physical units
            self.continuous_space = spaces.Box(
                low=np.array([FRET_MIN, TORQUE_MIN], dtype=np.float32),
                high=np.array([FRET_MAX, TORQUE_MAX], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
            
            self.flat_space = spaces.Box(
                low=np.array([0, 0, 0, FRET_MIN, TORQUE_MIN], dtype=np.float32),
                high=np.array([1, 1, 1, FRET_MAX, TORQUE_MAX], dtype=np.float32),
                shape=(5,),
                dtype=np.float32
            )
        
        # Combined Dict space
        self.dict_space = spaces.Dict({
            'string': self.string_space,
            'continuous': self.continuous_space,
        })
    
    def sample(self) -> RLFretAction:
        """Sample a random action from the space."""
        string_choice = np.random.randint(0, len(PLAYABLE_STRINGS))
        string_idx = PLAYABLE_STRINGS[string_choice]
        fret_position = np.random.uniform(FRET_MIN, FRET_MAX)
        torque = np.random.uniform(TORQUE_MIN, TORQUE_MAX)
        return RLFretAction(string_idx, fret_position, torque)
    
    def sample_harmonic(self) -> RLFretAction:
        """Sample an action targeting a harmonic node."""
        string_choice = np.random.randint(0, len(PLAYABLE_STRINGS))
        string_idx = PLAYABLE_STRINGS[string_choice]
        fret_position = float(np.random.choice(HARMONIC_FRETS_IN_RANGE))
        # Light torque for harmonics
        torque = np.random.uniform(TORQUE_LIGHT, TORQUE_NORMAL)
        return RLFretAction(string_idx, fret_position, torque)
    
    def from_normalized(self, action: np.ndarray) -> RLFretAction:
        """
        Convert a normalized action array to RLFretAction.
        
        Args:
            action: Array of shape (5,) with [string_logits(3), fret, torque]
                   All values in [-1, 1] range.
        
        Returns:
            RLFretAction with physical units.
        """
        # Extract string choice from logits (argmax)
        string_logits = action[:3]
        string_choice = int(np.argmax(string_logits))
        string_idx = PLAYABLE_STRINGS[string_choice]
        
        # Scale fret from [-1, 1] to [0, FRET_MAX]
        fret_normalized = action[3]
        fret_position = (fret_normalized + 1.0) / 2.0 * (FRET_MAX - FRET_MIN) + FRET_MIN
        
        # Scale torque from [-1, 1] to [0, TORQUE_MAX]
        torque_normalized = action[4]
        torque = (torque_normalized + 1.0) / 2.0 * (TORQUE_MAX - TORQUE_MIN) + TORQUE_MIN
        
        return RLFretAction(string_idx, fret_position, torque)
    
    def from_simple_normalized(self, action: np.ndarray) -> RLFretAction:
        """
        Convert a simple 3D normalized action to RLFretAction.
        
        Args:
            action: Array of shape (3,) with [string_continuous, fret, torque]
                   All values in [-1, 1] range.
        
        Returns:
            RLFretAction with physical units.
        """
        # Discretize string from continuous [-1, 1]
        string_continuous = (action[0] + 1.0) / 2.0  # -> [0, 1]
        string_choice = int(np.floor(string_continuous * len(PLAYABLE_STRINGS)))
        string_choice = np.clip(string_choice, 0, len(PLAYABLE_STRINGS) - 1)
        string_idx = PLAYABLE_STRINGS[string_choice]
        
        # Scale fret from [-1, 1] to [0, FRET_MAX]
        fret_normalized = action[1]
        fret_position = (fret_normalized + 1.0) / 2.0 * (FRET_MAX - FRET_MIN) + FRET_MIN
        
        # Scale torque from [-1, 1] to [0, TORQUE_MAX]
        torque_normalized = action[2]
        torque = (torque_normalized + 1.0) / 2.0 * (TORQUE_MAX - TORQUE_MIN) + TORQUE_MIN
        
        return RLFretAction(string_idx, fret_position, torque)
    
    def to_normalized(self, action: RLFretAction) -> np.ndarray:
        """
        Convert RLFretAction to normalized array representation.
        
        Returns:
            Array of shape (5,) with values in [-1, 1] range.
        """
        # One-hot encode string selection
        string_logits = np.zeros(3, dtype=np.float32)
        string_choice = PLAYABLE_STRINGS.index(action.string_idx)
        string_logits[string_choice] = 1.0
        
        # Normalize fret to [-1, 1]
        fret_normalized = (action.fret_position - FRET_MIN) / (FRET_MAX - FRET_MIN) * 2.0 - 1.0
        
        # Normalize torque to [-1, 1]
        torque_normalized = (action.torque - TORQUE_MIN) / (TORQUE_MAX - TORQUE_MIN) * 2.0 - 1.0
        
        return np.array([
            *string_logits,
            fret_normalized,
            torque_normalized
        ], dtype=np.float32)
    
    def get_harmonic_target_fret(self, target_harmonic: int) -> float:
        """Get the exact fret position for a target harmonic."""
        if target_harmonic not in HARMONIC_FRETS_IN_RANGE:
            raise ValueError(f"Invalid harmonic: {target_harmonic}. Must be in {HARMONIC_FRETS_IN_RANGE}")
        return float(target_harmonic)


# ----------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create action space
    action_space = GuitarBotActionSpace(use_normalized=True)
    
    print("=== GuitarBot RL Action Space (Fractional Frets) ===\n")
    print(f"Playable strings: {PLAYABLE_STRINGS}")
    print(f"Fret range: {FRET_MIN} - {FRET_MAX} fractional frets")
    print(f"Slider range: {SLIDER_MIN_MM} - {SLIDER_MAX_MM} mm")
    print(f"Torque range: {TORQUE_MIN} - {TORQUE_MAX}")
    print(f"Harmonic frets: {HARMONIC_FRETS_IN_RANGE}")
    print()
    
    # Test fret <-> mm conversion
    print("Fret to MM conversion:")
    for fret in [0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9]:
        mm = fret_to_mm(fret)
        back = mm_to_fret(mm)
        print(f"  Fret {fret:4.1f} -> {mm:6.1f} mm -> Fret {back:4.2f}")
    print()
    
    # Sample random action
    action = action_space.sample()
    print(f"Random action:")
    print(f"  {action}")
    print(f"  -> OSC args: /RLFret {action.to_osc_args()}")
    print(f"  -> Encoder pos: {action.to_encoder_position()}")
    print(f"  -> At harmonic: {action.is_at_harmonic}")
    print()
    
    # Sample harmonic action
    harmonic_action = action_space.sample_harmonic()
    print(f"Harmonic action:")
    print(f"  {harmonic_action}")
    print(f"  -> Fret {harmonic_action.fret_position} = {harmonic_action.slider_mm:.1f} mm")
    print()
    
    # Test normalized conversion roundtrip
    normalized = action_space.to_normalized(action)
    print(f"Normalized: {normalized}")
    
    reconstructed = action_space.from_normalized(normalized)
    print(f"Reconstructed: {reconstructed}")
