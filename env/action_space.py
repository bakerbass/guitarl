"""  
action_space.py
RL Action Space definitions for GuitarBot low-level control.

Uses FRACTIONAL FRETS instead of raw millimeters to encode musical structure.
This helps the agent learn that frets 4, 5, 7, 12 are harmonic nodes.

The action space for the /rlfret command provides fine-grained control:
  - String selection (discrete: 0, 2, 4 - strings with pluckers)
  - Fret position (continuous: 0.0 to 9.0 fractional frets)
  - Press/Unpress decision (discrete: press or release the string)
  - Torque magnitude (continuous: TORQUE_SAFE_MIN to TORQUE_MAX, only used when pressing)

The press/unpress split avoids the 0-15 dead zone in the microcontroller's
safety logic (processTrajPoints), which overrides torque mode to position mode
when the presser encoder is <= 15 ticks and the commanded value is <= 0.
By separating intent from magnitude, the agent never accidentally produces
values in that dead zone.

OSC Message Format: /RLFret <string_idx> <fret_position> <torque> [pluck_velocity]
Example: /RLFret 0 5.0 400  (string 0, fret 5, normal press)
Example: /RLFret 2 4.5 100 80  (string 2, between frets 4-5, light touch, velocity 80)
"""

from dataclasses import dataclass
from enum import IntEnum
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
TORQUE_UNPRESSED = -650  # Presser fully released
TORQUE_SAFE_MIN = 16     # Minimum torque that avoids the position-mode override
                         # (Arduino switches to position mode when pos<=15 AND cmd<=0)
TORQUE_MIN = -650        # Legacy alias
TORQUE_MAX = 650         # Maximum safe torque
TORQUE_LIGHT = 50        # Light touch for harmonics
TORQUE_NORMAL = 400      # Normal fretting pressure
TORQUE_HEAVY = 650       # Heavy press for bends


class PresserAction(IntEnum):
    """High-level presser intent — avoids the 0-15 torque dead zone."""
    UNPRESS = 0   # Release the string (send TORQUE_UNPRESSED)
    PRESS = 1     # Press the string (torque in [TORQUE_SAFE_MIN, TORQUE_MAX])

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
    
    The agent specifies:
      - string_idx: which string to play
      - fret_position: where along the neck (fractional frets)
      - press_action: PRESS or UNPRESS (high-level intent)
      - torque: magnitude when pressing (TORQUE_SAFE_MIN..TORQUE_MAX)
                ignored when unpressing (TORQUE_UNPRESSED is sent instead)
    
    Attributes:
        string_idx: Which string to fret (0, 2, or 4)
        fret_position: Fractional fret position (0.0 - 9.0)
        press_action: PresserAction.PRESS or PresserAction.UNPRESS
        torque: Fretting pressure when pressing (TORQUE_SAFE_MIN - TORQUE_MAX)
    """
    string_idx: int
    fret_position: float
    press_action: PresserAction
    torque: float
    
    def __post_init__(self):
        """Validate and clamp values to physical constraints."""
        # Validate string selection
        if self.string_idx not in PLAYABLE_STRINGS:
            raise ValueError(
                f"Invalid string_idx {self.string_idx}. "
                f"Must be one of {PLAYABLE_STRINGS}"
            )
        
        # Ensure press_action is a PresserAction
        self.press_action = PresserAction(int(self.press_action))
        
        # Clamp fret position
        self.fret_position = float(np.clip(self.fret_position, FRET_MIN, FRET_MAX))
        
        # Clamp torque to safe pressing range
        if self.press_action == PresserAction.PRESS:
            self.torque = float(np.clip(self.torque, TORQUE_SAFE_MIN, TORQUE_MAX))
        else:
            # When unpressing, torque value is ignored — we always send TORQUE_UNPRESSED
            self.torque = float(TORQUE_UNPRESSED)
    
    @property
    def effective_torque(self) -> float:
        """The torque value actually sent to the robot."""
        if self.press_action == PresserAction.PRESS:
            return self.torque
        return float(TORQUE_UNPRESSED)
    
    @property
    def is_pressing(self) -> bool:
        """Whether this action presses the string."""
        return self.press_action == PresserAction.PRESS
    
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
        return (self.string_idx, self.fret_position, self.effective_torque)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'string_idx': self.string_idx,
            'fret_position': self.fret_position,
            'slider_mm': self.slider_mm,
            'press_action': self.press_action.name,
            'torque': self.torque,
            'effective_torque': self.effective_torque,
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
    Separates press/unpress intent from torque magnitude to avoid
    the 0-15 torque dead zone on the microcontroller.
    
    Action representation (normalized, shape=(6,)):
        - [0:3]: String selection logits (argmax -> string 0, 2, or 4)
        - [3]:   Fret position normalized to [-1, 1] -> [0, 9]
        - [4]:   Press/Unpress decision: > 0 = PRESS, <= 0 = UNPRESS
        - [5]:   Torque magnitude normalized to [-1, 1] -> [TORQUE_SAFE_MIN, TORQUE_MAX]
                 (only used when pressing)
    
    For simpler continuous control without discrete string selection,
    use simple_continuous_space: Box(low=[-1,-1,-1,-1], high=[1,1,1,1]).
        - [0]: String continuous (discretized internally)
        - [1]: Fret position
        - [2]: Press/Unpress decision
        - [3]: Torque magnitude
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
            # Normalized continuous space [-1, 1] for fret, press decision, and torque
            self.continuous_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
            
            # Flattened Box space for algorithms that prefer single Box
            # [string_logits(3), fret, press_decision, torque_magnitude]
            self.flat_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32
            )
            
            # Simpler 4D continuous space (string as continuous, fret, press, torque)
            self.simple_continuous_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )
        else:
            # Raw physical units
            self.continuous_space = spaces.Box(
                low=np.array([FRET_MIN, 0.0, TORQUE_SAFE_MIN], dtype=np.float32),
                high=np.array([FRET_MAX, 1.0, TORQUE_MAX], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
            
            self.flat_space = spaces.Box(
                low=np.array([0, 0, 0, FRET_MIN, 0.0, TORQUE_SAFE_MIN], dtype=np.float32),
                high=np.array([1, 1, 1, FRET_MAX, 1.0, TORQUE_MAX], dtype=np.float32),
                shape=(6,),
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
        press_action = PresserAction(np.random.randint(0, 2))
        torque = np.random.uniform(TORQUE_SAFE_MIN, TORQUE_MAX)
        return RLFretAction(string_idx, fret_position, press_action, torque)
    
    def sample_harmonic(self) -> RLFretAction:
        """Sample an action targeting a harmonic node (always pressing)."""
        string_choice = np.random.randint(0, len(PLAYABLE_STRINGS))
        string_idx = PLAYABLE_STRINGS[string_choice]
        fret_position = float(np.random.choice(HARMONIC_FRETS_IN_RANGE))
        # Light torque for harmonics, always pressing
        torque = np.random.uniform(TORQUE_LIGHT, TORQUE_NORMAL)
        return RLFretAction(string_idx, fret_position, PresserAction.PRESS, torque)
    
    def from_normalized(self, action: np.ndarray) -> RLFretAction:
        """
        Convert a normalized action array to RLFretAction.
        
        Args:
            action: Array of shape (6,) with
                    [string_logits(3), fret, press_decision, torque_magnitude]
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
        
        # Press decision: > 0 means PRESS, <= 0 means UNPRESS
        press_action = PresserAction.PRESS if action[4] > 0.0 else PresserAction.UNPRESS
        
        # Scale torque from [-1, 1] to [TORQUE_SAFE_MIN, TORQUE_MAX]
        torque_normalized = action[5]
        torque = (torque_normalized + 1.0) / 2.0 * (TORQUE_MAX - TORQUE_SAFE_MIN) + TORQUE_SAFE_MIN
        
        return RLFretAction(string_idx, fret_position, press_action, torque)
    
    def from_simple_normalized(self, action: np.ndarray) -> RLFretAction:
        """
        Convert a simple 4D normalized action to RLFretAction.
        
        Args:
            action: Array of shape (4,) with
                    [string_continuous, fret, press_decision, torque_magnitude]
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
        
        # Press decision: > 0 means PRESS, <= 0 means UNPRESS
        press_action = PresserAction.PRESS if action[2] > 0.0 else PresserAction.UNPRESS
        
        # Scale torque from [-1, 1] to [TORQUE_SAFE_MIN, TORQUE_MAX]
        torque_normalized = action[3]
        torque = (torque_normalized + 1.0) / 2.0 * (TORQUE_MAX - TORQUE_SAFE_MIN) + TORQUE_SAFE_MIN
        
        return RLFretAction(string_idx, fret_position, press_action, torque)
    
    def to_normalized(self, action: RLFretAction) -> np.ndarray:
        """
        Convert RLFretAction to normalized array representation.
        
        Returns:
            Array of shape (6,) with values in [-1, 1] range.
        """
        # One-hot encode string selection
        string_logits = np.zeros(3, dtype=np.float32)
        string_choice = PLAYABLE_STRINGS.index(action.string_idx)
        string_logits[string_choice] = 1.0
        
        # Normalize fret to [-1, 1]
        fret_normalized = (action.fret_position - FRET_MIN) / (FRET_MAX - FRET_MIN) * 2.0 - 1.0
        
        # Press decision: PRESS = +1, UNPRESS = -1
        press_normalized = 1.0 if action.press_action == PresserAction.PRESS else -1.0
        
        # Normalize torque to [-1, 1] (using safe pressing range)
        torque_normalized = (action.torque - TORQUE_SAFE_MIN) / (TORQUE_MAX - TORQUE_SAFE_MIN) * 2.0 - 1.0
        
        return np.array([
            *string_logits,
            fret_normalized,
            press_normalized,
            torque_normalized,
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
    
    print("=== GuitarBot RL Action Space (Press/Unpress + Fractional Frets) ===\n")
    print(f"Playable strings: {PLAYABLE_STRINGS}")
    print(f"Fret range: {FRET_MIN} - {FRET_MAX} fractional frets")
    print(f"Slider range: {SLIDER_MIN_MM} - {SLIDER_MAX_MM} mm")
    print(f"Torque range (pressing): {TORQUE_SAFE_MIN} - {TORQUE_MAX}")
    print(f"Torque (unpressed): {TORQUE_UNPRESSED}")
    print(f"Harmonic frets: {HARMONIC_FRETS_IN_RANGE}")
    print()
    
    # Test fret <-> mm conversion
    print("Fret to MM conversion:")
    for fret in [0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9]:
        mm = fret_to_mm(fret)
        back = mm_to_fret(mm)
        print(f"  Fret {fret:4.1f} -> {mm:6.1f} mm -> Fret {back:4.2f}")
    print()
    
    # Sample random action (may be press or unpress)
    action = action_space.sample()
    print(f"Random action:")
    print(f"  {action}")
    print(f"  -> press_action: {action.press_action.name}")
    print(f"  -> effective_torque: {action.effective_torque}")
    print(f"  -> OSC args: /RLFret {action.to_osc_args()}")
    print(f"  -> Encoder pos: {action.to_encoder_position()}")
    print(f"  -> At harmonic: {action.is_at_harmonic}")
    print()
    
    # Sample harmonic action (always pressing)
    harmonic_action = action_space.sample_harmonic()
    print(f"Harmonic action:")
    print(f"  {harmonic_action}")
    print(f"  -> Fret {harmonic_action.fret_position} = {harmonic_action.slider_mm:.1f} mm")
    print(f"  -> press_action: {harmonic_action.press_action.name}")
    print()
    
    # Test press vs unpress
    press_action = RLFretAction(0, 5.0, PresserAction.PRESS, 200)
    unpress_action = RLFretAction(0, 5.0, PresserAction.UNPRESS, 200)
    print(f"Press action:   torque={press_action.torque}, effective={press_action.effective_torque}")
    print(f"Unpress action: torque={unpress_action.torque}, effective={unpress_action.effective_torque}")
    print()
    
    # Test normalized conversion roundtrip
    normalized = action_space.to_normalized(press_action)
    print(f"Normalized (6D): {normalized}")
    
    reconstructed = action_space.from_normalized(normalized)
    print(f"Reconstructed: {reconstructed}")
    print(f"  press_action: {reconstructed.press_action.name}, torque: {reconstructed.torque:.1f}")
