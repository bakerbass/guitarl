"""
OSC client for GuitarBot control.

Provides high-level interface to control the GuitarBot via OSC messages.
Supports both standard /fret (midi-based) and /rlfret (low-level RL) commands.
"""

import time
import logging
from typing import List, Optional, Tuple
from pythonosc import udp_client
import numpy as np

from .action_space import (
    RLFretAction,
    PresserAction,
    fret_to_mm, 
    mm_to_fret,
    PLAYABLE_STRINGS,
    STRING_TO_PLUCKER,
    FRET_MIN,
    FRET_MAX,
    TORQUE_MIN,
    TORQUE_MAX,
    TORQUE_SAFE_MIN,
    HARMONIC_FRETS_IN_RANGE,
)


logger = logging.getLogger(__name__)


class GuitarBotOSCClient:
    """
    OSC client for controlling the GuitarBot.
    
    Communicates via OSC protocol using /fret and /rlfret messages.
    Supports motor control, state queries, and chord/note commands.
    """
    
    # StringSim OSC port
    DEFAULT_PORT = 8000
    DEFAULT_HOST = "127.0.0.1"
    
    # Motor ranges
    NUM_STRINGS = 6
    SLIDER_MM_RANGE = (0.0, 234.0)  # Fret position in mm
    FORCE_RANGE = (0.0, 1.0)  # Normalized presser force (for /fret command)
    TORQUE_RANGE = (TORQUE_MIN, TORQUE_MAX)  # Raw torque (for /rlfret command)
    
    # Fret positions (mm) for harmonics
    HARMONIC_FRETS = {
        4: 112.0,  # Major 17th
        5: 139.0,  # Major 14th
        7: 187.0,  # Perfect 12th
    }
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, 
                 connection_timeout: float = 2.0):
        """
        Initialize OSC client.
        
        Args:
            host: StringSim host address
            port: OSC port (default 8000)
            connection_timeout: Timeout for connection validation
        """
        self.host = host
        self.port = port
        self.connection_timeout = connection_timeout
        
        try:
            self.client = udp_client.SimpleUDPClient(host, port)
            logger.info(f"OSC client initialized: {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to initialize OSC client: {e}")
            raise
    
    def send_fret(self, string_index: int, position_mm: float, 
                  force: float, timestamp: Optional[float] = None) -> bool:
        """
        Send /fret message to control a single string.
        
        Args:
            string_index: String index (0-5, where 0=High E, 5=Low E)
            position_mm: Slider position in mm (0-234)
            force: Presser force (0-1, normalized)
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            True if message sent successfully
        """
        # Validate inputs
        if not (0 <= string_index < self.NUM_STRINGS):
            logger.error(f"Invalid string index: {string_index}")
            return False
        
        position_mm = np.clip(position_mm, *self.SLIDER_MM_RANGE)
        force = np.clip(force, *self.FORCE_RANGE)
        
        # Build message
        if timestamp is None:
            message = [string_index, float(position_mm), float(force)]
        else:
            message = [string_index, float(position_mm), float(force), float(timestamp)]
        
        try:
            self.client.send_message("/fret", message)
            logger.debug(f"Sent /fret: string={string_index}, pos={position_mm:.1f}mm, force={force:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to send /fret: {e}")
            return False
    
    def send_rlfret(self, action: RLFretAction, timestamp: Optional[float] = None) -> bool:
        """
        Send /RLFret message for RL-based low-level control.
        
        This is the primary interface for RL agents.
        
        OSC Format: /RLFret <string_idx> <fret_position> <torque>
        
        Args:
            action: RLFretAction with string, fret position, and torque
            timestamp: Optional timestamp
            
        Returns:
            True if message sent successfully
        """
        string_idx, fret_pos, torque = action.to_osc_args()
        
        # Validate string is playable (has a plucker)
        if string_idx not in PLAYABLE_STRINGS:
            logger.error(f"String {string_idx} has no plucker. Use strings {PLAYABLE_STRINGS}")
            return False
        
        # Build message
        if timestamp is None:
            message = [int(string_idx), float(fret_pos), float(torque)]
        else:
            message = [int(string_idx), float(fret_pos), float(torque), float(timestamp)]
        
        try:
            self.client.send_message("/RLFret", message)
            logger.info(
                f"Sent /RLFret: string={string_idx}, fret={action.fret_position:.2f}, "
                f"torque={torque:.0f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send /RLFret: {e}")
            return False
    
    def send_rlfret_raw(self, string_idx: int, fret_position: float, 
                        torque: float, press: bool = True,
                        timestamp: Optional[float] = None) -> bool:
        """
        Send /rlfret message with raw parameters (convenience wrapper).
        
        Args:
            string_idx: String index (0, 2, or 4 - must have plucker)
            fret_position: Fractional fret position (0.0 - 9.0)
            torque: Fretting torque (TORQUE_SAFE_MIN - TORQUE_MAX)
            press: If True, press the string; if False, release it
            timestamp: Optional timestamp
            
        Returns:
            True if message sent successfully
        """
        try:
            press_action = PresserAction.PRESS if press else PresserAction.UNPRESS
            action = RLFretAction(string_idx, fret_position, press_action, torque)
            return self.send_rlfret(action, timestamp)
        except ValueError as e:
            logger.error(f"Invalid rlfret parameters: {e}")
            return False

    def reset(self, wait_time: float = 0.1) -> bool:
        """
        Reset all motors to neutral position.
        
        Args:
            wait_time: Time to wait after reset (seconds)
            
        Returns:
            True if reset successful
        """
        try:
            self.client.send_message("/Reset", [])
            logger.info("Sent /Reset command")
            time.sleep(wait_time)
            return True
        except Exception as e:
            logger.error(f"Failed to send /Reset: {e}")
            return False
    
    def get_state(self) -> bool:
        """
        Request current motor state.
        
        Note: StringSim currently broadcasts state rather than responding directly.
        This sends the request, but state must be captured via OSC listener.
        
        Returns:
            True if request sent successfully
        """
        try:
            self.client.send_message("/get_state", [])
            logger.debug("Sent /get_state request")
            return True
        except Exception as e:
            logger.error(f"Failed to send /get_state: {e}")
            return False
    
    def send_chord(self, chord_symbol: str, timestamp: Optional[float] = None) -> bool:
        """
        Send chord command (higher-level control).
        
        Args:
            chord_symbol: Chord symbol (e.g., "C", "Am", "G7")
            timestamp: Optional timestamp
            
        Returns:
            True if message sent successfully
        """
        message = [chord_symbol]
        if timestamp is not None:
            message.append(float(timestamp))
        
        try:
            self.client.send_message("/chord", message)
            logger.debug(f"Sent /chord: {chord_symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to send /chord: {e}")
            return False
    
    def send_harmonic(self, string_index: int, fret: int, 
                      force: float = 0.3, wait_time: float = 0.05) -> bool:
        """
        Convenience method to play a natural harmonic.
        
        Args:
            string_index: String to play (0-5)
            fret: Harmonic fret number (4, 5, or 7)
            force: Light force for harmonic (default 0.3)
            wait_time: Time to wait after sending command
            
        Returns:
            True if successful
        """
        if fret not in self.HARMONIC_FRETS:
            logger.error(f"Invalid harmonic fret: {fret}. Must be 4, 5, or 7")
            return False
        
        position_mm = self.HARMONIC_FRETS[fret]
        success = self.send_fret(string_index, position_mm, force)
        
        if success and wait_time > 0:
            time.sleep(wait_time)
        
        return success
    
    def validate_connection(self) -> bool:
        """
        Validate connection to StringSim by sending a reset command.
        
        Returns:
            True if connection appears valid
        """
        try:
            self.reset(wait_time=0.1)
            logger.info("Connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def send_rl_harmonic(self, string_idx: int, fret: int, 
                         torque: float = 100.0, wait_time: float = 0.05) -> bool:
        """
        Convenience method to play a harmonic using /rlfret.
        
        Uses light torque appropriate for harmonics.
        
        Args:
            string_idx: String to play (0, 2, or 4 - must have plucker)
            fret: Harmonic fret number (4, 5, or 7)
            torque: Light torque for harmonic (default 100)
            wait_time: Time to wait after sending command
            
        Returns:
            True if successful
        """
        if fret not in HARMONIC_FRETS_IN_RANGE:
            logger.error(f"Invalid harmonic fret: {fret}. Must be in {HARMONIC_FRETS_IN_RANGE}")
            return False
        
        if string_idx not in PLAYABLE_STRINGS:
            logger.error(f"String {string_idx} has no plucker. Use {PLAYABLE_STRINGS}")
            return False
        
        action = RLFretAction(string_idx, float(fret), PresserAction.PRESS, torque)
        success = self.send_rlfret(action)
        
        if success and wait_time > 0:
            time.sleep(wait_time)
        
        return success
    
    def fret_to_mm_legacy(self, fret: int) -> float:
        """
        Convert integer fret number to mm position (legacy method).
        
        For fractional fret support, use action_space.fret_to_mm().
        
        Args:
            fret: Fret number (0-9)
            
        Returns:
            Position in mm
        """
        # Use the action_space version for consistency
        return fret_to_mm(float(fret))
    
    def close(self):
        """Clean up resources."""
        logger.info("OSC client closed")
