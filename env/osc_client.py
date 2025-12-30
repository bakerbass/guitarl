"""
OSC client wrapper for StringSim guitar simulator.

Provides high-level interface to control the StringSim plugin via OSC messages.
"""

import time
import logging
from typing import List, Optional, Tuple
from pythonosc import udp_client
import numpy as np


logger = logging.getLogger(__name__)


class StringSimOSCClient:
    """
    OSC client for controlling StringSim guitar simulator.
    
    Communicates with StringSim on port 8000 using OSC protocol.
    Supports motor control, state queries, and chord/note commands.
    """
    
    # StringSim OSC port
    DEFAULT_PORT = 8000
    DEFAULT_HOST = "127.0.0.1"
    
    # Motor ranges
    NUM_STRINGS = 6
    SLIDER_MM_RANGE = (0.0, 234.0)  # Fret position in mm
    FORCE_RANGE = (0.0, 1.0)  # Normalized presser force
    
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
    
    def reset(self, wait_time: float = 0.1) -> bool:
        """
        Reset all motors to neutral position.
        
        Args:
            wait_time: Time to wait after reset (seconds)
            
        Returns:
            True if reset successful
        """
        try:
            self.client.send_message("/reset", [])
            logger.info("Sent /reset command")
            time.sleep(wait_time)
            return True
        except Exception as e:
            logger.error(f"Failed to send /reset: {e}")
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
    
    def fret_to_mm(self, fret: int) -> float:
        """
        Convert fret number to mm position.
        
        Uses the SLIDER_MM_PER_FRET lookup table from StringSim.
        
        Args:
            fret: Fret number (0-9)
            
        Returns:
            Position in mm
        """
        SLIDER_MM_PER_FRET = {
            0: 0.0,    # Open string
            1: 19.0,
            2: 52.0,
            3: 85.0,
            4: 112.0,
            5: 139.0,
            6: 164.0,
            7: 187.0,
            8: 211.0,
            9: 234.0,
        }
        
        if fret < 0:
            return -1.0  # Muted
        
        return SLIDER_MM_PER_FRET.get(fret, 234.0)  # Clamp to max
    
    def close(self):
        """Clean up resources."""
        logger.info("OSC client closed")
