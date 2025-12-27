#!/usr/bin/env python3
"""
Manual keyboard controller
"""

from typing import Dict, Any
from .base import Policy


class ManualPolicy(Policy):
    """Manual keyboard control policy"""
    
    def __init__(self):
        self.name = "Manual"
        self._pygame = None
    
    def _ensure_pygame(self):
        """Lazy import pygame only when needed"""
        if self._pygame is None:
            try:
                import pygame
                self._pygame = pygame
            except ImportError:
                raise ImportError("pygame is required for ManualPolicy but not installed")
    
    def reset(self) -> None:
        """Reset policy state"""
        pass
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """Get action from keyboard input"""
        self._ensure_pygame()
        
        # Get current key state
        keys = self._pygame.key.get_pressed()
        
        # Throttle input - direct control (0 or 1)
        throttle = 1.0 if keys[self._pygame.K_w] else 0.0
        
        # Steering input
        steer = 0.0
        if keys[self._pygame.K_a]:
            steer = -1.0
        elif keys[self._pygame.K_d]:
            steer = 1.0
        
        return {
            'steer': steer,
            'throttle': throttle
        }