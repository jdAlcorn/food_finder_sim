#!/usr/bin/env python3
"""
Manual keyboard controller
"""

import pygame
from typing import Dict, Any
from .base import Policy


class ManualPolicy(Policy):
    """Manual keyboard control policy"""
    
    def __init__(self):
        self.name = "Manual"
    
    def reset(self) -> None:
        """Reset policy state"""
        pass
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """Get action from keyboard input"""
        # Get current key state
        keys = pygame.key.get_pressed()
        
        # Throttle input - direct control (0 or 1)
        throttle = 1.0 if keys[pygame.K_w] else 0.0
        
        # Steering input
        steer = 0.0
        if keys[pygame.K_a]:
            steer = -1.0
        elif keys[pygame.K_d]:
            steer = 1.0
        
        return {
            'steer': steer,
            'throttle': throttle
        }