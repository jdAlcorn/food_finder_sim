#!/usr/bin/env python3
"""
Base policy/controller interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Policy(ABC):
    """Base class for all policies/controllers"""
    
    def reset(self) -> None:
        """Reset policy state (optional override)"""
        pass
    
    @abstractmethod
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action from current simulation state
        
        Args:
            sim_state: Dictionary containing simulation state with keys:
                - 'agent_state': Dict with x, y, vx, vy, theta, omega, throttle, speed
                - 'food_position': Dict with x, y
                - 'vision_distances': List of ray distances
                - 'vision_hit_types': List of hit types ('wall', 'food', None)
                - 'vision_hit_wall_ids': List of wall IDs (0-3) or None
                - 'time': Current simulation time
                - 'step': Current step count
                - 'food_collected': Total food collected
        
        Returns:
            Dict with action keys:
                - 'steer': float in [-1, 1] for steering input
                - 'throttle': float in [0, 1] for throttle input
        """
        pass