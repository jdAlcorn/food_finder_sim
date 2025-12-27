#!/usr/bin/env python3
"""
Simple scripted controller for testing load/watch functionality
"""

import math
from typing import Dict, Any
from .base import Policy


class ScriptedPolicy(Policy):
    """Simple scripted policy that seeks food"""
    
    def __init__(self, seek_strength: float = 2.0, throttle_level: float = 0.8):
        self.name = "Scripted"
        self.seek_strength = seek_strength
        self.throttle_level = throttle_level
    
    def reset(self) -> None:
        """Reset policy state"""
        pass
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """Simple food-seeking behavior"""
        agent_state = sim_state['agent_state']
        food_pos = sim_state['food_position']
        
        # Calculate angle to food
        dx = food_pos['x'] - agent_state['x']
        dy = food_pos['y'] - agent_state['y']
        angle_to_food = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = angle_to_food - agent_state['theta']
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Simple steering toward food
        steer = max(-1.0, min(1.0, angle_diff * self.seek_strength))
        
        # Constant throttle
        throttle = self.throttle_level
        
        return {
            'steer': steer,
            'throttle': throttle
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'seek_strength': self.seek_strength,
            'throttle_level': self.throttle_level
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'ScriptedPolicy':
        """Create policy from parameters"""
        return cls(
            seek_strength=params.get('seek_strength', 2.0),
            throttle_level=params.get('throttle_level', 0.8)
        )