#!/usr/bin/env python3
"""
DEPRECATED: Core 2D Continuous Simulation Engine

This module is deprecated. Use the unified simulation interface instead:
- from src.sim import Simulation  # Now uses batched backend with B=1
- from src.sim import BatchedSimulation  # For B>1 environments
- from src.sim import create_simulation  # Factory function

The old single-env simulation has been replaced with a batched implementation
that supports B=1 as a special case for backward compatibility.
"""

import warnings
from typing import Dict, Any


class SimulationConfig:
    """Configuration for simulation parameters"""
    def __init__(self):
        # World
        self.world_width = 800
        self.world_height = 800
        
        # Agent
        self.agent_radius = 12
        self.agent_max_thrust = 500.0
        self.agent_max_turn_accel = 13.0
        self.agent_linear_drag = 2.0
        self.agent_angular_drag = 5.0
        self.wall_restitution = 0.6
        
        # Food
        self.food_radius = 8
        
        # Vision
        self.fov_degrees = 120
        self.num_rays = 128
        self.max_range = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'world_width': self.world_width,
            'world_height': self.world_height,
            'agent_radius': self.agent_radius,
            'agent_max_thrust': self.agent_max_thrust,
            'agent_max_turn_accel': self.agent_max_turn_accel,
            'agent_linear_drag': self.agent_linear_drag,
            'agent_angular_drag': self.agent_angular_drag,
            'wall_restitution': self.wall_restitution,
            'food_radius': self.food_radius,
            'fov_degrees': self.fov_degrees,
            'num_rays': self.num_rays,
            'max_range': self.max_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# Deprecated classes - kept for compatibility but issue warnings
class Agent:
    """DEPRECATED: Use unified simulation interface instead"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Agent class is deprecated. Use unified simulation interface: "
            "from src.sim import Simulation",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Agent class removed. Use unified simulation interface.")


class Food:
    """DEPRECATED: Use unified simulation interface instead"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Food class is deprecated. Use unified simulation interface: "
            "from src.sim import Simulation",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Food class removed. Use unified simulation interface.")


class VisionSystem:
    """DEPRECATED: Use unified simulation interface instead"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VisionSystem class is deprecated. Use unified simulation interface: "
            "from src.sim import Simulation",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("VisionSystem class removed. Use unified simulation interface.")


class Simulation:
    """DEPRECATED: Use unified simulation interface instead"""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "src.sim.core.Simulation is deprecated. Use: "
            "from src.sim import Simulation  # Now uses batched backend",
            DeprecationWarning,
            stacklevel=2
        )
        # Redirect to unified interface
        from .unified import SimulationSingle
        return SimulationSingle(*args, **kwargs)