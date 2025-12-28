#!/usr/bin/env python3
"""
Unified simulation entry point
Makes batched simulation the single source of truth with B=1 wrapper for compatibility
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from .core import SimulationConfig
from .batched import BatchedSimulation


class SimulationSingle:
    """
    Single-environment wrapper around BatchedSimulation
    Provides backward compatibility with the old Simulation API
    """
    
    def __init__(self, config: SimulationConfig = None, seed: int = 42):
        """
        Initialize single simulation using batched backend with B=1
        
        Args:
            config: Simulation configuration
            seed: Random seed for deterministic behavior
        """
        self.config = config or SimulationConfig()
        self.seed = seed
        
        # Create batched simulation with B=1
        self.batched_sim = BatchedSimulation(batch_size=1, config=self.config)
        
        # Reset with the provided seed
        self.reset(seed)
    
    def reset(self, seed: int = None):
        """Reset simulation to initial state"""
        if seed is not None:
            self.seed = seed
        
        # Reset batched sim with single seed
        self.batched_sim.reset(seeds=[self.seed])
    
    def step(self, dt: float, action: Dict[str, float]) -> Dict[str, Any]:
        """
        Step simulation forward by dt seconds
        
        Args:
            dt: Time step in seconds
            action: Action dict with 'steer' and 'throttle' keys
            
        Returns:
            Step info dict compatible with old Simulation API
        """
        # Convert single action to batch format
        batch_actions = {
            'steer': np.array([action['steer']], dtype=np.float32),
            'throttle': np.array([action['throttle']], dtype=np.float32)
        }
        
        # Step batched simulation
        step_info = self.batched_sim.step(batch_actions, dt)
        
        # Convert batch results back to single format
        return self._unbatch_step_info(step_info)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state in old API format"""
        # Get batched state
        batch_state = self.batched_sim.get_state()
        
        # Convert to single format
        return self._unbatch_state(batch_state)
    
    def get_observations(self, v_scale: float = 400.0, omega_scale: float = 10.0) -> np.ndarray:
        """Get observation vector for the single environment"""
        batch_obs = self.batched_sim.get_observations(v_scale, omega_scale)
        return batch_obs[0]  # Extract single observation from batch
    
    def _unbatch_step_info(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batched step info to single format"""
        return {
            'food_collected_this_step': bool(batch_info['food_collected_this_step'][0]),
            'agent_state': {
                'x': float(batch_info['agent_states']['x'][0]),
                'y': float(batch_info['agent_states']['y'][0]),
                'vx': float(batch_info['agent_states']['vx'][0]),
                'vy': float(batch_info['agent_states']['vy'][0]),
                'theta': float(batch_info['agent_states']['theta'][0]),
                'omega': float(batch_info['agent_states']['omega'][0]),
                'throttle': float(batch_info['agent_states']['throttle'][0])
            },
            'food_position': {
                'x': float(batch_info['food_positions'][0][0]),
                'y': float(batch_info['food_positions'][0][1])
            },
            'vision_distances': batch_info['vision_distances'][0].tolist(),
            'vision_hit_types': self._convert_materials_to_hit_types(batch_info['vision_materials'][0]),
            'vision_hit_wall_ids': [None] * len(batch_info['vision_distances'][0]),  # Not used in batched
            'food_collected': int(batch_info['food_collected'][0]),
            'time': float(batch_info['time'][0]),
            'step': int(batch_info['step'][0])
        }
    
    def _unbatch_state(self, batch_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batched state to single format"""
        return {
            'agent_state': {
                'x': float(batch_state['agent_states']['x'][0]),
                'y': float(batch_state['agent_states']['y'][0]),
                'vx': float(batch_state['agent_states']['vx'][0]),
                'vy': float(batch_state['agent_states']['vy'][0]),
                'theta': float(batch_state['agent_states']['theta'][0]),
                'omega': float(batch_state['agent_states']['omega'][0]),
                'throttle': float(batch_state['agent_states']['throttle'][0])
            },
            'food_position': {
                'x': float(batch_state['food_positions'][0][0]),
                'y': float(batch_state['food_positions'][0][1])
            },
            'vision_distances': batch_state['vision_distances'][0].tolist(),
            'vision_hit_types': self._convert_materials_to_hit_types(batch_state['vision_materials'][0]),
            'vision_hit_wall_ids': [None] * len(batch_state['vision_distances'][0]),
            'food_collected': int(batch_state['food_collected'][0]),
            'time': float(batch_state['time'][0]),
            'step': int(batch_state['step'][0])
        }
    
    def _convert_materials_to_hit_types(self, materials: np.ndarray) -> List[Optional[str]]:
        """Convert material IDs to hit type strings for backward compatibility"""
        from .batched.scene import MATERIAL_NONE, MATERIAL_WALL, MATERIAL_FOOD
        
        hit_types = []
        for material in materials:
            if material == MATERIAL_NONE:
                hit_types.append(None)
            elif material == MATERIAL_WALL:
                hit_types.append('wall')
            elif material == MATERIAL_FOOD:
                hit_types.append('food')
            elif material >= 3:  # Obstacle materials (3+) should be treated as walls
                hit_types.append('wall')
            else:
                hit_types.append(None)
        
        return hit_types
    
    def render_state(self, env_idx: int = 0):
        """Get render state for GUI (env_idx ignored, always 0)"""
        return self.get_state()


# Backward compatibility alias
Simulation = SimulationSingle


def create_simulation(config: SimulationConfig = None, seed: int = 42, 
                     batch_size: int = None) -> Union[SimulationSingle, BatchedSimulation]:
    """
    Factory function for creating simulations
    
    Args:
        config: Simulation configuration
        seed: Random seed (for single sim) or base seed (for batched)
        batch_size: If None, creates single sim; if int, creates batched sim
        
    Returns:
        SimulationSingle (B=1) or BatchedSimulation (B>1)
    """
    if batch_size is None or batch_size == 1:
        return SimulationSingle(config, seed)
    else:
        sim = BatchedSimulation(batch_size, config)
        # Generate seeds for all environments
        seeds = [seed + i for i in range(batch_size)]
        sim.reset(seeds)
        return sim