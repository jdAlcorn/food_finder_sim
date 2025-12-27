#!/usr/bin/env python3
"""
Neural Network Policy Stub
Implements full observation extraction and transformation pipeline for future NN integration.
Currently returns safe default actions.
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List
from .base import Policy


class NeuralPolicyStub(Policy):
    """
    Neural network policy stub that builds egocentric observation vectors.
    Returns safe default actions until a real NN is plugged in.
    """
    
    def __init__(self, v_scale: float = 400.0, omega_scale: float = 10.0, 
                 default_throttle: float = 0.2):
        """
        Initialize the neural policy stub.
        
        Args:
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale  
            default_throttle: Default throttle value for safe actions
        """
        self.name = "NeuralStub"
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.default_throttle = default_throttle
        
        # Observation tracking
        self.last_obs: Optional[np.ndarray] = None
        self.obs_logged = False  # Log observation shape once
        
        # Expected observation dimensions
        self.num_rays = 128  # Will be updated from sim config
        self.vision_channels = 3  # close, food, wall
        self.proprioception_dim = 4  # v_forward, v_sideways, omega, throttle
        
    def reset(self) -> None:
        """Reset policy state"""
        self.last_obs = None
        self.obs_logged = False
    
    def _extract_vision_features(self, vision_distances: List[float], 
                                vision_hit_types: List[Optional[str]], 
                                max_range: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract vision features from raw vision data.
        
        Returns:
            Tuple of (vision_close, vision_food, vision_wall) arrays
        """
        num_rays = len(vision_distances)
        self.num_rays = num_rays  # Update for observation size calculation
        
        # Initialize arrays
        vision_close = np.zeros(num_rays, dtype=np.float32)
        vision_food = np.zeros(num_rays, dtype=np.float32)
        vision_wall = np.zeros(num_rays, dtype=np.float32)
        
        for i in range(num_rays):
            distance = vision_distances[i]
            hit_type = vision_hit_types[i]
            
            # Handle None distances (treat as max range)
            if distance is None:
                distance = max_range
            
            # Vision close: 1 - normalized distance (closer = higher value)
            vision_close[i] = 1.0 - np.clip(distance / max_range, 0.0, 1.0)
            
            # Vision food: binary indicator
            if hit_type == 'food':
                vision_food[i] = 1.0
            
            # Vision wall: binary indicator (handle 'wall' or 'wall_X' formats)
            if hit_type is not None and (hit_type == 'wall' or hit_type.startswith('wall')):
                vision_wall[i] = 1.0
        
        return vision_close, vision_food, vision_wall
    
    def _extract_proprioception(self, agent_state: Dict[str, float]) -> np.ndarray:
        """
        Extract proprioceptive features (agent's internal state in egocentric frame).
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Proprioception vector [v_forward_norm, v_sideways_norm, omega_norm, throttle]
        """
        # Extract state
        vx = agent_state['vx']
        vy = agent_state['vy']
        theta = agent_state['theta']
        omega = agent_state['omega']
        throttle = agent_state['throttle']
        
        # Compute agent's local coordinate frame
        forward_x = math.cos(theta)
        forward_y = math.sin(theta)
        right_x = -math.sin(theta)  # Perpendicular to forward (right side)
        right_y = math.cos(theta)
        
        # Project velocity onto agent's local frame
        v_forward = vx * forward_x + vy * forward_y
        v_sideways = vx * right_x + vy * right_y
        
        # Normalize velocities and angular velocity
        v_forward_norm = v_forward / self.v_scale
        v_sideways_norm = v_sideways / self.v_scale
        omega_norm = omega / self.omega_scale
        
        # Clamp to reasonable range to avoid outliers
        v_forward_norm = np.clip(v_forward_norm, -2.0, 2.0)
        v_sideways_norm = np.clip(v_sideways_norm, -2.0, 2.0)
        omega_norm = np.clip(omega_norm, -2.0, 2.0)
        
        # Throttle is already in [0, 1] range
        throttle_norm = np.clip(throttle, 0.0, 1.0)
        
        return np.array([v_forward_norm, v_sideways_norm, omega_norm, throttle_norm], 
                       dtype=np.float32)
    
    def _build_observation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """
        Build complete egocentric observation vector.
        
        Args:
            sim_state: Full simulation state
            
        Returns:
            Observation vector of shape (num_rays * 3 + 4,)
        """
        # Extract vision data
        vision_distances = sim_state['vision_distances']
        vision_hit_types = sim_state['vision_hit_types']
        agent_state = sim_state['agent_state']
        
        # Get max range from vision system (assume it's consistent)
        # Use a reasonable default if not available
        max_range = 300.0  # Default from typical config
        if len(vision_distances) > 0:
            # Infer max range from data (distances at max range should be consistent)
            max_distances = [d for d in vision_distances if d is not None]
            if max_distances:
                potential_max = max(max_distances)
                if potential_max >= 250:  # Reasonable threshold
                    max_range = potential_max
        
        # Extract vision features
        vision_close, vision_food, vision_wall = self._extract_vision_features(
            vision_distances, vision_hit_types, max_range
        )
        
        # Extract proprioception
        proprioception = self._extract_proprioception(agent_state)
        
        # Concatenate all features
        observation = np.concatenate([
            vision_close,    # 128 floats
            vision_food,     # 128 floats  
            vision_wall,     # 128 floats
            proprioception   # 4 floats
        ])
        
        # Verify observation integrity
        expected_length = self.num_rays * self.vision_channels + self.proprioception_dim
        assert len(observation) == expected_length, f"Observation length mismatch: {len(observation)} != {expected_length}"
        assert not np.any(np.isnan(observation)), "Observation contains NaN values"
        assert not np.any(np.isinf(observation)), "Observation contains infinite values"
        
        return observation
    
    def get_last_obs(self) -> Optional[np.ndarray]:
        """Get the last computed observation for debugging/inspection"""
        return self.last_obs
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract observation and return safe default action.
        
        Args:
            sim_state: Full simulation state
            
        Returns:
            Action dictionary with steer and throttle
        """
        # Build observation vector
        observation = self._build_observation(sim_state)
        self.last_obs = observation
        
        # Log observation info once for debugging
        if not self.obs_logged:
            print(f"NeuralPolicyStub: Built observation vector")
            print(f"  Shape: {observation.shape}")
            print(f"  Vision rays: {self.num_rays}")
            print(f"  Vision channels: {self.vision_channels} (close, food, wall)")
            print(f"  Proprioception: {self.proprioception_dim} (v_forward, v_sideways, omega, throttle)")
            print(f"  Total length: {len(observation)}")
            print(f"  Value range: [{observation.min():.3f}, {observation.max():.3f}]")
            
            # Show sample values
            vision_close_sample = observation[:5]
            proprioception_sample = observation[-4:]
            print(f"  Vision close sample (first 5): {vision_close_sample}")
            print(f"  Proprioception: {proprioception_sample}")
            
            self.obs_logged = True
        
        # Return safe default action
        # TODO: Replace with neural network inference:
        # net_out = self.model(observation)
        # steer = np.clip(net_out[0], -1.0, 1.0)
        # throttle = np.clip(sigmoid(net_out[1]), 0.0, 1.0)
        
        return {
            'steer': 0.0,  # No steering
            'throttle': self.default_throttle  # Gentle forward motion
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'default_throttle': self.default_throttle
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'NeuralPolicyStub':
        """Create policy from parameters"""
        return cls(
            v_scale=params.get('v_scale', 400.0),
            omega_scale=params.get('omega_scale', 10.0),
            default_throttle=params.get('default_throttle', 0.2)
        )