#!/usr/bin/env python3
"""
Shared observation building utilities
Extracted from neural network policies for reuse in training
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def extract_vision_features(vision_distances: List[float], vision_hit_types: List[Optional[str]], 
                           max_range: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract vision features from raw vision data
    
    Returns:
        Tuple of (vision_close, vision_food, vision_wall) arrays
    """
    num_rays = len(vision_distances)
    
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
        
        # Vision wall: binary indicator
        if hit_type is not None and (hit_type == 'wall' or hit_type.startswith('wall')):
            vision_wall[i] = 1.0
    
    return vision_close, vision_food, vision_wall


def extract_proprioception(agent_state: Dict[str, float], v_scale: float = 400.0, 
                          omega_scale: float = 10.0) -> np.ndarray:
    """
    Extract proprioceptive features (agent's internal state in egocentric frame)
    
    Args:
        agent_state: Agent state dictionary
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        
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
    v_forward_norm = v_forward / v_scale
    v_sideways_norm = v_sideways / v_scale
    omega_norm = omega / omega_scale
    
    # Clamp to reasonable range to avoid outliers
    v_forward_norm = np.clip(v_forward_norm, -2.0, 2.0)
    v_sideways_norm = np.clip(v_sideways_norm, -2.0, 2.0)
    omega_norm = np.clip(omega_norm, -2.0, 2.0)
    
    # Throttle is already in [0, 1] range
    throttle_norm = np.clip(throttle, 0.0, 1.0)
    
    return np.array([v_forward_norm, v_sideways_norm, omega_norm, throttle_norm], 
                   dtype=np.float32)


def build_observation(sim_state: Dict[str, Any], v_scale: float = 400.0, 
                     omega_scale: float = 10.0) -> np.ndarray:
    """
    Build complete egocentric observation vector
    
    Args:
        sim_state: Full simulation state
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        
    Returns:
        Observation vector of shape (num_rays * 3 + 4,)
    """
    # Extract vision data
    vision_distances = sim_state['vision_distances']
    vision_hit_types = sim_state['vision_hit_types']
    agent_state = sim_state['agent_state']
    
    # Get max range from vision system
    max_range = 300.0  # Default
    if len(vision_distances) > 0:
        max_distances = [d for d in vision_distances if d is not None]
        if max_distances:
            potential_max = max(max_distances)
            if potential_max >= 250:
                max_range = potential_max
    
    # Extract features
    vision_close, vision_food, vision_wall = extract_vision_features(
        vision_distances, vision_hit_types, max_range
    )
    proprioception = extract_proprioception(agent_state, v_scale, omega_scale)
    
    # Concatenate all features
    observation = np.concatenate([
        vision_close,    # 128 floats
        vision_food,     # 128 floats  
        vision_wall,     # 128 floats
        proprioception   # 4 floats
    ])
    
    # Verify observation integrity
    expected_length = len(vision_distances) * 3 + 4
    assert len(observation) == expected_length, f"Observation length mismatch: {len(observation)} != {expected_length}"
    assert not np.any(np.isnan(observation)), "Observation contains NaN values"
    assert not np.any(np.isinf(observation)), "Observation contains infinite values"
    
    return observation