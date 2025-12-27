#!/usr/bin/env python3
"""
Batched observation building
Converts batched simulation state to neural network observation format
"""

import numpy as np
from typing import Tuple
from .scene import MATERIAL_WALL, MATERIAL_FOOD


def build_obs_batch(agent_states: dict, distances: np.ndarray, materials: np.ndarray,
                   v_scale: float = 400.0, omega_scale: float = 10.0) -> np.ndarray:
    """
    Build batched egocentric observation vectors
    
    Args:
        agent_states: Dict with agent state arrays (x, y, vx, vy, theta, omega, throttle)
        distances: Ray distances [B, R]
        materials: Ray material IDs [B, R]
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        
    Returns:
        Observation batch [B, 388] where 388 = 128*3 + 4
    """
    B, R = distances.shape
    max_range = 300.0  # Should match config.max_range
    
    # Extract vision features
    vision_close, vision_food, vision_wall = extract_vision_features_batch(
        distances, materials, max_range
    )
    
    # Extract proprioception
    proprioception = extract_proprioception_batch(
        agent_states, v_scale, omega_scale
    )
    
    # Concatenate all features
    observation = np.concatenate([
        vision_close,    # [B, R]
        vision_food,     # [B, R]  
        vision_wall,     # [B, R]
        proprioception   # [B, 4]
    ], axis=1)
    
    # Verify observation shape
    expected_dim = R * 3 + 4
    assert observation.shape == (B, expected_dim), f"Expected shape ({B}, {expected_dim}), got {observation.shape}"
    
    return observation


def extract_vision_features_batch(distances: np.ndarray, materials: np.ndarray, 
                                 max_range: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract batched vision features from ray casting results
    
    Args:
        distances: Ray distances [B, R]
        materials: Ray material IDs [B, R]
        max_range: Maximum ray range
        
    Returns:
        Tuple of (vision_close, vision_food, vision_wall) each [B, R]
    """
    B, R = distances.shape
    
    # Vision close: 1 - normalized distance (closer = higher value)
    vision_close = 1.0 - np.clip(distances / max_range, 0.0, 1.0)
    
    # Vision food: binary indicator for food hits
    vision_food = (materials == MATERIAL_FOOD).astype(np.float32)
    
    # Vision wall: binary indicator for wall hits
    vision_wall = (materials == MATERIAL_WALL).astype(np.float32)
    
    return vision_close, vision_food, vision_wall


def extract_proprioception_batch(agent_states: dict, v_scale: float = 400.0, 
                                omega_scale: float = 10.0) -> np.ndarray:
    """
    Extract batched proprioceptive features (agent's internal state in egocentric frame)
    
    Args:
        agent_states: Dict with agent state arrays
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        
    Returns:
        Proprioception vectors [B, 4] with [v_forward_norm, v_sideways_norm, omega_norm, throttle]
    """
    # Extract state arrays
    vx = agent_states['vx']
    vy = agent_states['vy']
    theta = agent_states['theta']
    omega = agent_states['omega']
    throttle = agent_states['throttle']
    
    B = len(vx)
    
    # Compute agent's local coordinate frame
    forward_x = np.cos(theta)
    forward_y = np.sin(theta)
    right_x = -np.sin(theta)  # Perpendicular to forward (right side)
    right_y = np.cos(theta)
    
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
    
    # Stack into proprioception vectors
    proprioception = np.stack([
        v_forward_norm,
        v_sideways_norm, 
        omega_norm,
        throttle_norm
    ], axis=1)
    
    return proprioception