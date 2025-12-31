#!/usr/bin/env python3
"""
Sensor-only intrinsic curiosity module for RL training
Implements prediction error-based intrinsic motivation using only sensor data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import random


class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model f(o_t, a_t) -> o_{t+1}
    Predicts next observation from current observation and action
    """
    
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dims: List[int] = [256, 128]):
        """
        Initialize forward dynamics model
        
        Args:
            obs_dim: Observation vector dimension
            action_dim: Action dimension (2 for throttle, steering)
            hidden_dims: Hidden layer dimensions for MLP
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        input_dim = obs_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        # Output layer predicts next observation
        layers.append(nn.Linear(input_dim, obs_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict next observation
        
        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action taken [batch_size, action_dim]
            
        Returns:
            Predicted next observation [batch_size, obs_dim]
        """
        # Concatenate observation and action
        input_vec = torch.cat([obs, action], dim=-1)
        
        # Predict next observation
        pred_next_obs = self.network(input_vec)
        
        return pred_next_obs


class ReplayBuffer:
    """
    Simple replay buffer for storing (obs, action, next_obs) transitions
    """
    
    def __init__(self, capacity: int = 50000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        """
        Add a transition to the buffer
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
        """
        self.buffer.append((obs.copy(), action.copy(), next_obs.copy()))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (obs_batch, action_batch, next_obs_batch)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        obs_batch = np.stack([item[0] for item in batch])
        action_batch = np.stack([item[1] for item in batch])
        next_obs_batch = np.stack([item[2] for item in batch])
        
        return (
            torch.tensor(obs_batch, dtype=torch.float32),
            torch.tensor(action_batch, dtype=torch.float32),
            torch.tensor(next_obs_batch, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


def build_obs_vector(step_info: Dict[str, Any], v_scale: float = 400.0, 
                    omega_scale: float = 10.0, max_vision_range: float = 300.0) -> np.ndarray:
    """
    Build fixed-length observation vector from sensor data only
    
    Args:
        step_info: Step info from simulation
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        max_vision_range: Maximum vision range for normalization
        
    Returns:
        Fixed-length observation vector
    """
    # Extract vision data
    vision_distances = step_info['vision_distances']
    vision_hit_types = step_info['vision_hit_types']
    agent_state = step_info['agent_state']
    
    num_rays = len(vision_distances)
    
    # Normalize distances (replace None with max range, scale to [0,1])
    normalized_distances = np.zeros(num_rays, dtype=np.float32)
    for i, distance in enumerate(vision_distances):
        if distance is None:
            normalized_distances[i] = 1.0  # Max range
        else:
            normalized_distances[i] = np.clip(distance / max_vision_range, 0.0, 1.0)
    
    # Encode hit types (one-hot per ray: [none, food, wall])
    hit_type_encoding = np.zeros((num_rays, 3), dtype=np.float32)
    for i, hit_type in enumerate(vision_hit_types):
        if hit_type is None:
            hit_type_encoding[i, 0] = 1.0  # None
        elif hit_type == 'food':
            hit_type_encoding[i, 1] = 1.0  # Food
        else:  # wall or other
            hit_type_encoding[i, 2] = 1.0  # Wall
    
    # Flatten hit type encoding
    hit_type_flat = hit_type_encoding.flatten()  # num_rays * 3
    
    # Extract proprioceptive features (normalized)
    vx = agent_state['vx']
    vy = agent_state['vy']
    theta = agent_state['theta']
    omega = agent_state['omega']
    throttle = agent_state['throttle']
    
    # Compute agent's local coordinate frame velocities
    import math
    forward_x = math.cos(theta)
    forward_y = math.sin(theta)
    right_x = -math.sin(theta)
    right_y = math.cos(theta)
    
    v_forward = vx * forward_x + vy * forward_y
    v_sideways = vx * right_x + vy * right_y
    
    # Normalize and clip
    v_forward_norm = np.clip(v_forward / v_scale, -2.0, 2.0)
    v_sideways_norm = np.clip(v_sideways / v_scale, -2.0, 2.0)
    omega_norm = np.clip(omega / omega_scale, -2.0, 2.0)
    throttle_norm = np.clip(throttle, 0.0, 1.0)
    
    proprioception = np.array([v_forward_norm, v_sideways_norm, omega_norm, throttle_norm], 
                             dtype=np.float32)
    
    # Concatenate all features
    obs_vector = np.concatenate([
        normalized_distances,  # num_rays floats
        hit_type_flat,        # num_rays * 3 floats
        proprioception        # 4 floats
    ])
    
    return obs_vector


def encode_action(action: Dict[str, float]) -> np.ndarray:
    """
    Encode action as continuous vector
    
    Args:
        action: Action dict with 'throttle' and 'steer' keys
        
    Returns:
        Action vector [throttle, steer]
    """
    return np.array([action['throttle'], action['steer']], dtype=np.float32)


class IntrinsicCuriosityModule:
    """
    Intrinsic curiosity module using sensor-only prediction error
    """
    
    def __init__(self, obs_dim: int, device: torch.device = torch.device('cpu'),
                 eta: float = 0.05, r_max: float = 2.0, train_every: int = 1,
                 batch_size: int = 128, lr: float = 1e-3, buffer_capacity: int = 50000,
                 warmup_steps: int = 1000, max_vision_range: float = 300.0):
        """
        Initialize intrinsic curiosity module
        
        Args:
            obs_dim: Observation vector dimension
            device: PyTorch device
            eta: Intrinsic reward scaling factor
            r_max: Maximum intrinsic reward (clipping)
            train_every: Train forward model every N steps
            batch_size: Training batch size
            lr: Learning rate for forward model
            buffer_capacity: Replay buffer capacity
            warmup_steps: Steps before starting training
            max_vision_range: Maximum vision range for normalization
        """
        self.obs_dim = obs_dim
        self.device = device
        self.eta = eta
        self.r_max = r_max
        self.train_every = train_every
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_vision_range = max_vision_range
        
        # Forward dynamics model
        self.forward_model = ForwardDynamicsModel(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training state
        self.step_count = 0
        self.last_forward_loss = 0.0
        
        # Statistics tracking
        self.intrinsic_rewards = deque(maxlen=1000)
        self.forward_losses = deque(maxlen=1000)
        self.curiosity_active_steps = 0
        self.total_steps = 0
    
    def compute_intrinsic_reward(self, step_info: Dict[str, Any], action: Dict[str, float],
                                next_step_info: Dict[str, Any], food_visible: bool,
                                steps_since_food_seen: int, v_scale: float = 400.0,
                                omega_scale: float = 10.0) -> Tuple[float, Dict[str, float]]:
        """
        Compute intrinsic reward based on prediction error
        
        Args:
            step_info: Current step info
            action: Action taken
            next_step_info: Next step info
            food_visible: Whether food is currently visible
            steps_since_food_seen: Steps since food was last seen
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            
        Returns:
            Tuple of (intrinsic_reward, debug_info)
        """
        # Build observation vectors
        obs_vec = build_obs_vector(step_info, v_scale, omega_scale, self.max_vision_range)
        next_obs_vec = build_obs_vector(next_step_info, v_scale, omega_scale, self.max_vision_range)
        action_vec = encode_action(action)
        
        # Add to replay buffer
        self.replay_buffer.add(obs_vec, action_vec, next_obs_vec)
        
        # Gate curiosity: no intrinsic reward if food is visible
        # This prevents the agent from getting curious about food interactions
        if food_visible or steps_since_food_seen == 0:
            intrinsic_reward = 0.0
            curiosity_active = False
        else:
            # Compute prediction error
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_obs_tensor = torch.tensor(next_obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Predict next observation
                pred_next_obs = self.forward_model(obs_tensor, action_tensor)
                
                # Compute MSE prediction error
                mse_error = F.mse_loss(pred_next_obs, next_obs_tensor, reduction='mean').item()
                
                # Scale and clip intrinsic reward
                intrinsic_reward = self.eta * np.clip(mse_error, 0.0, self.r_max)
                curiosity_active = True
        
        # Update statistics
        self.intrinsic_rewards.append(intrinsic_reward)
        self.total_steps += 1
        if curiosity_active:
            self.curiosity_active_steps += 1
        
        # Train forward model periodically
        if self.step_count % self.train_every == 0 and len(self.replay_buffer) > self.warmup_steps:
            self._train_forward_model()
        
        self.step_count += 1
        
        debug_info = {
            'prediction_error': mse_error if not food_visible else 0.0,
            'curiosity_active': curiosity_active,
            'buffer_size': len(self.replay_buffer),
            'forward_loss': self.last_forward_loss
        }
        
        return intrinsic_reward, debug_info
    
    def _train_forward_model(self):
        """Train the forward dynamics model on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        obs_batch, action_batch, next_obs_batch = self.replay_buffer.sample(self.batch_size)
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        next_obs_batch = next_obs_batch.to(self.device)
        
        # Forward pass
        pred_next_obs = self.forward_model(obs_batch, action_batch)
        
        # Compute loss
        loss = F.mse_loss(pred_next_obs, next_obs_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss
        self.last_forward_loss = loss.item()
        self.forward_losses.append(self.last_forward_loss)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get curiosity module statistics"""
        stats = {
            'avg_intrinsic_reward': np.mean(self.intrinsic_rewards) if self.intrinsic_rewards else 0.0,
            'avg_forward_loss': np.mean(self.forward_losses) if self.forward_losses else 0.0,
            'curiosity_active_fraction': self.curiosity_active_steps / max(self.total_steps, 1),
            'buffer_size': len(self.replay_buffer),
            'step_count': self.step_count
        }
        return stats
    
    def reset_episode_stats(self):
        """Reset per-episode statistics"""
        self.curiosity_active_steps = 0
        self.total_steps = 0