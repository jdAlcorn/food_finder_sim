#!/usr/bin/env python3
"""
RL-based MLP policy for continuous action control
Implements Actor-Critic with feedforward MLP (no recurrence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base import Policy
from .obs import build_observation

MIN_LOG_STD = -2.5

class RLMLPNetwork(nn.Module):
    """
    Actor-Critic network with MLP for RL training (no recurrence)
    
    Architecture:
    - Encoder MLP: obs -> encoded features
    - Actor head: encoded features -> action mean (2D continuous)
    - Critic head: encoded features -> value estimate
    """
    
    def __init__(self, input_dim: int = 388, encoder_dims: Tuple[int, ...] = (256, 128),
                 init_seed: int = None):
        """
        Initialize RL MLP network
        
        Args:
            input_dim: Input observation dimension (default 388)
            encoder_dims: MLP encoder hidden layer dimensions
            init_seed: Random seed for weight initialization
        """
        super().__init__()
        
        if init_seed is not None:
            torch.manual_seed(init_seed)
        
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        
        # MLP encoder to process observations
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_output_dim = prev_dim
        
        # Actor head (outputs action mean)
        self.actor_head = nn.Linear(self.encoder_output_dim, 2)  # [throttle, steering]
        
        # Critic head (outputs value estimate)
        self.critic_head = nn.Linear(self.encoder_output_dim, 1)
        
        # Learnable log_std for Gaussian policy (global parameter)
        self.log_std = nn.Parameter(torch.zeros(2))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (action_mean, log_std, value)
        """
        batch_size = obs.shape[0]
        
        # Encode observations
        encoded = self.encoder(obs)  # [batch_size, encoder_output_dim]
        
        # Actor and critic heads - output raw logits for squashing
        action_logits = self.actor_head(encoded)  # [batch_size, 2]
        value = self.critic_head(encoded).squeeze(-1)  # [batch_size]
        
        # Expand log_std to match batch size
        log_std = self.log_std.expand(batch_size, -1)  # [batch_size, 2]
        log_std = torch.clamp(log_std, min=MIN_LOG_STD)
        
        return action_logits, log_std, value
    
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy using squashed Gaussian
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, log_std, value = self.forward(obs)
        
        if deterministic:
            # Use mean of pre-squash distribution, then squash
            u_mean = action_logits
            u = u_mean
        else:
            # Sample from pre-squash Gaussian
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_logits, std)
            u = dist.sample()
        
        # Squash with tanh
        action_tanh = torch.tanh(u)
        
        # Map to action bounds:
        # throttle: [-1,1] -> [0,1] via (tanh + 1) / 2
        # steering: [-1,1] -> [-1,1] (already correct)
        action = torch.stack([
            (action_tanh[:, 0] + 1.0) / 2.0,  # throttle: [0, 1]
            action_tanh[:, 1]                 # steering: [-1, 1]
        ], dim=1)
        
        if deterministic:
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            # Compute log_prob with tanh correction
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_logits, std)
            log_prob_u = dist.log_prob(u).sum(dim=-1)  # Sum over action dimensions
            
            # Tanh correction: log_prob(a) = log_prob(u) - sum(log(1 - tanh(u)^2 + eps))
            tanh_correction = torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
            log_prob = log_prob_u - tanh_correction
        
        return action, log_prob, value


class RLMLPPolicy(Policy):
    """
    RL-based MLP policy that implements the Policy interface
    Uses Actor-Critic with MLP for continuous action control (no recurrence)
    """
    
    def __init__(self, encoder_dims: Tuple[int, ...] = (256, 128), device: str = "cpu", 
                 v_scale: float = 400.0, omega_scale: float = 10.0, init_seed: int = None):
        """
        Initialize RL MLP policy
        
        Args:
            encoder_dims: MLP encoder hidden layer dimensions
            device: Device to run model on ("cpu" or "cuda")
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            init_seed: Random seed for deterministic initialization
        """
        self.name = "RLMLP"
        self.encoder_dims = encoder_dims
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.init_seed = init_seed
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Create network
        self.network = RLMLPNetwork(
            input_dim=388,
            encoder_dims=encoder_dims,
            init_seed=init_seed
        ).to(self.device)
        
        # Episode step counter (for compatibility)
        self._episode_step = 0
    
    def reset(self) -> None:
        """Reset policy state for new episode"""
        self._episode_step = 0
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action from current simulation state (Policy interface)
        
        Args:
            sim_state: Dictionary containing simulation state
            
        Returns:
            Dict with 'steer' and 'throttle' keys
        """
        # Build observation from sim_state
        observation = build_observation(sim_state, self.v_scale, self.omega_scale)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action (deterministic for evaluation)
        with torch.no_grad():
            action, _, _ = self.network.act(obs_tensor, deterministic=True)
        
        self._episode_step += 1
        
        # Convert to numpy and extract single values
        action_np = action.cpu().numpy()[0]
        throttle_val = float(action_np[0])
        steer_val = float(action_np[1])
        
        # Safety clamps
        throttle_val = np.clip(throttle_val, 0.0, 1.0)
        steer_val = np.clip(steer_val, -1.0, 1.0)
        
        return {'throttle': throttle_val, 'steer': steer_val}
    
    def act_training(self, sim_state: Dict[str, Any], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with training data (log_prob, value) from simulation state
        
        Args:
            sim_state: Dictionary containing simulation state
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action_tensor, log_prob, value)
        """
        # Build observation from sim_state
        observation = build_observation(sim_state, self.v_scale, self.omega_scale)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action with training data
        action, log_prob, value = self.network.act(obs_tensor, deterministic=deterministic)
        
        self._episode_step += 1
        
        return action, log_prob, value
    
    def act_batch(self, observations: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch action prediction for training
        
        Args:
            observations: Observation batch [batch_size, 388]
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        return self.network.act(observations, deterministic)
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Get value estimates for observations"""
        _, _, values = self.network.forward(observations)
        return values
    
    def save_weights(self, path: str):
        """Save model weights to file"""
        torch.save(self.network.state_dict(), path)
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'encoder_dims': list(self.encoder_dims),
            'device': str(self.device),
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'RLMLPPolicy':
        """Create policy from parameters dict"""
        return cls(
            encoder_dims=tuple(params['encoder_dims']),
            device=params['device'],
            v_scale=params['v_scale'],
            omega_scale=params['omega_scale'],
            init_seed=params['init_seed']
        )