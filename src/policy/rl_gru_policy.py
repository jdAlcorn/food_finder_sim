#!/usr/bin/env python3
"""
RL-based GRU policy for continuous action control
Implements Actor-Critic with GRU for memory and partial observability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base import Policy
from .obs import build_observation


class RLGRUNetwork(nn.Module):
    """
    Actor-Critic network with GRU for RL training
    
    Architecture:
    - Encoder MLP: obs -> encoded features
    - GRU layer: encoded features + hidden -> gru_output, new_hidden
    - Actor head: gru_output -> action mean (2D continuous)
    - Critic head: gru_output -> value estimate
    """
    
    def __init__(self, input_dim: int = 388, encoder_dims: Tuple[int, ...] = (256, 128),
                 hidden_size: int = 64, num_layers: int = 1, init_seed: int = None):
        """
        Initialize RL GRU network
        
        Args:
            input_dim: Input observation dimension (default 388)
            encoder_dims: MLP encoder hidden layer dimensions
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            init_seed: Random seed for weight initialization
        """
        super().__init__()
        
        if init_seed is not None:
            torch.manual_seed(init_seed)
        
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
        
        # GRU layer for memory
        self.gru = nn.GRU(
            input_size=self.encoder_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Actor head (outputs action mean)
        self.actor_head = nn.Linear(hidden_size, 2)  # [throttle, steering]
        
        # Critic head (outputs value estimate)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Learnable log_std for Gaussian policy (global parameter)
        self.log_std = nn.Parameter(torch.zeros(2))
    
    def forward(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            hidden: Optional hidden state [num_layers, batch_size, hidden_size]
            
        Returns:
            Tuple of (action_mean, log_std, value, new_hidden)
        """
        batch_size = obs.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        # Encode observations
        encoded = self.encoder(obs)  # [batch_size, encoder_output_dim]
        
        # Add sequence dimension for GRU (single timestep)
        encoded = encoded.unsqueeze(1)  # [batch_size, 1, encoder_output_dim]
        
        # GRU forward pass
        gru_output, new_hidden = self.gru(encoded, hidden)  # output: [batch_size, 1, hidden_size]
        
        # Remove sequence dimension
        gru_output = gru_output.squeeze(1)  # [batch_size, hidden_size]
        
        # Actor and critic heads
        action_mean = self.actor_head(gru_output)  # [batch_size, 2]
        value = self.critic_head(gru_output).squeeze(-1)  # [batch_size]
        
        # Apply action constraints
        # throttle: [0, 1] via sigmoid
        # steering: [-1, 1] via tanh
        action_mean = torch.stack([
            torch.sigmoid(action_mean[:, 0]),  # throttle
            torch.tanh(action_mean[:, 1])      # steering
        ], dim=1)
        
        # Expand log_std to match batch size
        log_std = self.log_std.expand(batch_size, -1)  # [batch_size, 2]
        
        return action_mean, log_std, value, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Initialize hidden state for GRU"""
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                          dtype=torch.float32, device=device)
    
    def act(self, obs: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            hidden: Hidden state [num_layers, batch_size, hidden_size]
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            Tuple of (action, log_prob, value, new_hidden)
        """
        action_mean, log_std, value, new_hidden = self.forward(obs, hidden)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            # Sample from Gaussian distribution
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
            
            # Clamp actions to valid ranges
            action = torch.stack([
                torch.clamp(action[:, 0], 0.0, 1.0),  # throttle
                torch.clamp(action[:, 1], -1.0, 1.0)  # steering
            ], dim=1)
        
        return action, log_prob, value, new_hidden


class RLGRUPolicy(Policy):
    """
    RL-based GRU policy that implements the Policy interface
    Uses Actor-Critic with GRU for memory and continuous action control
    """
    
    def __init__(self, encoder_dims: Tuple[int, ...] = (256, 128), hidden_size: int = 64,
                 num_layers: int = 1, device: str = "cpu", v_scale: float = 400.0, 
                 omega_scale: float = 10.0, init_seed: int = None):
        """
        Initialize RL GRU policy
        
        Args:
            encoder_dims: MLP encoder hidden layer dimensions
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            device: Device to run model on ("cpu" or "cuda")
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            init_seed: Random seed for deterministic initialization
        """
        self.name = "RLGRU"
        self.encoder_dims = encoder_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.init_seed = init_seed
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Create network
        self.network = RLGRUNetwork(
            input_dim=388,
            encoder_dims=encoder_dims,
            hidden_size=hidden_size,
            num_layers=num_layers,
            init_seed=init_seed
        ).to(self.device)
        
        # Hidden state for single environment usage
        self._hidden_state = None
        self._episode_step = 0
    
    def reset(self) -> None:
        """Reset policy state for new episode"""
        self._hidden_state = self.network.init_hidden(1, self.device)
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
        
        # Initialize hidden state if needed
        if self._hidden_state is None:
            self.reset()
        
        # Get action (deterministic for evaluation)
        with torch.no_grad():
            action, _, _, new_hidden = self.network.act(obs_tensor, self._hidden_state, deterministic=True)
        
        # Update hidden state
        self._hidden_state = new_hidden.detach()
        self._episode_step += 1
        
        # Convert to numpy and extract single values
        action_np = action.cpu().numpy()[0]
        throttle_val = float(action_np[0])
        steer_val = float(action_np[1])
        
        # Safety clamps
        throttle_val = np.clip(throttle_val, 0.0, 1.0)
        steer_val = np.clip(steer_val, -1.0, 1.0)
        
        return {'throttle': throttle_val, 'steer': steer_val}
    
    def act_batch(self, observations: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch action prediction for training
        
        Args:
            observations: Observation batch [batch_size, 388]
            hidden: Hidden state [num_layers, batch_size, hidden_size]
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, values, new_hidden)
        """
        return self.network.act(observations, hidden, deterministic)
    
    def get_value(self, observations: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Get value estimates for observations"""
        _, _, values, _ = self.network.forward(observations, hidden)
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
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'device': str(self.device),
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'RLGRUPolicy':
        """Create policy from parameters dict"""
        return cls(
            encoder_dims=tuple(params['encoder_dims']),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            device=params['device'],
            v_scale=params['v_scale'],
            omega_scale=params['omega_scale'],
            init_seed=params['init_seed']
        )