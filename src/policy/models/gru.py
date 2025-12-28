#!/usr/bin/env python3
"""
GRU-based recurrent policy for Evolution Strategies training
Implements memory-enabled policy that maintains hidden state across timesteps
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RecurrentGRUPolicy(nn.Module):
    """
    GRU-based recurrent policy for action prediction with memory
    
    Architecture:
    - MLP trunk: obs -> encoded features
    - GRU layer: encoded features + hidden -> gru_output, new_hidden
    - Output head: gru_output -> (steer, throttle)
    
    Input: 388-dim observation vector (vision + proprioception)
    Output: (steer, throttle) actions + hidden state
    """
    
    def __init__(self, input_dim: int = 388, trunk_dims: Tuple[int, ...] = (256, 128),
                 hidden_size: int = 64, num_layers: int = 1, output_dim: int = 2,
                 init_seed: int = None):
        """
        Initialize GRU-based recurrent policy
        
        Args:
            input_dim: Input observation dimension (default 388)
            trunk_dims: MLP trunk hidden layer dimensions
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            output_dim: Output dimension (2 for steer, throttle)
            init_seed: Random seed for weight initialization
        """
        super().__init__()
        
        if init_seed is not None:
            torch.manual_seed(init_seed)
        
        self.input_dim = input_dim
        self.trunk_dims = trunk_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # MLP trunk to encode observations
        trunk_layers = []
        prev_dim = input_dim
        
        for hidden_dim in trunk_dims:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_output_dim = prev_dim
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.trunk_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output head
        self.output_head = nn.Linear(hidden_size, output_dim)
        
        # Initialize weights with PyTorch defaults (already done by default)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional hidden state
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            hidden: Optional hidden state [num_layers, batch_size, hidden_size]
            
        Returns:
            Tuple of (steer, throttle, new_hidden)
            - steer: [-1, 1] range via tanh
            - throttle: [0, 1] range via sigmoid
            - new_hidden: Updated hidden state [num_layers, batch_size, hidden_size]
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Encode observations through MLP trunk
        encoded = self.trunk(x)  # [batch_size, trunk_output_dim]
        
        # Add sequence dimension for GRU (single timestep)
        encoded = encoded.unsqueeze(1)  # [batch_size, 1, trunk_output_dim]
        
        # GRU forward pass
        gru_output, new_hidden = self.gru(encoded, hidden)  # output: [batch_size, 1, hidden_size]
        
        # Remove sequence dimension
        gru_output = gru_output.squeeze(1)  # [batch_size, hidden_size]
        
        # Output head
        y = self.output_head(gru_output)  # [batch_size, 2]
        
        # Split outputs and apply activations
        steer_raw = y[:, 0]  # [batch_size]
        throttle_raw = y[:, 1]  # [batch_size]
        
        steer = torch.tanh(steer_raw)  # [-1, 1]
        throttle = torch.sigmoid(throttle_raw)  # [0, 1]
        
        return steer, throttle, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Initialize hidden state for GRU
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Zero-initialized hidden state [num_layers, batch_size, hidden_size]
        """
        if device is None:
            device = next(self.parameters()).device
        
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                          dtype=torch.float32, device=device)
    
    def step(self, obs: torch.Tensor, hidden: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single step forward pass for rollout evaluation
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            hidden: Hidden state [num_layers, batch_size, hidden_size]
            
        Returns:
            Tuple of ((steer, throttle), new_hidden)
        """
        with torch.no_grad():
            steer, throttle, new_hidden = self.forward(obs, hidden)
            # Detach hidden state to prevent gradient accumulation
            new_hidden = new_hidden.detach()
        
        return (steer, throttle), new_hidden
    
    def get_config(self) -> dict:
        """Get model configuration for serialization"""
        return {
            'model_type': 'RecurrentGRUPolicy',
            'input_dim': self.input_dim,
            'trunk_dims': self.trunk_dims,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim
        }


class RecurrentPolicyWrapper:
    """
    Wrapper for easy integration of recurrent policy into existing ES evaluation loops
    Provides stateful interface that manages hidden state automatically
    """
    
    def __init__(self, model: RecurrentGRUPolicy, device: str = "cpu"):
        """
        Initialize wrapper
        
        Args:
            model: RecurrentGRUPolicy instance
            device: Device for computations
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Hidden state management
        self._hidden_state = None
        self._batch_size = None
    
    def reset_hidden(self, batch_size: int = 1):
        """
        Reset hidden state for new episode(s)
        
        Args:
            batch_size: Number of parallel environments
        """
        self._batch_size = batch_size
        self._hidden_state = self.model.init_hidden(batch_size, self.device)
    
    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step with automatic hidden state management
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (steer, throttle)
        """
        batch_size = obs.shape[0]
        
        # Initialize hidden state if needed
        if self._hidden_state is None or self._batch_size != batch_size:
            self.reset_hidden(batch_size)
        
        # Forward pass
        (steer, throttle), self._hidden_state = self.model.step(obs, self._hidden_state)
        
        return steer, throttle
    
    def __call__(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compatibility interface for existing code
        
        Args:
            obs: Observation tensor [batch_size, input_dim]
            hidden: Optional hidden state (if None, uses internal state)
            
        Returns:
            Tuple of (steer, throttle)
        """
        if hidden is not None:
            # Stateless mode - use provided hidden state
            steer, throttle, _ = self.model.forward(obs, hidden)
            return steer, throttle
        else:
            # Stateful mode - use internal hidden state management
            return self.step(obs)