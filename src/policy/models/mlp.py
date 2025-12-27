#!/usr/bin/env python3
"""
Simple MLP model for continuous 2D simulation
Maps 388-dim observation to (steer, throttle) actions
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for action prediction
    
    Input: 388-dim observation vector (vision + proprioception)
    Output: (steer, throttle) actions
    """
    
    def __init__(self, input_dim: int = 388, hidden_dims: Tuple[int, ...] = (256, 128), 
                 output_dim: int = 2, init_seed: int = None):
        """
        Initialize MLP model
        
        Args:
            input_dim: Input observation dimension (default 388)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (2 for steer, throttle)
            init_seed: Random seed for weight initialization
        """
        super().__init__()
        
        if init_seed is not None:
            torch.manual_seed(init_seed)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - applied in forward)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with PyTorch defaults (already done by default)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (steer, throttle) tensors
            - steer: [-1, 1] range via tanh
            - throttle: [0, 1] range via sigmoid
        """
        # Forward through network
        y = self.network(x)  # [batch_size, 2]
        
        # Split outputs
        steer_raw = y[:, 0]  # [batch_size]
        throttle_raw = y[:, 1]  # [batch_size]
        
        # Apply activations for proper ranges
        steer = torch.tanh(steer_raw)  # [-1, 1]
        throttle = torch.sigmoid(throttle_raw)  # [0, 1]
        
        return steer, throttle
    
    def get_config(self) -> dict:
        """Get model configuration for serialization"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim
        }