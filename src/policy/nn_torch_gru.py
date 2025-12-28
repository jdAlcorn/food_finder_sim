#!/usr/bin/env python3
"""
PyTorch GRU-based recurrent policy implementation
Integrates with existing Policy interface while adding memory capabilities
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base import Policy
from .models.gru import RecurrentGRUPolicy, RecurrentPolicyWrapper


class TorchGRUPolicy(Policy):
    """
    PyTorch GRU-based recurrent policy that uses neural network with memory for action prediction.
    Compatible with existing TorchMLPPolicy interface while adding recurrent capabilities.
    """
    
    def __init__(self, trunk_dims: Tuple[int, ...] = (256, 128), hidden_size: int = 64,
                 num_layers: int = 1, device: str = "cpu", v_scale: float = 400.0, 
                 omega_scale: float = 10.0, init_seed: int = None):
        """
        Initialize PyTorch GRU recurrent policy
        
        Args:
            trunk_dims: MLP trunk hidden layer dimensions
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            device: Device to run model on ("cpu" or "cuda")
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            init_seed: Random seed for deterministic initialization
        """
        self.name = "TorchGRU"
        self.trunk_dims = trunk_dims
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
        
        # Create model
        self.model = RecurrentGRUPolicy(
            input_dim=388,
            trunk_dims=trunk_dims,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=2,
            init_seed=init_seed
        )
        
        # Create wrapper for easy state management
        self.wrapper = RecurrentPolicyWrapper(self.model, str(self.device))
        
        # Observation tracking (for compatibility)
        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[Dict[str, float]] = None
        self.obs_logged = False
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action from current simulation state (Policy interface)
        
        Args:
            sim_state: Dictionary containing simulation state
            
        Returns:
            Dict with 'steer' and 'throttle' keys
        """
        # Extract observation from sim_state (similar to other policies)
        try:
            from src.policy.obs import build_observation
        except ImportError:
            from ..policy.obs import build_observation
        observation = build_observation(sim_state, self.v_scale, self.omega_scale)
        
        # Use predict method
        return self.predict(observation)
    
    def reset(self, batch_size: int = None) -> None:
        """
        Reset hidden state for new episode(s)
        Compatible with base Policy.reset() interface
        
        Args:
            batch_size: Number of parallel environments (default: 1 for single env)
        """
        if batch_size is None:
            batch_size = 1  # Default for single environment (viewer, etc.)
        
        self.wrapper.reset_hidden(batch_size)
        self.last_obs = None
        self.last_action = None
        self.obs_logged = False
    
    def predict(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Predict action from observation (single environment interface)
        
        Args:
            observation: Observation array of shape [388]
            
        Returns:
            Action dictionary with 'steer' and 'throttle' keys
        """
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action from wrapper (manages hidden state automatically)
        with torch.no_grad():
            steer, throttle = self.wrapper.step(obs_tensor)
        
        # Convert to numpy and extract single values
        steer_val = float(steer.cpu().numpy()[0])
        throttle_val = float(throttle.cpu().numpy()[0])
        
        # Safety clamps
        steer_val = np.clip(steer_val, -1.0, 1.0)
        throttle_val = np.clip(throttle_val, 0.0, 1.0)
        
        action = {'steer': steer_val, 'throttle': throttle_val}
        
        # Update tracking
        self.last_obs = observation.copy()
        self.last_action = action.copy()
        
        return action
    
    def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict actions for batch of observations (batched interface)
        
        Args:
            observations: Observation batch of shape [batch_size, 388]
            
        Returns:
            Tuple of (steer_array, throttle_array) each of shape [batch_size]
        """
        # Convert to tensor
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        
        # Get actions from wrapper
        with torch.no_grad():
            steer, throttle = self.wrapper.step(obs_tensor)
        
        # Convert to numpy
        steer_array = steer.cpu().numpy()
        throttle_array = throttle.cpu().numpy()
        
        # Safety clamps
        steer_array = np.clip(steer_array, -1.0, 1.0)
        throttle_array = np.clip(throttle_array, 0.0, 1.0)
        
        return steer_array, throttle_array
    
    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration for serialization"""
        return {
            'policy_type': 'TorchGRU',
            'trunk_dims': self.trunk_dims,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed,
            'device': str(self.device)
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get underlying model configuration"""
        return self.model.get_config()
    
    def set_weights(self, weights: np.ndarray):
        """Set model weights from flattened array (for ES compatibility)"""
        try:
            from src.training.es.params import set_flat_params
        except ImportError:
            from ..training.es.params import set_flat_params
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        set_flat_params(self.model, weight_tensor)
    
    def get_weights(self) -> np.ndarray:
        """Get model weights as flattened array (for ES compatibility)"""
        try:
            from src.training.es.params import get_flat_params
        except ImportError:
            from ..training.es.params import get_flat_params
        return get_flat_params(self.model).numpy()
    
    def save_weights(self, path: str):
        """Save model weights to file"""
        import torch
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        import torch
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'trunk_dims': list(self.trunk_dims),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'device': str(self.device),
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'TorchGRUPolicy':
        """Create policy from parameters dict"""
        return cls(
            trunk_dims=tuple(params['trunk_dims']),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            device=params['device'],
            v_scale=params['v_scale'],
            omega_scale=params['omega_scale'],
            init_seed=params['init_seed']
        )