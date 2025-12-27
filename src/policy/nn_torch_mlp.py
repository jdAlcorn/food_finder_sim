#!/usr/bin/env python3
"""
PyTorch MLP Neural Network Policy
Uses a trained MLP to map observations to actions
"""

import math
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from .base import Policy
from .models.mlp import SimpleMLP


class TorchMLPPolicy(Policy):
    """
    PyTorch MLP policy that uses neural network for action prediction.
    Reuses observation extraction from NN policy stub.
    """
    
    def __init__(self, hidden_dims: Tuple[int, ...] = (256, 128), device: str = "cpu",
                 v_scale: float = 400.0, omega_scale: float = 10.0, 
                 init_seed: int = None):
        """
        Initialize PyTorch MLP policy
        
        Args:
            hidden_dims: Hidden layer dimensions for MLP
            device: Device to run model on ("cpu" or "cuda")
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            init_seed: Random seed for deterministic initialization
        """
        self.name = "TorchMLP"
        self.hidden_dims = hidden_dims
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.init_seed = init_seed
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Create model
        self.model = SimpleMLP(
            input_dim=388,
            hidden_dims=hidden_dims,
            output_dim=2,
            init_seed=init_seed
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Observation tracking
        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[Dict[str, float]] = None
        self.obs_logged = False
        
        # Expected observation dimensions
        self.num_rays = 128
        self.vision_channels = 3
        self.proprioception_dim = 4
        
        print(f"TorchMLPPolicy initialized on {self.device}")
        if init_seed is not None:
            print(f"Using deterministic seed: {init_seed}")
    
    def reset(self) -> None:
        """Reset policy state"""
        self.last_obs = None
        self.last_action = None
        self.obs_logged = False
    
    def _extract_vision_features(self, vision_distances: list, vision_hit_types: list, 
                                max_range: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract vision features from raw vision data (reused from stub)"""
        num_rays = len(vision_distances)
        self.num_rays = num_rays
        
        vision_close = np.zeros(num_rays, dtype=np.float32)
        vision_food = np.zeros(num_rays, dtype=np.float32)
        vision_wall = np.zeros(num_rays, dtype=np.float32)
        
        for i in range(num_rays):
            distance = vision_distances[i]
            hit_type = vision_hit_types[i]
            
            if distance is None:
                distance = max_range
            
            vision_close[i] = 1.0 - np.clip(distance / max_range, 0.0, 1.0)
            
            if hit_type == 'food':
                vision_food[i] = 1.0
            
            if hit_type is not None and (hit_type == 'wall' or hit_type.startswith('wall')):
                vision_wall[i] = 1.0
        
        return vision_close, vision_food, vision_wall
    
    def _extract_proprioception(self, agent_state: Dict[str, float]) -> np.ndarray:
        """Extract proprioceptive features (reused from stub)"""
        vx = agent_state['vx']
        vy = agent_state['vy']
        theta = agent_state['theta']
        omega = agent_state['omega']
        throttle = agent_state['throttle']
        
        # Agent's local coordinate frame
        forward_x = math.cos(theta)
        forward_y = math.sin(theta)
        right_x = -math.sin(theta)
        right_y = math.cos(theta)
        
        # Project velocity onto local frame
        v_forward = vx * forward_x + vy * forward_y
        v_sideways = vx * right_x + vy * right_y
        
        # Normalize and clamp
        v_forward_norm = np.clip(v_forward / self.v_scale, -2.0, 2.0)
        v_sideways_norm = np.clip(v_sideways / self.v_scale, -2.0, 2.0)
        omega_norm = np.clip(omega / self.omega_scale, -2.0, 2.0)
        throttle_norm = np.clip(throttle, 0.0, 1.0)
        
        return np.array([v_forward_norm, v_sideways_norm, omega_norm, throttle_norm], 
                       dtype=np.float32)
    
    def _build_observation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """Build complete egocentric observation vector (reused from stub)"""
        vision_distances = sim_state['vision_distances']
        vision_hit_types = sim_state['vision_hit_types']
        agent_state = sim_state['agent_state']
        
        # Infer max range
        max_range = 300.0
        if len(vision_distances) > 0:
            max_distances = [d for d in vision_distances if d is not None]
            if max_distances:
                potential_max = max(max_distances)
                if potential_max >= 250:
                    max_range = potential_max
        
        # Extract features
        vision_close, vision_food, vision_wall = self._extract_vision_features(
            vision_distances, vision_hit_types, max_range
        )
        proprioception = self._extract_proprioception(agent_state)
        
        # Concatenate
        observation = np.concatenate([
            vision_close,
            vision_food,
            vision_wall,
            proprioception
        ])
        
        # Validate
        expected_length = self.num_rays * self.vision_channels + self.proprioception_dim
        assert len(observation) == expected_length, f"Observation length mismatch: {len(observation)} != {expected_length}"
        assert not np.any(np.isnan(observation)), "Observation contains NaN values"
        assert not np.any(np.isinf(observation)), "Observation contains infinite values"
        
        return observation
    
    def get_last_obs(self) -> Optional[np.ndarray]:
        """Get last observation for debugging"""
        return self.last_obs
    
    def get_last_action(self) -> Optional[Dict[str, float]]:
        """Get last action for debugging"""
        return self.last_action
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract observation and predict action using neural network
        
        Args:
            sim_state: Full simulation state
            
        Returns:
            Action dictionary with steer and throttle
        """
        # Build observation
        observation = self._build_observation(sim_state)
        self.last_obs = observation
        
        # Log observation info once
        if not self.obs_logged:
            print(f"TorchMLPPolicy: Processing observation")
            print(f"  Shape: {observation.shape}")
            print(f"  Value range: [{observation.min():.3f}, {observation.max():.3f}]")
            print(f"  Device: {self.device}")
            self.obs_logged = True
        
        try:
            # Convert to torch tensor
            x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 388]
            
            # Run inference
            with torch.no_grad():
                steer, throttle = self.model(x)
                steer_val = float(steer.item())
                throttle_val = float(throttle.item())
            
            # Safety checks
            if np.isnan(steer_val) or np.isnan(throttle_val):
                print("Warning: NaN detected in model output, using safe defaults")
                steer_val = 0.0
                throttle_val = 0.0
            
            # Clamp to safe ranges (should already be done by model activations)
            steer_val = np.clip(steer_val, -1.0, 1.0)
            throttle_val = np.clip(throttle_val, 0.0, 1.0)
            
            # Create action dict
            action = {
                'steer': steer_val,
                'throttle': throttle_val
            }
            
            self.last_action = action
            return action
            
        except Exception as e:
            print(f"Error in neural network inference: {e}")
            print("Falling back to safe default action")
            
            # Fallback to safe action
            action = {'steer': 0.0, 'throttle': 0.0}
            self.last_action = action
            return action
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights from file"""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            print("Using randomly initialized weights")
    
    def save_weights(self, weights_path: str) -> None:
        """Save model weights to file"""
        torch.save(self.model.state_dict(), weights_path)
        print(f"Saved model weights to {weights_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'hidden_dims': list(self.hidden_dims),
            'device': str(self.device),
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'TorchMLPPolicy':
        """Create policy from parameters"""
        return cls(
            hidden_dims=tuple(params.get('hidden_dims', [256, 128])),
            device=params.get('device', 'cpu'),
            v_scale=params.get('v_scale', 400.0),
            omega_scale=params.get('omega_scale', 10.0),
            init_seed=params.get('init_seed', None)
        )