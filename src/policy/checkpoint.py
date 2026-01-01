#!/usr/bin/env python3
"""
Checkpoint save/load functionality for policies
"""

import json
import os
import shutil
from typing import Dict, Any, Tuple
from .base import Policy
from .manual import ManualPolicy
from .scripted import ScriptedPolicy
from .nn_policy_stub import NeuralPolicyStub
from .nn_torch_mlp import TorchMLPPolicy
from .nn_torch_gru import TorchGRUPolicy
from .rl_gru_policy import RLGRUPolicy
from .rl_mlp_policy import RLMLPPolicy
from src.sim.core import SimulationConfig


def save_policy(path: str, policy_name: str, policy_params: Dict[str, Any], 
                sim_config: SimulationConfig, metadata: Dict[str, Any] = None,
                policy_instance: Policy = None) -> None:
    """
    Save policy checkpoint to JSON file
    
    Args:
        path: File path to save to
        policy_name: Name of the policy type
        policy_params: Policy-specific parameters
        sim_config: Simulation configuration
        metadata: Optional metadata dict
        policy_instance: Optional policy instance (for saving weights)
    """
    # Handle weights path for torch policies
    weights_path = None
    if policy_name in ['TorchMLP', 'TorchGRU', 'RLGRU', 'RLMLP'] and policy_instance is not None:
        # Generate weights path
        base_path = os.path.splitext(path)[0]
        weights_path = f"{base_path}_weights.pt"
        policy_params['weights_path'] = weights_path
        
        # Save model weights
        if hasattr(policy_instance, 'save_weights'):
            policy_instance.save_weights(weights_path)
    
    checkpoint = {
        'policy_name': policy_name,
        'policy_params': policy_params,
        'sim_config': sim_config.to_dict(),
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Use atomic write to prevent race conditions with viewer
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Atomic rename - use shutil.move for cross-platform compatibility
    shutil.move(temp_path, path)
    
    print(f"Saved policy checkpoint to {path}")
    if weights_path:
        print(f"Saved model weights to {weights_path}")


def load_policy(path: str) -> Tuple[Policy, SimulationConfig, Dict[str, Any]]:
    """
    Load policy checkpoint from JSON file
    
    Args:
        path: File path to load from
    
    Returns:
        Tuple of (policy_instance, sim_config, metadata)
    """
    with open(path, 'r') as f:
        checkpoint = json.load(f)
    
    policy_name = checkpoint['policy_name']
    policy_params = checkpoint['policy_params']
    sim_config_dict = checkpoint['sim_config']
    metadata = checkpoint.get('metadata', {})
    
    # Create simulation config
    sim_config = SimulationConfig.from_dict(sim_config_dict)
    
    # Create policy instance
    if policy_name == 'Manual':
        policy = ManualPolicy()
    elif policy_name == 'Scripted':
        policy = ScriptedPolicy.from_params(policy_params)
    elif policy_name == 'NeuralStub':
        policy = NeuralPolicyStub.from_params(policy_params)
    elif policy_name == 'TorchMLP':
        policy = TorchMLPPolicy.from_params(policy_params)
        
        # Load weights if available
        weights_path = policy_params.get('weights_path')
        if weights_path and os.path.exists(weights_path):
            policy.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}, using random initialization")
    elif policy_name == 'TorchGRU':
        policy = TorchGRUPolicy.from_params(policy_params)
        
        # Load weights if available
        weights_path = policy_params.get('weights_path')
        if weights_path and os.path.exists(weights_path):
            policy.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}, using random initialization")
    elif policy_name == 'RLGRU':
        policy = RLGRUPolicy.from_params(policy_params)
        
        # Load weights if available
        weights_path = policy_params.get('weights_path')
        if weights_path and os.path.exists(weights_path):
            policy.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}, using random initialization")
    elif policy_name == 'RLMLP':
        policy = RLMLPPolicy.from_params(policy_params)
        
        # Load weights if available
        weights_path = policy_params.get('weights_path')
        if weights_path and os.path.exists(weights_path):
            policy.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}, using random initialization")
    else:
        raise ValueError(f"Unknown policy type: {policy_name}")
    
    print(f"Loaded {policy_name} policy from {path}")
    return policy, sim_config, metadata


def create_example_checkpoint(path: str = "checkpoints/example.json") -> None:
    """Create an example checkpoint for testing"""
    policy = ScriptedPolicy(seek_strength=1.5, throttle_level=0.9)
    config = SimulationConfig()
    metadata = {
        'created_by': 'example_generator',
        'description': 'Simple food-seeking scripted policy',
        'version': '1.0'
    }
    
    save_policy(path, 'Scripted', policy.get_params(), config, metadata)


def create_torch_mlp_checkpoint(path: str = "checkpoints/torch_mlp.json") -> None:
    """Create a PyTorch MLP checkpoint for testing"""
    policy = TorchMLPPolicy(
        hidden_dims=(256, 128),
        device='cpu',
        v_scale=400.0,
        omega_scale=10.0,
        init_seed=42  # Deterministic initialization
    )
    config = SimulationConfig()
    metadata = {
        'created_by': 'torch_mlp_generator',
        'description': 'PyTorch MLP policy with random initialization',
        'version': '1.0',
        'observation_dim': 388,
        'model_type': 'SimpleMLP',
        'hidden_dims': [256, 128]
    }
    
    save_policy(path, 'TorchMLP', policy.get_params(), config, metadata, policy)


if __name__ == "__main__":
    # Create example checkpoints
    create_example_checkpoint()
    create_torch_mlp_checkpoint()