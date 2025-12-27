#!/usr/bin/env python3
"""
Checkpoint save/load functionality for policies
"""

import json
import os
from typing import Dict, Any, Tuple
from .base import Policy
from .manual import ManualPolicy
from .scripted import ScriptedPolicy
from .nn_policy_stub import NeuralPolicyStub
from src.sim.core import SimulationConfig


def save_policy(path: str, policy_name: str, policy_params: Dict[str, Any], 
                sim_config: SimulationConfig, metadata: Dict[str, Any] = None) -> None:
    """
    Save policy checkpoint to JSON file
    
    Args:
        path: File path to save to
        policy_name: Name of the policy type
        policy_params: Policy-specific parameters
        sim_config: Simulation configuration
        metadata: Optional metadata dict
    """
    checkpoint = {
        'policy_name': policy_name,
        'policy_params': policy_params,
        'sim_config': sim_config.to_dict(),
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Saved policy checkpoint to {path}")


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


def create_neural_stub_checkpoint(path: str = "checkpoints/neural_stub.json") -> None:
    """Create a neural stub checkpoint for testing"""
    policy = NeuralPolicyStub(v_scale=400.0, omega_scale=10.0, default_throttle=0.3)
    config = SimulationConfig()
    metadata = {
        'created_by': 'neural_stub_generator',
        'description': 'Neural network policy stub with observation extraction',
        'version': '1.0',
        'observation_dim': 128 * 3 + 4,  # vision + proprioception
        'ready_for_nn': True
    }
    
    save_policy(path, 'NeuralStub', policy.get_params(), config, metadata)


if __name__ == "__main__":
    # Create example checkpoints
    create_example_checkpoint()
    create_neural_stub_checkpoint()