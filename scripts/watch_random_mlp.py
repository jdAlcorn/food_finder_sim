#!/usr/bin/env python3
"""
Watch a randomly initialized PyTorch MLP policy
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_single import run_simulation_gui
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.sim.core import SimulationConfig


def main():
    """Run simulation with randomly initialized PyTorch MLP policy"""
    print("Running PyTorch MLP Policy (Random Initialization)")
    print("=" * 50)
    print("This runs a neural network with random weights")
    print("The agent will move based on untrained network outputs")
    print("Expect erratic but bounded behavior (steer ∈ [-1,1], throttle ∈ [0,1])")
    print()
    
    # Create policy with deterministic random initialization
    policy = TorchMLPPolicy(
        hidden_dims=(256, 128),
        device='cpu',
        v_scale=400.0,
        omega_scale=10.0,
        init_seed=42  # Deterministic for reproducibility
    )
    
    # Create default config
    config = SimulationConfig()
    
    print("Policy created successfully")
    print(f"Model device: {policy.device}")
    print(f"Hidden dimensions: {policy.hidden_dims}")
    print("Starting simulation...")
    print()
    
    # Run GUI
    run_simulation_gui(policy, config, fps=60, policy_name="TorchMLP (Random)")


if __name__ == "__main__":
    main()