#!/usr/bin/env python3
"""
Run multiple simulations in parallel view
"""

import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_multi import run_multi_simulation
from src.policy.manual import ManualPolicy
from src.policy.scripted import ScriptedPolicy
from src.policy.nn_policy_stub import NeuralPolicyStub
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.policy.checkpoint import load_policy
from src.sim.core import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description='Run multiple simulations in parallel')
    parser.add_argument('--num-sims', type=int, default=16,
                       help='Number of simulations to run (default: 16)')
    parser.add_argument('--policy', choices=['manual', 'scripted', 'neural_stub', 'torch_mlp', 'checkpoint'], 
                       default='scripted', help='Policy type to use')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file (required if policy=checkpoint)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--width', type=int, default=1200,
                       help='Window width (default: 1200)')
    parser.add_argument('--height', type=int, default=800,
                       help='Window height (default: 800)')
    parser.add_argument('--vision', action='store_true',
                       help='Show vision rays (can be toggled with V key)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.policy == 'checkpoint' and not args.checkpoint:
        print("Error: --checkpoint required when using checkpoint policy")
        sys.exit(1)
    
    # Select policy class and config
    config = SimulationConfig()
    
    if args.policy == 'manual':
        policy_class = ManualPolicy
        print("Note: Manual policy in multi-sim uses the same keyboard input for all agents")
    elif args.policy == 'scripted':
        policy_class = ScriptedPolicy
    elif args.policy == 'neural_stub':
        policy_class = NeuralPolicyStub
    elif args.policy == 'torch_mlp':
        policy_class = TorchMLPPolicy
    elif args.policy == 'checkpoint':
        try:
            # Load one instance to get the policy class
            policy_instance, loaded_config, metadata = load_policy(args.checkpoint)
            config = loaded_config  # Use config from checkpoint
            
            # Create a policy class that recreates the loaded policy
            if hasattr(policy_instance, 'get_params'):
                params = policy_instance.get_params()
                policy_class = lambda: type(policy_instance).from_params(params)
            else:
                policy_class = type(policy_instance)
            
            print(f"Loaded policy: {type(policy_instance).__name__}")
            if metadata:
                print(f"Metadata: {metadata}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
    print(f"Starting {args.num_sims} simulations with {policy_class.__name__ if hasattr(policy_class, '__name__') else 'loaded'} policy")
    print("Controls: V=toggle vision, S=toggle stats, R=reset all, ESC=quit")
    
    # Run multi-simulation viewer
    run_multi_simulation(
        num_sims=args.num_sims,
        policy_class=policy_class,
        config=config,
        fps=args.fps,
        window_width=args.width,
        window_height=args.height,
        show_vision=args.vision
    )


if __name__ == "__main__":
    main()