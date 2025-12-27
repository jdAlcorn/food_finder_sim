#!/usr/bin/env python3
"""
Watch a loaded agent/policy in the simulation
"""

import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_single import run_simulation_gui
from src.policy.checkpoint import load_policy, create_example_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Watch a loaded agent/policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy checkpoint file')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--vision', action='store_true', default=True,
                       help='Start with vision display on (default: True)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found: {args.checkpoint}")
        print("Creating example checkpoint...")
        create_example_checkpoint(args.checkpoint)
        print(f"Created example checkpoint at {args.checkpoint}")
    
    try:
        # Load policy and config
        policy, config, metadata = load_policy(args.checkpoint)
        
        print(f"Loaded policy: {policy.name}")
        if metadata:
            print(f"Metadata: {metadata}")
        
        # Run GUI with loaded policy
        run_simulation_gui(policy, config, fps=args.fps, policy_name=policy.name)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()