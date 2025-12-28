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
from src.eval.load_world import load_world, list_available_worlds
from src.eval.world_integration import apply_world_to_simulation
from src.sim.batched import BatchedSimulation


def main():
    parser = argparse.ArgumentParser(description='Watch a loaded agent/policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy checkpoint file')
    parser.add_argument('--world', type=str, default=None,
                       help='World ID to load (default: empty world)')
    parser.add_argument('--list-worlds', action='store_true',
                       help='List available worlds and exit')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--vision', action='store_true', default=True,
                       help='Start with vision display on (default: True)')
    
    args = parser.parse_args()
    
    # List worlds if requested
    if args.list_worlds:
        worlds = list_available_worlds()
        print("Available worlds:")
        for world_id in worlds:
            print(f"  - {world_id}")
        return
    
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
        
        # Apply world if specified
        sim = None
        if args.world:
            try:
                print(f"Loading world: {args.world}")
                world = load_world(args.world)
                
                # Create a single-environment batched simulation to apply world
                batched_sim = BatchedSimulation(1, config)
                
                # Apply world geometry and physics
                updated_config = apply_world_to_simulation(batched_sim, world, [])
                
                # Create unified simulation wrapper for GUI compatibility
                from src.sim.unified import SimulationSingle
                sim = SimulationSingle(updated_config)
                sim.batched_sim = batched_sim
                
                print(f"World loaded: {world.description}")
                obstacle_count = (len(world.geometry.rectangles) + 
                                len(world.geometry.circles) + 
                                len(world.geometry.segments))
                if obstacle_count > 0:
                    print(f"World has {obstacle_count} obstacles")
                
                # Update config for GUI
                config = updated_config
                
            except Exception as e:
                print(f"Error loading world '{args.world}': {e}")
                print("Falling back to default empty world")
                sim = None
        
        # Run GUI with loaded policy
        policy_name = f"{policy.name} ({args.world})" if args.world else policy.name
        run_simulation_gui(policy, config, fps=args.fps, policy_name=policy_name, sim=sim)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()