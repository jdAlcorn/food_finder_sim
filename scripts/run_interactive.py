#!/usr/bin/env python3
"""
Run interactive simulation with manual keyboard control
Supports loading world specifications for obstacle environments
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_single import run_simulation_gui
from src.policy.manual import ManualPolicy
from src.sim.core import SimulationConfig
from src.eval.load_world import load_world, list_available_worlds
from src.eval.world_integration import apply_world_to_simulation
from src.sim.batched import BatchedSimulation


def main():
    """Run interactive simulation with manual control"""
    parser = argparse.ArgumentParser(description='Run interactive simulation with manual control')
    parser.add_argument('--world', type=str, default=None,
                       help='World ID to load (default: empty world)')
    parser.add_argument('--list-worlds', action='store_true',
                       help='List available worlds and exit')
    
    args = parser.parse_args()
    
    # List worlds if requested
    if args.list_worlds:
        worlds = list_available_worlds()
        print("Available worlds:")
        for world_id in worlds:
            print(f"  - {world_id}")
        return
    
    print("Starting interactive simulation with manual control...")
    print("Controls: W=thrust, A/D=steer, V=toggle vision, ESC=quit")
    
    # Create manual policy and config
    policy = ManualPolicy()
    config = SimulationConfig()
    
    # Load world if specified
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
            
            # Copy the world-configured scene to the unified sim
            sim.batched_sim = batched_sim
            
            print(f"World loaded: {world.description}")
            obstacle_count = (len(world.geometry.rectangles) + 
                            len(world.geometry.circles) + 
                            len(world.geometry.segments))
            if obstacle_count > 0:
                print(f"World has {obstacle_count} obstacles")
            
        except Exception as e:
            print(f"Error loading world '{args.world}': {e}")
            print("Falling back to default empty world")
            sim = None
            config = SimulationConfig()
    
    # Run GUI
    policy_name = f"Manual ({args.world})" if args.world else "Manual"
    run_simulation_gui(policy, config, fps=60, policy_name=policy_name, sim=sim)


if __name__ == "__main__":
    main()