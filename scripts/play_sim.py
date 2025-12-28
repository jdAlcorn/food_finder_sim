#!/usr/bin/env python3
"""
Unified simulation player - watch agents and/or play test cases
Combines functionality of watch_agent.py and play_testcase.py
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.load_suite import load_suite
from src.sim import SimulationConfig
from src.sim.unified import SimulationSingle
from src.policy.manual import ManualPolicy
from src.policy.scripted import ScriptedPolicy
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.policy.checkpoint import load_policy
from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
from src.eval.load_world import load_world, list_available_worlds
from src.sim.batched import BatchedSimulation


def setup_simulation_from_testcase(test_case, config):
    """Set up simulation from a test case"""
    # Resolve world for this test case
    try:
        world = resolve_test_case_world(test_case)
        print(f"Resolved world: {world.world_id}")
        if world.world_id != "default_empty":
            print(f"World description: {world.description}")
        
        obstacle_count = len(world.geometry.rectangles) + len(world.geometry.circles) + len(world.geometry.segments)
        if obstacle_count > 0:
            print(f"World obstacles: {obstacle_count} total")
        
        # Apply world to simulation
        batched_sim = BatchedSimulation(1, config)
        updated_config = apply_world_to_simulation(batched_sim, world, [test_case])
        
        # Create unified simulation wrapper
        sim = SimulationSingle(updated_config)
        sim.batched_sim = batched_sim
        
        # Set up test case initial state
        agent_states = [{
            'x': test_case.agent_start.x,
            'y': test_case.agent_start.y,
            'theta': test_case.agent_start.theta,
            'vx': test_case.agent_start.vx,
            'vy': test_case.agent_start.vy,
            'omega': test_case.agent_start.omega,
            'throttle': test_case.agent_start.throttle
        }]
        
        food_states = [{
            'x': test_case.food.x,
            'y': test_case.food.y
        }]
        
        # Reset simulation to test case state
        sim.batched_sim.reset_to_states(agent_states, food_states)
        
        return sim, updated_config
        
    except Exception as e:
        print(f"Warning: Could not resolve test case world: {e}")
        return None, config


def setup_simulation_from_world(world_id, config):
    """Set up simulation from a world ID"""
    try:
        print(f"Loading world: {world_id}")
        world = load_world(world_id)
        
        # Create a single-environment batched simulation to apply world
        batched_sim = BatchedSimulation(1, config)
        
        # Apply world geometry and physics
        updated_config = apply_world_to_simulation(batched_sim, world, [])
        
        # Create unified simulation wrapper for GUI compatibility
        sim = SimulationSingle(updated_config)
        sim.batched_sim = batched_sim
        
        print(f"World loaded: {world.description}")
        obstacle_count = (len(world.geometry.rectangles) + 
                        len(world.geometry.circles) + 
                        len(world.geometry.segments))
        if obstacle_count > 0:
            print(f"World has {obstacle_count} obstacles")
        
        return sim, updated_config
        
    except Exception as e:
        print(f"Error loading world '{world_id}': {e}")
        return None, config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Play simulation with agent and/or test case')
    
    # Policy options
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument('--agent', '--checkpoint', type=str, dest='checkpoint',
                             help='Path to agent checkpoint file')
    policy_group.add_argument('--policy', choices=['manual', 'scripted', 'random_mlp'], 
                             default='manual',
                             help='Built-in policy type (default: manual)')
    
    # Policy parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for random_mlp policy (default: 42)')
    
    # Test case options
    parser.add_argument('--suite', type=str, default=None,
                       help='Path to test suite JSON file')
    parser.add_argument('--case-id', type=str, default=None,
                       help='ID of test case to play (requires --suite)')
    
    # World options
    parser.add_argument('--world', type=str, default=None,
                       help='World ID to load (overrides test case world)')
    parser.add_argument('--list-worlds', action='store_true',
                       help='List available worlds and exit')
    
    # Simulation options
    parser.add_argument('--dt', type=float, default=None,
                       help='Fixed timestep for physics simulation (default: variable based on FPS)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    
    args = parser.parse_args()
    
    # List worlds if requested
    if args.list_worlds:
        worlds = list_available_worlds()
        print("Available worlds:")
        for world_id in worlds:
            print(f"  - {world_id}")
        return
    
    # Validate arguments
    if args.case_id and not args.suite:
        print("Error: --case-id requires --suite")
        sys.exit(1)
    
    print("Simulation Player")
    print("=" * 50)
    
    # Load test case if specified
    test_case = None
    if args.suite and args.case_id:
        try:
            suite = load_suite(args.suite)
            test_case = suite.get_case_by_id(args.case_id)
            if test_case is None:
                print(f"Test case '{args.case_id}' not found in suite")
                print("Available test cases:")
                for case in suite.test_cases:
                    print(f"  - {case.id}: {case.notes}")
                return
            
            print(f"Test case: {test_case.id}")
            print(f"Description: {test_case.notes}")
            print(f"Max steps: {test_case.max_steps}")
            
        except Exception as e:
            print(f"Error loading test suite: {e}")
            return
    
    # Create policy
    if args.checkpoint:
        try:
            policy, loaded_config, metadata = load_policy(args.checkpoint)
            config = loaded_config
            
            print(f"Loaded agent: {os.path.basename(args.checkpoint)}")
            print(f"Policy type: {type(policy).__name__}")
            if metadata:
                if 'generation' in metadata:
                    print(f"Generation: {metadata['generation']}")
                if 'best_fitness' in metadata:
                    print(f"Best fitness: {metadata['best_fitness']:.2f}")
            
        except Exception as e:
            print(f"Error loading agent from {args.checkpoint}: {e}")
            return
    else:
        # Use built-in policy
        config = SimulationConfig()
        
        if args.policy == 'manual':
            policy = ManualPolicy()
            print("Using manual control")
        elif args.policy == 'scripted':
            policy = ScriptedPolicy()
            print("Using scripted policy (food-seeking)")
        elif args.policy == 'random_mlp':
            policy = TorchMLPPolicy(
                hidden_dims=(256, 128),
                device='cpu',
                v_scale=400.0,
                omega_scale=10.0,
                init_seed=args.seed
            )
            print(f"Using random MLP policy (seed: {args.seed})")
        else:
            print(f"Unknown policy type: {args.policy}")
            return
    
    print()
    
    # Set up simulation
    sim = None
    
    # Priority: test case > explicit world > default
    if test_case:
        sim, config = setup_simulation_from_testcase(test_case, config)
        
        # Override with explicit world if specified
        if args.world:
            print(f"Overriding test case world with: {args.world}")
            sim, config = setup_simulation_from_world(args.world, config)
            
    elif args.world:
        sim, config = setup_simulation_from_world(args.world, config)
    
    # Import and run GUI
    try:
        from src.viz.pygame_single import run_simulation_gui
        
        print("Starting interactive GUI...")
        if args.dt is not None:
            print(f"Using fixed timestep: dt = {args.dt:.6f}")
        else:
            print("Using variable timestep based on FPS")
        
        # Build policy name
        policy_parts = []
        if args.checkpoint:
            policy_parts.append(os.path.basename(args.checkpoint).replace('.json', ''))
        else:
            policy_parts.append(args.policy.title())
        
        if test_case:
            policy_parts.append(f"({test_case.id})")
        elif args.world:
            policy_parts.append(f"({args.world})")
        
        policy_name = " ".join(policy_parts)
        
        # Show controls
        if args.checkpoint or args.policy != 'manual':
            print("Controls:")
            print("  V: Toggle vision display")
            print("  ESC: Exit")
        else:
            print("Controls:")
            print("  Arrow keys: Steer")
            print("  Space: Throttle")
            print("  V: Toggle vision display")
            print("  ESC: Exit")
        print()
        
        run_simulation_gui(
            policy=policy,
            config=config,
            fps=args.fps,
            policy_name=policy_name,
            sim=sim,
            dt=args.dt
        )
        
    except ImportError:
        print("GUI not available (pygame not installed)")
    except Exception as e:
        print(f"Error running GUI: {e}")


if __name__ == "__main__":
    main()