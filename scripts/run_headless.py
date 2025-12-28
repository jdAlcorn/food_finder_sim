#!/usr/bin/env python3
"""
Headless simulation runner - no pygame, CLI output only
Run single simulation with detailed logging and statistics
"""

import sys
import os
import time
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sim import Simulation, SimulationConfig
from src.policy.manual import ManualPolicy
from src.policy.scripted import ScriptedPolicy
from src.policy.nn_policy_stub import NeuralPolicyStub
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.policy.checkpoint import load_policy
from src.eval.load_world import load_world, list_available_worlds
from src.eval.world_integration import apply_world_to_simulation
from src.sim.batched import BatchedSimulation


def run_headless_simulation(policy, config: SimulationConfig, max_steps: int = 5000, 
                           dt: float = 1/60, seed: int = None, verbose: bool = True,
                           print_interval: int = 100) -> dict:
    """
    Run a single simulation without graphics
    
    Args:
        policy: Policy instance to use
        config: Simulation configuration
        max_steps: Maximum simulation steps
        dt: Time step size
        seed: Random seed for simulation
        verbose: Whether to print detailed progress
        print_interval: Steps between progress prints
        
    Returns:
        Dictionary with simulation results
    """
    # Create simulation
    sim = Simulation(config, seed=seed)
    policy.reset()
    
    if verbose:
        print(f"Starting headless simulation")
        print(f"Policy: {policy.name if hasattr(policy, 'name') else type(policy).__name__}")
        print(f"Max steps: {max_steps}, dt: {dt:.4f}")
        print(f"World size: {config.world_width}x{config.world_height}")
        print(f"Seed: {seed}")
        print("-" * 50)
    
    # Track statistics
    start_time = time.time()
    food_collection_times = []
    action_stats = {'steer': [], 'throttle': []}
    
    # Run simulation
    for step in range(max_steps):
        # Get current state
        sim_state = sim.get_state()
        
        # Get action from policy
        action = policy.act(sim_state)
        
        # Track action statistics
        action_stats['steer'].append(action.get('steer', 0.0))
        action_stats['throttle'].append(action.get('throttle', 0.0))
        
        # Step simulation
        step_info = sim.step(dt, action)
        
        # Track food collection
        if step_info['food_collected_this_step']:
            food_collection_times.append(step_info['time'])
            if verbose:
                print(f"Step {step:4d}: FOOD COLLECTED! Total: {step_info['food_collected']}, Time: {step_info['time']:.2f}s")
        
        # Print progress
        if verbose and step % print_interval == 0:
            agent_state = step_info['agent_state']
            print(f"Step {step:4d}: Pos=({agent_state['x']:.1f},{agent_state['y']:.1f}), "
                  f"Speed={agent_state['speed']:.1f}, Food={step_info['food_collected']}, "
                  f"Action=(steer={action.get('steer', 0):.3f}, throttle={action.get('throttle', 0):.3f})")
        
        # Optional early termination
        # if step_info['food_collected'] >= 10:
        #     print(f"Early termination: collected {step_info['food_collected']} food")
        #     break
    
    end_time = time.time()
    
    # Calculate final statistics
    final_state = sim.get_state()
    wall_clock_time = end_time - start_time
    
    # Action statistics
    import numpy as np
    steer_stats = {
        'mean': np.mean(action_stats['steer']),
        'std': np.std(action_stats['steer']),
        'min': np.min(action_stats['steer']),
        'max': np.max(action_stats['steer'])
    }
    throttle_stats = {
        'mean': np.mean(action_stats['throttle']),
        'std': np.std(action_stats['throttle']),
        'min': np.min(action_stats['throttle']),
        'max': np.max(action_stats['throttle'])
    }
    
    results = {
        'seed': seed,
        'steps': sim.step_count,
        'simulation_time': sim.time,
        'wall_clock_time': wall_clock_time,
        'food_collected': sim.food_collected,
        'food_collection_times': food_collection_times,
        'final_agent_state': final_state['agent_state'],
        'steps_per_second': sim.step_count / wall_clock_time,
        'action_stats': {
            'steer': steer_stats,
            'throttle': throttle_stats
        }
    }
    
    if verbose:
        print("-" * 50)
        print("SIMULATION COMPLETE")
        print(f"Total steps: {results['steps']}")
        print(f"Simulation time: {results['simulation_time']:.2f}s")
        print(f"Wall clock time: {results['wall_clock_time']:.2f}s")
        print(f"Speed: {results['steps_per_second']:.0f} steps/second")
        print(f"Food collected: {results['food_collected']}")
        
        if food_collection_times:
            avg_collection_time = np.mean(np.diff([0] + food_collection_times))
            print(f"Average time between food: {avg_collection_time:.2f}s")
        
        print(f"Final position: ({results['final_agent_state']['x']:.1f}, {results['final_agent_state']['y']:.1f})")
        print(f"Final speed: {results['final_agent_state']['speed']:.1f}")
        
        print("\nAction Statistics:")
        print(f"  Steer: mean={steer_stats['mean']:.3f}, std={steer_stats['std']:.3f}, range=[{steer_stats['min']:.3f}, {steer_stats['max']:.3f}]")
        print(f"  Throttle: mean={throttle_stats['mean']:.3f}, std={throttle_stats['std']:.3f}, range=[{throttle_stats['min']:.3f}, {throttle_stats['max']:.3f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run headless simulation with detailed output')
    parser.add_argument('--policy', choices=['scripted', 'neural_stub', 'torch_mlp', 'checkpoint'], 
                       default='scripted', help='Policy type to use')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file (required if policy=checkpoint)')
    parser.add_argument('--world', type=str, default=None,
                       help='World ID to load (default: empty world)')
    parser.add_argument('--list-worlds', action='store_true',
                       help='List available worlds and exit')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum simulation steps (default: 5000)')
    parser.add_argument('--dt', type=float, default=1/60,
                       help='Simulation timestep (default: 1/60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (only final results)')
    parser.add_argument('--print-interval', type=int, default=100,
                       help='Steps between progress prints (default: 100)')
    
    args = parser.parse_args()
    
    # List worlds if requested
    if args.list_worlds:
        worlds = list_available_worlds()
        print("Available worlds:")
        for world_id in worlds:
            print(f"  - {world_id}")
        return
    
    # Validate arguments
    if args.policy == 'checkpoint' and not args.checkpoint:
        print("Error: --checkpoint required when using checkpoint policy")
        sys.exit(1)
    
    # Create policy
    config = SimulationConfig()
    
    if args.policy == 'scripted':
        policy = ScriptedPolicy()
    elif args.policy == 'neural_stub':
        policy = NeuralPolicyStub()
    elif args.policy == 'torch_mlp':
        policy = TorchMLPPolicy(init_seed=args.seed)  # Deterministic NN
    elif args.policy == 'checkpoint':
        try:
            policy, loaded_config, metadata = load_policy(args.checkpoint)
            config = loaded_config  # Use config from checkpoint
            print(f"Loaded policy: {type(policy).__name__}")
            if metadata:
                print(f"Metadata: {metadata}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
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
            
            # Create unified simulation wrapper
            from src.sim.unified import SimulationSingle
            sim = SimulationSingle(updated_config)
            sim.batched_sim = batched_sim
            
            print(f"World loaded: {world.description}")
            obstacle_count = (len(world.geometry.rectangles) + 
                            len(world.geometry.circles) + 
                            len(world.geometry.segments))
            if obstacle_count > 0:
                print(f"World has {obstacle_count} obstacles")
            
            # Update config for headless runner
            config = updated_config
            
        except Exception as e:
            print(f"Error loading world '{args.world}': {e}")
            print("Falling back to default empty world")
            sim = None
    
    # Run simulation
    results = run_headless_simulation(
        policy=policy,
        config=config,
        max_steps=args.max_steps,
        dt=args.dt,
        seed=args.seed,
        verbose=not args.quiet,
        print_interval=args.print_interval
    )
    
    # Always print final summary (even in quiet mode)
    if args.quiet:
        print(f"Steps: {results['steps']}, Food: {results['food_collected']}, "
              f"Time: {results['simulation_time']:.2f}s, Speed: {results['steps_per_second']:.0f} steps/s")


if __name__ == "__main__":
    main()