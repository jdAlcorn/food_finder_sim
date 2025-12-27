#!/usr/bin/env python3
"""
Batch simulation runner for training/testing
No graphics - pure CLI with fast execution
"""

import time
import argparse
from typing import List, Dict, Any
from simulation_core import Simulation, SimulationConfig


def random_policy(sim_state: Dict[str, Any]) -> tuple[float, float]:
    """
    Example random policy for testing
    Returns (steer_input, throttle_input)
    """
    import random
    steer = random.uniform(-1.0, 1.0)
    throttle = random.uniform(0.0, 1.0)
    return steer, throttle


def simple_food_seeking_policy(sim_state: Dict[str, Any]) -> tuple[float, float]:
    """
    Simple policy that tries to move toward food
    Returns (steer_input, throttle_input)
    """
    import math
    
    agent_state = sim_state['agent_state']
    food_pos = sim_state['food_position']
    
    # Calculate angle to food
    dx = food_pos['x'] - agent_state['x']
    dy = food_pos['y'] - agent_state['y']
    angle_to_food = math.atan2(dy, dx)
    
    # Calculate angle difference
    angle_diff = angle_to_food - agent_state['theta']
    
    # Normalize angle difference to [-pi, pi]
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    # Simple steering toward food
    steer = max(-1.0, min(1.0, angle_diff * 2.0))
    
    # Always thrust forward
    throttle = 1.0
    
    return steer, throttle


def run_single_simulation(config: SimulationConfig, policy_func, max_steps: int = 10000, 
                         dt: float = 1/60, seed: int = None, verbose: bool = False) -> Dict[str, Any]:
    """Run a single simulation episode"""
    sim = Simulation(config, seed=seed)
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Get current state
        sim_state = sim.get_state()
        
        # Get action from policy
        steer_input, throttle_input = policy_func(sim_state)
        
        # Step simulation
        step_info = sim.step(dt, steer_input, throttle_input)
        
        if verbose and step_info['food_collected_this_step']:
            print(f"Step {step}: Food collected! Total: {step_info['food_collected']}")
        
        # Optional early termination conditions
        # if step_info['food_collected'] >= 10:  # Stop after collecting 10 food
        #     break
    
    end_time = time.time()
    
    final_state = sim.get_state()
    
    return {
        'seed': seed,
        'steps': sim.step_count,
        'simulation_time': sim.time,
        'wall_clock_time': end_time - start_time,
        'food_collected': sim.food_collected,
        'final_agent_state': final_state['agent_state'],
        'steps_per_second': sim.step_count / (end_time - start_time)
    }


def run_batch_simulations(num_sims: int, config: SimulationConfig = None, 
                         policy_func = None, max_steps: int = 10000,
                         dt: float = 1/60, verbose: bool = False) -> List[Dict[str, Any]]:
    """Run multiple simulations in batch"""
    
    if config is None:
        config = SimulationConfig()
    
    if policy_func is None:
        policy_func = random_policy
    
    results = []
    
    print(f"Running {num_sims} simulations...")
    start_time = time.time()
    
    for i in range(num_sims):
        if verbose or (i + 1) % max(1, num_sims // 10) == 0:
            print(f"Simulation {i + 1}/{num_sims}")
        
        result = run_single_simulation(
            config=config,
            policy_func=policy_func,
            max_steps=max_steps,
            dt=dt,
            seed=i,  # Use simulation index as seed for reproducibility
            verbose=False
        )
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Print summary statistics
    total_food = sum(r['food_collected'] for r in results)
    avg_food = total_food / num_sims
    avg_steps_per_sec = sum(r['steps_per_second'] for r in results) / num_sims
    
    print(f"\nBatch Results:")
    print(f"Total simulations: {num_sims}")
    print(f"Total wall clock time: {total_time:.2f}s")
    print(f"Average food collected per sim: {avg_food:.2f}")
    print(f"Total food collected: {total_food}")
    print(f"Average simulation speed: {avg_steps_per_sec:.0f} steps/second")
    print(f"Total simulation steps: {sum(r['steps'] for r in results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run batch simulations')
    parser.add_argument('--num-sims', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--max-steps', type=int, default=5000, help='Max steps per simulation')
    parser.add_argument('--policy', choices=['random', 'food-seeking'], default='random', 
                       help='Policy to use')
    parser.add_argument('--dt', type=float, default=1/60, help='Simulation timestep')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Select policy
    if args.policy == 'random':
        policy = random_policy
    elif args.policy == 'food-seeking':
        policy = simple_food_seeking_policy
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    
    # Create config
    config = SimulationConfig()
    
    # Run batch
    results = run_batch_simulations(
        num_sims=args.num_sims,
        config=config,
        policy_func=policy,
        max_steps=args.max_steps,
        dt=args.dt,
        verbose=args.verbose
    )
    
    # Optionally save results
    # import json
    # with open('batch_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()