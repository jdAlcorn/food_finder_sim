#!/usr/bin/env python3
"""
Interactive test case player
Load and play a specific test case in GUI mode
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
from src.policy.checkpoint import load_policy


def play_testcase(suite_path: str, case_id: str, agent_path: str = None, dt: float = None):
    """
    Play a specific test case interactively
    
    Args:
        suite_path: Path to test suite JSON file
        case_id: ID of test case to play
        agent_path: Optional path to agent checkpoint (defaults to manual control)
        dt: Optional fixed timestep for physics simulation
    """
    # Load test suite
    try:
        suite = load_suite(suite_path)
    except Exception as e:
        print(f"Error loading test suite: {e}")
        return
    
    # Find test case
    test_case = suite.get_case_by_id(case_id)
    if test_case is None:
        print(f"Test case '{case_id}' not found in suite '{suite.suite_id}'")
        print(f"Available test cases:")
        for case in suite.test_cases:
            print(f"  - {case.id}: {case.notes}")
        return
    
    print(f"Playing test case: {test_case.id}")
    print(f"Description: {test_case.notes}")
    print(f"Max steps: {test_case.max_steps}")
    
    # Create policy
    if agent_path:
        try:
            policy, _, metadata = load_policy(agent_path)
            policy_name = f"{os.path.basename(agent_path)} ({case_id})"
            print(f"Loaded agent: {agent_path}")
            if 'generation' in metadata:
                print(f"Generation: {metadata['generation']}")
            if 'best_fitness' in metadata:
                print(f"Best fitness: {metadata['best_fitness']:.2f}")
        except Exception as e:
            print(f"Error loading agent from {agent_path}: {e}")
            return
    else:
        policy = ManualPolicy()
        policy_name = f"Manual ({case_id})"
        print("Using manual control")
    
    print()
    
    # Create simulation
    config = SimulationConfig()
    sim = SimulationSingle(config)
    
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
    
    # Import and run GUI
    try:
        from src.viz.pygame_single import run_simulation_gui
        
        print("Starting interactive GUI...")
        if dt is not None:
            print(f"Using fixed timestep: dt = {dt:.6f}")
        else:
            print("Using variable timestep based on FPS")
        
        if agent_path:
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
            fps=60,
            policy_name=policy_name,
            sim=sim,  # Pass pre-configured simulation
            dt=dt     # Pass fixed timestep if provided
        )
        
    except ImportError:
        print("GUI not available (pygame not installed)")
    except Exception as e:
        print(f"Error running GUI: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Play a test case interactively')
    
    parser.add_argument('--suite', type=str, default='data/test_suites/basic_v1.json',
                       help='Path to test suite JSON file')
    parser.add_argument('--case-id', type=str, required=True,
                       help='ID of test case to play')
    parser.add_argument('--agent', type=str, default=None,
                       help='Path to agent checkpoint file (default: manual control)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Fixed timestep for physics simulation (default: variable based on FPS)')
    
    args = parser.parse_args()
    
    # Check if suite file exists
    if not os.path.exists(args.suite):
        print(f"Test suite file not found: {args.suite}")
        return
    
    # Check if agent file exists (if specified)
    if args.agent and not os.path.exists(args.agent):
        print(f"Agent checkpoint file not found: {args.agent}")
        return
    
    play_testcase(args.suite, args.case_id, args.agent, args.dt)


if __name__ == "__main__":
    main()