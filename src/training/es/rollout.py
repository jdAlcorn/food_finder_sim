#!/usr/bin/env python3
"""
Rollout evaluation for Evolution Strategies
Deterministic fitness evaluation of parameter vectors
"""

import torch
import numpy as np
import math
from typing import Dict, Any, Callable
from src.sim.core import Simulation, SimulationConfig
from src.policy.obs import build_observation
from .params import set_flat_params


def rollout_fitness(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                   sim_config: SimulationConfig, seed: int, T: int, dt: float, 
                   max_range: float = 300.0, v_scale: float = 400.0, 
                   omega_scale: float = 10.0, food_reward_multiplier: float = 1000.0,
                   proximity_reward_scale: float = 1.0) -> float:
    """
    Evaluate fitness of a parameter vector through simulation rollout
    
    Args:
        theta_flat: Flattened model parameters
        model_ctor: Model constructor function
        model_kwargs: Model constructor arguments
        sim_config: Simulation configuration
        seed: Random seed for deterministic evaluation
        T: Number of simulation steps
        dt: Fixed timestep
        max_range: Maximum vision range for fitness calculation
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        food_reward_multiplier: Points awarded per food collected
        proximity_reward_scale: Scale factor for proximity reward
        
    Returns:
        Fitness value (higher is better)
    """
    try:
        # Create model and load parameters
        model = model_ctor(**model_kwargs)
        model.eval()
        
        # Load flattened parameters into model
        set_flat_params(model, theta_flat)
        
        # Create simulation
        sim = Simulation(sim_config, seed=seed)
        
        # Track fitness components
        distances_to_food = []
        food_count = 0
        
        # Run episode
        with torch.no_grad():
            for step in range(T):
                # Get current state
                sim_state = sim.get_state()
                
                # Build observation
                obs = build_observation(sim_state, v_scale, omega_scale)
                
                # Convert to tensor and run model
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                steer_raw, throttle_raw = model(obs_tensor)
                
                # Extract actions (activations already applied in model)
                steer = float(steer_raw.item())
                throttle = float(throttle_raw.item())
                
                # Safety clamps (should be redundant)
                steer = np.clip(steer, -1.0, 1.0)
                throttle = np.clip(throttle, 0.0, 1.0)
                
                # Step simulation
                action = {'steer': steer, 'throttle': throttle}
                step_info = sim.step(dt, action)
                
                # Track distance to food (current food position)
                agent_pos = (step_info['agent_state']['x'], step_info['agent_state']['y'])
                food_pos = (step_info['food_position']['x'], step_info['food_position']['y'])
                dist_to_food = math.sqrt((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)
                distances_to_food.append(dist_to_food)
                
                # Count food collected
                if step_info['food_collected_this_step']:
                    food_count += 1
        
        # Calculate fitness components
        final_distance = distances_to_food[-1] if distances_to_food else max_range
        
        # Fitness components (food count heavily weighted)
        food_reward = food_count * food_reward_multiplier  # Points per food collected
        proximity_reward = max(0, max_range - final_distance) * proximity_reward_scale  # Reward for being close at end
        step_penalty = -0.001 * T  # Small penalty for episode length
        
        fitness = food_reward + proximity_reward + step_penalty
        
        return float(fitness)
        
    except Exception as e:
        # Return very low fitness for failed rollouts
        print(f"Rollout failed with error: {e}")
        return -10000.0


def evaluate_multiple_seeds(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                           sim_config: SimulationConfig, base_seed: int, num_seeds: int, T: int, dt: float,
                           **kwargs) -> float:
    """
    Evaluate fitness across multiple random seeds and return average
    
    Args:
        theta_flat: Flattened model parameters
        model_ctor: Model constructor function
        model_kwargs: Model constructor arguments
        sim_config: Simulation configuration
        base_seed: Base seed for deterministic seed generation
        num_seeds: Number of seeds to evaluate
        T: Number of simulation steps per episode
        dt: Fixed timestep
        **kwargs: Additional arguments for rollout_fitness
        
    Returns:
        Average fitness across all seeds
    """
    fitnesses = []
    
    for i in range(num_seeds):
        # Generate deterministic seed
        seed = base_seed + i * 1000  # Spread seeds apart
        
        fitness = rollout_fitness(
            theta_flat=theta_flat,
            model_ctor=model_ctor,
            model_kwargs=model_kwargs,
            sim_config=sim_config,
            seed=seed,
            T=T,
            dt=dt,
            **kwargs
        )
        
        fitnesses.append(fitness)
    
    return float(np.mean(fitnesses))


# Worker function for multiprocessing
def evaluate_candidate_worker(args):
    """
    Worker function for multiprocessing evaluation
    
    Args:
        args: Tuple of (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
                       base_seed, num_seeds, T, dt, kwargs)
    
    Returns:
        Tuple of (candidate_idx, fitness)
    """
    (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
     base_seed, num_seeds, T, dt, kwargs) = args
    
    try:
        # Generate unique base seed for this candidate
        candidate_base_seed = base_seed + candidate_idx * 10000
        
        fitness = evaluate_multiple_seeds(
            theta_flat=theta_flat,
            model_ctor=model_ctor,
            model_kwargs=model_kwargs,
            sim_config=sim_config,
            base_seed=candidate_base_seed,
            num_seeds=num_seeds,
            T=T,
            dt=dt,
            **kwargs
        )
        
        return candidate_idx, fitness
        
    except Exception as e:
        print(f"Worker {candidate_idx} failed: {e}")
        return candidate_idx, -10000.0


def test_rollout():
    """Test rollout evaluation"""
    from src.policy.models.mlp import SimpleMLP
    
    # Create test setup
    model_ctor = SimpleMLP
    model_kwargs = {'input_dim': 388, 'hidden_dims': (256, 128), 'output_dim': 2}
    sim_config = SimulationConfig()
    
    # Create random parameters
    model = model_ctor(**model_kwargs)
    from .params import get_flat_params
    theta = get_flat_params(model)
    
    print(f"Testing rollout with {theta.numel()} parameters")
    
    # Test single rollout
    fitness = rollout_fitness(
        theta_flat=theta,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        seed=42,
        T=100,
        dt=1/60
    )
    
    print(f"Single rollout fitness: {fitness:.2f}")
    
    # Test multiple seeds
    avg_fitness = evaluate_multiple_seeds(
        theta_flat=theta,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        base_seed=42,
        num_seeds=3,
        T=100,
        dt=1/60
    )
    
    print(f"Average fitness (3 seeds): {avg_fitness:.2f}")
    print("Rollout tests passed!")


if __name__ == "__main__":
    test_rollout()