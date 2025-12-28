#!/usr/bin/env python3
"""
Test suite evaluator for deterministic fitness evaluation
"""

import torch
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Callable
from .testcases import TestSuite, TestCase
from ..sim.batched import BatchedSimulation
from ..sim.core import SimulationConfig
from ..training.es.params import set_flat_params


def evaluate_candidate_on_suite(
    theta_flat: torch.Tensor,
    model_ctor: Callable,
    model_kwargs: Dict[str, Any],
    sim_config: SimulationConfig,
    suite: TestSuite,
    dt: float,
    batch_size: int = 32,
    device: str = "cpu",
    v_scale: float = 400.0,
    omega_scale: float = 10.0,
    max_range: float = None
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Evaluate a candidate policy on a test suite
    
    Args:
        theta_flat: Flattened model parameters
        model_ctor: Model constructor function
        model_kwargs: Model constructor arguments
        sim_config: Simulation configuration
        suite: Test suite to evaluate on
        dt: Simulation timestep
        batch_size: Batch size for evaluation (number of test cases per batch)
        device: Device for PyTorch operations
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        max_range: Maximum range for proximity scoring (default: sim_config.max_range)
        
    Returns:
        Tuple of (fitness_mean, per_case_scores, metadata)
        - fitness_mean: Average fitness across all test cases
        - per_case_scores: Array of scores for each test case
        - metadata: Dict with additional info (reached flags, etc.)
    """
    if max_range is None:
        max_range = sim_config.max_range
    
    # Create model and load parameters
    model = model_ctor(**model_kwargs)
    model.eval()
    set_flat_params(model, theta_flat)
    
    # Move model to device
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)
    
    # Evaluate test cases in batches
    all_scores = []
    all_reached = []
    
    test_cases = list(suite.test_cases)
    
    for batch_start in range(0, len(test_cases), batch_size):
        batch_end = min(batch_start + batch_size, len(test_cases))
        batch_cases = test_cases[batch_start:batch_end]
        current_batch_size = len(batch_cases)
        
        # Evaluate this batch
        batch_scores, batch_reached = _evaluate_test_case_batch(
            model, batch_cases, sim_config, dt, current_batch_size,
            device, v_scale, omega_scale, max_range
        )
        
        all_scores.extend(batch_scores)
        all_reached.extend(batch_reached)
    
    # Compute overall statistics
    per_case_scores = np.array(all_scores)
    fitness_mean = float(np.mean(per_case_scores))
    
    metadata = {
        'per_case_reached': np.array(all_reached),
        'num_reached': int(np.sum(all_reached)),
        'num_total': len(test_cases),
        'success_rate': float(np.mean(all_reached))
    }
    
    return fitness_mean, per_case_scores, metadata


def _evaluate_test_case_batch(
    model: torch.nn.Module,
    test_cases: List[TestCase],
    sim_config: SimulationConfig,
    dt: float,
    batch_size: int,
    device: str,
    v_scale: float,
    omega_scale: float,
    max_range: float
) -> Tuple[List[float], List[bool]]:
    """
    Evaluate a batch of test cases
    
    Returns:
        Tuple of (scores, reached_flags)
    """
    # Create batched simulation
    batched_sim = BatchedSimulation(batch_size, sim_config)
    
    # Prepare initial states
    agent_states = []
    food_states = []
    max_steps_list = []
    dt_list = []
    
    for case in test_cases:
        # Agent state
        agent_state = {
            'x': case.agent_start.x,
            'y': case.agent_start.y,
            'theta': case.agent_start.theta,
            'vx': case.agent_start.vx,
            'vy': case.agent_start.vy,
            'omega': case.agent_start.omega,
            'throttle': case.agent_start.throttle
        }
        agent_states.append(agent_state)
        
        # Food state
        food_state = {
            'x': case.food.x,
            'y': case.food.y
        }
        food_states.append(food_state)
        
        max_steps_list.append(case.max_steps)
        dt_list.append(case.dt if case.dt is not None else dt)
    
    # Reset simulation to test case states
    batched_sim.reset_to_states(agent_states, food_states)
    
    # Track evaluation metrics
    reached = np.zeros(batch_size, dtype=bool)
    step_reached = np.full(batch_size, -1, dtype=int)
    min_distances = np.full(batch_size, float('inf'), dtype=float)
    
    # Determine maximum steps for this batch
    max_steps = max(max_steps_list)
    
    # Run rollout
    with torch.no_grad():
        for step in range(max_steps):
            # Check which environments are still active
            active_mask = np.array([
                step < max_steps_list[i] and not reached[i] 
                for i in range(batch_size)
            ])
            
            if not np.any(active_mask):
                break
            
            # Get observations for all environments
            observations = batched_sim.get_observations(v_scale, omega_scale)
            
            # Convert to tensor and run model
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            if device != "cpu" and torch.cuda.is_available():
                obs_tensor = obs_tensor.to(device)
            
            steer_raw, throttle_raw = model(obs_tensor)
            
            # Move back to CPU for simulation
            steer = steer_raw.detach().cpu().numpy()
            throttle = throttle_raw.detach().cpu().numpy()
            
            # Safety clamps
            steer = np.clip(steer, -1.0, 1.0)
            throttle = np.clip(throttle, 0.0, 1.0)
            
            # Step simulation (use first dt for simplicity, could be per-env if needed)
            actions = {'steer': steer, 'throttle': throttle}
            step_info = batched_sim.step(actions, dt_list[0])
            
            # Check for food collection and update metrics
            for i in range(batch_size):
                if not active_mask[i]:
                    continue
                
                # Calculate distance to food
                agent_x = step_info['agent_states']['x'][i]
                agent_y = step_info['agent_states']['y'][i]
                food_x = step_info['food_positions'][i][0]
                food_y = step_info['food_positions'][i][1]
                
                distance = math.sqrt((agent_x - food_x)**2 + (agent_y - food_y)**2)
                min_distances[i] = min(min_distances[i], distance)
                
                # Check if food was reached
                collision_distance = sim_config.agent_radius + test_cases[i].food.radius
                if distance <= collision_distance and not reached[i]:
                    reached[i] = True
                    step_reached[i] = step
    
    # Calculate scores using the scoring function
    scores = []
    for i in range(batch_size):
        score = _calculate_test_case_score(
            reached[i], step_reached[i], min_distances[i], 
            max_steps_list[i], max_range
        )
        scores.append(score)
    
    return scores, reached.tolist()


def _calculate_test_case_score(
    reached: bool,
    step_reached: int,
    min_distance: float,
    max_steps: int,
    max_range: float
) -> float:
    """
    Calculate score for a single test case
    
    Scoring function:
    - If reached: score = 1.0 + (max_steps - step_reached) / max_steps  # in [1, 2]
    - Else: score = 0.25 * (1.0 - clip(min_dist / max_range, 0, 1))     # in [0, 0.25]
    
    Args:
        reached: Whether food was reached
        step_reached: Step when food was reached (-1 if not reached)
        min_distance: Minimum distance achieved to food
        max_steps: Maximum steps allowed for this test case
        max_range: Maximum range for proximity scoring
        
    Returns:
        Score for this test case
    """
    if reached:
        # Success bonus + efficiency bonus
        efficiency_bonus = (max_steps - step_reached) / max_steps
        score = 1.0 + efficiency_bonus  # Range: [1.0, 2.0]
    else:
        # Partial credit based on proximity
        normalized_distance = np.clip(min_distance / max_range, 0.0, 1.0)
        proximity_score = 1.0 - normalized_distance
        score = 0.25 * proximity_score  # Range: [0.0, 0.25]
    
    return float(score)


def evaluate_suite_summary(suite: TestSuite, per_case_scores: np.ndarray, 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for suite evaluation
    
    Args:
        suite: Test suite that was evaluated
        per_case_scores: Scores for each test case
        metadata: Metadata from evaluation
        
    Returns:
        Dict with summary statistics
    """
    reached_flags = metadata['per_case_reached']
    
    # Overall statistics
    summary = {
        'suite_id': suite.suite_id,
        'suite_version': suite.version,
        'num_cases': len(suite),
        'fitness_mean': float(np.mean(per_case_scores)),
        'fitness_std': float(np.std(per_case_scores)),
        'fitness_min': float(np.min(per_case_scores)),
        'fitness_max': float(np.max(per_case_scores)),
        'success_rate': metadata['success_rate'],
        'num_reached': metadata['num_reached'],
        'num_failed': len(suite) - metadata['num_reached']
    }
    
    # Per-category analysis (if test case IDs follow naming conventions)
    categories = {}
    for i, case in enumerate(suite.test_cases):
        category = case.id.split('_')[0]  # e.g., "center", "corner", "wall"
        if category not in categories:
            categories[category] = {'scores': [], 'reached': []}
        categories[category]['scores'].append(per_case_scores[i])
        categories[category]['reached'].append(reached_flags[i])
    
    category_stats = {}
    for category, data in categories.items():
        scores = np.array(data['scores'])
        reached = np.array(data['reached'])
        category_stats[category] = {
            'count': len(scores),
            'mean_score': float(np.mean(scores)),
            'success_rate': float(np.mean(reached)),
            'num_reached': int(np.sum(reached))
        }
    
    summary['category_stats'] = category_stats
    
    return summary