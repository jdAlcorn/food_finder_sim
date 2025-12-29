#!/usr/bin/env python3
"""
Test suite evaluator for deterministic fitness evaluation
"""

import torch
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Callable
from .testcases import TestSuite, TestCase
from .world_integration import setup_simulation_with_worlds
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
    max_range: float = None,
    fail_weight: float = 0.20,
    proximity_scale: float = 25.0,
    exploration_weight: float = 0.1,
    exploration_cell_size: float = 50.0,
    exploration_max_bonus: float = 100.0,
    exploration_movement_threshold: float = 2.0
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
        fail_weight: Weight for progress score on failed test cases (default: 0.20)
        proximity_scale: Scale parameter for exponential proximity reward (default: 15.0)
        
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
    all_diagnostics = []
    
    test_cases = list(suite.test_cases)
    
    for batch_start in range(0, len(test_cases), batch_size):
        batch_end = min(batch_start + batch_size, len(test_cases))
        batch_cases = test_cases[batch_start:batch_end]
        current_batch_size = len(batch_cases)
        
        # Evaluate this batch
        batch_scores, batch_reached, batch_diagnostics = _evaluate_test_case_batch(
            model, batch_cases, sim_config, dt, current_batch_size,
            device, v_scale, omega_scale, max_range, fail_weight, proximity_scale,
            exploration_weight, exploration_cell_size, exploration_max_bonus, exploration_movement_threshold
        )
        
        all_scores.extend(batch_scores)
        all_reached.extend(batch_reached)
        all_diagnostics.append(batch_diagnostics)
    
    # Compute overall statistics
    per_case_scores = np.array(all_scores)
    fitness_mean = float(np.mean(per_case_scores))
    
    # Aggregate diagnostics across batches
    total_passes = sum(d['passes_count'] for d in all_diagnostics)
    all_pass_times = []
    all_fail_progresses = []
    all_min_dist_ratios = []
    all_min_distances = []
    
    for d in all_diagnostics:
        if d['mean_pass_time'] is not None:
            # Reconstruct individual pass times (approximation for logging)
            all_pass_times.extend([d['mean_pass_time']] * d['passes_count'])
        if d['mean_fail_progress'] is not None:
            fail_count = len(test_cases) // len(all_diagnostics) - d['passes_count']
            all_fail_progresses.extend([d['mean_fail_progress']] * fail_count)
        if d['mean_min_dist_ratio'] is not None:
            fail_count = len(test_cases) // len(all_diagnostics) - d['passes_count']
            all_min_dist_ratios.extend([d['mean_min_dist_ratio']] * fail_count)
        
        # Collect all min distances for per-case analysis
        all_min_distances.extend(d['min_distances'])
    
    metadata = {
        'per_case_reached': np.array(all_reached),
        'per_case_scores': per_case_scores,
        'per_case_min_distances': np.array(all_min_distances),
        'num_reached': int(np.sum(all_reached)),
        'num_total': len(test_cases),
        'success_rate': float(np.mean(all_reached)),
        # Dense fitness diagnostics
        'passes_count': total_passes,
        'mean_pass_time': float(np.mean(all_pass_times)) if all_pass_times else None,
        'mean_fail_progress': float(np.mean(all_fail_progresses)) if all_fail_progresses else None,
        'mean_min_dist_ratio': float(np.mean(all_min_dist_ratios)) if all_min_dist_ratios else None,
        'fail_weight': fail_weight
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
    max_range: float,
    fail_weight: float = 0.20,
    proximity_scale: float = 25.0,
    exploration_weight: float = 0.1,
    exploration_cell_size: float = 50.0,
    exploration_max_bonus: float = 100.0,
    exploration_movement_threshold: float = 2.0
) -> Tuple[List[float], List[bool], Dict[str, Any]]:
    """
    Evaluate a batch of test cases with dense progress-based fitness
    Supports both feedforward (MLP) and recurrent (GRU) models
    
    Args:
        fail_weight: Weight for progress score on failed test cases (default: 0.20)
        proximity_scale: Scale parameter for exponential proximity reward (default: 15.0)
    
    Returns:
        Tuple of (scores, reached_flags, diagnostics)
    """
    # Check if model is recurrent (has init_hidden method)
    is_recurrent = hasattr(model, 'init_hidden')
    hidden_state = None
    
    if is_recurrent:
        # Initialize hidden state for recurrent model
        hidden_state = model.init_hidden(batch_size, device=torch.device(device))
    
    # Create batched simulation with world support
    batched_sim, updated_sim_config = setup_simulation_with_worlds(test_cases, sim_config)
    
    # Configure exploration parameters
    batched_sim.exploration_weight = exploration_weight
    batched_sim.exploration_cell_size = exploration_cell_size
    batched_sim.exploration_max_bonus = exploration_max_bonus
    batched_sim.exploration_movement_threshold = exploration_movement_threshold
    
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
    
    # Calculate initial distances (d0) for progress tracking
    initial_distances = np.zeros(batch_size, dtype=float)
    for i in range(batch_size):
        agent_x = agent_states[i]['x']
        agent_y = agent_states[i]['y']
        food_x = food_states[i]['x']
        food_y = food_states[i]['y']
        initial_distances[i] = math.sqrt((agent_x - food_x)**2 + (agent_y - food_y)**2)
    
    # Track evaluation metrics
    reached = np.zeros(batch_size, dtype=bool)
    step_reached = np.full(batch_size, -1, dtype=int)
    min_distances = initial_distances.copy()  # Initialize with d0
    final_distances = np.zeros(batch_size, dtype=float)
    exploration_scores = np.zeros(batch_size, dtype=float)
    unique_cells_visited = np.zeros(batch_size, dtype=int)
    
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
            
            if is_recurrent:
                # Recurrent model: forward pass with hidden state
                steer_raw, throttle_raw, hidden_state = model(obs_tensor, hidden_state)
                # Detach hidden state to prevent gradient accumulation
                hidden_state = hidden_state.detach()
            else:
                # Feedforward model: standard forward pass
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
                final_distances[i] = distance  # Track final distance for diagnostics
                
                # Check if food was reached
                collision_distance = sim_config.agent_radius + test_cases[i].food.radius
                if distance <= collision_distance and not reached[i]:
                    reached[i] = True
                    step_reached[i] = step
    
    # Get final exploration metrics
    final_step_info = batched_sim.get_state()
    exploration_scores = final_step_info['exploration_scores']
    unique_cells_visited = final_step_info['unique_cells_visited']
    
    # Calculate scores using exponential proximity for fail cases
    scores = []
    pass_times = []
    fail_progresses = []
    min_dist_ratios = []
    
    eps = 1e-6  # Avoid divide-by-zero
    
    for i in range(batch_size):
        if reached[i]:
            # Pass scoring: 1.0 + efficiency bonus [1.0, 2.0]
            efficiency_bonus = (max_steps_list[i] - step_reached[i]) / max_steps_list[i]
            base_score = 1.0 + efficiency_bonus
        else:
            # Fail scoring: exponential proximity reward [0.0, fail_weight]
            # 
            # WHY exponential proximity:
            # - Relative progress (1 - min_dist/initial_dist) saturates near 1.0 when 
            #   initial distances are large, collapsing fitness variance
            # - Exponential proximity exp(-min_dist/scale) increases resolution near the goal,
            #   giving meaningful fitness differences for small improvements in navigation
            min_dist_clamped = max(min_distances[i], 0.0)  # Ensure non-negative
            proximity = np.exp(-min_dist_clamped / proximity_scale)
            base_score = fail_weight * proximity
            
            # Preserve existing bookkeeping for logging/debugging
            d0 = max(initial_distances[i], eps)
            progress = np.clip(1.0 - (min_distances[i] / d0), 0.0, 1.0)
            fail_progresses.append(progress)
            min_dist_ratios.append(min_distances[i] / d0)
        
        # Add exploration bonus to both pass and fail cases
        total_score = base_score + exploration_scores[i]
        scores.append(total_score)
        
        # Track pass times for successful cases
        if reached[i]:
            pass_times.append(step_reached[i])
    
    # Compile diagnostics
    diagnostics = {
        'passes_count': int(np.sum(reached)),
        'mean_pass_time': float(np.mean(pass_times)) if pass_times else None,
        'mean_fail_progress': float(np.mean(fail_progresses)) if fail_progresses else None,
        'mean_min_dist_ratio': float(np.mean(min_dist_ratios)) if min_dist_ratios else None,
        'initial_distances': initial_distances.tolist(),
        'min_distances': min_distances.tolist(),
        'final_distances': final_distances.tolist(),
        'exploration_scores': exploration_scores.tolist(),
        'unique_cells_visited': unique_cells_visited.tolist(),
        'mean_exploration_score': float(np.mean(exploration_scores)),
        'mean_unique_cells': float(np.mean(unique_cells_visited))
    }
    
    return scores, reached.tolist(), diagnostics


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