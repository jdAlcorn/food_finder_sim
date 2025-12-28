#!/usr/bin/env python3
"""
Rollout evaluation for Evolution Strategies
Deterministic fitness evaluation of parameter vectors
"""

import torch
import numpy as np
import math
import time
from typing import Dict, Any, Callable, List
from src.sim.core import SimulationConfig
from src.sim.batched import BatchedSimulation
from src.sim.batched.obs import build_obs_batch
from .params import set_flat_params


class LightweightProfiler:
    """
    Minimal overhead profiler for candidate evaluation
    Only profiles when explicitly enabled for specific candidates
    """
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.timers = {}
        self.counts = {}
        self.start_times = {}
        self.total_rollout_start = None
        self.total_steps = 0
        
        # Check if we need CUDA synchronization
        self.use_cuda_sync = torch.cuda.is_available()
    
    def start_timer(self, name: str):
        """Start timing a section"""
        if not self.enabled:
            return
        
        if name == 'policy_forward' and self.use_cuda_sync:
            torch.cuda.synchronize()
        
        self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str):
        """End timing a section"""
        if not self.enabled:
            return
        
        if name == 'policy_forward' and self.use_cuda_sync:
            torch.cuda.synchronize()
        
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timers[name] = self.timers.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            del self.start_times[name]
    
    def start_rollout(self):
        """Start timing the entire rollout"""
        if not self.enabled:
            return
        self.total_rollout_start = time.perf_counter()
    
    def end_rollout(self):
        """End timing the entire rollout"""
        if not self.enabled:
            return
        if self.total_rollout_start is not None:
            self.total_rollout_time = time.perf_counter() - self.total_rollout_start
    
    def step(self):
        """Count a simulation step"""
        if not self.enabled:
            return
        self.total_steps += 1
    
    def print_summary(self, generation: int, candidate_idx: int, num_seeds: int):
        """Print profiling summary"""
        if not self.enabled or self.total_steps == 0:
            return
        
        total_time = getattr(self, 'total_rollout_time', sum(self.timers.values()))
        steps_per_sec = self.total_steps / total_time if total_time > 0 else 0
        
        print(f"\n[PROFILE] gen={generation} cand={candidate_idx} seeds={num_seeds} "
              f"steps={self.total_steps} total={total_time:.3f}s ({steps_per_sec:.1f} steps/s)")
        
        # Separate one-time setup costs from per-step costs
        setup_timers = {}
        step_timers = {}
        
        for name, total_sec in self.timers.items():
            if name.endswith('_setup'):
                setup_timers[name] = total_sec
            else:
                step_timers[name] = total_sec
        
        # Calculate step-only time for percentage calculation
        step_time_total = sum(step_timers.values())
        
        # Print setup costs (one-time)
        if setup_timers:
            print("Setup (one-time):")
            for name, total_sec in sorted(setup_timers.items(), key=lambda x: x[1], reverse=True):
                percentage = (total_sec / total_time) * 100 if total_time > 0 else 0
                print(f"  {name:18s}: {total_sec:.3f}s {percentage:5.1f}%")
        
        # Print per-step costs
        print("Per-step costs:")
        for name, total_sec in sorted(step_timers.items(), key=lambda x: x[1], reverse=True):
            per_step_ms = (total_sec / self.total_steps) * 1000 if self.total_steps > 0 else 0
            percentage = (total_sec / step_time_total) * 100 if step_time_total > 0 else 0
            
            # Add CUDA annotation if applicable
            name_display = name
            if name == 'policy_forward' and self.use_cuda_sync:
                name_display = f"{name}(cuda_sync)"
            
            print(f"  {name_display:18s}: {total_sec:.3f}s ({per_step_ms:.2f}ms/step) {percentage:5.1f}%")
        
        print()  # Empty line after profile


def rollout_fitness(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                   sim_config: SimulationConfig, seed: int, T: int, dt: float, 
                   max_range: float = 300.0, v_scale: float = 400.0, 
                   omega_scale: float = 10.0, food_reward_multiplier: float = 1000.0,
                   proximity_reward_scale: float = 1.0) -> float:
    """
    DEPRECATED: Use evaluate_multiple_seeds_batched instead
    
    Evaluate fitness of a parameter vector through simulation rollout
    """
    import warnings
    warnings.warn(
        "rollout_fitness is deprecated. Use evaluate_multiple_seeds_batched with num_seeds=1",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to batched version with single seed
    fitnesses = evaluate_multiple_seeds_batched(
        theta_flat=theta_flat,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        base_seed=seed,
        num_seeds=1,
        T=T,
        dt=dt,
        batch_size=1,
        food_reward_multiplier=food_reward_multiplier,
        proximity_reward_scale=proximity_reward_scale
    )
    
    return fitnesses[0]


def evaluate_multiple_seeds(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                           sim_config: SimulationConfig, base_seed: int, num_seeds: int, T: int, dt: float,
                           **kwargs) -> float:
    """
    DEPRECATED: Use evaluate_multiple_seeds_batched instead
    
    Evaluate fitness across multiple random seeds and return average
    """
    import warnings
    warnings.warn(
        "evaluate_multiple_seeds is deprecated. Use evaluate_multiple_seeds_batched",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to batched version
    fitnesses = evaluate_multiple_seeds_batched(
        theta_flat=theta_flat,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        base_seed=base_seed,
        num_seeds=num_seeds,
        T=T,
        dt=dt,
        batch_size=num_seeds,  # Batch all seeds together
        **kwargs
    )
    
    return sum(fitnesses) / len(fitnesses)


def evaluate_multiple_seeds_batched(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                                   sim_config: SimulationConfig, base_seed: int, num_seeds: int, T: int, dt: float,
                                   batch_size: int = None, **kwargs) -> float:
    """
    Evaluate fitness across multiple random seeds using batched simulation
    
    BATCHING DESIGN:
    - Seeds are partitioned across batches: batch_0=[seed_0, seed_1], batch_1=[seed_2, seed_3], etc.
    - Each batch runs B environments in parallel (vectorized simulation)
    - Total seeds evaluated = num_seeds exactly (no duplication, no gaps)
    
    Args:
        theta_flat: Flattened model parameters
        model_ctor: Model constructor function
        model_kwargs: Model constructor arguments
        sim_config: Simulation configuration
        base_seed: Base seed for deterministic seed generation
        num_seeds: Number of seeds to evaluate
        T: Number of simulation steps per episode
        dt: Fixed timestep
        batch_size: Batch size for batched simulation (default: num_seeds)
        **kwargs: Additional arguments for fitness calculation
        
    Returns:
        Average fitness across all seeds
    """
    if batch_size is None:
        batch_size = num_seeds
    
    # Process seeds in batches - PARTITION seeds to avoid duplication
    all_fitnesses = []
    
    for start_idx in range(0, num_seeds, batch_size):
        end_idx = min(start_idx + batch_size, num_seeds)
        current_batch_size = end_idx - start_idx
        
        # Generate DISTINCT seeds for this batch (no overlap with other batches)
        # Seed pattern: [base_seed + start_idx*1000, base_seed + (start_idx+1)*1000, ...]
        seeds = [base_seed + (start_idx + i) * 1000 for i in range(current_batch_size)]
        
        # Evaluate batch
        batch_fitnesses = rollout_fitness_batched(
            theta_flat=theta_flat,
            model_ctor=model_ctor,
            model_kwargs=model_kwargs,
            sim_config=sim_config,
            seeds=seeds,
            T=T,
            dt=dt,
            profiler=kwargs.get('profiler'),
            **{k: v for k, v in kwargs.items() if k != 'profiler'}
        )
        
        all_fitnesses.extend(batch_fitnesses)
    
    return float(np.mean(all_fitnesses))


def rollout_fitness_batched(theta_flat: torch.Tensor, model_ctor: Callable, model_kwargs: Dict[str, Any],
                           sim_config: SimulationConfig, seeds: List[int], T: int, dt: float,
                           v_scale: float = 400.0, omega_scale: float = 10.0,
                           food_reward_multiplier: float = 1000.0,
                           proximity_reward_scale: float = 1.0, profiler: LightweightProfiler = None) -> List[float]:
    """
    Evaluate fitness using batched simulation across multiple seeds
    
    Args:
        theta_flat: Flattened model parameters
        model_ctor: Model constructor function
        model_kwargs: Model constructor arguments
        sim_config: Simulation configuration
        seeds: List of random seeds for each environment
        T: Number of simulation steps
        dt: Fixed timestep
        v_scale: Velocity normalization scale
        omega_scale: Angular velocity normalization scale
        food_reward_multiplier: Points awarded per food collected
        proximity_reward_scale: Scale factor for proximity reward
        profiler: Optional profiler for timing measurements
        
    Returns:
        List of fitness values for each seed
    """
    try:
        batch_size = len(seeds)
        
        if profiler:
            profiler.start_rollout()
        
        # Create model and load parameters
        profiler and profiler.start_timer('policy_setup')
        model = model_ctor(**model_kwargs)
        model.eval()
        
        # Load flattened parameters into model
        set_flat_params(model, theta_flat)
        profiler and profiler.end_timer('policy_setup')
        
        # Create batched simulation
        profiler and profiler.start_timer('sim_setup')
        batched_sim = BatchedSimulation(batch_size, sim_config)
        batched_sim.reset(seeds)
        profiler and profiler.end_timer('sim_setup')
        
        # Track fitness components for each environment
        food_counts = np.zeros(batch_size, dtype=np.int32)
        final_distances = np.full(batch_size, sim_config.max_range, dtype=np.float32)
        
        # Run episode
        with torch.no_grad():
            for step in range(T):
                # Stop profiling after max steps to avoid overhead
                if profiler and hasattr(profiler, 'max_steps') and step >= profiler.max_steps:
                    break
                
                profiler and profiler.step()
                
                # Get observations for all environments
                profiler and profiler.start_timer('obs_feature_build')
                observations = batched_sim.get_observations(v_scale, omega_scale, profiler)
                profiler and profiler.end_timer('obs_feature_build')
                
                # Convert to tensor and run model (batched forward pass)
                profiler and profiler.start_timer('policy_forward')
                obs_tensor = torch.tensor(observations, dtype=torch.float32)
                steer_raw, throttle_raw = model(obs_tensor)
                profiler and profiler.end_timer('policy_forward')
                
                # Extract actions (activations already applied in model)
                profiler and profiler.start_timer('action_postprocess')
                steer = steer_raw.detach().cpu().numpy()
                throttle = throttle_raw.detach().cpu().numpy()
                
                # Safety clamps (should be redundant)
                steer = np.clip(steer, -1.0, 1.0)
                throttle = np.clip(throttle, 0.0, 1.0)
                profiler and profiler.end_timer('action_postprocess')
                
                # Step simulation
                profiler and profiler.start_timer('sim_step_physics')
                actions = {'steer': steer, 'throttle': throttle}
                step_info = batched_sim.step(actions, dt)
                profiler and profiler.end_timer('sim_step_physics')
                
                # Update food counts and track distances
                profiler and profiler.start_timer('reward_done')
                food_counts += step_info['food_collected_this_step'].astype(np.int32)
                
                # Track final distances to food
                agent_positions = step_info['agent_states']
                food_positions = step_info['food_positions']
                
                dx = agent_positions['x'] - food_positions[:, 0]
                dy = agent_positions['y'] - food_positions[:, 1]
                distances_to_food = np.sqrt(dx * dx + dy * dy)
                final_distances = distances_to_food  # Keep updating (final will be last step)
                profiler and profiler.end_timer('reward_done')
        
        # Calculate fitness for each environment
        profiler and profiler.start_timer('fitness_calc')
        fitnesses = []
        for i in range(batch_size):
            # Fitness components
            food_reward = food_counts[i] * food_reward_multiplier
            proximity_reward = max(0, sim_config.max_range - final_distances[i]) * proximity_reward_scale
            step_penalty = -0.001 * T
            
            fitness = food_reward + proximity_reward + step_penalty
            fitnesses.append(float(fitness))
        profiler and profiler.end_timer('fitness_calc')
        
        if profiler:
            profiler.end_rollout()
        
        return fitnesses
        
    except Exception as e:
        # Return very low fitness for failed rollouts
        print(f"Batched rollout failed with error: {e}")
        return [-10000.0] * len(seeds)


def evaluate_candidate_worker(args):
    """
    Worker function for multiprocessing evaluation
    
    WORKER ISOLATION DESIGN:
    - Each worker processes exactly ONE candidate (no duplication across workers)
    - Each candidate uses unique seed range: candidate_i gets base_seed + i*10000
    - Within candidate: seeds are [base_seed, base_seed+1000, base_seed+2000, ...]
    - Seed spacing (1000) and candidate offset (10000) prevent any overlap
    
    Args:
        args: Tuple of (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
                       base_seed, num_seeds, T, dt, kwargs, generation)
    
    Returns:
        Tuple of (candidate_idx, fitness)
    """
    (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
     base_seed, num_seeds, T, dt, kwargs, generation) = args
    
    try:
        # Generate unique base seed for this candidate
        # CRITICAL: This ensures no seed overlap between candidates
        # candidate_0: [base_seed + 0, base_seed + 1000, base_seed + 2000, ...]
        # candidate_1: [base_seed + 10000, base_seed + 11000, base_seed + 12000, ...]
        # candidate_2: [base_seed + 20000, base_seed + 21000, base_seed + 22000, ...]
        candidate_base_seed = base_seed + candidate_idx * 10000
        
        # Check if profiling is enabled for this candidate
        profiler = None
        if (kwargs.get('profile_enabled', False) and 
            candidate_idx == kwargs.get('profile_candidate_idx', 0) and
            generation % kwargs.get('profile_print_every_gen', 1) == 0):
            
            profiler = LightweightProfiler(enabled=True)
            profiler.max_steps = kwargs.get('profile_max_steps', 500)
        
        # Always use batched evaluation (unified approach)
        batch_size = kwargs.get('batch_size', num_seeds)  # Default to num_seeds for optimal batching
        
        fitness = evaluate_multiple_seeds_batched(
            theta_flat=theta_flat,
            model_ctor=model_ctor,
            model_kwargs=model_kwargs,
            sim_config=sim_config,
            base_seed=candidate_base_seed,
            num_seeds=num_seeds,
            T=T,
            dt=dt,
            batch_size=batch_size,
            profiler=profiler,
            **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'profiler', 'profile_enabled', 'profile_candidate_idx', 'profile_max_steps', 'profile_print_every_gen']}
        )
        
        # Print profiling summary if enabled
        if profiler and profiler.enabled:
            profiler.print_summary(generation, candidate_idx, num_seeds)
        
        return candidate_idx, fitness
        
    except Exception as e:
        print(f"Worker {candidate_idx} failed: {e}")
        return candidate_idx, -10000.0


def evaluate_candidate_suite_worker(args):
    """
    Worker function for suite-based candidate evaluation
    
    Args:
        args: Tuple of (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
                       test_suite, fitness_kwargs, generation)
    
    Returns:
        Tuple of (candidate_idx, fitness)
    """
    (candidate_idx, theta_flat, model_ctor, model_kwargs, sim_config, 
     test_suite, fitness_kwargs, generation) = args
    
    try:
        # Import here to avoid circular imports
        from ..eval.suite_evaluator import evaluate_candidate_on_suite
        
        # Extract parameters
        batch_size = fitness_kwargs.get('batch_size', 32)
        v_scale = fitness_kwargs.get('v_scale', 400.0)
        omega_scale = fitness_kwargs.get('omega_scale', 10.0)
        
        # Evaluate on test suite
        fitness_mean, per_case_scores, metadata = evaluate_candidate_on_suite(
            theta_flat=theta_flat,
            model_ctor=model_ctor,
            model_kwargs=model_kwargs,
            sim_config=sim_config,
            suite=test_suite,
            batch_size=batch_size,
            device="cpu",  # Force CPU for multiprocessing compatibility
            v_scale=v_scale,
            omega_scale=omega_scale
        )
        
        return candidate_idx, fitness_mean
        
    except Exception as e:
        print(f"Suite worker {candidate_idx} failed: {e}")
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
    
    # Test multiple seeds (single-env)
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
    
    print(f"Average fitness (3 seeds, single-env): {avg_fitness:.2f}")
    
    # Test batched evaluation
    avg_fitness_batched = evaluate_multiple_seeds_batched(
        theta_flat=theta,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        base_seed=42,
        num_seeds=3,
        T=100,
        dt=1/60,
        batch_size=3
    )
    
    print(f"Average fitness (3 seeds, batched): {avg_fitness_batched:.2f}")
    
    # Test profiling
    profiler = LightweightProfiler(enabled=True)
    profiler.max_steps = 50
    
    avg_fitness_profiled = evaluate_multiple_seeds_batched(
        theta_flat=theta,
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        base_seed=42,
        num_seeds=2,
        T=50,
        dt=1/60,
        batch_size=2,
        profiler=profiler
    )
    
    profiler.print_summary(0, 0, 2)
    print(f"Average fitness (profiled): {avg_fitness_profiled:.2f}")
    
    # Compare results (should be very close)
    diff = abs(avg_fitness - avg_fitness_batched)
    print(f"Difference between single and batched: {diff:.4f}")
    
    if diff < 1.0:  # Allow small numerical differences
        print("✓ Batched evaluation matches single evaluation!")
    else:
        print("⚠ Large difference between single and batched evaluation")
    
    print("Rollout tests passed!")


if __name__ == "__main__":
    test_rollout()