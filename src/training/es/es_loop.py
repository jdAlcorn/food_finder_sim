#!/usr/bin/env python3
"""
Evolution Strategies training loop
Simple ES implementation with multiprocessing support
"""

import torch
import numpy as np
import multiprocessing as mp
from typing import Dict, Any, Callable, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from src.sim.core import SimulationConfig
from .rollout import evaluate_candidate_worker, evaluate_candidate_suite_worker


class EvolutionStrategiesTrainer:
    """
    Simple Evolution Strategies trainer
    """
    
    def __init__(self, model_ctor: Callable, model_kwargs: Dict[str, Any], 
                 sim_config: SimulationConfig, pop_size: int = 128, sigma: float = 0.02,
                 alpha: float = 0.01, T: int = 600, dt: float = 1/60, seed0: int = 42,
                 num_workers: int = None, eval_seeds_per_candidate: int = 1, 
                 antithetic: bool = True, fitness_kwargs: Dict[str, Any] = None,
                 eval_batch_size: int = None, eval_mode: str = "suite", 
                 test_suite_path: str = None):
        """
        Initialize ES trainer
        
        Args:
            model_ctor: Model constructor function
            model_kwargs: Model constructor arguments
            sim_config: Simulation configuration
            pop_size: Population size
            sigma: Noise standard deviation
            alpha: Learning rate
            T: Episode length in steps
            dt: Fixed timestep
            seed0: Base random seed
            num_workers: Number of worker processes (default: cpu_count//2)
            eval_seeds_per_candidate: Number of seeds to evaluate per candidate (respawn mode only)
            antithetic: Whether to use antithetic sampling
            fitness_kwargs: Additional arguments for fitness function
            eval_batch_size: Batch size for batched evaluation (default: eval_seeds_per_candidate)
            eval_mode: Evaluation mode - "suite" or "respawn" (default: "suite")
            test_suite_path: Path to test suite JSON file (required for suite mode)
        """
        self.model_ctor = model_ctor
        self.model_kwargs = model_kwargs
        self.sim_config = sim_config
        self.pop_size = pop_size
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.dt = dt
        self.seed0 = seed0
        self.eval_seeds_per_candidate = eval_seeds_per_candidate
        self.antithetic = antithetic
        self.fitness_kwargs = fitness_kwargs or {}
        self.eval_batch_size = eval_batch_size or eval_seeds_per_candidate
        self.eval_mode = eval_mode
        
        # Load test suite if using suite mode
        if eval_mode == "suite":
            if test_suite_path is None:
                test_suite_path = "data/test_suites/basic_v1.json"
            
            from src.eval.load_suite import load_suite
            self.test_suite = load_suite(test_suite_path)
            print(f"Loaded test suite: {self.test_suite.suite_id} v{self.test_suite.version}")
            print(f"  Test cases: {len(self.test_suite)}")
            print(f"  Description: {self.test_suite.description}")
            
            # For suite mode, batch size is for test case batching
            self.fitness_kwargs['batch_size'] = self.eval_batch_size
        else:
            # Respawn mode - always use batched evaluation
            self.test_suite = None
            self.fitness_kwargs['batch_size'] = self.eval_batch_size
        
        # Set up multiprocessing
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() // 2)
        self.num_workers = num_workers
        
        # Initialize persistent executor for multiprocessing
        # Create once and reuse across all generations to avoid spawn overhead
        self.executor = None
        if self.num_workers > 1:
            import time
            start_time = time.time()
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
            executor_time = time.time() - start_time
            print(f"  Executor creation time: {executor_time:.3f}s")
        
        # Initialize random state
        self.rng = np.random.RandomState(seed0)
        
        # Get parameter count
        dummy_model = model_ctor(**model_kwargs)
        from .params import count_parameters
        self.param_count = count_parameters(dummy_model)
        
        print(f"ES Trainer initialized:")
        print(f"  Population size: {pop_size}")
        print(f"  Parameter count: {self.param_count}")
        print(f"  Sigma: {sigma}, Alpha: {alpha}")
        print(f"  Episode length: {T} steps, dt: {dt:.4f}")
        print(f"  Workers: {num_workers}")
        print(f"  Evaluation mode: {eval_mode}")
        if eval_mode == "respawn":
            print(f"  Seeds per candidate: {eval_seeds_per_candidate}")
        print(f"  Antithetic sampling: {antithetic}")
        print(f"  Batched evaluation: Always enabled")
        print(f"  Eval batch size: {self.eval_batch_size}")
        print(f"  Persistent executor: {'Yes' if self.executor else 'No (single-threaded)'}")
    
    def step(self, theta_flat: torch.Tensor, iteration: int) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, float]:
        """
        Single ES training step
        
        Args:
            theta_flat: Current parameter vector
            iteration: Current iteration number (for seed generation)
            
        Returns:
            Tuple of (new_theta, stats_dict, best_candidate_theta, best_fitness)
        """
        # Ensure theta is on CPU
        theta_flat = theta_flat.detach().cpu()
        
        # Sample perturbations
        if self.antithetic:
            # Sample half the population, then mirror
            half_pop = self.pop_size // 2
            eps_half = self.rng.randn(half_pop, self.param_count).astype(np.float32)
            eps = np.vstack([eps_half, -eps_half])
            if self.pop_size % 2 == 1:
                # Add one more random sample if odd population size
                eps_extra = self.rng.randn(1, self.param_count).astype(np.float32)
                eps = np.vstack([eps, eps_extra])
        else:
            eps = self.rng.randn(self.pop_size, self.param_count).astype(np.float32)
        
        # Generate candidate parameters
        candidates = []
        for i in range(self.pop_size):
            candidate_theta = theta_flat + self.sigma * torch.from_numpy(eps[i])
            candidates.append(candidate_theta)
        
        # Evaluate candidates in parallel
        results = self._evaluate_candidates_parallel(candidates, iteration)
        
        # Extract fitnesses and diagnostics
        fitnesses = []
        all_diagnostics = []
        for result in results:
            if len(result) == 3:  # Suite mode with diagnostics
                candidate_idx, fitness, diagnostics = result
                fitnesses.append(fitness)
                all_diagnostics.append((candidate_idx, diagnostics))
            else:  # Respawn mode or old format
                candidate_idx, fitness = result
                fitnesses.append(fitness)
        
        # Convert to numpy for ES update
        fitnesses = np.array(fitnesses)
        
        # Normalize fitness (fitness shaping)
        fitness_mean = np.mean(fitnesses)
        fitness_std = np.std(fitnesses) + 1e-8
        fitness_normalized = (fitnesses - fitness_mean) / fitness_std
        
        # ES gradient estimate
        gradient = np.dot(eps.T, fitness_normalized) / (self.pop_size * self.sigma)
        
        # Update parameters
        theta_new = theta_flat + self.alpha * torch.from_numpy(gradient.astype(np.float32))
        
        # Track best candidate
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_candidate_theta = candidates[best_idx]
        
        # Get diagnostics for best candidate (suite mode only)
        best_diagnostics = {}
        if all_diagnostics:
            for candidate_idx, diagnostics in all_diagnostics:
                if candidate_idx == best_idx:
                    best_diagnostics = diagnostics
                    break
        
        # Compile statistics
        stats = {
            'fitness_mean': float(fitness_mean),
            'fitness_std': float(fitness_std),
            'fitness_max': float(np.max(fitnesses)),
            'fitness_min': float(np.min(fitnesses)),
            'best_fitness': float(best_fitness),
            'gradient_norm': float(np.linalg.norm(gradient)),
            'param_norm': float(torch.norm(theta_flat).item()),
            'sigma': self.sigma,
            'alpha': self.alpha
        }
        
        # Add suite diagnostics if available
        if best_diagnostics:
            stats.update({
                'passes_count': best_diagnostics.get('passes_count', 0),
                'mean_pass_time': best_diagnostics.get('mean_pass_time'),
                'mean_fail_progress': best_diagnostics.get('mean_fail_progress'),
                'mean_min_dist_ratio': best_diagnostics.get('mean_min_dist_ratio'),
                'fail_weight': best_diagnostics.get('fail_weight', 0.20)
            })
        
        return theta_new, stats, best_candidate_theta, best_fitness
    
    def _evaluate_candidates_parallel(self, candidates: List[torch.Tensor], iteration: int) -> List[float]:
        """
        Evaluate candidates in parallel using persistent multiprocessing executor
        
        PARALLELIZATION DESIGN:
        - WORKER LEVEL: Each candidate gets exactly one task (no duplication)
        - BATCH LEVEL: Within each candidate, seeds are batched for vectorized simulation
        - SEED ISOLATION: Candidates use non-overlapping seed ranges (10000-seed offset)
        
        Args:
            candidates: List of parameter vectors to evaluate
            iteration: Current iteration (for seed generation)
            
        Returns:
            List of fitness values
        """
        import time
        eval_start_time = time.time()
        
        if self.eval_mode == "suite":
            # Suite-based evaluation
            worker_args = []
            for i, candidate_theta in enumerate(candidates):
                args = (
                    i,  # candidate_idx
                    candidate_theta,
                    self.model_ctor,
                    self.model_kwargs,
                    self.sim_config,
                    self.test_suite,
                    self.fitness_kwargs,  # Contains batch_size for test case batching
                    self.dt,  # Add dt parameter
                    iteration  # Add generation number for logging
                )
                worker_args.append(args)
            
            # Evaluate using suite worker function
            if self.executor is None:
                # Single-threaded
                results = [evaluate_candidate_suite_worker(args) for args in worker_args]
            else:
                # Multi-threaded using persistent executor
                results = list(self.executor.map(evaluate_candidate_suite_worker, worker_args))
        
        else:
            # Respawn-based evaluation (original method)
            worker_args = []
            for i, candidate_theta in enumerate(candidates):
                # Generate unique base seed for this candidate and iteration
                base_seed = self.seed0 + iteration * 100000 + i * 10000
                
                args = (
                    i,  # candidate_idx
                    candidate_theta,
                    self.model_ctor,
                    self.model_kwargs,
                    self.sim_config,
                    base_seed,
                    self.eval_seeds_per_candidate,
                    self.T,
                    self.dt,
                    self.fitness_kwargs,
                    iteration
                )
                worker_args.append(args)
            
            # Evaluate using respawn worker function
            if self.executor is None:
                # Single-threaded
                results = [evaluate_candidate_worker(args) for args in worker_args]
            else:
                # Multi-threaded using persistent executor
                results = list(self.executor.map(evaluate_candidate_worker, worker_args))
        
        eval_time = time.time() - eval_start_time
        
        # Sort results by candidate index
        results.sort(key=lambda x: x[0])
        
        # Log timing (only occasionally to avoid spam)
        if iteration % 10 == 0:
            mode_info = f"suite ({len(self.test_suite)} cases)" if self.eval_mode == "suite" else f"respawn ({self.eval_seeds_per_candidate} seeds)"
            print(f"  Generation {iteration} evaluation time: {eval_time:.3f}s ({mode_info})")
        
        return results
    
    def update_hyperparameters(self, **kwargs):
        """Update hyperparameters during training"""
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
    
    def shutdown(self):
        """
        Clean shutdown of the trainer, including executor cleanup
        Should be called when training is complete or interrupted
        """
        if self.executor is not None:
            print("Shutting down worker pool...")
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None
            print("Worker pool shutdown complete.")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.shutdown()


def test_es_trainer():
    """Test ES trainer"""
    from src.policy.models.mlp import SimpleMLP
    from .params import get_flat_params
    
    # Create test setup
    model_ctor = SimpleMLP
    model_kwargs = {'input_dim': 388, 'hidden_dims': (64, 32), 'output_dim': 2}  # Smaller for testing
    sim_config = SimulationConfig()
    
    # Create trainer with context manager for proper cleanup
    with EvolutionStrategiesTrainer(
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        pop_size=8,  # Small for testing
        sigma=0.1,
        alpha=0.01,
        T=50,  # Short episodes for testing
        dt=1/60,
        num_workers=2,
        eval_seeds_per_candidate=1
    ) as trainer:
        
        # Initialize parameters
        model = model_ctor(**model_kwargs)
        theta = get_flat_params(model)
        
        print(f"Testing ES with {theta.numel()} parameters")
        
        # Run a few steps
        for i in range(3):
            print(f"\nStep {i+1}:")
            theta, stats, best_theta, best_fitness = trainer.step(theta, i)
            
            print(f"  Mean fitness: {stats['fitness_mean']:.2f}")
            print(f"  Best fitness: {stats['best_fitness']:.2f}")
            print(f"  Gradient norm: {stats['gradient_norm']:.4f}")
    
    print("ES trainer test passed!")


if __name__ == "__main__":
    test_es_trainer()