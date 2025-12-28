#!/usr/bin/env python3
"""
Evolution Strategies training script
CLI entrypoint for training neural network policies using ES
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sim.core import SimulationConfig
from src.policy.models.mlp import SimpleMLP
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.policy.checkpoint import save_policy, load_policy
from src.training.es.es_loop import EvolutionStrategiesTrainer
from src.training.es.params import get_flat_params


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train neural network policy using Evolution Strategies')
    
    # Training parameters
    parser.add_argument('--generations', type=int, default=1000, 
                       help='Number of ES generations to run (default: 1000)')
    parser.add_argument('--pop-size', type=int, default=128,
                       help='Population size (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.02,
                       help='Noise standard deviation (default: 0.02)')
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    
    # Episode parameters
    parser.add_argument('--episode-length', type=int, default=600,
                       help='Episode length in steps (default: 600)')
    parser.add_argument('--dt', type=float, default=1/60,
                       help='Fixed timestep (default: 1/60)')
    parser.add_argument('--eval-seeds', type=int, default=1,
                       help='Number of seeds per candidate evaluation (default: 1)')
    
    # Model parameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden layer dimensions (default: 256 128)')
    parser.add_argument('--v-scale', type=float, default=400.0,
                       help='Velocity normalization scale (default: 400.0)')
    parser.add_argument('--omega-scale', type=float, default=10.0,
                       help='Angular velocity normalization scale (default: 10.0)')
    
    # Fitness parameters
    parser.add_argument('--food-reward', type=float, default=1000.0,
                       help='Points awarded per food collected (default: 1000.0)')
    parser.add_argument('--proximity-reward-scale', type=float, default=1.0,
                       help='Scale factor for proximity reward (default: 1.0)')
    
    # System parameters
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: cpu_count//2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints (default: checkpoints)')
    parser.add_argument('--save-prefix', type=str, default='es_trained',
                       help='Prefix for saved checkpoint files (default: es_trained)')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save checkpoint every N generations (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Print stats every N generations (default: 10)')
    parser.add_argument('--csv-log', type=str, default=None,
                       help='CSV file to log training statistics')
    
    # ES parameters
    parser.add_argument('--antithetic', action='store_true', default=True,
                       help='Use antithetic sampling (default: True)')
    parser.add_argument('--no-antithetic', dest='antithetic', action='store_false',
                       help='Disable antithetic sampling')
    
    # Batched evaluation parameters (always enabled now)
    parser.add_argument('--eval-batch-size', type=int, default=None,
                       help='Batch size for batched evaluation (default: same as eval-seeds)')
    
    # Profiling parameters
    parser.add_argument('--profile-one-candidate-per-gen', action='store_true', default=False,
                       help='Enable lightweight profiling of one candidate per generation')
    parser.add_argument('--profile-candidate-idx', type=int, default=0,
                       help='Which candidate index to profile (default: 0)')
    parser.add_argument('--profile-max-steps', type=int, default=500,
                       help='Maximum steps to profile per candidate (default: 500)')
    parser.add_argument('--profile-print-every-gen', type=int, default=1,
                       help='Print profile every N generations (default: 1)')
    
    return parser.parse_args()


def create_model_constructor_and_kwargs(hidden_dims, v_scale, omega_scale):
    """Create model constructor and kwargs for ES trainer"""
    model_ctor = SimpleMLP
    model_kwargs = {
        'input_dim': 388,  # Fixed observation dimension
        'hidden_dims': tuple(hidden_dims),
        'output_dim': 2
    }
    return model_ctor, model_kwargs


def initialize_training(args) -> tuple:
    """Initialize training components"""
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create simulation config
    sim_config = SimulationConfig()
    
    # Create model constructor
    model_ctor, model_kwargs = create_model_constructor_and_kwargs(
        args.hidden_dims, args.v_scale, args.omega_scale
    )
    
    # Initialize parameters
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Load checkpoint and extract parameters
        policy, _, metadata = load_policy(args.resume)
        if not isinstance(policy, TorchMLPPolicy):
            raise ValueError(f"Resume checkpoint must be TorchMLP policy, got {type(policy)}")
        
        theta = get_flat_params(policy.model)
        start_generation = metadata.get('generation', 0)
        best_fitness_so_far = metadata.get('best_fitness', float('-inf'))
        
        print(f"Resumed from generation {start_generation}, best fitness: {best_fitness_so_far:.2f}")
    else:
        # Initialize random parameters
        model = model_ctor(**model_kwargs)
        theta = get_flat_params(model)
        start_generation = 0
        best_fitness_so_far = float('-inf')
        
        print(f"Initialized random parameters: {theta.numel()} parameters")
    
    # Create ES trainer
    trainer = EvolutionStrategiesTrainer(
        model_ctor=model_ctor,
        model_kwargs=model_kwargs,
        sim_config=sim_config,
        pop_size=args.pop_size,
        sigma=args.sigma,
        alpha=args.alpha,
        T=args.episode_length,
        dt=args.dt,
        seed0=args.seed,
        num_workers=args.workers,
        eval_seeds_per_candidate=args.eval_seeds,
        antithetic=args.antithetic,
        fitness_kwargs={
            'food_reward_multiplier': args.food_reward,
            'proximity_reward_scale': args.proximity_reward_scale,
            'profile_enabled': args.profile_one_candidate_per_gen,
            'profile_candidate_idx': args.profile_candidate_idx,
            'profile_max_steps': args.profile_max_steps,
            'profile_print_every_gen': args.profile_print_every_gen
        },
        eval_batch_size=args.eval_batch_size
    )
    
    return trainer, theta, start_generation, best_fitness_so_far, sim_config, model_ctor, model_kwargs


def save_checkpoint(args, generation, theta, best_theta, best_fitness, sim_config, 
                   model_ctor, model_kwargs, stats_history, is_best=False):
    """Save training checkpoint"""
    # Create policy instance with best parameters
    policy = TorchMLPPolicy(
        hidden_dims=tuple(args.hidden_dims),
        device='cpu',
        v_scale=args.v_scale,
        omega_scale=args.omega_scale
    )
    
    # Load best parameters
    from src.training.es.params import set_flat_params
    set_flat_params(policy.model, best_theta)
    
    # Create metadata
    metadata = {
        'generation': generation,
        'best_fitness': float(best_fitness),
        'training_method': 'Evolution Strategies',
        'pop_size': args.pop_size,
        'sigma': args.sigma,
        'alpha': args.alpha,
        'episode_length': args.episode_length,
        'eval_seeds': args.eval_seeds,
        'hidden_dims': args.hidden_dims,
        'v_scale': args.v_scale,
        'omega_scale': args.omega_scale,
        'total_parameters': theta.numel(),
        'stats_history': stats_history[-10:] if len(stats_history) > 10 else stats_history  # Last 10 entries
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"{args.save_prefix}_gen_{generation:04d}.json")
    save_policy(checkpoint_path, 'TorchMLP', policy.get_params(), sim_config, metadata, policy)
    
    # Also save as "latest"
    latest_path = os.path.join(args.output_dir, f"{args.save_prefix}_latest.json")
    save_policy(latest_path, 'TorchMLP', policy.get_params(), sim_config, metadata, policy)
    
    # If this is the best model, also save as "best"
    if is_best:
        best_path = os.path.join(args.output_dir, f"{args.save_prefix}_best.json")
        save_policy(best_path, 'TorchMLP', policy.get_params(), sim_config, metadata, policy)
        return best_path
    
    return checkpoint_path


def setup_csv_logging(csv_path):
    """Setup CSV logging"""
    if csv_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Write header
        with open(csv_path, 'w') as f:
            f.write('generation,fitness_mean,fitness_std,fitness_max,fitness_min,best_fitness,'
                   'gradient_norm,param_norm,sigma,alpha,elapsed_time\n')
        
        return True
    return False


def log_to_csv(csv_path, generation, stats, elapsed_time):
    """Log statistics to CSV"""
    if csv_path:
        with open(csv_path, 'a') as f:
            f.write(f"{generation},{stats['fitness_mean']:.4f},{stats['fitness_std']:.4f},"
                   f"{stats['fitness_max']:.4f},{stats['fitness_min']:.4f},{stats['best_fitness']:.4f},"
                   f"{stats['gradient_norm']:.6f},{stats['param_norm']:.4f},"
                   f"{stats['sigma']:.6f},{stats['alpha']:.6f},{elapsed_time:.2f}\n")


def main():
    """Main training loop"""
    args = parse_args()
    
    print("Evolution Strategies Training")
    print("=" * 50)
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Episode length: {args.episode_length} steps")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Batched evaluation: Always enabled")
    batch_size = args.eval_batch_size or args.eval_seeds
    print(f"Eval batch size: {batch_size}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Initialize training
    trainer, theta, start_generation, best_fitness_so_far, sim_config, model_ctor, model_kwargs = initialize_training(args)
    
    # Setup logging
    csv_logging = setup_csv_logging(args.csv_log)
    
    # Training state
    best_theta = theta.clone()
    stats_history = []
    start_time = time.time()
    
    try:
        # Training loop
        for generation in range(start_generation, start_generation + args.generations):
            gen_start_time = time.time()
            
            # ES step
            theta, stats, candidate_best_theta, candidate_best_fitness = trainer.step(theta, generation)
            
            # Update best
            if candidate_best_fitness > best_fitness_so_far:
                best_fitness_so_far = candidate_best_fitness
                best_theta = candidate_best_theta.clone()
                
                # Save best model immediately when new best is found
                best_checkpoint_path = save_checkpoint(
                    args, generation + 1, theta, best_theta, best_fitness_so_far,
                    sim_config, model_ctor, model_kwargs, stats_history,
                    is_best=True
                )
                print(f"🏆 New best fitness {best_fitness_so_far:.2f}! Saved: {best_checkpoint_path}")
            
            # Record stats
            gen_elapsed = time.time() - gen_start_time
            stats['generation'] = generation
            stats['elapsed_time'] = gen_elapsed
            stats['best_fitness_so_far'] = best_fitness_so_far
            stats_history.append(stats)
            
            # Logging
            if generation % args.log_interval == 0 or generation == start_generation:
                total_elapsed = time.time() - start_time
                print(f"Gen {generation:4d}: "
                      f"mean={stats['fitness_mean']:7.2f} "
                      f"best={stats['best_fitness']:7.2f} "
                      f"best_ever={best_fitness_so_far:7.2f} "
                      f"std={stats['fitness_std']:6.2f} "
                      f"grad_norm={stats['gradient_norm']:8.4f} "
                      f"time={gen_elapsed:5.2f}s "
                      f"total={total_elapsed/60:5.1f}m")
            
            # CSV logging
            if csv_logging:
                log_to_csv(args.csv_log, generation, stats, gen_elapsed)
            
            # Save checkpoint
            if (generation + 1) % args.save_interval == 0 or generation == start_generation + args.generations - 1:
                checkpoint_path = save_checkpoint(
                    args, generation + 1, theta, best_theta, best_fitness_so_far,
                    sim_config, model_ctor, model_kwargs, stats_history,
                    is_best=False
                )
                print(f"Saved checkpoint: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final checkpoint
        checkpoint_path = save_checkpoint(
            args, generation, theta, best_theta, best_fitness_so_far,
            sim_config, model_ctor, model_kwargs, stats_history,
            is_best=False
        )
        print(f"Saved final checkpoint: {checkpoint_path}")
    
    finally:
        # Always clean up the trainer's executor
        trainer.shutdown()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final best fitness: {best_fitness_so_far:.2f}")
    print(f"Final checkpoint saved")


if __name__ == "__main__":
    main()