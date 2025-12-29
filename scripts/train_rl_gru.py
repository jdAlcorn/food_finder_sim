#!/usr/bin/env python3
"""
RL training script for GRU policy
CPU-first training with continuous environment, online updates, and reward shaping
"""

import argparse
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sim.core import SimulationConfig
from src.sim.unified import SimulationSingle
from src.policy.rl_gru_policy import RLGRUPolicy
from src.policy.checkpoint import save_policy
from src.eval.load_suite import load_suite
from src.eval.testcases import TestCase
from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
from src.sim.batched import BatchedSimulation


def respawn_food(sim: SimulationSingle, test_case: TestCase, rng: np.random.RandomState, 
                 world_bounds: Tuple[float, float, float, float] = (50, 750, 50, 750),
                 min_distance_from_agent: float = 100.0) -> Tuple[float, float]:
    """
    Respawn food at a random but valid location
    
    Args:
        sim: Simulation instance
        test_case: Test case for reference
        rng: Random number generator for deterministic spawning
        world_bounds: (min_x, max_x, min_y, max_y) world boundaries
        min_distance_from_agent: Minimum distance from agent
        
    Returns:
        Tuple of (food_x, food_y)
    """
    agent_state = sim.get_state()['agent_state']
    agent_x, agent_y = agent_state['x'], agent_state['y']
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # Generate random position within world bounds
        food_x = rng.uniform(world_bounds[0], world_bounds[1])
        food_y = rng.uniform(world_bounds[2], world_bounds[3])
        
        # Check distance from agent
        distance_from_agent = np.sqrt((food_x - agent_x)**2 + (food_y - agent_y)**2)
        
        if distance_from_agent >= min_distance_from_agent:
            # Valid position found
            return food_x, food_y
    
    # Fallback: use original test case position if no valid position found
    print(f"Warning: Could not find valid food spawn position after {max_attempts} attempts, using original position")
    return test_case.food.x, test_case.food.y


def compute_gae(rewards: List[float], values: List[float], next_value: float, 
                dones: List[bool], gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state (for bootstrapping)
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    
    # Compute advantages using GAE
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_val = next_value if not dones[i] else 0
        else:
            next_val = values[i + 1] if not dones[i] else 0
        
        delta = rewards[i] + gamma * next_val - values[i]
        gae = delta + gamma * lam * gae * (1 - dones[i])
        advantages.insert(0, gae)
    
    # Compute returns
    for i in range(len(advantages)):
        returns.append(advantages[i] + values[i])
    
    return advantages, returns


def compute_reward(sim_state: Dict[str, Any], prev_sim_state: Optional[Dict[str, Any]], 
                   food_collected_this_step: bool, steps_since_food_seen: int) -> Tuple[float, int]:
    """
    Compute reward with sensor-based shaping
    
    Args:
        sim_state: Current simulation state
        prev_sim_state: Previous simulation state (None for first step)
        food_collected_this_step: Whether food was collected this step
        steps_since_food_seen: Steps since food was last visible
        
    Returns:
        Tuple of (reward, updated_steps_since_food_seen)
    """
    reward = 0.0
    
    # 1. Terminal reward (dominant)
    if food_collected_this_step:
        reward += 1000.0  # Large positive reward for reaching food
        steps_since_food_seen = 0
        return reward, steps_since_food_seen
    
    # 2. Food visibility and proximity shaping (sensor-derived only)
    vision_hit_types = sim_state['vision_hit_types']
    vision_distances = sim_state['vision_distances']
    
    # Check if food is visible in any ray
    food_visible = False
    min_food_distance = float('inf')
    
    for i, hit_type in enumerate(vision_hit_types):
        if hit_type == 'food':
            food_visible = True
            distance = vision_distances[i]
            if distance is not None and distance < min_food_distance:
                min_food_distance = distance
    
    if food_visible:
        steps_since_food_seen = 0
        
        # Small bonus for seeing food
        reward += 1.0
        
        # Progress reward when getting closer to visible food
        if prev_sim_state is not None:
            prev_vision_hit_types = prev_sim_state['vision_hit_types']
            prev_vision_distances = prev_sim_state['vision_distances']
            
            # Find minimum food distance in previous state
            prev_min_food_distance = float('inf')
            for i, hit_type in enumerate(prev_vision_hit_types):
                if hit_type == 'food':
                    distance = prev_vision_distances[i]
                    if distance is not None and distance < prev_min_food_distance:
                        prev_min_food_distance = distance
            
            # Progress reward (capped to prevent farming)
            if prev_min_food_distance < float('inf') and min_food_distance < float('inf'):
                progress = prev_min_food_distance - min_food_distance
                progress_reward = np.clip(progress, 0, 50.0)  # Cap at 50.0 instead of 10.0
                reward += progress_reward
    else:
        steps_since_food_seen += 1
    
    # 3. Small time penalty to encourage efficiency
    reward -= 0.001
    
    return reward, steps_since_food_seen


def run_episode(policy: RLGRUPolicy, sim: SimulationSingle, test_case: TestCase, max_steps: int, 
                dt: float, episode_rng: np.random.RandomState) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], 
                                   List[torch.Tensor], List[bool], Dict[str, Any]]:
    """
    Run a single episode and collect rollout data
    
    Returns:
        Tuple of (observations, actions, rewards, log_probs, dones, episode_info)
    """
    observations = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    
    # Episode tracking
    total_reward = 0.0
    food_collected = 0
    steps_since_food_seen = 0
    food_seen_this_episode = False
    
    # Reset simulation and policy
    sim.reset()
    policy.reset()
    
    # Reset to test case state
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
    
    sim.batched_sim.reset_to_states(agent_states, food_states)
    
    # Get initial state
    sim_state = sim.get_state()
    prev_sim_state = None
    
    for step in range(max_steps):
        # Build observation
        from src.policy.obs import build_observation
        obs = build_observation(sim_state, policy.v_scale, policy.omega_scale)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value, new_hidden = policy.network.act(
                obs_tensor, policy._hidden_state, deterministic=False
            )
        
        # Update hidden state
        policy._hidden_state = new_hidden.detach()
        
        # Convert action to simulation format
        action_np = action.cpu().numpy()[0]
        sim_action = {
            'throttle': float(np.clip(action_np[0], 0.0, 1.0)),
            'steer': float(np.clip(action_np[1], -1.0, 1.0))
        }
        
        # Step simulation
        step_info = sim.step(dt, sim_action)
        food_collected_this_step = step_info['food_collected_this_step']
        
        # Compute reward
        reward, steps_since_food_seen = compute_reward(
            step_info, prev_sim_state, food_collected_this_step, steps_since_food_seen
        )
        
        # Debug: Check if food is visible
        vision_hit_types = step_info['vision_hit_types']
        food_visible_this_step = 'food' in vision_hit_types if vision_hit_types else False
        if food_visible_this_step:
            food_seen_this_episode = True
        
        # Store rollout data
        observations.append(obs_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        rewards.append(reward)
        log_probs.append(log_prob.squeeze(0))
        values.append(value.squeeze(0))
        
        # Update tracking
        total_reward += reward
        
        # Check for food collection and respawn if needed
        if food_collected_this_step:
            food_collected += 1
            
            # Respawn food at new location for continuous learning
            new_food_x, new_food_y = respawn_food(sim, test_case, episode_rng)
            
            # Get current agent state to preserve it
            current_agent_state = step_info['agent_state']
            agent_states = [{
                'x': current_agent_state['x'],
                'y': current_agent_state['y'],
                'theta': current_agent_state['theta'],
                'vx': current_agent_state['vx'],
                'vy': current_agent_state['vy'],
                'omega': current_agent_state['omega'],
                'throttle': current_agent_state['throttle']
            }]
            
            food_states = [{'x': new_food_x, 'y': new_food_y}]
            sim.batched_sim.reset_to_states(agent_states, food_states)
            
            # Reset steps since food seen for new food
            steps_since_food_seen = 0
        
        # Check for episode termination (only on max steps, not food collection)
        done = step >= max_steps - 1
        dones.append(done)
        
        if done:
            break
        
        # Update state
        prev_sim_state = sim_state
        sim_state = step_info
    
    episode_info = {
        'total_reward': total_reward,
        'food_collected': food_collected,
        'episode_length': len(rewards),
        'final_steps_since_food_seen': steps_since_food_seen,
        'food_seen_this_episode': food_seen_this_episode
    }
    
    return observations, actions, rewards, log_probs, values, dones, episode_info


def update_policy(policy: RLGRUPolicy, optimizer: torch.optim.Optimizer,
                  observations: List[torch.Tensor], actions: List[torch.Tensor],
                  rewards: List[float], log_probs: List[torch.Tensor], 
                  values: List[torch.Tensor], dones: List[bool],
                  gamma: float = 0.99, lam: float = 0.95, 
                  clip_grad_norm: float = 0.5) -> Dict[str, float]:
    """
    Update policy using collected rollout data
    
    Returns:
        Dict with training metrics
    """
    # Convert to tensors
    obs_batch = torch.stack(observations)
    action_batch = torch.stack(actions)
    old_log_prob_batch = torch.stack(log_probs)
    value_batch = torch.stack(values)
    
    # Compute next value for bootstrapping (if episode didn't terminate)
    if not dones[-1]:
        with torch.no_grad():
            next_obs = obs_batch[-1].unsqueeze(0)
            next_value = policy.get_value(next_obs, policy._hidden_state).item()
    else:
        next_value = 0.0
    
    # Compute advantages and returns using GAE
    advantages, returns = compute_gae(
        rewards, [v.item() for v in value_batch], next_value, dones, gamma, lam
    )
    
    advantage_batch = torch.tensor(advantages, dtype=torch.float32, device=policy.device)
    return_batch = torch.tensor(returns, dtype=torch.float32, device=policy.device)
    
    # Normalize advantages
    advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
    
    # Reset hidden state for forward pass
    batch_size = obs_batch.shape[0]
    hidden = policy.network.init_hidden(batch_size, policy.device)
    
    # Forward pass through network
    action_mean, log_std, new_values, _ = policy.network.forward(obs_batch, hidden)
    
    # Compute new log probabilities
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(action_mean, std)
    new_log_probs = dist.log_prob(action_batch).sum(dim=-1)
    
    # Policy loss (simplified A2C, no PPO clipping for now)
    policy_loss = -(new_log_probs * advantage_batch).mean()
    
    # Value loss
    value_loss = F.mse_loss(new_values, return_batch)
    
    # Entropy bonus for exploration
    entropy = dist.entropy().sum(dim=-1).mean()
    entropy_loss = -0.01 * entropy  # Small entropy coefficient
    
    # Total loss
    total_loss = policy_loss + 0.5 * value_loss + entropy_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy.network.parameters(), clip_grad_norm)
    
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'total_loss': total_loss.item(),
        'mean_advantage': advantage_batch.mean().item(),
        'mean_return': return_batch.mean().item()
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL GRU policy with reward shaping')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes to train (default: 10000)')
    parser.add_argument('--max-steps', type=int, default=600,
                       help='Maximum steps per episode (default: 600)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--lam', type=float, default=0.95,
                       help='GAE lambda parameter (default: 0.95)')
    parser.add_argument('--clip-grad-norm', type=float, default=0.5,
                       help='Gradient clipping norm (default: 0.5)')
    
    # Model parameters
    parser.add_argument('--encoder-dims', type=int, nargs='+', default=[256, 128],
                       help='Encoder MLP hidden dimensions (default: 256 128)')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='GRU hidden state size (default: 64)')
    parser.add_argument('--num-layers', type=int, default=1,
                       help='Number of GRU layers (default: 1)')
    parser.add_argument('--v-scale', type=float, default=400.0,
                       help='Velocity normalization scale (default: 400.0)')
    parser.add_argument('--omega-scale', type=float, default=10.0,
                       help='Angular velocity normalization scale (default: 10.0)')
    
    # Environment parameters
    parser.add_argument('--dt', type=float, default=1/60,
                       help='Fixed timestep (default: 1/60)')
    parser.add_argument('--test-suite', type=str, default='data/test_suites/basic_v1.json',
                       help='Test suite for environment setup (default: data/test_suites/basic_v1.json)')
    parser.add_argument('--test-case-id', type=str, default=None,
                       help='Specific test case ID to use (default: random selection)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda, default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints (default: checkpoints)')
    parser.add_argument('--save-prefix', type=str, default='rl_gru_trained',
                       help='Prefix for saved checkpoint files (default: rl_gru_trained)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log progress every N episodes (default: 10)')
    
    # Viewer integration
    parser.add_argument('--viewer', action='store_true',
                       help='Enable viewer for best agent visualization')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("RL GRU Training")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Encoder dims: {args.encoder_dims}")
    print(f"GRU hidden size: {args.hidden_size}")
    print()
    
    # Load test suite
    try:
        test_suite = load_suite(args.test_suite)
        print(f"Loaded test suite: {test_suite.suite_id} v{test_suite.version}")
        print(f"Test cases: {len(test_suite.test_cases)}")
        
        # Select test case
        if args.test_case_id:
            test_case = None
            for case in test_suite.test_cases:
                if case.id == args.test_case_id:
                    test_case = case
                    break
            if test_case is None:
                print(f"Test case '{args.test_case_id}' not found")
                available_ids = [case.id for case in test_suite.test_cases]
                print(f"Available test cases: {', '.join(available_ids)}")
                return
        else:
            # Use first test case for now (could randomize later)
            test_case = test_suite.test_cases[0]
        
        print(f"Using test case: {test_case.id}")
        print()
        
    except Exception as e:
        print(f"Error loading test suite: {e}")
        return
    
    # Create simulation
    config = SimulationConfig()
    
    # Set up simulation with test case
    try:
        world = resolve_test_case_world(test_case)
        batched_sim = BatchedSimulation(1, config)
        updated_config = apply_world_to_simulation(batched_sim, world, [test_case])
        
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
        
        sim.batched_sim.reset_to_states(agent_states, food_states)
        
    except Exception as e:
        print(f"Error setting up simulation: {e}")
        return
    
    # Create policy
    policy = RLGRUPolicy(
        encoder_dims=tuple(args.encoder_dims),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
        v_scale=args.v_scale,
        omega_scale=args.omega_scale,
        init_seed=args.seed
    )
    
    # Create optimizer
    optimizer = optim.Adam(policy.network.parameters(), lr=args.lr)
    
    # Training tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    food_collected_counts = deque(maxlen=100)
    best_reward = float('-inf')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.save_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Saving to: {run_dir}")
    print()
    
    # Training loop
    start_time = time.time()
    
    # Create episode-specific RNG for deterministic food respawning
    episode_rng = np.random.RandomState(args.seed + 1000)
    
    for episode in range(args.episodes):
        # Run episode
        observations, actions, rewards, log_probs, values, dones, episode_info = run_episode(
            policy, sim, test_case, args.max_steps, args.dt, episode_rng
        )
        
        # Update policy
        train_metrics = update_policy(
            policy, optimizer, observations, actions, rewards, log_probs, values, dones,
            args.gamma, args.lam, args.clip_grad_norm
        )
        
        # Track metrics
        episode_reward = episode_info['total_reward']
        episode_length = episode_info['episode_length']
        food_collected = episode_info['food_collected']
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        food_collected_counts.append(food_collected)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            
            # Save best checkpoint
            best_path = os.path.join(run_dir, f"{args.save_prefix}_best.json")
            metadata = {
                'episode': episode,
                'best_reward': best_reward,
                'episode_length': episode_length,
                'food_collected': food_collected,
                'training_time': time.time() - start_time
            }
            save_policy(best_path, 'RLGRU', policy.get_params(), config, metadata, policy)
        
        # Logging
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_food = np.mean(food_collected_counts) if food_collected_counts else 0
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"Length: {episode_length:3d} | "
                  f"Food: {food_collected} | "
                  f"Seen: {'Y' if episode_info['food_seen_this_episode'] else 'N'} | "
                  f"Policy Loss: {train_metrics['policy_loss']:.4f} | "
                  f"Value Loss: {train_metrics['value_loss']:.4f}")
        
        # Save periodic checkpoints
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(run_dir, f"{args.save_prefix}_ep_{episode:06d}.json")
            metadata = {
                'episode': episode,
                'reward': episode_reward,
                'best_reward': best_reward,
                'episode_length': episode_length,
                'food_collected': food_collected,
                'training_time': time.time() - start_time
            }
            save_policy(checkpoint_path, 'RLGRU', policy.get_params(), config, metadata, policy)
    
    # Save final checkpoint
    final_path = os.path.join(run_dir, f"{args.save_prefix}_final.json")
    metadata = {
        'episode': args.episodes,
        'final_reward': episode_reward,
        'best_reward': best_reward,
        'training_time': time.time() - start_time,
        'total_episodes': args.episodes
    }
    save_policy(final_path, 'RLGRU', policy.get_params(), config, metadata, policy)
    
    print()
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final reward: {episode_reward:.2f}")
    print(f"Training time: {time.time() - start_time:.1f}s")
    print(f"Checkpoints saved to: {run_dir}")
    
    # Launch viewer if requested
    if args.viewer:
        print()
        print("Launching viewer with best agent...")
        try:
            # Load best policy
            best_path = os.path.join(run_dir, f"{args.save_prefix}_best.json")
            if os.path.exists(best_path):
                from src.policy.checkpoint import load_policy
                best_policy, best_config, best_metadata = load_policy(best_path)
                
                # Set up simulation for viewer (reset to original test case)
                world = resolve_test_case_world(test_case)
                viewer_batched_sim = BatchedSimulation(1, best_config)
                viewer_updated_config = apply_world_to_simulation(viewer_batched_sim, world, [test_case])
                
                viewer_sim = SimulationSingle(viewer_updated_config)
                viewer_sim.batched_sim = viewer_batched_sim
                
                # Reset to test case initial state
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
                
                viewer_sim.batched_sim.reset_to_states(agent_states, food_states)
                
                from src.viz.pygame_single import run_simulation_gui
                
                run_simulation_gui(
                    policy=best_policy,
                    config=viewer_updated_config,
                    fps=60,
                    policy_name=f"RL GRU (Best: {best_reward:.1f}, Food: {best_metadata.get('food_collected', 0)})",
                    sim=viewer_sim
                )
            else:
                print("Best checkpoint not found")
                
        except ImportError:
            print("Viewer not available (pygame not installed)")
        except Exception as e:
            print(f"Error launching viewer: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()