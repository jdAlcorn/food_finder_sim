#!/usr/bin/env python3
"""
RL training script for MLP policy
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
from src.policy.rl_mlp_policy import RLMLPPolicy
from src.policy.checkpoint import save_policy, load_policy
from src.eval.load_suite import load_suite
from src.eval.testcases import TestCase
from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
from src.sim.batched import BatchedSimulation
from src.training.test_case_scheduler import TestCaseScheduler, load_scheduler_config

# Import shared functions from GRU trainer (excluding log_to_csv and setup_csv_logging which we'll redefine)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_rl_gru import (
    setup_simulation_for_test_case, extract_run_folder_from_resume_path, get_run_folder,
    compute_gae
)


def setup_csv_logging(run_dir: str, csv_filename: str) -> str:
    """Setup CSV logging in the run folder for MLP trainer with focused reward system"""
    if csv_filename:
        csv_path = os.path.join(run_dir, csv_filename)
    else:
        csv_path = os.path.join(run_dir, "training_log.csv")
    
    # Write header (will overwrite existing file)
    with open(csv_path, 'w') as f:
        f.write('episode,test_case_id,reward,avg_reward_100,best_reward,episode_length,'
               'success,success_step,success_rate_100,total_terminal_reward,'
               'total_reacquire_reward,total_center_progress,total_distance_progress,'
               'total_lost_sight_penalty,total_no_food_penalty,'
               'min_food_distance_seen,final_food_distance,food_visible_fraction,'
               'policy_loss,value_loss,entropy,total_loss,grad_norm_pre_clip,'
               'grad_norm_post_clip,training_time_elapsed\n')
    
    return csv_path


def log_to_csv(csv_path: str, episode: int, current_test_case_id: str, episode_info: Dict[str, Any], 
               episode_reward: float, avg_reward: float, best_reward: float, success_rate: float,
               train_metrics: Dict[str, float], training_time_elapsed: float):
    """Log episode statistics to CSV for MLP trainer with focused reward system"""
    if csv_path:
        with open(csv_path, 'a') as f:
            # Format success step (N/A if None)
            success_step = episode_info['success_step'] if episode_info['success_step'] is not None else -1
            
            # Format distance values (N/A if None)
            min_dist = episode_info['min_food_distance_seen'] if episode_info['min_food_distance_seen'] is not None else -1
            final_dist = episode_info['final_food_distance'] if episode_info['final_food_distance'] is not None else -1
            
            line = (f"{episode},{current_test_case_id},{episode_reward:.6f},{avg_reward:.6f},"
                   f"{best_reward:.6f},{episode_info['episode_length']},"
                   f"{1 if episode_info['success'] else 0},{success_step},{success_rate:.6f},"
                   f"{episode_info['total_terminal_reward']:.6f},"
                   f"{episode_info['total_reacquire_reward']:.6f},"
                   f"{episode_info['total_center_progress']:.6f},"
                   f"{episode_info['total_distance_progress']:.6f},"
                   f"{episode_info['total_lost_sight_penalty']:.6f},"
                   f"{episode_info['total_no_food_penalty']:.6f},"
                   f"{min_dist:.6f},{final_dist:.6f},"
                   f"{episode_info['food_visible_fraction']:.6f},"
                   f"{train_metrics['policy_loss']:.6f},{train_metrics['value_loss']:.6f},"
                   f"{train_metrics['entropy']:.6f},{train_metrics['total_loss']:.6f},"
                   f"{train_metrics['grad_norm_pre_clip']:.6f},"
                   f"{train_metrics['grad_norm_post_clip']:.6f},"
                   f"{training_time_elapsed:.2f}\n")
            
            f.write(line)


def compute_reward_focused(step_info: Dict[str, Any], prev_step_info: Optional[Dict[str, Any]], 
                          food_collected_this_step: bool, steps_since_food_seen: int, 
                          step_idx: int, max_steps: int = 600, reward_scale: float = 1.0) -> Tuple[float, int, Dict[str, float]]:
    """
    Focused reward system based on food visibility and positioning
    
    Rewards:
    1. Seeing food after not seeing it last step
    2. Food moving closer to center of vision
    3. Visual distance to food decreasing
    
    Penalties:
    4. Food leaving vision after seeing it last step
    5. Food moving further from center of vision
    6. Distance to food increasing
    7. Small penalty for each step without seeing food
    
    Args:
        step_info: Current step info from sim.step()
        prev_step_info: Previous step info (None for first step)
        food_collected_this_step: Whether food was collected this step
        steps_since_food_seen: Steps since food was last visible
        step_idx: Current step index in episode
        max_steps: Maximum steps in episode
        reward_scale: Scale factor for all rewards
        
    Returns:
        Tuple of (reward, updated_steps_since_food_seen, reward_components_dict)
    """
    # ============================================================================
    # REWARD COMPONENT TOGGLES - Set to False to disable specific reward types
    # ============================================================================
    ENABLE_TERMINAL_REWARD = False          # Food collection reward (1000 points)
    ENABLE_REACQUIRE_REWARD = False         # Seeing food after losing it (10 points)
    ENABLE_CENTER_PROGRESS_REWARD = False   # Food moving toward/away from center
    ENABLE_DISTANCE_PROGRESS_REWARD = True # Visual distance decreasing/increasing
    ENABLE_LOST_SIGHT_PENALTY = False       # Penalty for losing sight of food (-15 points)
    ENABLE_NO_FOOD_PENALTY = False          # Per-step penalty without food (-0.1 points)
    
    from src.sim.vision_utils import calculate_food_vision_metrics, calculate_center_vision_reward
    
    reward = 0.0
    reward_components = {
        'terminal': 0.0,
        'reacquire': 0.0,          # Seeing food after not seeing it
        'center_progress': 0.0,     # Food moving toward/away from center
        'distance_progress': 0.0,   # Visual distance decreasing/increasing
        'lost_sight': 0.0,         # Penalty for losing sight of food
        'no_food_penalty': 0.0,    # Penalty for not seeing food
    }
    
    # 1. Terminal reward (dominant)
    if food_collected_this_step and ENABLE_TERMINAL_REWARD:
        terminal_reward = 1000.0 * reward_scale
        reward += terminal_reward
        reward_components['terminal'] = terminal_reward
        steps_since_food_seen = 0
        return reward, steps_since_food_seen, reward_components
    
    # Calculate current vision metrics
    curr_metrics = calculate_food_vision_metrics(
        step_info['agent_state'], 
        step_info['food_position'],
        step_info['vision_distances'],
        step_info['vision_hit_types']
    )
    
    # Calculate previous vision metrics if available
    prev_metrics = None
    if prev_step_info is not None:
        prev_metrics = calculate_food_vision_metrics(
            prev_step_info['agent_state'],
            prev_step_info['food_position'], 
            prev_step_info['vision_distances'],
            prev_step_info['vision_hit_types']
        )
    
    curr_food_visible = curr_metrics['food_visible']
    prev_food_visible = prev_metrics['food_visible'] if prev_metrics else False
    
    if curr_food_visible:
        steps_since_food_seen = 0
        
        # 1. Reacquire reward: Seeing food after not seeing it
        if not prev_food_visible and ENABLE_REACQUIRE_REWARD:
            reacquire_reward = 10.0 * reward_scale
            reward += reacquire_reward
            reward_components['reacquire'] = reacquire_reward
        
        # 2. Center progress reward: Food moving toward/away from center of vision
        if prev_metrics and prev_food_visible and ENABLE_CENTER_PROGRESS_REWARD:
            center_reward = calculate_center_vision_reward(
                prev_metrics['food_rays_from_center'],
                curr_metrics['food_rays_from_center'],
                reward_scale
            )
            reward += center_reward
            reward_components['center_progress'] = center_reward
        
        # 3. Distance progress reward: Visual distance decreasing/increasing
        if (prev_metrics and prev_food_visible and prev_metrics['closest_food_distance'] is not None 
            and ENABLE_DISTANCE_PROGRESS_REWARD):
            prev_distance = prev_metrics['closest_food_distance']
            curr_distance = curr_metrics['closest_food_distance']
            
            if curr_distance is not None:
                # Positive reward for getting closer, negative for getting farther
                distance_progress = prev_distance - curr_distance
                distance_reward = np.clip(distance_progress, -10.0, 20.0) * reward_scale
                reward += distance_reward
                reward_components['distance_progress'] = distance_reward
    
    else:
        steps_since_food_seen += 1
        
        # 4. Lost sight penalty: Food leaving vision after seeing it
        if prev_food_visible and ENABLE_LOST_SIGHT_PENALTY:
            lost_sight_penalty = -150.0 * reward_scale
            reward += lost_sight_penalty
            reward_components['lost_sight'] = lost_sight_penalty
        
        # 7. No food penalty: Small penalty for each step without seeing food
        if ENABLE_NO_FOOD_PENALTY:
            no_food_penalty = -0.1 * reward_scale
            reward += no_food_penalty
            reward_components['no_food_penalty'] = no_food_penalty
    
    return reward, steps_since_food_seen, reward_components


def initialize_policy_and_optimizer(args) -> Tuple[RLMLPPolicy, torch.optim.Optimizer, int, float, deque, deque, deque]:
    """Initialize policy and optimizer, optionally resuming from checkpoint"""
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        
        # Load checkpoint
        policy, config, metadata = load_policy(args.resume)
        
        # Verify policy type
        if not hasattr(policy, 'network') or hasattr(policy.network, 'gru'):
            raise ValueError(f"Resume checkpoint must be an RL MLP policy, got {type(policy)}")
        
        # Create optimizer with loaded policy parameters
        optimizer = optim.Adam(policy.network.parameters(), lr=args.lr)
        
        # Extract training state from metadata
        start_episode = metadata.get('episode', 0)
        best_reward = metadata.get('best_reward', float('-inf'))
        
        # Initialize tracking deques with some history if available
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        food_collected_counts = deque(maxlen=100)
        
        # If we have recent episode history in metadata, restore it
        if 'recent_rewards' in metadata:
            episode_rewards.extend(metadata['recent_rewards'])
        if 'recent_lengths' in metadata:
            episode_lengths.extend(metadata['recent_lengths'])
        if 'recent_food_collected' in metadata:
            food_collected_counts.extend(metadata['recent_food_collected'])
        
        print(f"Resumed from episode {start_episode}, best reward: {best_reward:.2f}")
        print(f"Policy device: {policy.device}")
        
        return policy, optimizer, start_episode, best_reward, episode_rewards, episode_lengths, food_collected_counts
    
    else:
        # Create new policy
        policy = RLMLPPolicy(
            encoder_dims=tuple(args.encoder_dims),
            device=args.device,
            v_scale=args.v_scale,
            omega_scale=args.omega_scale,
            init_seed=args.seed
        )
        
        # Create optimizer
        optimizer = optim.Adam(policy.network.parameters(), lr=args.lr)
        
        # Initialize training tracking
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        food_collected_counts = deque(maxlen=100)
        
        start_episode = 0
        best_reward = float('-inf')
        
        print(f"Initialized new policy with {sum(p.numel() for p in policy.network.parameters())} parameters")
        
        return policy, optimizer, start_episode, best_reward, episode_rewards, episode_lengths, food_collected_counts


def save_checkpoint_with_training_state(run_dir: str, args, episode: int, policy: RLMLPPolicy, 
                                       best_reward: float, episode_rewards: deque, episode_lengths: deque, 
                                       food_collected_counts: deque, config, training_time: float, 
                                       is_best: bool = False, is_final: bool = False) -> str:
    """Save checkpoint with training state for resumption"""
    # Determine checkpoint filename
    if is_best:
        checkpoint_path = os.path.join(run_dir, f"{args.save_prefix}_best.json")
    elif is_final:
        checkpoint_path = os.path.join(run_dir, f"{args.save_prefix}_final.json")
    else:
        checkpoint_path = os.path.join(run_dir, f"{args.save_prefix}_ep_{episode:06d}.json")
    
    # Create metadata with training state
    metadata = {
        'episode': episode,
        'best_reward': best_reward,
        'training_time': training_time,
        'training_method': 'RL with MLP',
        'policy_type': 'RLMLP',
        
        # Model hyperparameters
        'encoder_dims': args.encoder_dims,
        'v_scale': args.v_scale,
        'omega_scale': args.omega_scale,
        
        # Training hyperparameters
        'lr': args.lr,
        'gamma': args.gamma,
        'lam': args.lam,
        'clip_grad_norm': args.clip_grad_norm,
        'value_loss_coef': args.value_loss_coef,
        'reward_scale': args.reward_scale,
        'randomize_start': args.randomize_start,
        'max_steps': args.max_steps,
        
        # Recent training history for resumption
        'recent_rewards': list(episode_rewards)[-20:] if episode_rewards else [],
        'recent_lengths': list(episode_lengths)[-20:] if episode_lengths else [],
        'recent_food_collected': list(food_collected_counts)[-20:] if food_collected_counts else [],
        
        # Statistics
        'total_episodes': episode,
        'avg_reward_last_100': np.mean(episode_rewards) if episode_rewards else 0.0,
        'success_rate_last_100': sum(1 for x in food_collected_counts if x > 0) / len(food_collected_counts) if food_collected_counts else 0.0,
        
        # Run info
        'run_folder': os.path.basename(run_dir),
        'device': args.device,
        'seed': args.seed
    }
    
    # Add current episode info if not final
    if not is_final and episode_rewards:
        metadata['current_reward'] = episode_rewards[-1]
        metadata['current_length'] = episode_lengths[-1] if episode_lengths else 0
        metadata['current_food_collected'] = food_collected_counts[-1] if food_collected_counts else 0
    
    # Save checkpoint
    save_policy(checkpoint_path, 'RLMLP', policy.get_params(), config, metadata, policy)
    
    return checkpoint_path


def run_episode(policy: RLMLPPolicy, sim: SimulationSingle, test_case: TestCase, max_steps: int, 
                dt: float, episode_rng: np.random.RandomState, reward_scale: float = 1.0,
                normalize_rewards: bool = False, randomize_start: float = 0.0) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], 
                                   List[torch.Tensor], List[torch.Tensor], List[bool], Dict[str, Any]]:
    """
    Run a single episode and collect rollout data (MLP version - no hidden states)
    
    Returns:
        Tuple of (observations, actions, rewards, log_probs, values, dones, episode_info)
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
    
    # Detailed diagnostics
    success_step = None
    total_terminal_reward = 0.0
    total_reacquire_reward = 0.0
    total_center_progress = 0.0
    total_distance_progress = 0.0
    total_lost_sight_penalty = 0.0
    total_no_food_penalty = 0.0
    
    # Debug tracking for progress calculation
    progress_debug_prev_distances = []
    progress_debug_current_distances = []
    progress_debug_raw_progress = []
    min_food_distance_seen = float('inf')
    final_food_distance = float('inf')
    food_visible_steps = 0
    total_steps = 0
    
    # Trainer-side memory for smoothed progress tracking (NOT part of agent observation)
    last_seen_food_distance = None
    
    # Reset simulation and policy
    sim.reset()
    policy.reset()
    
    # Reset to test case state with optional randomization
    agent_x = test_case.agent_start.x
    agent_y = test_case.agent_start.y
    agent_theta = test_case.agent_start.theta
    
    # Apply randomization if enabled
    if randomize_start > 0.0:
        # Randomize position by ±randomize_start pixels
        agent_x += episode_rng.uniform(-randomize_start, randomize_start)
        agent_y += episode_rng.uniform(-randomize_start, randomize_start)
        
        # Randomize angle by ±randomize_start degrees (convert to radians)
        angle_noise_rad = np.radians(episode_rng.uniform(-randomize_start, randomize_start))
        agent_theta += angle_noise_rad
        
        # Normalize angle to [-π, π]
        while agent_theta > np.pi:
            agent_theta -= 2 * np.pi
        while agent_theta < -np.pi:
            agent_theta += 2 * np.pi
    
    agent_states = [{
        'x': agent_x,
        'y': agent_y,
        'theta': agent_theta,
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
        # Get action from policy using proper interface
        with torch.no_grad():
            action, log_prob, value = policy.act_training(sim_state, deterministic=False)
        
        # Convert action to simulation format - NO CLIPPING since action is already bounded
        action_np = action.cpu().numpy()[0]
        sim_action = {
            'throttle': float(action_np[0]),  # Already in [0,1]
            'steer': float(action_np[1])      # Already in [-1,1]
        }
        
        # Step simulation
        step_info = sim.step(dt, sim_action)
        food_collected_this_step = step_info['food_collected_this_step']
        
        # Compute reward with detailed tracking
        reward, steps_since_food_seen, reward_components = compute_reward_focused(
            step_info, prev_sim_state, food_collected_this_step, steps_since_food_seen, step,
            max_steps, reward_scale
        )
        
        # Track reward components
        total_terminal_reward += reward_components['terminal']
        total_reacquire_reward += reward_components['reacquire']
        total_center_progress += reward_components['center_progress']
        total_distance_progress += reward_components['distance_progress']
        total_lost_sight_penalty += reward_components['lost_sight']
        total_no_food_penalty += reward_components['no_food_penalty']
        
        # Track food distance diagnostics
        vision_hit_types = step_info['vision_hit_types']
        vision_distances = step_info['vision_distances']
        food_visible_this_step = False
        current_min_food_distance = float('inf')
        
        # Store minimum food distance if we're closer than we've ever been to it
        for i, hit_type in enumerate(vision_hit_types):
            if hit_type == 'food':
                food_visible_this_step = True
                distance = vision_distances[i]
                if distance is not None and distance < current_min_food_distance:
                    current_min_food_distance = distance
        
        if food_visible_this_step:
            food_seen_this_episode = True
            food_visible_steps += 1
            if current_min_food_distance < min_food_distance_seen:
                min_food_distance_seen = current_min_food_distance
            final_food_distance = current_min_food_distance  # Update final distance when visible
        
        total_steps += 1
        
        # Store rollout data
        # We need the observation tensor for training, so build it here
        from src.policy.obs import build_observation
        obs = build_observation(sim_state, policy.v_scale, policy.omega_scale)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=policy.device)
        
        observations.append(obs_tensor)
        actions.append(action.squeeze(0))
        rewards.append(reward)
        log_probs.append(log_prob.squeeze(0))
        values.append(value.squeeze(0))
        
        # Update tracking
        total_reward += reward
        
        # Check for episode termination
        done = False
        if food_collected_this_step:
            food_collected += 1
            success_step = step
            done = True  # Episode ends on food collection
        elif step >= max_steps - 1:
            done = True  # Time limit reached
        
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
        'food_seen_this_episode': food_seen_this_episode,
        # Detailed diagnostics
        'success': food_collected > 0,
        'success_step': success_step,
        'total_terminal_reward': total_terminal_reward,
        'total_reacquire_reward': total_reacquire_reward,
        'total_center_progress': total_center_progress,
        'total_distance_progress': total_distance_progress,
        'total_lost_sight_penalty': total_lost_sight_penalty,
        'total_no_food_penalty': total_no_food_penalty,
        'min_food_distance_seen': min_food_distance_seen if min_food_distance_seen < float('inf') else None,
        'final_food_distance': final_food_distance if final_food_distance < float('inf') else None,
        'food_visible_fraction': food_visible_steps / total_steps if total_steps > 0 else 0.0,
    }
    
    return observations, actions, rewards, log_probs, values, dones, episode_info


def update_policy(policy: RLMLPPolicy, optimizer: torch.optim.Optimizer,
                  observations: List[torch.Tensor], actions: List[torch.Tensor],
                  rewards: List[float], log_probs: List[torch.Tensor], 
                  values: List[torch.Tensor], dones: List[bool],
                  gamma: float = 0.99, lam: float = 0.95, 
                  clip_grad_norm: float = 0.5, value_loss_coef: float = 0.5,
                  episode_num: int = 0) -> Dict[str, float]:
    """
    Update policy using collected rollout data (MLP version - no sequence processing)
    
    Returns:
        Dict with training metrics
    """
    # Convert to tensors
    obs_batch = torch.stack(observations)  # [T, obs_dim]
    action_batch = torch.stack(actions)    # [T, action_dim]
    old_log_prob_batch = torch.stack(log_probs)  # [T]
    value_batch = torch.stack(values)      # [T]
    
    T = obs_batch.shape[0]
    
    # Compute next value for bootstrapping (if episode didn't terminate)
    if not dones[-1]:
        with torch.no_grad():
            next_obs = obs_batch[-1].unsqueeze(0)
            next_value = policy.get_value(next_obs).item()
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
    
    # Forward pass through network (batch processing for MLP)
    action_logits_batch, log_std_batch, new_value_batch = policy.network.forward(obs_batch)
    
    # Compute new log probabilities using squashed Gaussian (vectorized)
    std_batch = torch.exp(log_std_batch)  # [T, 2]
    
    # Reverse the squashing to get u from executed actions
    # Policy mapping: throttle = (tanh + 1) / 2, steering = tanh
    # Inverse: tanh_throttle = 2*throttle - 1, tanh_steering = steering
    action_tanh = torch.stack([
        2.0 * action_batch[:, 0] - 1.0,  # throttle: [0,1] -> [-1,1]
        action_batch[:, 1]               # steering: [-1,1] -> [-1,1]
    ], dim=1)  # [T, 2]
    
    # Inverse tanh (atanh) to get u, with clamping for numerical stability
    action_tanh_clamped = torch.clamp(action_tanh, -0.999, 0.999)
    u_batch = torch.atanh(action_tanh_clamped)  # [T, 2]
    
    # Compute log prob of u under Gaussian
    dist_batch = torch.distributions.Normal(action_logits_batch, std_batch)
    log_prob_u_batch = dist_batch.log_prob(u_batch).sum(dim=-1)  # [T]
    
    # Apply tanh correction
    tanh_correction = torch.log(1 - action_tanh_clamped.pow(2) + 1e-6).sum(dim=-1)  # [T]
    new_log_prob_batch = log_prob_u_batch - tanh_correction
    
    # Policy loss (simplified A2C, no PPO clipping for now)
    policy_loss = -(new_log_prob_batch * advantage_batch).mean()
    
    # Value loss
    value_loss = F.mse_loss(new_value_batch, return_batch)
    
    # Entropy bonus for exploration (vectorized)
    entropy = dist_batch.entropy().sum(dim=-1).mean()  # [T] -> scalar
    entropy_loss = -0.01 * entropy  # Small entropy coefficient
    
    # Total loss
    total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Compute gradient norm before clipping
    grad_norm_pre_clip = 0.0
    for name, param in policy.network.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            grad_norm_pre_clip += param_grad_norm ** 2
    
    grad_norm_pre_clip = grad_norm_pre_clip ** 0.5
    
    # Gradient clipping
    grad_norm_post_clip = torch.nn.utils.clip_grad_norm_(policy.network.parameters(), clip_grad_norm)
    
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'total_loss': total_loss.item(),
        'mean_advantage': advantage_batch.mean().item(),
        'mean_return': return_batch.mean().item(),
        'grad_norm_pre_clip': grad_norm_pre_clip,
        'grad_norm_post_clip': grad_norm_post_clip.item(),
        'grad_clipped': grad_norm_pre_clip > clip_grad_norm,
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL MLP policy with reward shaping')
    
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
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                       help='Scale all rewards by this factor (default: 1.0)')
    parser.add_argument('--normalize-rewards', action='store_true',
                       help='Normalize rewards to [-1, 1] range using tanh scaling')
    

    # Model parameters
    parser.add_argument('--encoder-dims', type=int, nargs='+', default=[256, 128],
                       help='Encoder MLP hidden dimensions (default: 256 128)')
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
                       help='Specific test case ID to use (overrides scheduler and suite cycling)')
    parser.add_argument('--scheduler-config', type=str, default=None,
                       help='Path to test case scheduler config JSON file (enables weighted sampling)')
    parser.add_argument('--scheduler-seed', type=int, default=None,
                       help='Seed for test case scheduler (default: uses main seed + 1000)')
    parser.add_argument('--randomize-start', type=float, default=0.0,
                       help='Randomize agent start position/angle. Value is max deviation: position ±N pixels, angle ±N degrees (default: 0.0 = no randomization)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda, default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints (default: checkpoints)')
    parser.add_argument('--save-prefix', type=str, default='rl_mlp_trained',
                       help='Prefix for saved checkpoint files (default: rl_mlp_trained)')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log progress every N episodes (default: 10)')
    parser.add_argument('--detailed-log-interval', type=int, default=50,
                       help='Log detailed diagnostics every N episodes (default: 50)')
    parser.add_argument('--csv-log', type=str, default=None,
                       help='CSV filename to log training statistics (default: training_log.csv, saved in run folder)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
    # Viewer integration
    parser.add_argument('--viewer', action='store_true',
                       help='Enable viewer for best agent visualization')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("RL MLP Training")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Encoder dims: {args.encoder_dims}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    if args.normalize_rewards:
        print(f"Reward normalization enabled: rewards will be scaled to [-1, 1]")
    if args.randomize_start > 0.0:
        print(f"Start randomization enabled: ±{args.randomize_start} pixels/degrees")
    print()
    
    # Load test suite and setup (same as GRU trainer)
    try:
        test_suite = load_suite(args.test_suite)
        print(f"Loaded test suite: {test_suite.suite_id} v{test_suite.version}")
        print(f"Test cases: {len(test_suite.test_cases)}")
        
        # Set up test case selection strategy
        scheduler = None
        single_test_case = None
        
        if args.test_case_id:
            # Single specific test case mode
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
            
            single_test_case = test_case
            print(f"Using single test case: {test_case.id}")
            
        elif args.scheduler_config:
            # Weighted sampling mode
            try:
                weights = load_scheduler_config(args.scheduler_config)
                scheduler_seed = args.scheduler_seed if args.scheduler_seed is not None else args.seed + 1000
                scheduler = TestCaseScheduler(test_suite.test_cases, weights, scheduler_seed)
                print(f"Using weighted scheduler from: {args.scheduler_config}")
                
                # Show statistics for the planned training
                stats = scheduler.get_statistics(args.episodes)
                print(f"Planned distribution over {args.episodes} episodes:")
                for case_id, percentage in stats['case_percentages'].items():
                    if percentage > 0:
                        print(f"  {case_id}: {percentage:.1f}%")
                        
            except Exception as e:
                print(f"Error loading scheduler config: {e}")
                return
                
        else:
            # Sequential cycling mode (default when no specific case or scheduler)
            scheduler_seed = args.scheduler_seed if args.scheduler_seed is not None else args.seed + 1000
            scheduler = TestCaseScheduler(test_suite.test_cases, None, scheduler_seed)
            print(f"Using sequential cycling through all {len(test_suite.test_cases)} test cases")
        
        print()
        
    except Exception as e:
        print(f"Error loading test suite: {e}")
        return
    
    # Create base simulation config
    config = SimulationConfig()
    
    # Set up initial simulation
    try:
        if single_test_case:
            # Single test case mode - set up once
            sim = setup_simulation_for_test_case(single_test_case, config)
            print(f"Simulation configured for single test case: {single_test_case.id}")
        else:
            # Multi-test case mode - set up with first case, will reconfigure per episode
            initial_test_case = test_suite.test_cases[0]
            sim = setup_simulation_for_test_case(initial_test_case, config)
            print(f"Simulation configured for multi-test case training")
        
    except Exception as e:
        print(f"Error setting up simulation: {e}")
        return
    
    # Get or create run folder
    run_dir = get_run_folder(args)
    
    # Initialize policy and optimizer (with optional resume)
    policy, optimizer, start_episode, best_reward, episode_rewards, episode_lengths, food_collected_counts = initialize_policy_and_optimizer(args)
    
    print(f"Saving to: {run_dir}")
    
    # Setup CSV logging
    csv_path = setup_csv_logging(run_dir, args.csv_log)
    print(f"CSV logging to: {csv_path}")
    print()
    
    # Training loop
    start_time = time.time()
    
    # Create episode-specific RNG for deterministic food respawning
    episode_rng = np.random.RandomState(args.seed + 1000)
    
    # Track previous test case for simulation reconfiguration
    previous_test_case_id = None
    
    for episode in range(start_episode, start_episode + args.episodes):
        # Reset to test case state
        if single_test_case:
            # Single test case mode
            current_test_case = single_test_case
        else:
            # Multi-test case mode - use scheduler
            current_test_case = scheduler.get_next_test_case(episode)
            
            # Reconfigure simulation for the new test case if it changed
            if episode == start_episode or current_test_case.id != previous_test_case_id:
                sim = setup_simulation_for_test_case(current_test_case, config)
            
            previous_test_case_id = current_test_case.id
        
        # Run episode
        observations, actions, rewards, log_probs, values, dones, episode_info = run_episode(
            policy, sim, current_test_case, args.max_steps, args.dt, episode_rng,
            args.reward_scale, args.normalize_rewards, args.randomize_start
        )
        
        # Update policy (no hidden states for MLP)
        train_metrics = update_policy(
            policy, optimizer, observations, actions, rewards, log_probs, values, dones,
            args.gamma, args.lam, args.clip_grad_norm, args.value_loss_coef, episode
        )
        
        # Track metrics
        episode_reward = episode_info['total_reward']
        episode_length = episode_info['episode_length']
        food_collected = episode_info['food_collected']
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        food_collected_counts.append(food_collected)
        
        # Calculate metrics for CSV logging
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        recent_episodes = min(100, len(food_collected_counts))
        recent_successes = sum(1 for x in list(food_collected_counts)[-recent_episodes:] if x > 0)
        success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0.0
        
        # Log to CSV
        log_to_csv(csv_path, episode, current_test_case.id, episode_info, episode_reward, 
                   avg_reward, best_reward, success_rate, train_metrics, time.time() - start_time)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            
            # Save best checkpoint
            best_checkpoint_path = save_checkpoint_with_training_state(
                run_dir, args, episode, policy, best_reward, episode_rewards, episode_lengths, 
                food_collected_counts, config, time.time() - start_time, is_best=True
            )
        
        # Logging with detailed diagnostics
        if episode % args.log_interval == 0:
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_food = np.mean(food_collected_counts) if food_collected_counts else 0
            
            print(f"Episode {episode:5d} | "
                  f"Case: {current_test_case.id[:20]:20s} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"Length: {episode_length:3d} | "
                  f"Success: {'Y' if episode_info['success'] else 'N'} | "
                  f"Step: {episode_info['success_step'] if episode_info['success_step'] is not None else 'N/A'} | "
                  f"SR: {success_rate:.2f}")
            
            # Detailed diagnostics
            if episode % args.detailed_log_interval == 0:
                print(f"  Rewards: T={episode_info['total_terminal_reward']:.1f}, "
                      f"R={episode_info['total_reacquire_reward']:.1f}, "
                      f"C={episode_info['total_center_progress']:.1f}, "
                      f"D={episode_info['total_distance_progress']:.1f}, "
                      f"L={episode_info['total_lost_sight_penalty']:.1f}, "
                      f"N={episode_info['total_no_food_penalty']:.1f}")
                min_dist_str = f"{episode_info['min_food_distance_seen']:.1f}" if episode_info['min_food_distance_seen'] is not None else "N/A"
                final_dist_str = f"{episode_info['final_food_distance']:.1f}" if episode_info['final_food_distance'] is not None else "N/A"
                
                print(f"  Food: MinDist={min_dist_str}, "
                      f"FinalDist={final_dist_str}, "
                      f"VisFrac={episode_info['food_visible_fraction']:.2f}")
                print(f"  Training: PL={train_metrics['policy_loss']:.4f}, "
                      f"VL={train_metrics['value_loss']:.4f}, "
                      f"GradPre={train_metrics['grad_norm_pre_clip']:.4f}, "
                      f"GradPost={train_metrics['grad_norm_post_clip']:.4f}, "
                      f"Clipped={'Y' if train_metrics['grad_clipped'] else 'N'}")
                
                print()
        
        # Save periodic checkpoints
        if episode % args.save_interval == 0 and episode > start_episode:
            checkpoint_path = save_checkpoint_with_training_state(
                run_dir, args, episode, policy, best_reward, episode_rewards, episode_lengths, 
                food_collected_counts, config, time.time() - start_time
            )
    
    # Save final checkpoint
    final_checkpoint_path = save_checkpoint_with_training_state(
        run_dir, args, start_episode + args.episodes, policy, best_reward, episode_rewards, episode_lengths, 
        food_collected_counts, config, time.time() - start_time, is_final=True
    )
    
    print()
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final reward: {episode_reward:.2f}")
    print(f"Training time: {time.time() - start_time:.1f}s")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()