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
from src.policy.checkpoint import save_policy, load_policy
from src.eval.load_suite import load_suite
from src.eval.testcases import TestCase
from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
from src.sim.batched import BatchedSimulation
from src.training.test_case_scheduler import TestCaseScheduler, load_scheduler_config


def setup_simulation_for_test_case(test_case: TestCase, config: SimulationConfig) -> SimulationSingle:
    """
    Set up simulation for a specific test case
    
    Args:
        test_case: Test case to set up
        config: Base simulation configuration
        
    Returns:
        Configured simulation ready for the test case
    """
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
    
    return sim


def extract_run_folder_from_resume_path(resume_path: str) -> str:
    """Extract the run folder from a resume checkpoint path"""
    # Resume path should be like: checkpoints/rl_gru_trained_20241230_143022/rl_gru_trained_best.json
    # We want to extract: checkpoints/rl_gru_trained_20241230_143022
    return os.path.dirname(resume_path)
    """Extract the run folder from a resume checkpoint path"""
    # Resume path should be like: checkpoints/rl_gru_trained_20241230_143022/rl_gru_trained_best.json
    # We want to extract: checkpoints/rl_gru_trained_20241230_143022
    return os.path.dirname(resume_path)


def get_run_folder(args) -> str:
    """Get the run folder - either create new or extract from resume path"""
    if args.resume:
        # Resuming - use the same folder as the original run
        run_folder = extract_run_folder_from_resume_path(args.resume)
        print(f"Resuming run in folder: {run_folder}")
        return run_folder
    else:
        # New run - create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"{args.save_prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run folder: {run_dir}")
        return run_dir


def initialize_policy_and_optimizer(args) -> Tuple[RLGRUPolicy, torch.optim.Optimizer, int, float, deque, deque, deque]:
    """Initialize policy and optimizer, optionally resuming from checkpoint"""
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        
        # Load checkpoint
        policy, config, metadata = load_policy(args.resume)
        
        # Verify policy type
        if not hasattr(policy, 'network') or not hasattr(policy.network, 'gru'):
            raise ValueError(f"Resume checkpoint must be an RL GRU policy, got {type(policy)}")
        
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
        
        # Initialize training tracking
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        food_collected_counts = deque(maxlen=100)
        
        start_episode = 0
        best_reward = float('-inf')
        
        print(f"Initialized new policy with {sum(p.numel() for p in policy.network.parameters())} parameters")
        
        return policy, optimizer, start_episode, best_reward, episode_rewards, episode_lengths, food_collected_counts


def save_checkpoint_with_training_state(run_dir: str, args, episode: int, policy: RLGRUPolicy, 
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
        'training_method': 'RL with GRU',
        'policy_type': 'RLGRU',
        
        # Model hyperparameters
        'encoder_dims': args.encoder_dims,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'v_scale': args.v_scale,
        'omega_scale': args.omega_scale,
        
        # Training hyperparameters
        'lr': args.lr,
        'gamma': args.gamma,
        'lam': args.lam,
        'clip_grad_norm': args.clip_grad_norm,
        'value_loss_coef': args.value_loss_coef,
        'reward_scale': args.reward_scale,
        'time_bonus_coef': args.time_bonus_coef,
        'step_penalty': args.step_penalty,
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
    save_policy(checkpoint_path, 'RLGRU', policy.get_params(), config, metadata, policy)
    
    return checkpoint_path


# Note: respawn_food function removed as episodes now terminate on food collection


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


def compute_reward(step_info: Dict[str, Any], prev_step_info: Optional[Dict[str, Any]], 
                   food_collected_this_step: bool, steps_since_food_seen: int, 
                   step_idx: int, max_steps: int = 600, reward_scale: float = 1.0,
                   time_bonus_coef: float = 0.0, step_penalty: float = 0.001,
                   last_seen_food_distance: Optional[float] = None) -> Tuple[float, int, Dict[str, float], Optional[float]]:
    """
    Compute reward with sensor-based shaping (consistent schema) and detailed component tracking
    
    REWARD DESIGN CHANGES (to eliminate local optima and farming):
    - REMOVED per-step "food visible" bonus to prevent stationary farming behavior
    - ADDED reacquire-only bonus when food becomes visible after being invisible
    - ADDED smoothed progress tracking with grace period for brief vision flicker
    - ADDED small negative progress penalty to discourage backing away from food
    
    Args:
        step_info: Current step info from sim.step()
        prev_step_info: Previous step info (None for first step)
        food_collected_this_step: Whether food was collected this step
        steps_since_food_seen: Steps since food was last visible
        step_idx: Current step index in episode
        max_steps: Maximum steps in episode
        reward_scale: Scale factor for all rewards
        time_bonus_coef: Coefficient for time-to-success bonus
        step_penalty: Per-step penalty
        last_seen_food_distance: Trainer-side memory of last observed food distance
        
    Returns:
        Tuple of (reward, updated_steps_since_food_seen, reward_components_dict, updated_last_seen_food_distance)
    """
    reward = 0.0
    reward_components = {
        'terminal': 0.0,
        'reacquire_bonus': 0.0,  # Renamed from visible_bonus
        'progress': 0.0,
        'step_penalty': 0.0,
        'time_bonus': 0.0
    }
    
    # 1. Terminal reward (dominant)
    if food_collected_this_step:
        terminal_reward = 1000.0 * reward_scale
        time_bonus = time_bonus_coef * (max_steps - step_idx) * reward_scale
        reward += terminal_reward + time_bonus
        reward_components['terminal'] = terminal_reward
        reward_components['time_bonus'] = time_bonus
        steps_since_food_seen = 0
        return reward, steps_since_food_seen, reward_components, None
    
    # 2. Food visibility and proximity shaping (sensor-derived only)
    vision_hit_types = step_info['vision_hit_types']
    vision_distances = step_info['vision_distances']
    
    # Check if food is visible in any ray
    food_visible = False
    min_food_distance = float('inf')
    
    for i, hit_type in enumerate(vision_hit_types):
        if hit_type == 'food':
            food_visible = True
            distance = vision_distances[i]
            if distance is not None and distance < min_food_distance:
                min_food_distance = distance
    
    # Check if food was visible in previous step
    prev_food_visible = False
    if prev_step_info is not None:
        prev_vision_hit_types = prev_step_info['vision_hit_types']
        for hit_type in prev_vision_hit_types:
            if hit_type == 'food':
                prev_food_visible = True
                break
    
    if food_visible:
        steps_since_food_seen = 0
        
        # REACQUIRE-ONLY bonus: Only when food becomes visible after not being visible
        # Eliminates per-step farming of visibility bonus
        if not prev_food_visible:
            reacquire_bonus = 2.0 * reward_scale  # Slightly higher than old per-step bonus
            reward += reacquire_bonus
            reward_components['reacquire_bonus'] = reacquire_bonus
        
        # Update last seen distance for progress tracking
        last_seen_food_distance = min_food_distance
        
        # Progress reward using smoothed distance tracking
        if last_seen_food_distance is not None and last_seen_food_distance < float('inf'):
            progress = last_seen_food_distance - min_food_distance
            # Allow small negative progress (backing away penalty) but cap positive progress
            progress_reward = np.clip(progress, -2.0, 20.0) * reward_scale  # Symmetric range with penalty
            reward += progress_reward
            reward_components['progress'] = progress_reward
            
    else:
        steps_since_food_seen += 1
        
        # Grace period: retain last_seen_food_distance for brief vision flicker
        # After 5 steps without seeing food, clear the distance memory
        if steps_since_food_seen > 5:
            last_seen_food_distance = None
    
    # 3. Time penalty to encourage efficiency
    step_penalty_val = -step_penalty * reward_scale
    reward += step_penalty_val
    reward_components['step_penalty'] = step_penalty_val
    
    return reward, steps_since_food_seen, reward_components, last_seen_food_distance


def run_episode(policy: RLGRUPolicy, sim: SimulationSingle, test_case: TestCase, max_steps: int, 
                dt: float, episode_rng: np.random.RandomState, reward_scale: float = 1.0,
                time_bonus_coef: float = 0.0, step_penalty: float = 0.001) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], 
                                   List[torch.Tensor], List[torch.Tensor], List[bool], Dict[str, Any]]:
    """
    Run a single episode and collect rollout data
    
    Returns:
        Tuple of (observations, actions, rewards, log_probs, values, dones, episode_info)
    """
    observations = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    hidden_states = []  # Store hidden states for potential future use (e.g., debugging, analysis)
    
    # Episode tracking
    total_reward = 0.0
    food_collected = 0
    steps_since_food_seen = 0
    food_seen_this_episode = False
    
    # Detailed diagnostics
    success_step = None
    total_terminal_reward = 0.0
    total_reacquire_bonus = 0.0  # reacquire_bonus from function
    total_progress_reward = 0.0
    total_step_penalty = 0.0
    min_food_distance_seen = float('inf')
    final_food_distance = float('inf')
    food_visible_steps = 0
    total_steps = 0
    
    # Trainer-side memory for smoothed progress tracking (NOT part of agent observation)
    last_seen_food_distance = None
    
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
    
    # Move observation building outside the loop
    from src.policy.obs import build_observation
    
    for step in range(max_steps):
        # Build observation
        obs = build_observation(sim_state, policy.v_scale, policy.omega_scale)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
        
        # Store hidden state before action
        hidden_states.append(policy._hidden_state.clone())
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value, new_hidden = policy.network.act(
                obs_tensor, policy._hidden_state, deterministic=False
            )
        
        # Update hidden state
        policy._hidden_state = new_hidden.detach()
        
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
        reward, steps_since_food_seen, reward_components, last_seen_food_distance = compute_reward(
            step_info, prev_sim_state, food_collected_this_step, steps_since_food_seen, step,
            max_steps, reward_scale, time_bonus_coef, step_penalty, last_seen_food_distance
        )
        
        # Track reward components
        total_terminal_reward += reward_components['terminal']
        total_reacquire_bonus += reward_components['reacquire_bonus']
        total_progress_reward += reward_components['progress']
        total_step_penalty += reward_components['step_penalty']
        
        # Track food distance diagnostics
        vision_hit_types = step_info['vision_hit_types']
        vision_distances = step_info['vision_distances']
        food_visible_this_step = False
        current_min_food_distance = float('inf')
        
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
        observations.append(obs_tensor.squeeze(0))
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
        'hidden_states': hidden_states,
        # Detailed diagnostics
        'success': food_collected > 0,
        'success_step': success_step,
        'total_terminal_reward': total_terminal_reward,
        'total_reacquire_bonus': total_reacquire_bonus,  # reacquire_bonus
        'total_progress_reward': total_progress_reward,
        'total_step_penalty': total_step_penalty,
        'min_food_distance_seen': min_food_distance_seen if min_food_distance_seen < float('inf') else None,
        'final_food_distance': final_food_distance if final_food_distance < float('inf') else None,
        'food_visible_fraction': food_visible_steps / total_steps if total_steps > 0 else 0.0
    }
    
    return observations, actions, rewards, log_probs, values, dones, episode_info


def update_policy(policy: RLGRUPolicy, optimizer: torch.optim.Optimizer,
                  observations: List[torch.Tensor], actions: List[torch.Tensor],
                  rewards: List[float], log_probs: List[torch.Tensor], 
                  values: List[torch.Tensor], dones: List[bool],
                  hidden_states: List[torch.Tensor],
                  gamma: float = 0.99, lam: float = 0.95, 
                  clip_grad_norm: float = 0.5, value_loss_coef: float = 0.5,
                  episode_num: int = 0) -> Dict[str, float]:
    """
    Update policy using collected rollout data with proper GRU sequence training
    
    Returns:
        Dict with training metrics
    """
    # Convert to tensors
    obs_seq = torch.stack(observations)  # [T, obs_dim]
    action_seq = torch.stack(actions)    # [T, action_dim]
    old_log_prob_seq = torch.stack(log_probs)  # [T]
    value_seq = torch.stack(values)      # [T]
    
    T = obs_seq.shape[0]
    
    # Compute next value for bootstrapping (if episode didn't terminate)
    if not dones[-1]:
        with torch.no_grad():
            next_obs = obs_seq[-1].unsqueeze(0)
            next_value = policy.get_value(next_obs, policy._hidden_state).item()
    else:
        next_value = 0.0
    
    # Compute advantages and returns using GAE
    advantages, returns = compute_gae(
        rewards, [v.item() for v in value_seq], next_value, dones, gamma, lam
    )
    
    advantage_seq = torch.tensor(advantages, dtype=torch.float32, device=policy.device)
    return_seq = torch.tensor(returns, dtype=torch.float32, device=policy.device)
    
    # Normalize advantages
    advantage_seq = (advantage_seq - advantage_seq.mean()) / (advantage_seq.std() + 1e-8)
    
    # Initialize hidden state for sequence forward pass
    h0 = policy.network.init_hidden(1, policy.device)
    
    # Forward pass through network over the entire sequence
    # forward_sequence expects [T, obs_dim] and returns [1, T, *] (batch-first)
    action_logits_seq, log_std_seq, new_value_seq, _ = policy.network.forward_sequence(obs_seq, h0)
    
    # Squeeze batch dimension since we have batch_size=1
    # forward_sequence returns [1, T, *], we want [T, *]
    action_logits_seq = action_logits_seq.squeeze(0)  # [T, 2]
    log_std_seq = log_std_seq.squeeze(0)              # [T, 2]
    new_value_seq = new_value_seq.squeeze(0)          # [T]
    
    # Compute new log probabilities using squashed Gaussian (vectorized)
    std_seq = torch.exp(log_std_seq)  # [T, 2]
    
    # Reverse the squashing to get u from executed actions
    # Policy mapping: throttle = (tanh + 1) / 2, steering = tanh
    # Inverse: tanh_throttle = 2*throttle - 1, tanh_steering = steering
    action_tanh = torch.stack([
        2.0 * action_seq[:, 0] - 1.0,  # throttle: [0,1] -> [-1,1]
        action_seq[:, 1]               # steering: [-1,1] -> [-1,1]
    ], dim=1)  # [T, 2]
    
    # Inverse tanh (atanh) to get u, with clamping for numerical stability
    action_tanh_clamped = torch.clamp(action_tanh, -0.999, 0.999)
    u_seq = torch.atanh(action_tanh_clamped)  # [T, 2]
    
    # Compute log prob of u under Gaussian
    dist_seq = torch.distributions.Normal(action_logits_seq, std_seq)
    log_prob_u_seq = dist_seq.log_prob(u_seq).sum(dim=-1)  # [T]
    
    # Apply tanh correction
    tanh_correction = torch.log(1 - action_tanh_clamped.pow(2) + 1e-6).sum(dim=-1)  # [T]
    new_log_prob_seq = log_prob_u_seq - tanh_correction
    
    # Policy loss (simplified A2C, no PPO clipping for now)
    policy_loss = -(new_log_prob_seq * advantage_seq).mean()
    
    # Value loss
    value_loss = F.mse_loss(new_value_seq, return_seq)
    
    # Entropy bonus for exploration (vectorized)
    entropy = dist_seq.entropy().sum(dim=-1).mean()  # [T] -> scalar
    entropy_loss = -0.01 * entropy  # Small entropy coefficient
    
    # Total loss
    total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Compute gradient norm before clipping
    grad_norm_pre_clip = 0.0
    gru_grad_norm = 0.0
    for name, param in policy.network.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            grad_norm_pre_clip += param_grad_norm ** 2
            
            # Track GRU-specific gradients
            if 'gru' in name:
                gru_grad_norm += param_grad_norm ** 2
    
    grad_norm_pre_clip = grad_norm_pre_clip ** 0.5
    gru_grad_norm = gru_grad_norm ** 0.5
    
    # Gradient clipping
    grad_norm_post_clip = torch.nn.utils.clip_grad_norm_(policy.network.parameters(), clip_grad_norm)
    
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'total_loss': total_loss.item(),
        'mean_advantage': advantage_seq.mean().item(),
        'mean_return': return_seq.mean().item(),
        'grad_norm_pre_clip': grad_norm_pre_clip,
        'grad_norm_post_clip': grad_norm_post_clip.item(),
        'grad_clipped': grad_norm_pre_clip > clip_grad_norm,
        'gru_grad_norm': gru_grad_norm
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
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                       help='Scale all rewards by this factor (default: 1.0)')
    parser.add_argument('--time-bonus-coef', type=float, default=0.0,
                       help='Time-to-success bonus coefficient (default: 0.0)')
    parser.add_argument('--step-penalty', type=float, default=0.001,
                       help='Per-step penalty (default: 0.001)')
    
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
                       help='Specific test case ID to use (overrides scheduler and suite cycling)')
    parser.add_argument('--scheduler-config', type=str, default=None,
                       help='Path to test case scheduler config JSON file (enables weighted sampling)')
    parser.add_argument('--scheduler-seed', type=int, default=None,
                       help='Seed for test case scheduler (default: uses main seed + 1000)')
    
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
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
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
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print()
    
    # Load test suite
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
    
    # Set up initial simulation (will be reconfigured per episode if using scheduler)
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
    print()
    
    # Training loop
    start_time = time.time()
    
    # Create episode-specific RNG for deterministic food respawning
    episode_rng = np.random.RandomState(args.seed + 1000)
    
    # Track previous test case for simulation reconfiguration
    previous_test_case_id = None
    
    for episode in range(start_episode, start_episode + args.episodes):
        # Select test case for this episode
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
            args.reward_scale, args.time_bonus_coef, args.step_penalty
        )
        
        # Update policy
        train_metrics = update_policy(
            policy, optimizer, observations, actions, rewards, log_probs, values, dones,
            episode_info['hidden_states'], args.gamma, args.lam, args.clip_grad_norm, args.value_loss_coef,
            episode
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
            best_checkpoint_path = save_checkpoint_with_training_state(
                run_dir, args, episode, policy, best_reward, episode_rewards, episode_lengths, 
                food_collected_counts, config, time.time() - start_time, is_best=True
            )
        
        # Logging with detailed diagnostics
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_food = np.mean(food_collected_counts) if food_collected_counts else 0
            
            # Success rate over last 100 episodes
            recent_episodes = min(100, len(food_collected_counts))
            recent_successes = sum(1 for x in list(food_collected_counts)[-recent_episodes:] if x > 0)
            success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0.0
            
            print(f"Episode {episode:5d} | "
                  f"Case: {current_test_case.id[:20]:20s} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"Length: {episode_length:3d} | "
                  f"Success: {'Y' if episode_info['success'] else 'N'} | "
                  f"Step: {episode_info['success_step'] if episode_info['success_step'] is not None else 'N/A'} | "
                  f"SR: {success_rate:.2f}")
            
            # Detailed diagnostics every 50 episodes
            if episode % (args.log_interval * 5) == 0:
                print(f"  Rewards: T={episode_info['total_terminal_reward']:.1f}, "
                      f"R={episode_info['total_reacquire_bonus']:.1f}, "  # R for Reacquire
                      f"P={episode_info['total_progress_reward']:.1f}, "
                      f"S={episode_info['total_step_penalty']:.3f}")
                min_dist_str = f"{episode_info['min_food_distance_seen']:.1f}" if episode_info['min_food_distance_seen'] is not None else "N/A"
                final_dist_str = f"{episode_info['final_food_distance']:.1f}" if episode_info['final_food_distance'] is not None else "N/A"
                
                print(f"  Food: MinDist={min_dist_str}, "
                      f"FinalDist={final_dist_str}, "
                      f"VisFrac={episode_info['food_visible_fraction']:.2f}")
                print(f"  Training: PL={train_metrics['policy_loss']:.4f}, "
                      f"VL={train_metrics['value_loss']:.4f}, "
                      f"GradPre={train_metrics['grad_norm_pre_clip']:.4f}, "
                      f"GradPost={train_metrics['grad_norm_post_clip']:.4f}, "
                      f"GRU_Grad={train_metrics['gru_grad_norm']:.4f}, "
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