#!/usr/bin/env python3
"""
Test script for batched simulation
Verifies that batched simulation produces consistent results with single simulation
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.core import Simulation, SimulationConfig
from src.sim.batched import BatchedSimulation
from src.policy.models.mlp import SimpleMLP
from src.training.es.params import get_flat_params, set_flat_params


def test_batched_vs_single():
    """Test that batched simulation matches single simulation"""
    print("Testing batched vs single simulation...")
    
    # Create config and model
    config = SimulationConfig()
    model = SimpleMLP(input_dim=388, hidden_dims=(64, 32), output_dim=2)
    theta = get_flat_params(model)
    
    # Test parameters
    num_envs = 4
    num_steps = 50
    dt = 1/60
    seeds = [42, 43, 44, 45]
    
    # Run single simulations
    single_results = []
    for seed in seeds:
        # Create single simulation
        sim = Simulation(config, seed=seed)
        
        # Create model copy and load parameters
        model_copy = SimpleMLP(input_dim=388, hidden_dims=(64, 32), output_dim=2)
        set_flat_params(model_copy, theta)
        model_copy.eval()
        
        food_collected = 0
        final_distance = 0
        
        with torch.no_grad():
            for step in range(num_steps):
                # Get state and build observation
                state = sim.get_state()
                
                # Simple observation extraction (matching batched version)
                distances = np.array(state['vision_distances'])
                hit_types = state['vision_hit_types']
                
                # Convert hit types to materials
                materials = np.zeros(len(distances), dtype=np.int16)
                for i, hit_type in enumerate(hit_types):
                    if hit_type == 'wall':
                        materials[i] = 1
                    elif hit_type == 'food':
                        materials[i] = 2
                
                # Build simple observation (just use distances for this test)
                obs = np.concatenate([
                    1.0 - np.clip(distances / config.max_range, 0, 1),  # vision_close
                    (materials == 2).astype(np.float32),               # vision_food
                    (materials == 1).astype(np.float32),               # vision_wall
                    [0, 0, 0, 0]  # Simple proprioception
                ])
                
                # Run model
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                steer_raw, throttle_raw = model_copy(obs_tensor)
                
                steer = float(steer_raw.item())
                throttle = float(throttle_raw.item())
                
                # Step simulation
                action = {'steer': np.clip(steer, -1, 1), 'throttle': np.clip(throttle, 0, 1)}
                step_info = sim.step(dt, action)
                
                if step_info['food_collected_this_step']:
                    food_collected += 1
                
                # Track final distance
                agent_pos = (step_info['agent_state']['x'], step_info['agent_state']['y'])
                food_pos = (step_info['food_position']['x'], step_info['food_position']['y'])
                final_distance = np.sqrt((agent_pos[0] - food_pos[0])**2 + (agent_pos[1] - food_pos[1])**2)
        
        single_results.append((food_collected, final_distance))
    
    print(f"Single simulation results: {single_results}")
    
    # Run batched simulation
    batched_sim = BatchedSimulation(num_envs, config)
    batched_sim.reset(seeds)
    
    # Load model parameters
    set_flat_params(model, theta)
    model.eval()
    
    batched_food_collected = np.zeros(num_envs, dtype=int)
    
    with torch.no_grad():
        for step in range(num_steps):
            # Get batched observations
            observations = batched_sim.get_observations()
            
            # Run model (batched)
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            steer_raw, throttle_raw = model(obs_tensor)
            
            steer = steer_raw.detach().cpu().numpy()
            throttle = throttle_raw.detach().cpu().numpy()
            
            # Step simulation
            actions = {
                'steer': np.clip(steer, -1, 1), 
                'throttle': np.clip(throttle, 0, 1)
            }
            step_info = batched_sim.step(actions, dt)
            
            # Track food collected
            batched_food_collected += step_info['food_collected_this_step'].astype(int)
    
    # Get final state
    final_state = batched_sim.get_state()
    agent_positions = final_state['agent_states']
    food_positions = final_state['food_positions']
    
    # Calculate final distances
    dx = agent_positions['x'] - food_positions[:, 0]
    dy = agent_positions['y'] - food_positions[:, 1]
    batched_final_distances = np.sqrt(dx * dx + dy * dy)
    
    batched_results = list(zip(batched_food_collected, batched_final_distances))
    print(f"Batched simulation results: {batched_results}")
    
    # Compare results
    print("\nComparison:")
    all_match = True
    for i, (single, batched) in enumerate(zip(single_results, batched_results)):
        single_food, single_dist = single
        batched_food, batched_dist = batched
        
        food_match = single_food == batched_food
        dist_close = abs(single_dist - batched_dist) < 1.0  # Allow small numerical differences
        
        print(f"Env {i}: Food {single_food} vs {batched_food} ({'✓' if food_match else '✗'}), "
              f"Distance {single_dist:.2f} vs {batched_dist:.2f} ({'✓' if dist_close else '✗'})")
        
        if not (food_match and dist_close):
            all_match = False
    
    if all_match:
        print("\n✓ All results match! Batched simulation is working correctly.")
    else:
        print("\n⚠ Some results don't match. Check batched simulation implementation.")
    
    return all_match


def test_batched_raycast():
    """Test batched ray casting"""
    print("\nTesting batched ray casting...")
    
    config = SimulationConfig()
    batch_size = 2
    
    # Create batched simulation
    batched_sim = BatchedSimulation(batch_size, config)
    batched_sim.reset([42, 43])
    
    # Set specific agent positions and headings for testing
    batched_sim.agent_x[0] = 400  # Center
    batched_sim.agent_y[0] = 400
    batched_sim.agent_theta[0] = 0  # Facing right
    
    batched_sim.agent_x[1] = 100  # Near left wall
    batched_sim.agent_y[1] = 100
    batched_sim.agent_theta[1] = np.pi/2  # Facing up
    
    # Compute vision
    distances, materials = batched_sim._compute_vision_batch()
    
    print(f"Vision distances shape: {distances.shape}")
    print(f"Vision materials shape: {materials.shape}")
    
    # Check that we get reasonable results
    assert distances.shape == (batch_size, config.num_rays)
    assert materials.shape == (batch_size, config.num_rays)
    assert np.all(distances >= 0)
    assert np.all(distances <= config.max_range)
    assert np.all((materials >= 0) & (materials <= 2))
    
    print("✓ Batched ray casting produces valid results")
    
    # Check that different environments produce different results
    env0_distances = distances[0]
    env1_distances = distances[1]
    
    if not np.array_equal(env0_distances, env1_distances):
        print("✓ Different environments produce different vision results")
    else:
        print("⚠ All environments produce identical vision (might be a problem)")


if __name__ == "__main__":
    print("Batched Simulation Test")
    print("=" * 30)
    
    try:
        # Test ray casting first
        test_batched_raycast()
        
        # Test full simulation comparison
        success = test_batched_vs_single()
        
        if success:
            print("\n🎉 All tests passed! Batched simulation is ready to use.")
        else:
            print("\n❌ Some tests failed. Check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()