#!/usr/bin/env python3
"""
Live viewer for training - continuously displays best agent on test cases
Runs in a separate process to avoid blocking training
"""

import os
import sys
import time
import json
import math
import multiprocessing as mp
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import pygame only when needed to avoid multiple initializations
pygame = None

from src.sim.core import SimulationConfig
from src.sim.unified import SimulationSingle
from src.policy.checkpoint import load_policy
from src.eval.load_suite import load_suite
from src.viz.vision_rendering import draw_vision_from_sim_state
from src.viz.rendering_utils import Colors, draw_simulation_state, draw_text_lines


@dataclass
class ViewerConfig:
    """Configuration for the live viewer"""
    episode_seconds: float = 10.0
    dt: float = 1/60
    fps: int = 60
    window_width: int = 800
    window_height: int = 800
    test_suite_path: str = "data/test_suites/basic_v1.json"


class TestCasePolicy:
    """Policy wrapper that loads from checkpoint and runs on test cases"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.policy = None
        self.load_policy()
    
    def load_policy(self):
        """Load policy from checkpoint"""
        try:
            self.policy, _, _ = load_policy(self.checkpoint_path)
            print(f"Viewer: Loaded policy from {self.checkpoint_path}")
        except Exception as e:
            print(f"Viewer: Error loading policy from {self.checkpoint_path}: {e}")
            self.policy = None
    
    def reset(self):
        """Reset policy state"""
        if self.policy:
            self.policy.reset()
    
    def act(self, sim_state: dict) -> dict:
        """Get action from policy"""
        if self.policy:
            return self.policy.act(sim_state)
        else:
            # Return safe default action if policy failed to load
            return {'steer': 0.0, 'throttle': 0.0}


def run_test_case_with_timeout(policy, test_case, config, viewer_config):
    """Run a single test case with the given policy for a fixed duration"""
    # Create simulation with world integration
    from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
    from src.sim.batched import BatchedSimulation
    
    try:
        # Resolve world for this test case
        world = resolve_test_case_world(test_case)
        
        # Create batched simulation and apply world
        batched_sim = BatchedSimulation(1, config)
        updated_config = apply_world_to_simulation(batched_sim, world, [test_case])
        
        # Create unified simulation wrapper
        sim = SimulationSingle(updated_config)
        sim.batched_sim = batched_sim
        
    except Exception as e:
        print(f"Viewer: Warning - Could not apply world geometry: {e}")
        sim = SimulationSingle(config)
    
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
    
    # Reset simulation to test case state
    sim.batched_sim.reset_to_states(agent_states, food_states)
    
    # Reset policy
    policy.reset()
    
    # Run simulation for fixed duration
    max_steps = int(viewer_config.episode_seconds / viewer_config.dt)
    
    for step in range(max_steps):
        # Get action from policy
        sim_state = sim.get_state()
        action = policy.act(sim_state)
        
        # Step simulation
        step_info = sim.step(viewer_config.dt, action)
        
        # Check if food was collected (early termination)
        if step_info.get('food_collected_this_step', False):
            print(f"Viewer: Food collected in test case {test_case.id} at step {step}")
            break
    
    return sim


def viewer_process(message_queue: mp.Queue, viewer_config: ViewerConfig, 
                  initial_checkpoint: Optional[str] = None):
    """
    Main viewer process function
    Continuously cycles through test cases showing the best agent
    """
    print("Viewer: Starting live viewer process")
    
    # Import and initialize pygame only in this process
    global pygame
    import pygame
    pygame.init()
    
    screen = pygame.display.set_mode((viewer_config.window_width, viewer_config.window_height))
    pygame.display.set_caption("ES Training - Live Best Agent Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Load test suite
    try:
        suite = load_suite(viewer_config.test_suite_path)
        print(f"Viewer: Loaded test suite with {len(suite)} test cases")
    except Exception as e:
        print(f"Viewer: Error loading test suite: {e}")
        return
    
    # Initialize policy
    current_checkpoint = initial_checkpoint
    policy = None
    if current_checkpoint:
        policy = TestCasePolicy(current_checkpoint)
    
    # Simulation config
    sim_config = SimulationConfig()
    sim_config.world_width = viewer_config.window_width
    sim_config.world_height = viewer_config.window_height
    
    # Viewer state
    running = True
    current_test_case_idx = 0
    test_case_start_time = time.time()
    show_vision = True
    
    # Persistent simulation state (FIX: Don't recreate every frame)
    current_sim = None
    current_test_case_id = None
    sim_step_count = 0
    food_collected = False
    
    # Colors (using shared Colors class)
    BLACK = Colors.BLACK
    WHITE = Colors.WHITE
    RED = Colors.RED
    GREEN = Colors.GREEN
    BLUE = Colors.BLUE
    
    # Main viewer loop
    while running:
        current_time = time.time()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    show_vision = not show_vision
                elif event.key == pygame.K_n:
                    # Skip to next test case
                    current_test_case_idx = (current_test_case_idx + 1) % len(suite.test_cases)
                    test_case_start_time = current_time
                    current_sim = None  # Force recreation
                    current_test_case_id = None
        
        # Check for new best agent updates
        try:
            while not message_queue.empty():
                message = message_queue.get_nowait()
                if message['type'] == 'new_best':
                    new_checkpoint = message['checkpoint_path']
                    print(f"Viewer: Received new best agent: {new_checkpoint}")
                    
                    # Load new policy
                    policy = TestCasePolicy(new_checkpoint)
                    current_checkpoint = new_checkpoint
                    
                    # Reset simulation to restart with new policy
                    current_sim = None
                    current_test_case_id = None
                    test_case_start_time = current_time
                elif message['type'] == 'shutdown':
                    print("Viewer: Received shutdown signal")
                    running = False
        except:
            pass  # Queue empty or other non-critical error
        
        # Skip rendering if no policy loaded
        if policy is None or policy.policy is None:
            screen.fill(BLACK)
            
            # Show waiting message
            text = font.render("Waiting for best agent...", True, WHITE)
            text_rect = text.get_rect(center=(viewer_config.window_width // 2, viewer_config.window_height // 2))
            screen.blit(text, text_rect)
            
            pygame.display.flip()
            clock.tick(viewer_config.fps)
            continue
        
        # Check if we should advance to next test case
        elapsed_time = current_time - test_case_start_time
        advance_to_next = False
        
        if elapsed_time >= viewer_config.episode_seconds or food_collected:
            advance_to_next = True
        
        # Get current test case
        test_case = suite.test_cases[current_test_case_idx]
        
        # Create or maintain simulation (FIX: Only create when needed)
        if current_sim is None or current_test_case_id != test_case.id:
            print(f"Viewer: Setting up simulation for test case: {test_case.id}")
            
            # Create new simulation for this test case
            from src.eval.world_integration import resolve_test_case_world, apply_world_to_simulation
            from src.sim.batched import BatchedSimulation
            
            try:
                # Resolve world for this test case
                world = resolve_test_case_world(test_case)
                
                # Create batched simulation and apply world
                batched_sim = BatchedSimulation(1, sim_config)
                updated_config = apply_world_to_simulation(batched_sim, world, [test_case])
                
                # Create unified simulation wrapper
                sim = SimulationSingle(updated_config)
                sim.batched_sim = batched_sim
                
            except Exception as e:
                print(f"Viewer: Warning - Could not apply world geometry: {e}")
                sim = SimulationSingle(sim_config)
            
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
            
            # Reset simulation to test case state
            sim.batched_sim.reset_to_states(agent_states, food_states)
            
            # Reset policy (FIX: Only reset when starting new test case)
            policy.reset()
            
            # Update persistent state
            current_sim = sim
            current_test_case_id = test_case.id
            sim_step_count = 0
            food_collected = False
        
        # Step simulation forward by one frame (FIX: Don't recreate, just step)
        if current_sim is not None:
            max_steps_for_case = int(viewer_config.episode_seconds / viewer_config.dt)
            
            if sim_step_count < max_steps_for_case and not food_collected:
                # Get current state and action
                sim_state = current_sim.get_state()
                action = policy.act(sim_state)
                
                # Step simulation by one frame
                step_info = current_sim.step(viewer_config.dt, action)
                sim_step_count += 1
                
                # Check for food collection
                if step_info.get('food_collected_this_step', False):
                    food_collected = True
                    print(f"Viewer: Food collected in test case {test_case.id} at step {sim_step_count}")
        
        # Advance to next test case if needed
        if advance_to_next:
            current_test_case_idx = (current_test_case_idx + 1) % len(suite.test_cases)
            test_case_start_time = current_time
            current_sim = None  # Force recreation for new test case
            current_test_case_id = None
        
        # Render current state
        screen.fill(BLACK)
        
        if current_sim is not None:
            # Get current simulation state for rendering
            sim_state = current_sim.get_state()
            
            # Draw simulation state (border, food, agent, obstacles)
            draw_simulation_state(screen, sim_state, sim_config, scene=current_sim.batched_sim.scene, env_idx=0)
            
            # Draw vision rays if enabled
            if show_vision:
                draw_vision_from_sim_state(screen, sim_state, sim_config)
        
        # Draw UI info
        info_lines = [
            f"Test Case: {test_case.id} ({current_test_case_idx + 1}/{len(suite.test_cases)})",
            f"Time: {elapsed_time:.1f}s / {viewer_config.episode_seconds:.1f}s",
            f"Steps: {sim_step_count}",
            f"Status: {'COLLECTED!' if food_collected else 'Running'}",
            f"Checkpoint: {os.path.basename(current_checkpoint) if current_checkpoint else 'None'}",
            "",
            "Controls: V=vision, N=next case, ESC=close viewer only"
        ]
        
        # Prepare colors for info lines
        info_colors = [GREEN if food_collected and i == 3 else WHITE for i in range(len(info_lines))]
        
        # Draw info lines
        draw_text_lines(screen, info_lines, font, (10, 10), colors=info_colors)
        
        pygame.display.flip()
        clock.tick(viewer_config.fps)
    
    print("Viewer: Shutting down (training continues)")
    pygame.quit()


def start_viewer_process(viewer_config: ViewerConfig, initial_checkpoint: Optional[str] = None) -> tuple:
    """
    Start the viewer process and return (process, message_queue)
    
    Returns:
        Tuple of (Process, Queue) for communication
    """
    message_queue = mp.Queue()
    
    process = mp.Process(
        target=viewer_process,
        args=(message_queue, viewer_config, initial_checkpoint),
        daemon=True  # Dies when main process dies
    )
    
    process.start()
    print(f"Viewer: Started viewer process (PID: {process.pid})")
    
    return process, message_queue


def notify_new_best(message_queue: mp.Queue, checkpoint_path: str):
    """Notify viewer of new best agent"""
    try:
        message = {
            'type': 'new_best',
            'checkpoint_path': checkpoint_path,
            'timestamp': time.time()
        }
        message_queue.put_nowait(message)
    except:
        pass  # Queue full or process dead - not critical


def shutdown_viewer(message_queue: mp.Queue):
    """Request viewer shutdown"""
    try:
        message = {
            'type': 'shutdown',
            'timestamp': time.time()
        }
        message_queue.put_nowait(message)
    except:
        pass  # Queue full or process dead - not critical