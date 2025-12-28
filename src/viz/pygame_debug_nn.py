#!/usr/bin/env python3
"""
Debug visualization for neural network observation extraction
Uses manual control for movement, but runs NN policy in parallel to observe data
"""

import pygame
import math
import sys
import time
from src.sim.core import Simulation, SimulationConfig
from src.policy.manual import ManualPolicy
from src.policy.nn_policy_stub import NeuralPolicyStub
from src.viz.vision_rendering import draw_vision_from_simulation

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

# Wall colors for visualization
WALL_COLORS = [
    (255, 255, 255),  # Top wall - White
    (0, 255, 255),    # Right wall - Cyan
    (255, 0, 255),    # Bottom wall - Magenta
    (255, 165, 0)     # Left wall - Orange
]


def build_depth_strip(distances, hit_types, hit_wall_ids, config):
    """Build the 1D depth image strip as a pygame surface"""
    strip = pygame.Surface((config.num_rays, 60))
    strip.fill(BLACK)
    
    for i, (distance, hit_type, wall_id) in enumerate(zip(distances, hit_types, hit_wall_ids)):
        if hit_type is not None:  # Hit something
            # Brightness based on distance
            brightness = max(0.0, 1.0 - distance / config.max_range)
            
            # Get color based on hit type
            if hit_type == 'food':
                color = RED
            elif hit_type == 'wall' and wall_id is not None:
                color = WALL_COLORS[wall_id]
            else:
                color = WHITE
            
            # Apply brightness
            bright_color = (
                int(color[0] * brightness),
                int(color[1] * brightness),
                int(color[2] * brightness)
            )
            
            # Fill the column
            pygame.draw.rect(strip, bright_color, (i, 0, 1, 60))
    
    return strip


def draw_agent(screen, agent_state, config):
    """Draw the agent"""
    x, y = agent_state['x'], agent_state['y']
    theta = agent_state['theta']
    
    # Draw agent circle
    pygame.draw.circle(screen, BLUE, (int(x), int(y)), config.agent_radius)
    
    # Draw heading indicator
    head_length = config.agent_radius + 8
    head_x = x + head_length * math.cos(theta)
    head_y = y + head_length * math.sin(theta)
    pygame.draw.line(screen, WHITE, (x, y), (head_x, head_y), 3)


def draw_food(screen, food_pos, config):
    """Draw the food"""
    pygame.draw.circle(screen, RED, (int(food_pos['x']), int(food_pos['y'])), config.food_radius)


def draw_vision(screen, sim, show_vision):
    """Draw the FOV cone and ray hits visualization"""
    if not show_vision:
        return
    
    draw_vision_from_simulation(screen, sim)


def draw_ui(screen, font, sim, fps, show_vision, nn_obs_info):
    """Draw UI information including NN observation data"""
    agent_state = sim.agent.get_state()
    
    # Standard agent info (left side)
    throttle_text = font.render(f"Throttle: {agent_state['throttle']:.2f}", True, WHITE)
    screen.blit(throttle_text, (10, 10))
    
    speed_text = font.render(f"Speed: {agent_state['speed']:.1f}", True, WHITE)
    screen.blit(speed_text, (10, 35))
    
    fps_text = font.render(f"FPS: {fps:.1f}", True, WHITE)
    screen.blit(fps_text, (10, 60))
    
    pos_text = font.render(f"Position: ({agent_state['x']:.1f}, {agent_state['y']:.1f})", True, WHITE)
    screen.blit(pos_text, (10, 85))
    
    heading_degrees = math.degrees(agent_state['theta'])
    heading_text = font.render(f"Heading: {heading_degrees:.1f}°", True, WHITE)
    screen.blit(heading_text, (10, 110))
    
    vision_text = font.render(f"Vision: {'ON' if show_vision else 'OFF'}", True, WHITE)
    screen.blit(vision_text, (10, 135))
    
    food_text = font.render(f"Food collected: {sim.food_collected}", True, WHITE)
    screen.blit(food_text, (10, 160))
    
    # NN Observation info (right side)
    if nn_obs_info:
        x_offset = sim.config.world_width - 350
        y_start = 10
        
        # Header
        nn_header = font.render("NN Observation Data:", True, GREEN)
        screen.blit(nn_header, (x_offset, y_start))
        
        # Observation stats
        obs_shape = font.render(f"Shape: {nn_obs_info['shape']}", True, WHITE)
        screen.blit(obs_shape, (x_offset, y_start + 25))
        
        obs_range = font.render(f"Range: [{nn_obs_info['min']:.3f}, {nn_obs_info['max']:.3f}]", True, WHITE)
        screen.blit(obs_range, (x_offset, y_start + 50))
        
        # Vision channel stats
        vision_close_text = font.render(f"Vision Close: {nn_obs_info['vision_close_stats']}", True, WHITE)
        screen.blit(vision_close_text, (x_offset, y_start + 80))
        
        vision_food_text = font.render(f"Vision Food: {nn_obs_info['vision_food_count']} hits", True, WHITE)
        screen.blit(vision_food_text, (x_offset, y_start + 105))
        
        vision_wall_text = font.render(f"Vision Wall: {nn_obs_info['vision_wall_count']} hits", True, WHITE)
        screen.blit(vision_wall_text, (x_offset, y_start + 130))
        
        # Proprioception
        prop_text = font.render("Proprioception:", True, GRAY)
        screen.blit(prop_text, (x_offset, y_start + 160))
        
        v_forward_text = font.render(f"  v_forward: {nn_obs_info['v_forward']:.3f}", True, WHITE)
        screen.blit(v_forward_text, (x_offset, y_start + 185))
        
        v_sideways_text = font.render(f"  v_sideways: {nn_obs_info['v_sideways']:.3f}", True, WHITE)
        screen.blit(v_sideways_text, (x_offset, y_start + 210))
        
        omega_text = font.render(f"  omega: {nn_obs_info['omega']:.3f}", True, WHITE)
        screen.blit(omega_text, (x_offset, y_start + 235))
        
        throttle_norm_text = font.render(f"  throttle: {nn_obs_info['throttle']:.3f}", True, WHITE)
        screen.blit(throttle_norm_text, (x_offset, y_start + 260))
    
    # Controls (bottom right)
    controls = [
        "Controls:",
        "W - Forward thrust (hold)",
        "A/D - Steer left/right", 
        "V - Toggle vision display",
        "P - Toggle NN observation printing",
        "ESC - Quit"
    ]
    for i, text in enumerate(controls):
        color = GRAY if i == 0 else WHITE
        control_text = font.render(text, True, color)
        screen.blit(control_text, (sim.config.world_width - 250, sim.config.world_height - 150 + i * 20))


def extract_nn_observation_info(nn_policy, sim_state):
    """Extract NN observation and return summary info for display"""
    # Get the observation from NN policy (without using its action)
    nn_policy.act(sim_state)  # This builds the observation
    obs = nn_policy.get_last_obs()
    
    if obs is None:
        return None
    
    # Extract different parts of the observation
    num_rays = len(sim_state['vision_distances'])
    vision_close = obs[:num_rays]
    vision_food = obs[num_rays:num_rays*2]
    vision_wall = obs[num_rays*2:num_rays*3]
    proprioception = obs[num_rays*3:]
    
    # Calculate stats
    vision_close_max = vision_close.max()
    vision_close_mean = vision_close.mean()
    vision_food_count = int(vision_food.sum())
    vision_wall_count = int(vision_wall.sum())
    
    return {
        'shape': obs.shape,
        'min': obs.min(),
        'max': obs.max(),
        'vision_close_stats': f"max={vision_close_max:.3f}, avg={vision_close_mean:.3f}",
        'vision_food_count': vision_food_count,
        'vision_wall_count': vision_wall_count,
        'v_forward': proprioception[0],
        'v_sideways': proprioception[1], 
        'omega': proprioception[2],
        'throttle': proprioception[3]
    }


def run_debug_nn_visualization(config: SimulationConfig = None, fps: int = 60):
    """
    Run debug visualization with manual control + NN observation monitoring
    """
    pygame.init()
    
    # Create simulation
    if config is None:
        config = SimulationConfig()
    sim = Simulation(config)
    
    screen = pygame.display.set_mode((config.world_width, config.world_height))
    pygame.display.set_caption("NN Debug Visualization - Manual Control + NN Observation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)
    
    # Create both policies
    manual_policy = ManualPolicy()  # For actual control
    nn_policy = NeuralPolicyStub()  # For observation extraction only
    
    # Reset both policies
    manual_policy.reset()
    nn_policy.reset()
    
    # Game state
    running = True
    show_vision = True
    print_nn_obs = False
    last_print_time = 0
    
    print("Debug NN Visualization Started")
    print("Manual control active - NN policy running in parallel for observation extraction")
    print("Press P to toggle NN observation console printing")
    
    while running:
        dt = clock.tick(fps) / 1000.0
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    show_vision = not show_vision
                    print(f"Vision display: {'ON' if show_vision else 'OFF'}")
                elif event.key == pygame.K_p:
                    print_nn_obs = not print_nn_obs
                    print(f"NN observation printing: {'ON' if print_nn_obs else 'OFF'}")
        
        # Get current simulation state
        sim_state = sim.get_state()
        
        # Get action from manual policy (this controls the agent)
        manual_action = manual_policy.act(sim_state)
        
        # Update simulation with manual action
        step_info = sim.step(dt, manual_action)
        
        # Extract NN observation info (for display/debugging only)
        nn_obs_info = extract_nn_observation_info(nn_policy, sim_state)
        
        # Print NN observation to console periodically
        if print_nn_obs and current_time - last_print_time > 1.0:  # Every 1 second
            if nn_obs_info:
                print(f"\n--- NN Observation ---")
                print(f"Shape: {nn_obs_info['shape']}")
                print(f"Range: [{nn_obs_info['min']:.3f}, {nn_obs_info['max']:.3f}]")
                print(f"Vision - Close: {nn_obs_info['vision_close_stats']}")
                print(f"Vision - Food hits: {nn_obs_info['vision_food_count']}")
                print(f"Vision - Wall hits: {nn_obs_info['vision_wall_count']}")
                print(f"Proprioception: v_fwd={nn_obs_info['v_forward']:.3f}, v_side={nn_obs_info['v_sideways']:.3f}, omega={nn_obs_info['omega']:.3f}, throttle={nn_obs_info['throttle']:.3f}")
            last_print_time = current_time
        
        # Print food collection
        if step_info['food_collected_this_step']:
            print(f"FOOD REACHED! Total collected: {step_info['food_collected']}")
        
        # Render
        screen.fill(BLACK)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, (0, 0, config.world_width, config.world_height), 2)
        
        # Draw game objects
        draw_food(screen, step_info['food_position'], config)
        draw_agent(screen, step_info['agent_state'], config)
        
        # Draw vision system
        draw_vision(screen, sim, show_vision)
        
        # Draw depth strip (only when vision is enabled)
        if show_vision:
            depth_strip = build_depth_strip(step_info['vision_distances'], 
                                          step_info['vision_hit_types'],
                                          step_info['vision_hit_wall_ids'], 
                                          config)
            
            strip_x = 10
            strip_y = config.world_height - 60 - 10
            scaled_strip = pygame.transform.scale(depth_strip, (config.num_rays * 3, 60))
            screen.blit(scaled_strip, (strip_x, strip_y))
            
            # Draw strip border
            pygame.draw.rect(screen, WHITE, (strip_x - 1, strip_y - 1, config.num_rays * 3 + 2, 62), 1)
            
            # Strip label
            strip_label = font.render("LiDAR Depth Strip", True, WHITE)
            screen.blit(strip_label, (strip_x, strip_y - 25))
        
        # Draw UI with NN observation info
        draw_ui(screen, font, sim, clock.get_fps(), show_vision, nn_obs_info)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    run_debug_nn_visualization()