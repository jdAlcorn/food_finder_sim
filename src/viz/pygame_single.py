#!/usr/bin/env python3
"""
Interactive GUI version of the 2D simulation
Uses src.sim.core for all physics/logic
"""

import pygame
import math
import sys
from typing import Protocol
from src.sim import Simulation, SimulationConfig
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


class PolicyProtocol(Protocol):
    """Protocol for policy/controller interface"""
    def reset(self) -> None:
        """Reset policy state (optional)"""
        ...
    
    def act(self, sim_state: dict) -> dict:
        """Get action from current simulation state"""
        ...


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

def draw_ui(screen, font, sim, fps, show_vision, policy_name="Unknown"):
    """Draw UI information"""
    sim_state = sim.get_state()
    agent_state = sim_state['agent_state']
    
    # Agent info
    throttle_text = font.render(f"Throttle: {agent_state['throttle']:.2f}", True, WHITE)
    screen.blit(throttle_text, (10, 10))
    
    # Calculate speed from velocity components
    speed = math.sqrt(agent_state['vx']**2 + agent_state['vy']**2)
    speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
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
    
    food_text = font.render(f"Food collected: {sim_state['food_collected']}", True, WHITE)
    screen.blit(food_text, (10, 160))
    
    policy_text = font.render(f"Policy: {policy_name}", True, WHITE)
    screen.blit(policy_text, (10, 185))
    
    # Controls
    controls = [
        "Controls:",
        "V - Toggle vision display",
        "ESC - Quit"
    ]
    if policy_name == "Manual":
        controls.extend([
            "W - Forward thrust (hold)",
            "A/D - Steer left/right"
        ])
    
    for i, text in enumerate(controls):
        color = GRAY if i == 0 else WHITE
        control_text = font.render(text, True, color)
        screen.blit(control_text, (sim.config.world_width - 220, 10 + i * 25))


def run_simulation_gui(policy: PolicyProtocol, config: SimulationConfig = None, 
                      fps: int = 60, policy_name: str = "Unknown", sim=None, dt: float = None):
    """Run the pygame GUI with given policy
    
    Args:
        policy: Policy to use for actions
        config: Simulation configuration
        fps: Target FPS for display
        policy_name: Name to display in window title
        sim: Pre-configured simulation (optional)
        dt: Fixed timestep to use (optional, defaults to variable timestep based on fps)
    """
    pygame.init()
    
    # Create or use provided simulation
    if sim is None:
        if config is None:
            config = SimulationConfig()
        sim = Simulation(config)
    else:
        # Use provided simulation and extract config
        config = sim.config
    
    screen = pygame.display.set_mode((config.world_width, config.world_height))
    pygame.display.set_caption("2D Continuous Simulation with Ray-Cast Vision")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Game state
    running = True
    show_vision = True
    
    # Reset policy
    policy.reset()
    
    while running:
        if dt is not None:
            # Use fixed timestep
            clock.tick(fps)  # Still limit FPS for display
            sim_dt = dt
        else:
            # Use variable timestep based on actual frame timing
            sim_dt = clock.tick(fps) / 1000.0  # Convert to seconds
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    show_vision = not show_vision
        
        # Get action from policy
        sim_state = sim.get_state()
        action = policy.act(sim_state)
        
        # Update simulation
        step_info = sim.step(sim_dt, action)
        
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
        
        # Draw UI
        draw_ui(screen, font, sim, clock.get_fps(), show_vision, policy_name)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    # Default to manual policy if run directly
    from src.policy.manual import ManualPolicy
    policy = ManualPolicy()
    run_simulation_gui(policy, policy_name="Manual")