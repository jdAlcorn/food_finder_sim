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
from src.viz.rendering_utils import Colors, create_depth_strip, draw_text_lines, handle_common_events


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
    return create_depth_strip(distances, hit_types, hit_wall_ids, config)


def draw_agent(screen, agent_state, config):
    """Draw the agent"""
    from src.viz.rendering_utils import draw_agent as draw_agent_util
    draw_agent_util(screen, agent_state, config)


def draw_food(screen, food_pos, config):
    """Draw the food"""
    from src.viz.rendering_utils import draw_food as draw_food_util
    draw_food_util(screen, food_pos, config)


def draw_vision(screen, sim, show_vision):
    """Draw the FOV cone and ray hits visualization"""
    if not show_vision:
        return
    
    draw_vision_from_simulation(screen, sim)

def draw_ui(screen, font, sim, fps, show_vision, policy_name="Unknown"):
    """Draw UI information"""
    sim_state = sim.get_state()
    agent_state = sim_state['agent_state']
    
    # Calculate speed from velocity components
    speed = math.sqrt(agent_state['vx']**2 + agent_state['vy']**2)
    heading_degrees = math.degrees(agent_state['theta'])
    
    # Prepare info lines
    info_lines = [
        f"Throttle: {agent_state['throttle']:.2f}",
        f"Speed: {speed:.1f}",
        f"FPS: {fps:.1f}",
        f"Position: ({agent_state['x']:.1f}, {agent_state['y']:.1f})",
        f"Heading: {heading_degrees:.1f}°",
        f"Vision: {'ON' if show_vision else 'OFF'}",
        f"Food collected: {sim_state['food_collected']}",
        f"Policy: {policy_name}"
    ]
    
    # Draw info lines
    draw_text_lines(screen, info_lines, font, (10, 10), Colors.WHITE)
    
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
    
    # Prepare control colors
    control_colors = [Colors.GRAY if i == 0 else Colors.WHITE for i in range(len(controls))]
    
    # Draw controls
    draw_text_lines(screen, controls, font, (sim.config.world_width - 220, 10), 
                   colors=control_colors)


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
            should_quit, show_vision, vision_toggled = handle_common_events(event, show_vision)
            if should_quit:
                running = False
            if vision_toggled:
                # Optional: print vision toggle message
                pass
        
        # Get action from policy
        sim_state = sim.get_state()
        action = policy.act(sim_state)
        
        # Update simulation
        step_info = sim.step(sim_dt, action)
        
        # Print food collection
        if step_info['food_collected_this_step']:
            print(f"FOOD REACHED! Total collected: {step_info['food_collected']}")
        
        # Render
        screen.fill(Colors.BLACK)
        
        # Draw border
        pygame.draw.rect(screen, Colors.WHITE, (0, 0, config.world_width, config.world_height), 2)
        
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
            pygame.draw.rect(screen, Colors.WHITE, (strip_x - 1, strip_y - 1, config.num_rays * 3 + 2, 62), 1)
            
            # Strip label
            strip_label = font.render("LiDAR Depth Strip", True, Colors.WHITE)
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