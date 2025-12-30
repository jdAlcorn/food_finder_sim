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

def draw_ui(screen, font, sim, fps, show_vision, policy_name="Unknown", test_case_info=None):
    """Draw UI information"""
    sim_state = sim.get_state()
    agent_state = sim_state['agent_state']
    
    # Calculate speed from velocity components
    speed = math.sqrt(agent_state['vx']**2 + agent_state['vy']**2)
    heading_degrees = math.degrees(agent_state['theta'])
    
    # Prepare info lines
    info_lines = [
        f"Step: {sim_state['step']}",
        f"Throttle: {agent_state['throttle']:.2f}",
        f"Speed: {speed:.1f}",
        f"FPS: {fps:.1f}",
        f"Position: ({agent_state['x']:.1f}, {agent_state['y']:.1f})",
        f"Heading: {heading_degrees:.1f}°",
        f"Vision: {'ON' if show_vision else 'OFF'}",
        f"Food collected: {sim_state['food_collected']}",
        f"Policy: {policy_name}"
    ]
    
    # Add test case info if available
    if test_case_info:
        info_lines.extend([
            f"Test Case: {test_case_info['id']} ({test_case_info['index']+1}/{test_case_info['total']})",
            f"Description: {test_case_info['notes'][:40]}{'...' if len(test_case_info['notes']) > 40 else ''}"
        ])
    
    # Draw info lines
    draw_text_lines(screen, info_lines, font, (10, 10), Colors.WHITE)
    
    # Controls
    controls = [
        "Controls:",
        "V - Toggle vision display",
        "ESC - Quit"
    ]
    
    # Add test case cycling controls
    if test_case_info:
        controls.extend([
            "N - Next test case",
            "P - Previous test case", 
            "R - Reset test case"
        ])
    
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
                      fps: int = 60, policy_name: str = "Unknown", sim=None, dt: float = None,
                      test_case_context=None):
    """Run the pygame GUI with given policy
    
    Args:
        policy: Policy to use for actions
        config: Simulation configuration
        fps: Target FPS for display
        policy_name: Name to display in window title
        sim: Pre-configured simulation (optional)
        dt: Fixed timestep to use (optional, defaults to variable timestep based on fps)
        test_case_context: Dict with test case cycling info (optional)
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
    
    # Test case cycling state
    current_test_case_info = None
    if test_case_context:
        current_test_case = test_case_context['test_cases'][test_case_context['current_index']]
        current_test_case_info = {
            'id': current_test_case.id,
            'index': test_case_context['current_index'],
            'total': len(test_case_context['test_cases']),
            'notes': current_test_case.notes
        }
    
    # Reset policy
    policy.reset()
    
    def switch_test_case(new_index):
        """Switch to a different test case"""
        nonlocal sim, config, current_test_case_info, policy_name
        
        if not test_case_context or new_index < 0 or new_index >= len(test_case_context['test_cases']):
            return
        
        test_case_context['current_index'] = new_index
        new_test_case = test_case_context['test_cases'][new_index]
        
        print(f"Switching to test case: {new_test_case.id}")
        
        # Import setup function
        from scripts.play_sim import setup_simulation_from_testcase, setup_simulation_from_world
        
        # Set up new simulation
        new_sim, new_config = setup_simulation_from_testcase(new_test_case, test_case_context['config'])
        
        # Override with world if specified
        if test_case_context['world_override']:
            new_sim, new_config = setup_simulation_from_world(test_case_context['world_override'], new_config)
        
        if new_sim:
            sim = new_sim
            config = new_config
            
            # Update test case info
            current_test_case_info = {
                'id': new_test_case.id,
                'index': new_index,
                'total': len(test_case_context['test_cases']),
                'notes': new_test_case.notes
            }
            
            # Update policy name
            base_policy_name = policy_name.split(' (')[0]  # Remove old test case info
            policy_name = f"{base_policy_name} ({new_test_case.id} - {new_index+1}/{len(test_case_context['test_cases'])})"
            
            # Reset policy for new test case
            policy.reset()
            
            print(f"  Description: {new_test_case.notes}")
    
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
            
            # Handle test case cycling
            if test_case_context and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:  # Next test case
                    next_index = (test_case_context['current_index'] + 1) % len(test_case_context['test_cases'])
                    switch_test_case(next_index)
                elif event.key == pygame.K_p:  # Previous test case
                    prev_index = (test_case_context['current_index'] - 1) % len(test_case_context['test_cases'])
                    switch_test_case(prev_index)
                elif event.key == pygame.K_r:  # Reset current test case
                    switch_test_case(test_case_context['current_index'])
        
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
        
        # Draw obstacles from scene (if simulation has batched scene)
        if hasattr(sim, 'batched_sim') and sim.batched_sim is not None:
            from src.viz.rendering_utils import draw_obstacles
            draw_obstacles(screen, sim.batched_sim.scene, env_idx=0)
        
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
        draw_ui(screen, font, sim, clock.get_fps(), show_vision, policy_name, current_test_case_info)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    # Default to manual policy if run directly
    from src.policy.manual import ManualPolicy
    policy = ManualPolicy()
    run_simulation_gui(policy, policy_name="Manual")