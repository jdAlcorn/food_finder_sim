#!/usr/bin/env python3
"""
Multi-simulation viewer - shows many miniaturized simulations running in parallel
"""

import pygame
import math
import sys
from typing import List, Type, Optional
from src.sim.core import Simulation, SimulationConfig
from src.policy.base import Policy

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


class MultiSimViewer:
    """Viewer for multiple parallel simulations"""
    
    def __init__(self, num_sims: int, policy_class: Type[Policy], 
                 sim_config: SimulationConfig = None, 
                 window_width: int = 1200, window_height: int = 800,
                 show_vision: bool = False, show_stats: bool = True):
        
        self.num_sims = num_sims
        self.policy_class = policy_class
        self.config = sim_config or SimulationConfig()
        self.show_vision = show_vision
        self.show_stats = show_stats
        
        # Calculate grid layout
        self.grid_cols = math.ceil(math.sqrt(num_sims))
        self.grid_rows = math.ceil(num_sims / self.grid_cols)
        
        # Calculate mini-window size
        stats_height = 100 if show_stats else 0
        available_width = window_width - 20  # margins
        available_height = window_height - stats_height - 20
        
        self.mini_width = available_width // self.grid_cols
        self.mini_height = available_height // self.grid_rows
        
        # Scale factor for rendering sim content in mini windows
        self.scale_x = self.mini_width / self.config.world_width
        self.scale_y = self.mini_height / self.config.world_height
        self.scale = min(self.scale_x, self.scale_y) * 0.9  # Leave some margin
        
        # Create simulations and policies
        self.simulations = []
        self.policies = []
        
        for i in range(num_sims):
            sim = Simulation(self.config, seed=i)  # Different seed for each
            policy = policy_class()
            policy.reset()
            
            self.simulations.append(sim)
            self.policies.append(policy)
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Multi-Sim Viewer ({num_sims} simulations)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        
        # Stats tracking
        self.total_food_collected = 0
        self.step_count = 0
    
    def world_to_mini(self, world_x: float, world_y: float, mini_x_offset: int, mini_y_offset: int) -> tuple:
        """Convert world coordinates to mini-window screen coordinates"""
        # Center the scaled world in the mini window
        scaled_width = self.config.world_width * self.scale
        scaled_height = self.config.world_height * self.scale
        
        x_margin = (self.mini_width - scaled_width) / 2
        y_margin = (self.mini_height - scaled_height) / 2
        
        screen_x = mini_x_offset + x_margin + world_x * self.scale
        screen_y = mini_y_offset + y_margin + world_y * self.scale
        
        return int(screen_x), int(screen_y)
    
    def draw_mini_simulation(self, sim: Simulation, mini_x: int, mini_y: int, sim_id: int):
        """Draw a single simulation in a mini window"""
        # Get simulation state
        state = sim.get_state()
        agent_state = state['agent_state']
        food_pos = state['food_position']
        
        # Draw mini window border
        border_color = WHITE if sim_id < len(self.simulations) else GRAY
        pygame.draw.rect(self.screen, border_color, 
                        (mini_x, mini_y, self.mini_width, self.mini_height), 1)
        
        # Draw world bounds (scaled)
        world_rect_x = mini_x + (self.mini_width - self.config.world_width * self.scale) / 2
        world_rect_y = mini_y + (self.mini_height - self.config.world_height * self.scale) / 2
        world_rect_w = self.config.world_width * self.scale
        world_rect_h = self.config.world_height * self.scale
        
        pygame.draw.rect(self.screen, GRAY, 
                        (world_rect_x, world_rect_y, world_rect_w, world_rect_h), 1)
        
        # Draw food
        food_screen_x, food_screen_y = self.world_to_mini(food_pos['x'], food_pos['y'], mini_x, mini_y)
        food_radius = max(1, int(self.config.food_radius * self.scale))
        pygame.draw.circle(self.screen, RED, (food_screen_x, food_screen_y), food_radius)
        
        # Draw agent
        agent_screen_x, agent_screen_y = self.world_to_mini(agent_state['x'], agent_state['y'], mini_x, mini_y)
        agent_radius = max(1, int(self.config.agent_radius * self.scale))
        pygame.draw.circle(self.screen, BLUE, (agent_screen_x, agent_screen_y), agent_radius)
        
        # Draw heading indicator
        head_length = agent_radius + 3
        head_x = agent_screen_x + head_length * math.cos(agent_state['theta'])
        head_y = agent_screen_y + head_length * math.sin(agent_state['theta'])
        pygame.draw.line(self.screen, WHITE, (agent_screen_x, agent_screen_y), (head_x, head_y), 1)
        
        # Draw vision if enabled
        if self.show_vision:
            self.draw_mini_vision(sim, mini_x, mini_y, agent_state)
        
        # Draw sim stats in corner
        stats_text = f"#{sim_id} F:{state['food_collected']}"
        text_surface = self.small_font.render(stats_text, True, WHITE)
        self.screen.blit(text_surface, (mini_x + 2, mini_y + 2))
    
    def draw_mini_vision(self, sim: Simulation, mini_x: int, mini_y: int, agent_state: dict):
        """Draw vision rays in mini window"""
        distances, hit_types, hit_wall_ids = sim.vision_system.compute_vision(sim.agent, sim.food)
        
        agent_x, agent_y = agent_state['x'], agent_state['y']
        theta = agent_state['theta']
        
        fov_rad = math.radians(sim.config.fov_degrees)
        
        # Draw every Nth ray to avoid clutter
        ray_skip = max(1, len(distances) // 16)  # Show ~16 rays max
        
        for i in range(0, len(distances), ray_skip):
            distance = distances[i]
            angle = theta + fov_rad * (i / (len(distances) - 1) - 0.5)
            
            # Calculate ray end point
            ray_length = min(distance, sim.config.max_range) * 0.3  # Shorter for visibility
            end_x = agent_x + ray_length * math.cos(angle)
            end_y = agent_y + ray_length * math.sin(angle)
            
            # Convert to screen coordinates
            start_screen = self.world_to_mini(agent_x, agent_y, mini_x, mini_y)
            end_screen = self.world_to_mini(end_x, end_y, mini_x, mini_y)
            
            # Draw ray
            color = (100, 100, 100) if distance >= sim.config.max_range else (150, 150, 150)
            pygame.draw.line(self.screen, color, start_screen, end_screen, 1)
    
    def draw_global_stats(self):
        """Draw global statistics"""
        if not self.show_stats:
            return
        
        # Calculate stats
        total_food = sum(sim.food_collected for sim in self.simulations)
        avg_food = total_food / len(self.simulations)
        
        # Find best and worst performers
        food_counts = [sim.food_collected for sim in self.simulations]
        best_food = max(food_counts) if food_counts else 0
        worst_food = min(food_counts) if food_counts else 0
        
        # Draw stats panel
        stats_y = self.screen.get_height() - 90
        pygame.draw.rect(self.screen, (20, 20, 20), (0, stats_y, self.screen.get_width(), 90))
        
        stats_lines = [
            f"Simulations: {self.num_sims}  |  Policy: {self.policy_class.__name__}",
            f"Total Food: {total_food}  |  Average: {avg_food:.1f}  |  Best: {best_food}  |  Worst: {worst_food}",
            f"Steps: {self.step_count}  |  Vision: {'ON' if self.show_vision else 'OFF'}",
            "Controls: V=toggle vision, S=toggle stats, R=reset all, ESC=quit"
        ]
        
        for i, line in enumerate(stats_lines):
            color = WHITE if i < 3 else GRAY
            text = self.font.render(line, True, color)
            self.screen.blit(text, (10, stats_y + 5 + i * 20))
    
    def reset_all(self):
        """Reset all simulations"""
        for i, (sim, policy) in enumerate(zip(self.simulations, self.policies)):
            sim.reset(seed=i)
            policy.reset()
        self.step_count = 0
        print("Reset all simulations")
    
    def run(self, fps: int = 30):
        """Run the multi-simulation viewer"""
        running = True
        dt = 1.0 / fps  # Fixed timestep for consistency
        
        print(f"Running {self.num_sims} simulations with {self.policy_class.__name__} policy")
        print("Controls: V=vision, S=stats, R=reset, ESC=quit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_v:
                        self.show_vision = not self.show_vision
                        print(f"Vision: {'ON' if self.show_vision else 'OFF'}")
                    elif event.key == pygame.K_s:
                        self.show_stats = not self.show_stats
                        print(f"Stats: {'ON' if self.show_stats else 'OFF'}")
                    elif event.key == pygame.K_r:
                        self.reset_all()
            
            # Update all simulations
            for sim, policy in zip(self.simulations, self.policies):
                sim_state = sim.get_state()
                action = policy.act(sim_state)
                step_info = sim.step(dt, action)
                
                # Optional: print food collection (but not too spammy)
                if step_info['food_collected_this_step'] and sim.food_collected <= 3:
                    sim_id = self.simulations.index(sim)
                    print(f"Sim #{sim_id}: Food collected! Total: {step_info['food_collected']}")
            
            self.step_count += 1
            
            # Render
            self.screen.fill(BLACK)
            
            # Draw each simulation in its mini window
            for i in range(self.num_sims):
                row = i // self.grid_cols
                col = i % self.grid_cols
                
                mini_x = 10 + col * self.mini_width
                mini_y = 10 + row * self.mini_height
                
                self.draw_mini_simulation(self.simulations[i], mini_x, mini_y, i)
            
            # Draw global stats
            self.draw_global_stats()
            
            pygame.display.flip()
            self.clock.tick(fps)
        
        pygame.quit()


def run_multi_simulation(num_sims: int, policy_class: Type[Policy], 
                        config: SimulationConfig = None, fps: int = 30,
                        window_width: int = 1200, window_height: int = 800,
                        show_vision: bool = False) -> None:
    """
    Run multiple simulations in parallel view
    
    Args:
        num_sims: Number of simulations to run
        policy_class: Policy class to use for all agents
        config: Simulation configuration (optional)
        fps: Frames per second
        window_width: Window width in pixels
        window_height: Window height in pixels
        show_vision: Whether to show vision rays
    """
    viewer = MultiSimViewer(
        num_sims=num_sims,
        policy_class=policy_class,
        sim_config=config,
        window_width=window_width,
        window_height=window_height,
        show_vision=show_vision
    )
    
    viewer.run(fps=fps)


if __name__ == "__main__":
    # Example usage
    from src.policy.scripted import ScriptedPolicy
    
    print("Running 16 simulations with scripted policy...")
    run_multi_simulation(
        num_sims=16,
        policy_class=ScriptedPolicy,
        fps=30,
        show_vision=False
    )