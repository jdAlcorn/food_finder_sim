#!/usr/bin/env python3
"""
Shared rendering utilities for pygame-based visualizations
Consolidates common drawing functions used across multiple visualization modules
"""

import math
import pygame
from typing import Dict, Any, List, Tuple, Optional
from src.sim.core import SimulationConfig


# Common colors used across visualizations
class Colors:
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


def draw_agent(screen: pygame.Surface, agent_state: Dict[str, Any], config: SimulationConfig,
               color: Tuple[int, int, int] = None, scale: float = 1.0, offset: Tuple[int, int] = (0, 0)):
    """
    Draw agent with position, orientation indicator
    
    Args:
        screen: Pygame surface to draw on
        agent_state: Agent state dict with x, y, theta
        config: Simulation configuration
        color: Agent color (defaults to BLUE)
        scale: Scale factor for rendering
        offset: (x, y) offset for rendering position
    """
    if color is None:
        color = Colors.BLUE
    
    x = agent_state['x'] * scale + offset[0]
    y = agent_state['y'] * scale + offset[1]
    theta = agent_state['theta']
    radius = max(1, int(config.agent_radius * scale))
    
    # Draw agent circle
    pygame.draw.circle(screen, color, (int(x), int(y)), radius)
    
    # Draw heading indicator
    head_length = radius + max(3, int(8 * scale))
    head_x = x + head_length * math.cos(theta)
    head_y = y + head_length * math.sin(theta)
    line_width = max(1, int(3 * scale))
    pygame.draw.line(screen, Colors.WHITE, (x, y), (head_x, head_y), line_width)


def draw_food(screen: pygame.Surface, food_pos: Dict[str, Any], config: SimulationConfig,
              color: Tuple[int, int, int] = None, scale: float = 1.0, offset: Tuple[int, int] = (0, 0)):
    """
    Draw food item
    
    Args:
        screen: Pygame surface to draw on
        food_pos: Food position dict with x, y
        config: Simulation configuration
        color: Food color (defaults to RED)
        scale: Scale factor for rendering
        offset: (x, y) offset for rendering position
    """
    if color is None:
        color = Colors.RED
    
    x = food_pos['x'] * scale + offset[0]
    y = food_pos['y'] * scale + offset[1]
    radius = max(1, int(config.food_radius * scale))
    
    pygame.draw.circle(screen, color, (int(x), int(y)), radius)


def draw_world_border(screen: pygame.Surface, config: SimulationConfig, 
                     color: Tuple[int, int, int] = None, scale: float = 1.0, 
                     offset: Tuple[int, int] = (0, 0), thickness: int = 2):
    """
    Draw world boundary rectangle
    
    Args:
        screen: Pygame surface to draw on
        config: Simulation configuration
        color: Border color (defaults to WHITE)
        scale: Scale factor for rendering
        offset: (x, y) offset for rendering position
        thickness: Border line thickness
    """
    if color is None:
        color = Colors.WHITE
    
    x = offset[0]
    y = offset[1]
    width = int(config.world_width * scale)
    height = int(config.world_height * scale)
    
    pygame.draw.rect(screen, color, (x, y, width, height), thickness)


def draw_obstacles(screen: pygame.Surface, scene, env_idx: int = 0, 
                  scale: float = 1.0, offset: Tuple[int, int] = (0, 0)):
    """
    Draw obstacles from batched scene for a specific environment
    
    Args:
        screen: Pygame surface to draw on
        scene: BatchedScene instance
        env_idx: Environment index to render (default: 0)
        scale: Scale factor for rendering
        offset: (x, y) offset for rendering position
    """
    # Draw segment obstacles (walls and line obstacles)
    for i in range(scene.Ks_max):
        if scene.segments_active[env_idx, i]:
            # Get segment endpoints
            p1 = scene.segments_p1[env_idx, i]
            p2 = scene.segments_p2[env_idx, i]
            color = tuple(scene.segments_color[env_idx, i])
            
            # Scale and offset coordinates
            x1 = p1[0] * scale + offset[0]
            y1 = p1[1] * scale + offset[1]
            x2 = p2[0] * scale + offset[0]
            y2 = p2[1] * scale + offset[1]
            
            # Draw segment as line
            thickness = max(1, int(3 * scale))
            pygame.draw.line(screen, color, (int(x1), int(y1)), (int(x2), int(y2)), thickness)
    
    # Draw circle obstacles
    for i in range(scene.Kc_max):
        if scene.circles_active[env_idx, i]:
            # Skip food (material ID 2) - it's drawn separately
            if scene.circles_material[env_idx, i] == 2:  # MATERIAL_FOOD
                continue
                
            # Get circle properties
            center = scene.circles_center[env_idx, i]
            radius = scene.circles_radius[env_idx, i]
            color = tuple(scene.circles_color[env_idx, i])
            
            # Scale and offset coordinates
            x = center[0] * scale + offset[0]
            y = center[1] * scale + offset[1]
            scaled_radius = max(1, int(radius * scale))
            
            # Draw circle
            pygame.draw.circle(screen, color, (int(x), int(y)), scaled_radius)


def draw_text_lines(screen: pygame.Surface, lines: List[str], font: pygame.font.Font,
                   position: Tuple[int, int], color: Tuple[int, int, int] = None,
                   line_spacing: int = 25, colors: List[Tuple[int, int, int]] = None):
    """
    Draw multiple lines of text
    
    Args:
        screen: Pygame surface to draw on
        lines: List of text lines to draw
        font: Pygame font to use
        position: (x, y) starting position
        color: Default text color (defaults to WHITE)
        line_spacing: Pixels between lines
        colors: Optional list of colors per line (overrides default color)
    """
    if color is None:
        color = Colors.WHITE
    
    x, y = position
    
    for i, line in enumerate(lines):
        line_color = colors[i] if colors and i < len(colors) else color
        text_surface = font.render(line, True, line_color)
        screen.blit(text_surface, (x, y + i * line_spacing))


def create_depth_strip(distances: List[float], hit_types: List[str], hit_wall_ids: List[int],
                      config: SimulationConfig, strip_height: int = 60) -> pygame.Surface:
    """
    Create 1D depth visualization strip
    
    Args:
        distances: Ray distances
        hit_types: Hit types for each ray
        hit_wall_ids: Wall IDs for wall hits
        config: Simulation configuration
        strip_height: Height of the strip in pixels
    
    Returns:
        Pygame surface containing the depth strip
    """
    strip = pygame.Surface((config.num_rays, strip_height))
    strip.fill(Colors.BLACK)
    
    for i, (distance, hit_type, wall_id) in enumerate(zip(distances, hit_types, hit_wall_ids)):
        if hit_type is not None:  # Hit something
            # Brightness based on distance
            brightness = max(0.0, 1.0 - distance / config.max_range)
            
            # Get color based on hit type
            if hit_type == 'food':
                color = Colors.RED
            elif hit_type == 'wall' and wall_id is not None and wall_id < len(Colors.WALL_COLORS):
                color = Colors.WALL_COLORS[wall_id]
            else:
                color = Colors.WHITE
            
            # Apply brightness
            bright_color = (
                int(color[0] * brightness),
                int(color[1] * brightness),
                int(color[2] * brightness)
            )
            
            # Fill the column
            pygame.draw.rect(strip, bright_color, (i, 0, 1, strip_height))
    
    return strip


def handle_common_events(event: pygame.event.Event, show_vision: bool) -> Tuple[bool, bool, bool]:
    """
    Handle common pygame events across visualizations
    
    Args:
        event: Pygame event
        show_vision: Current vision display state
    
    Returns:
        Tuple of (should_quit, new_show_vision, vision_toggled)
    """
    should_quit = False
    vision_toggled = False
    
    if event.type == pygame.QUIT:
        should_quit = True
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            should_quit = True
        elif event.key == pygame.K_v:
            show_vision = not show_vision
            vision_toggled = True
    
    return should_quit, show_vision, vision_toggled


class CoordinateTransform:
    """Helper class for coordinate transformations in multi-sim views"""
    
    def __init__(self, world_width: float, world_height: float, 
                 mini_width: int, mini_height: int):
        """
        Initialize coordinate transformer
        
        Args:
            world_width, world_height: World dimensions
            mini_width, mini_height: Mini window dimensions in pixels
        """
        self.world_width = world_width
        self.world_height = world_height
        self.mini_width = mini_width
        self.mini_height = mini_height
        
        # Calculate scale factor
        scale_x = mini_width / world_width
        scale_y = mini_height / world_height
        self.scale = min(scale_x, scale_y) * 0.9  # Leave some margin
        
        # Calculate centering offsets
        scaled_width = world_width * self.scale
        scaled_height = world_height * self.scale
        self.x_margin = (mini_width - scaled_width) / 2
        self.y_margin = (mini_height - scaled_height) / 2
    
    def world_to_screen(self, world_x: float, world_y: float, 
                       mini_x_offset: int, mini_y_offset: int) -> Tuple[int, int]:
        """
        Convert world coordinates to mini-window screen coordinates
        
        Args:
            world_x, world_y: World coordinates
            mini_x_offset, mini_y_offset: Mini window position on screen
        
        Returns:
            Screen coordinates as (x, y) tuple
        """
        screen_x = mini_x_offset + self.x_margin + world_x * self.scale
        screen_y = mini_y_offset + self.y_margin + world_y * self.scale
        return int(screen_x), int(screen_y)


def draw_simulation_state(screen: pygame.Surface, sim_state: Dict[str, Any], config: SimulationConfig,
                         scale: float = 1.0, offset: Tuple[int, int] = (0, 0),
                         show_border: bool = True, border_thickness: int = 2,
                         scene=None, env_idx: int = 0):
    """
    Draw complete simulation state (agent + food + border + obstacles)
    
    Args:
        screen: Pygame surface to draw on
        sim_state: Simulation state dict
        config: Simulation configuration
        scale: Scale factor for rendering
        offset: (x, y) offset for rendering position
        show_border: Whether to draw world border
        border_thickness: Border line thickness
        scene: Optional BatchedScene for obstacle rendering
        env_idx: Environment index for obstacle rendering
    """
    # Draw border
    if show_border:
        draw_world_border(screen, config, scale=scale, offset=offset, thickness=border_thickness)
    
    # Draw obstacles if scene is provided
    if scene is not None:
        draw_obstacles(screen, scene, env_idx=env_idx, scale=scale, offset=offset)
    
    # Draw food
    food_pos = sim_state['food_position']
    draw_food(screen, food_pos, config, scale=scale, offset=offset)
    
    # Draw agent
    agent_state = sim_state['agent_state']
    draw_agent(screen, agent_state, config, scale=scale, offset=offset)


def create_info_panel_background(screen: pygame.Surface, x: int, y: int, 
                                width: int, height: int, 
                                background_color: Tuple[int, int, int] = None,
                                border_color: Tuple[int, int, int] = None):
    """
    Create a background panel for UI information
    
    Args:
        screen: Pygame surface to draw on
        x, y: Panel position
        width, height: Panel dimensions
        background_color: Panel background color
        border_color: Panel border color (optional)
    """
    if background_color is None:
        background_color = (20, 20, 20)
    
    # Draw background
    pygame.draw.rect(screen, background_color, (x, y, width, height))
    
    # Draw border if specified
    if border_color is not None:
        pygame.draw.rect(screen, border_color, (x, y, width, height), 1)