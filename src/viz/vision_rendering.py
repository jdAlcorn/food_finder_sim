#!/usr/bin/env python3
"""
Vision cone rendering utilities
Consolidates vision rendering logic used across multiple visualization modules
"""

import math
import pygame
from typing import List, Optional, Tuple, Dict, Any
from src.sim.core import SimulationConfig


def draw_vision_cone(screen: pygame.Surface, agent_x: float, agent_y: float, agent_theta: float,
                    distances: List[float], hit_types: List[str], config: SimulationConfig,
                    show_rays: bool = True, show_polygon: bool = True, show_cone_outline: bool = True):
    """
    Draw complete vision cone visualization
    
    Args:
        screen: Pygame surface to draw on
        agent_x, agent_y: Agent position
        agent_theta: Agent orientation (radians)
        distances: Ray distances (None for max range)
        hit_types: Hit types for each ray ('wall', 'food', None)
        config: Simulation configuration
        show_rays: Whether to draw individual rays
        show_polygon: Whether to draw filled vision polygon
        show_cone_outline: Whether to draw FOV cone outline
    """
    if not distances:
        return
    
    # Colors
    CYAN = (0, 255, 255)
    YELLOW = (255, 255, 0)
    GRAY = (100, 100, 100)
    
    # Calculate FOV parameters
    fov_rad = math.radians(config.fov_degrees)
    num_rays = len(distances)
    
    # Calculate ray angles
    angles = [agent_theta + angle for angle in 
             [fov_rad * (i / (num_rays - 1) - 0.5) for i in range(num_rays)]]
    
    # Calculate hit points
    hit_points = []
    for distance, angle in zip(distances, angles):
        if distance is None:
            distance = config.max_range
        hit_x = agent_x + min(distance, config.max_range) * math.cos(angle)
        hit_y = agent_y + min(distance, config.max_range) * math.sin(angle)
        hit_points.append((hit_x, hit_y))
    
    # Draw filled vision polygon
    if show_polygon and len(hit_points) > 0:
        # Create polygon vertices: agent position + hit points
        polygon_points = [(agent_x, agent_y)]
        for hit_x, hit_y in hit_points:
            # Clamp to screen bounds for drawing
            hit_x = max(0, min(config.world_width, hit_x))
            hit_y = max(0, min(config.world_height, hit_y))
            polygon_points.append((hit_x, hit_y))
        
        if len(polygon_points) > 2:
            # Draw semi-transparent filled polygon
            temp_surface = pygame.Surface((config.world_width, config.world_height))
            temp_surface.set_alpha(30)
            temp_surface.fill((0, 0, 0))
            pygame.draw.polygon(temp_surface, CYAN, polygon_points)
            screen.blit(temp_surface, (0, 0))
            
            # Draw polygon outline
            pygame.draw.polygon(screen, CYAN, polygon_points, 1)
    
    # Draw hit points and rays
    if show_rays:
        for i, (distance, hit_type, angle) in enumerate(zip(distances, hit_types, angles)):
            # Handle None distances
            if distance is None:
                distance = config.max_range
                
            hit_x = agent_x + min(distance, config.max_range) * math.cos(angle)
            hit_y = agent_y + min(distance, config.max_range) * math.sin(angle)
            
            if distance < config.max_range:
                # Draw hit point
                pygame.draw.circle(screen, YELLOW, (int(hit_x), int(hit_y)), 2)
            
            # Draw every 8th ray for clarity
            if i % 8 == 0:
                color = GRAY
                # Clamp ray endpoint for drawing
                draw_hit_x = max(0, min(config.world_width, hit_x))
                draw_hit_y = max(0, min(config.world_height, hit_y))
                pygame.draw.line(screen, color, (agent_x, agent_y), (draw_hit_x, draw_hit_y), 1)
    
    # Draw FOV cone outline
    if show_cone_outline:
        cone_length = config.max_range
        left_angle = agent_theta - fov_rad/2
        right_angle = agent_theta + fov_rad/2
        
        left_x = agent_x + cone_length * math.cos(left_angle)
        left_y = agent_y + cone_length * math.sin(left_angle)
        right_x = agent_x + cone_length * math.cos(right_angle)
        right_y = agent_y + cone_length * math.sin(right_angle)
        
        # Draw FOV cone lines
        pygame.draw.line(screen, YELLOW, (agent_x, agent_y), (left_x, left_y), 1)
        pygame.draw.line(screen, YELLOW, (agent_x, agent_y), (right_x, right_y), 1)


def draw_vision_from_sim_state(screen: pygame.Surface, sim_state: Dict[str, Any], 
                              config: SimulationConfig, **kwargs):
    """
    Draw vision cone from simulation state dictionary
    
    Args:
        screen: Pygame surface to draw on
        sim_state: Simulation state dict with agent_state, vision_distances, vision_hit_types
        config: Simulation configuration
        **kwargs: Additional arguments passed to draw_vision_cone
    """
    agent_state = sim_state['agent_state']
    distances = sim_state.get('vision_distances', [])
    hit_types = sim_state.get('vision_hit_types', [])
    
    if distances:
        draw_vision_cone(
            screen=screen,
            agent_x=agent_state['x'],
            agent_y=agent_state['y'],
            agent_theta=agent_state['theta'],
            distances=distances,
            hit_types=hit_types,
            config=config,
            **kwargs
        )


def draw_vision_from_simulation(screen: pygame.Surface, sim, **kwargs):
    """
    Draw vision cone from simulation object
    
    Args:
        screen: Pygame surface to draw on
        sim: Simulation object with get_state() method
        **kwargs: Additional arguments passed to draw_vision_cone
    """
    sim_state = sim.get_state()
    draw_vision_from_sim_state(screen, sim_state, sim.config, **kwargs)


def create_mock_vision_data(agent_x: float, agent_y: float, agent_theta: float,
                           food_x: float, food_y: float, config: SimulationConfig) -> Tuple[List[float], List[str]]:
    """
    Create mock vision data for visualization when no simulation is running
    Useful for test case creators and static visualizations
    
    Args:
        agent_x, agent_y: Agent position
        agent_theta: Agent orientation (radians)
        food_x, food_y: Food position
        config: Simulation configuration
    
    Returns:
        Tuple of (distances, hit_types) lists
    """
    fov_rad = math.radians(config.fov_degrees)
    num_rays = config.num_rays
    
    # Calculate ray angles
    angles = [agent_theta + angle for angle in 
             [fov_rad * (i / (num_rays - 1) - 0.5) for i in range(num_rays)]]
    
    distances = []
    hit_types = []
    
    for angle in angles:
        # Cast ray to find intersections
        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        
        # Check wall intersections (simple box world)
        wall_distances = []
        
        # Left wall (x = 0)
        if ray_dx < 0:
            t = -agent_x / ray_dx
            if t > 0:
                y_intersect = agent_y + t * ray_dy
                if 0 <= y_intersect <= config.world_height:
                    wall_distances.append(t)
        
        # Right wall (x = world_width)
        if ray_dx > 0:
            t = (config.world_width - agent_x) / ray_dx
            if t > 0:
                y_intersect = agent_y + t * ray_dy
                if 0 <= y_intersect <= config.world_height:
                    wall_distances.append(t)
        
        # Top wall (y = 0)
        if ray_dy < 0:
            t = -agent_y / ray_dy
            if t > 0:
                x_intersect = agent_x + t * ray_dx
                if 0 <= x_intersect <= config.world_width:
                    wall_distances.append(t)
        
        # Bottom wall (y = world_height)
        if ray_dy > 0:
            t = (config.world_height - agent_y) / ray_dy
            if t > 0:
                x_intersect = agent_x + t * ray_dx
                if 0 <= x_intersect <= config.world_width:
                    wall_distances.append(t)
        
        # Check food intersection (simple circle)
        food_radius = 8.0  # Default food radius
        dx_to_food = food_x - agent_x
        dy_to_food = food_y - agent_y
        
        # Ray-circle intersection
        a = ray_dx * ray_dx + ray_dy * ray_dy
        b = 2 * (ray_dx * (-dx_to_food) + ray_dy * (-dy_to_food))
        c = dx_to_food * dx_to_food + dy_to_food * dy_to_food - food_radius * food_radius
        
        discriminant = b * b - 4 * a * c
        food_distance = None
        
        if discriminant >= 0:
            t1 = (-b - math.sqrt(discriminant)) / (2 * a)
            t2 = (-b + math.sqrt(discriminant)) / (2 * a)
            
            # Take the closest positive intersection
            if t1 > 0:
                food_distance = t1
            elif t2 > 0:
                food_distance = t2
        
        # Determine closest hit
        min_wall_distance = min(wall_distances) if wall_distances else config.max_range
        
        if food_distance is not None and food_distance < min_wall_distance:
            distances.append(food_distance)
            hit_types.append('food')
        elif min_wall_distance < config.max_range:
            distances.append(min_wall_distance)
            hit_types.append('wall')
        else:
            distances.append(config.max_range)
            hit_types.append(None)
    
    return distances, hit_types