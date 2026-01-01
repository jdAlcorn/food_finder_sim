#!/usr/bin/env python3
"""
Vision utilities for calculating food position and distance in agent's field of view
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


def calculate_food_vision_metrics(agent_state: Dict[str, float], food_position: Dict[str, float],
                                 vision_distances: List[Optional[float]], vision_hit_types: List[Optional[str]],
                                 fov_degrees: float = 120.0, num_rays: int = 128) -> Dict[str, Optional[float]]:
    """
    Calculate detailed metrics about food in the agent's vision field
    
    Args:
        agent_state: Agent state with x, y, theta
        food_position: Food position with x, y
        vision_distances: Ray distances from vision system
        vision_hit_types: Ray hit types from vision system
        fov_degrees: Field of view in degrees
        num_rays: Number of vision rays
        
    Returns:
        Dict with vision metrics:
        - food_visible: Whether food is visible in any ray
        - closest_food_distance: Distance to food via closest ray (None if not visible)
        - food_rays_from_center: Integer number of rays away from center (None if not visible)
        - actual_food_distance: Euclidean distance to food
    """
    # Extract agent position and orientation
    agent_x = agent_state['x']
    agent_y = agent_state['y']
    agent_theta = agent_state['theta']
    
    food_x = food_position['x']
    food_y = food_position['y']
    
    # Calculate actual Euclidean distance to food
    dx = food_x - agent_x
    dy = food_y - agent_y
    actual_food_distance = math.sqrt(dx * dx + dy * dy)
    
    # Check if food is visible in vision rays
    food_visible = False
    closest_food_distance = None
    closest_ray_idx = None
    
    for i, (distance, hit_type) in enumerate(zip(vision_distances, vision_hit_types)):
        if hit_type == 'food' and distance is not None:
            food_visible = True
            if closest_food_distance is None or distance < closest_food_distance:
                closest_food_distance = distance
                closest_ray_idx = i
    
    # Calculate rays from center
    food_rays_from_center = None
    if closest_ray_idx is not None:
        # Calculate center ray index
        # For even number of rays, center is between two middle rays
        # For odd number of rays, center is the exact middle ray
        if num_rays % 2 == 0:
            # Even number: center is between rays at indices (num_rays//2 - 1) and (num_rays//2)
            # We'll use the distance to the closer of these two indices
            center_left = num_rays // 2 - 1
            center_right = num_rays // 2
            distance_to_left = abs(closest_ray_idx - center_left)
            distance_to_right = abs(closest_ray_idx - center_right)
            food_rays_from_center = min(distance_to_left, distance_to_right)
        else:
            # Odd number: center is at index num_rays // 2
            center_idx = num_rays // 2
            food_rays_from_center = abs(closest_ray_idx - center_idx)
    
    return {
        'food_visible': food_visible,
        'closest_food_distance': closest_food_distance,
        'food_rays_from_center': food_rays_from_center,
        'actual_food_distance': actual_food_distance
    }


def calculate_center_vision_reward(prev_rays_from_center: Optional[int], 
                                  curr_rays_from_center: Optional[int],
                                  reward_scale: float = 1.0) -> float:
    """
    Calculate reward for food moving toward/away from center of vision
    
    Args:
        prev_rays_from_center: Previous number of rays away from center (None if not visible)
        curr_rays_from_center: Current number of rays away from center (None if not visible)
        reward_scale: Scale factor for reward
        
    Returns:
        Reward value (positive for moving toward center, negative for moving away)
        Note: Moving away from center is penalized more heavily than moving toward center is rewarded
    """
    if prev_rays_from_center is None or curr_rays_from_center is None:
        return 0.0
    
    if curr_rays_from_center < 20:
        return 5 * reward_scale
    else:
        return -5 * reward_scale
    # Calculate the change in rays from center
    # e.g. if food was far left (+20) and is now closer (+9) our bonus is +11
    # but what if +20 to -20? (20 - (-20)) = 40 oh boy
    center_progress = prev_rays_from_center - curr_rays_from_center
    
    if center_progress > 0:
        # Moving toward center (positive progress) - normal reward
        return center_progress * reward_scale
    elif center_progress < 0:
        # Moving away from center (negative progress) - amplified penalty
        penalty_multiplier = 2.0  # Penalize moving away 2x more than rewarding moving toward
        return center_progress * reward_scale * penalty_multiplier
    else:
        # No change
        return 0.0