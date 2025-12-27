#!/usr/bin/env python3
"""
Core 2D Continuous Simulation Engine
- Pure math/physics simulation
- No graphics dependencies
- Optimized for batch processing
- Deterministic with seed control
"""

import math
import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class SimulationConfig:
    """Configuration for simulation parameters"""
    def __init__(self):
        # World
        self.world_width = 800
        self.world_height = 800
        
        # Agent
        self.agent_radius = 12
        self.agent_max_thrust = 500.0
        self.agent_max_turn_accel = 13.0
        self.agent_linear_drag = 2.0
        self.agent_angular_drag = 5.0
        self.wall_restitution = 0.6
        
        # Food
        self.food_radius = 8
        
        # Vision
        self.fov_degrees = 120
        self.num_rays = 128
        self.max_range = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'world_width': self.world_width,
            'world_height': self.world_height,
            'agent_radius': self.agent_radius,
            'agent_max_thrust': self.agent_max_thrust,
            'agent_max_turn_accel': self.agent_max_turn_accel,
            'agent_linear_drag': self.agent_linear_drag,
            'agent_angular_drag': self.agent_angular_drag,
            'wall_restitution': self.wall_restitution,
            'food_radius': self.food_radius,
            'fov_degrees': self.fov_degrees,
            'num_rays': self.num_rays,
            'max_range': self.max_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class Agent:
    """Agent with physics-based movement and vision"""
    
    def __init__(self, x: float, y: float, config: SimulationConfig):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0  # heading angle in radians
        self.omega = 0.0  # angular velocity
        self.throttle = 0.0  # 0 to 1
        self.config = config
        
    def update(self, dt: float, action: Dict[str, float]):
        """Update agent physics with action dict"""
        steer_input = action.get('steer', 0.0)
        throttle_input = action.get('throttle', 0.0)
        
        # Direct throttle control
        self.throttle = max(0.0, throttle_input)
        
        # Forward direction
        fx = math.cos(self.theta)
        fy = math.sin(self.theta)
        
        # Apply thrust
        thrust_accel = self.throttle * self.config.agent_max_thrust
        ax = thrust_accel * fx
        ay = thrust_accel * fy
        
        # Update velocity
        self.vx += ax * dt
        self.vy += ay * dt
        
        # Apply linear drag
        drag_factor = max(0.0, 1.0 - self.config.agent_linear_drag * dt)
        self.vx *= drag_factor
        self.vy *= drag_factor
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Update angular motion
        self.omega += steer_input * self.config.agent_max_turn_accel * dt
        
        # Apply angular drag
        angular_drag_factor = max(0.0, 1.0 - self.config.agent_angular_drag * dt)
        self.omega *= angular_drag_factor
        
        # Update heading
        self.theta += self.omega * dt
        
        # Normalize theta to [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
    
    def handle_wall_collisions(self):
        """Handle collisions with world boundaries"""
        config = self.config
        
        # Left wall
        if self.x - config.agent_radius < 0:
            self.x = config.agent_radius
            self.vx *= -config.wall_restitution
            self.omega *= 0.8
        
        # Right wall
        if self.x + config.agent_radius > config.world_width:
            self.x = config.world_width - config.agent_radius
            self.vx *= -config.wall_restitution
            self.omega *= 0.8
        
        # Top wall
        if self.y - config.agent_radius < 0:
            self.y = config.agent_radius
            self.vy *= -config.wall_restitution
            self.omega *= 0.8
        
        # Bottom wall
        if self.y + config.agent_radius > config.world_height:
            self.y = config.world_height - config.agent_radius
            self.vy *= -config.wall_restitution
            self.omega *= 0.8
    
    def get_speed(self) -> float:
        """Get current speed"""
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)
    
    def get_state(self) -> Dict[str, float]:
        """Get agent state as dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'theta': self.theta,
            'omega': self.omega,
            'throttle': self.throttle,
            'speed': self.get_speed()
        }


class Food:
    """Food target"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.x = 0.0
        self.y = 0.0
        self.respawn()
    
    def respawn(self):
        """Spawn food at random location"""
        margin = self.config.agent_radius + self.config.food_radius + 20
        self.x = random.uniform(margin, self.config.world_width - margin)
        self.y = random.uniform(margin, self.config.world_height - margin)


class VisionSystem:
    """Ray-casting vision system"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def intersect_ray_circle(self, origin_x: float, origin_y: float, dir_x: float, dir_y: float,
                           circle_x: float, circle_y: float, radius: float) -> Optional[float]:
        """Ray-circle intersection"""
        oc_x = origin_x - circle_x
        oc_y = origin_y - circle_y
        
        a = dir_x * dir_x + dir_y * dir_y
        b = 2.0 * (oc_x * dir_x + oc_y * dir_y)
        c = oc_x * oc_x + oc_y * oc_y - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None
    
    def intersect_ray_segment(self, origin_x: float, origin_y: float, dir_x: float, dir_y: float,
                            p1_x: float, p1_y: float, p2_x: float, p2_y: float) -> Optional[float]:
        """Ray-line segment intersection"""
        seg_x = p2_x - p1_x
        seg_y = p2_y - p1_y
        
        h_x = p1_x - origin_x
        h_y = p1_y - origin_y
        
        a = dir_x * seg_y - dir_y * seg_x
        
        if abs(a) < 1e-10:
            return None
        
        f = 1.0 / a
        s = f * (h_x * seg_y - h_y * seg_x)
        t = f * (h_x * dir_y - h_y * dir_x)
        
        if s > 1e-6 and 0 <= t <= 1:
            return s
        
        return None
    
    def cast_ray(self, origin_x: float, origin_y: float, dir_x: float, dir_y: float,
                food: Food) -> Tuple[Optional[float], Optional[str], Optional[int]]:
        """Cast single ray and find nearest intersection"""
        nearest_distance = float('inf')
        hit_type = None
        hit_wall_id = None
        
        # Wall segments
        walls = [
            (0, 0, self.config.world_width, 0),  # Top wall
            (self.config.world_width, 0, self.config.world_width, self.config.world_height),  # Right
            (self.config.world_width, self.config.world_height, 0, self.config.world_height),  # Bottom
            (0, self.config.world_height, 0, 0)  # Left
        ]
        
        for wall_idx, (p1_x, p1_y, p2_x, p2_y) in enumerate(walls):
            distance = self.intersect_ray_segment(origin_x, origin_y, dir_x, dir_y, p1_x, p1_y, p2_x, p2_y)
            if distance is not None and distance < nearest_distance and distance > 1e-6:
                nearest_distance = distance
                hit_type = 'wall'
                hit_wall_id = wall_idx
        
        # Food intersection
        food_distance = self.intersect_ray_circle(origin_x, origin_y, dir_x, dir_y,
                                                food.x, food.y, self.config.food_radius)
        if food_distance is not None and food_distance < nearest_distance and food_distance > 1e-6:
            nearest_distance = food_distance
            hit_type = 'food'
            hit_wall_id = None
        
        if nearest_distance == float('inf'):
            return None, None, None
        
        return nearest_distance, hit_type, hit_wall_id
    
    def compute_vision(self, agent: Agent, food: Food) -> Tuple[List[float], List[str], List[Optional[int]]]:
        """Compute full vision sweep"""
        fov_rad = math.radians(self.config.fov_degrees)
        angles = np.linspace(-fov_rad/2, fov_rad/2, self.config.num_rays)
        
        distances = []
        hit_types = []
        hit_wall_ids = []
        
        for angle in angles:
            ray_angle = agent.theta + angle
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)
            
            distance, hit_type, hit_wall_id = self.cast_ray(agent.x, agent.y, dir_x, dir_y, food)
            
            if distance is not None and distance <= self.config.max_range:
                distances.append(distance)
                hit_types.append(hit_type)
                hit_wall_ids.append(hit_wall_id)
            else:
                distances.append(self.config.max_range)
                hit_types.append(None)
                hit_wall_ids.append(None)
        
        return distances, hit_types, hit_wall_ids


class Simulation:
    """Main simulation engine"""
    
    def __init__(self, config: SimulationConfig = None, seed: int = None):
        self.config = config or SimulationConfig()
        self.vision_system = VisionSystem(self.config)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize objects
        self.agent = Agent(self.config.world_width // 2, self.config.world_height // 2, self.config)
        self.food = Food(self.config)
        
        # State tracking
        self.time = 0.0
        self.food_collected = 0
        self.step_count = 0
    
    def check_food_collision(self) -> bool:
        """Check if agent reached food"""
        dx = self.agent.x - self.food.x
        dy = self.agent.y - self.food.y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= (self.config.agent_radius + self.config.food_radius)
    
    def step(self, dt: float, action: Dict[str, float]) -> Dict[str, Any]:
        """Single simulation step with action dict"""
        # Update agent
        self.agent.update(dt, action)
        self.agent.handle_wall_collisions()
        
        # Check food collision
        food_collected_this_step = False
        if self.check_food_collision():
            self.food_collected += 1
            self.food.respawn()
            food_collected_this_step = True
        
        # Compute vision
        distances, hit_types, hit_wall_ids = self.vision_system.compute_vision(self.agent, self.food)
        
        # Update time
        self.time += dt
        self.step_count += 1
        
        # Return step info
        return {
            'time': self.time,
            'step': self.step_count,
            'agent_state': self.agent.get_state(),
            'food_position': {'x': self.food.x, 'y': self.food.y},
            'food_collected': self.food_collected,
            'food_collected_this_step': food_collected_this_step,
            'vision_distances': distances,
            'vision_hit_types': hit_types,
            'vision_hit_wall_ids': hit_wall_ids
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get full simulation state"""
        distances, hit_types, hit_wall_ids = self.vision_system.compute_vision(self.agent, self.food)
        
        return {
            'time': self.time,
            'step': self.step_count,
            'agent_state': self.agent.get_state(),
            'food_position': {'x': self.food.x, 'y': self.food.y},
            'food_collected': self.food_collected,
            'vision_distances': distances,
            'vision_hit_types': hit_types,
            'vision_hit_wall_ids': hit_wall_ids
        }
    
    def reset(self, seed: int = None):
        """Reset simulation to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.agent = Agent(self.config.world_width // 2, self.config.world_height // 2, self.config)
        self.food = Food(self.config)
        self.time = 0.0
        self.food_collected = 0
        self.step_count = 0