#!/usr/bin/env python3
"""
Batched simulation for CPU vectorized physics and vision
Runs B environments in parallel using NumPy vectorization
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from ..core import SimulationConfig
from .scene import BatchedScene
from .raycast import BatchedRaycaster
from .obs import build_obs_batch


class BatchedSimulation:
    """
    Batched simulation running B environments in parallel
    Uses vectorized physics and ray casting for performance
    """
    
    def __init__(self, batch_size: int, config: SimulationConfig = None):
        """
        Initialize batched simulation
        
        Args:
            batch_size: Number of environments to run in parallel (B)
            config: Simulation configuration
        """
        self.B = batch_size
        self.config = config or SimulationConfig()
        
        # Initialize scene and raycaster
        self.scene = BatchedScene(batch_size, self.config)
        self.raycaster = BatchedRaycaster(self.config)
        
        # Agent state arrays [B]
        self.agent_x = np.zeros(batch_size, dtype=np.float32)
        self.agent_y = np.zeros(batch_size, dtype=np.float32)
        self.agent_vx = np.zeros(batch_size, dtype=np.float32)
        self.agent_vy = np.zeros(batch_size, dtype=np.float32)
        self.agent_theta = np.zeros(batch_size, dtype=np.float32)
        self.agent_omega = np.zeros(batch_size, dtype=np.float32)
        self.agent_throttle = np.zeros(batch_size, dtype=np.float32)
        
        # Food state arrays [B]
        self.food_x = np.zeros(batch_size, dtype=np.float32)
        self.food_y = np.zeros(batch_size, dtype=np.float32)
        
        # Episode tracking
        self.time = np.zeros(batch_size, dtype=np.float32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.food_collected = np.zeros(batch_size, dtype=np.int32)
        self.done = np.zeros(batch_size, dtype=bool)
        
        # Seed tracking for deterministic food respawning
        self.initial_seeds = np.zeros(batch_size, dtype=np.int64)
        
        # Initialize environments
        self.reset()
    
    def reset(self, seeds: List[int] = None):
        """
        Reset all environments to initial state
        
        SEED ISOLATION DESIGN:
        - Each environment gets a unique seed from the seeds list
        - Seeds must be distinct to ensure different random food spawns per environment
        - Deterministic: same seeds always produce same initial conditions
        
        Args:
            seeds: List of seeds for deterministic reset (length B, all distinct)
        """
        if seeds is None:
            seeds = list(range(self.B))
        
        assert len(seeds) == self.B, f"Expected {self.B} seeds, got {len(seeds)}"
        
        # Store initial seeds for deterministic respawning
        self.initial_seeds = np.array(seeds, dtype=np.int64)
        
        # Reset agent positions to center
        center_x = self.config.world_width / 2
        center_y = self.config.world_height / 2
        
        self.agent_x.fill(center_x)
        self.agent_y.fill(center_y)
        self.agent_vx.fill(0.0)
        self.agent_vy.fill(0.0)
        self.agent_theta.fill(0.0)
        self.agent_omega.fill(0.0)
        self.agent_throttle.fill(0.0)
        
        # Reset episode tracking
        self.time.fill(0.0)
        self.step_count.fill(0)
        self.food_collected.fill(0)
        self.done.fill(False)
        
        # Spawn food deterministically for each environment
        self._spawn_food_batch(seeds)
    
    def reset_to_states(self, agent_states: List[Dict], food_states: List[Dict], 
                       obstacle_states: List[Dict] = None):
        """
        Reset environments to explicit states (for test case evaluation)
        
        Args:
            agent_states: List of agent state dicts with keys: x, y, theta, vx, vy, omega, throttle
            food_states: List of food state dicts with keys: x, y
            obstacle_states: List of obstacle state dicts (unused for now, but kept for API)
        """
        assert len(agent_states) == self.B, f"Expected {self.B} agent states, got {len(agent_states)}"
        assert len(food_states) == self.B, f"Expected {self.B} food states, got {len(food_states)}"
        
        # Set agent states
        for i, state in enumerate(agent_states):
            self.agent_x[i] = state['x']
            self.agent_y[i] = state['y']
            self.agent_theta[i] = state['theta']
            self.agent_vx[i] = state.get('vx', 0.0)
            self.agent_vy[i] = state.get('vy', 0.0)
            self.agent_omega[i] = state.get('omega', 0.0)
            self.agent_throttle[i] = state.get('throttle', 0.0)
        
        # Set food states
        for i, state in enumerate(food_states):
            self.food_x[i] = state['x']
            self.food_y[i] = state['y']
        
        # Reset episode tracking
        self.time.fill(0.0)
        self.step_count.fill(0)
        self.food_collected.fill(0)
        self.done.fill(False)
        
        # Set dummy initial seeds for respawn determinism (not used in test cases typically)
        self.initial_seeds = np.arange(self.B, dtype=np.int64)
        
        # Update scene with food positions
        food_positions = np.stack([self.food_x, self.food_y], axis=1)
        self.scene.set_food_positions(food_positions)
        
        # TODO: Handle obstacle_states when obstacle support is added
        if obstacle_states is not None:
            # For now, just validate it's empty or None per environment
            for i, obs_state in enumerate(obstacle_states):
                if obs_state is not None and (obs_state.get('circles') or obs_state.get('segments')):
                    raise NotImplementedError("Obstacle support not yet implemented in batched simulation")
    
    def _spawn_food_batch(self, seeds: List[int]):
        """
        Spawn food for all environments using deterministic seeding
        
        Args:
            seeds: Seeds for each environment
        """
        margin = self.config.agent_radius + self.config.food_radius + 20
        min_coord = margin
        max_x = self.config.world_width - margin
        max_y = self.config.world_height - margin
        
        # Generate deterministic food positions
        for i, seed in enumerate(seeds):
            rng = np.random.RandomState(seed)
            self.food_x[i] = rng.uniform(min_coord, max_x)
            self.food_y[i] = rng.uniform(min_coord, max_y)
        
        # Update scene with food positions
        food_positions = np.stack([self.food_x, self.food_y], axis=1)
        self.scene.set_food_positions(food_positions)
    
    def step(self, actions: Dict[str, np.ndarray], dt: float) -> Dict[str, Any]:
        """
        Step all environments forward by dt
        
        Args:
            actions: Dict with 'steer' and 'throttle' arrays of shape [B]
            dt: Fixed timestep
            
        Returns:
            Dict with step information for all environments
        """
        steer = actions['steer']
        throttle = actions['throttle']
        
        assert steer.shape == (self.B,), f"Expected steer shape ({self.B},), got {steer.shape}"
        assert throttle.shape == (self.B,), f"Expected throttle shape ({self.B},), got {throttle.shape}"
        
        # Update agent physics (vectorized)
        self._update_agent_physics(steer, throttle, dt)
        
        # Handle wall collisions (vectorized)
        self._handle_wall_collisions()
        
        # Handle obstacle collisions (vectorized)
        self._handle_obstacle_collisions()
        
        # Check food collisions and respawn
        food_collected_mask = self._check_and_handle_food_collisions()
        
        # Update episode tracking
        self.time += dt
        self.step_count += 1
        
        # Compute vision for all environments
        distances, materials = self._compute_vision_batch()
        
        # Build step info
        step_info = {
            'time': self.time.copy(),
            'step': self.step_count.copy(),
            'agent_states': self._get_agent_states(),
            'food_positions': np.stack([self.food_x, self.food_y], axis=1),
            'food_collected': self.food_collected.copy(),
            'food_collected_this_step': food_collected_mask,
            'vision_distances': distances,
            'vision_materials': materials,
            'done': self.done.copy()
        }
        
        return step_info
    
    def _update_agent_physics(self, steer: np.ndarray, throttle: np.ndarray, dt: float):
        """
        Update agent physics for all environments (vectorized)
        
        Args:
            steer: Steering input [-1, 1] for each environment
            throttle: Throttle input [0, 1] for each environment
            dt: Timestep
        """
        # Direct throttle control
        self.agent_throttle = np.clip(throttle, 0.0, 1.0)
        
        # Forward direction
        fx = np.cos(self.agent_theta)
        fy = np.sin(self.agent_theta)
        
        # Apply thrust
        thrust_accel = self.agent_throttle * self.config.agent_max_thrust
        ax = thrust_accel * fx
        ay = thrust_accel * fy
        
        # Update velocity
        self.agent_vx += ax * dt
        self.agent_vy += ay * dt
        
        # Apply linear drag
        drag_factor = np.maximum(0.0, 1.0 - self.config.agent_linear_drag * dt)
        self.agent_vx *= drag_factor
        self.agent_vy *= drag_factor
        
        # Update position
        self.agent_x += self.agent_vx * dt
        self.agent_y += self.agent_vy * dt
        
        # Update angular motion
        steer_clamped = np.clip(steer, -1.0, 1.0)
        self.agent_omega += steer_clamped * self.config.agent_max_turn_accel * dt
        
        # Apply angular drag
        angular_drag_factor = np.maximum(0.0, 1.0 - self.config.agent_angular_drag * dt)
        self.agent_omega *= angular_drag_factor
        
        # Update heading
        self.agent_theta += self.agent_omega * dt
        
        # Normalize theta to [-pi, pi] (vectorized)
        self.agent_theta = np.arctan2(np.sin(self.agent_theta), np.cos(self.agent_theta))
    
    def _handle_wall_collisions(self):
        """
        Handle wall collisions for all environments (vectorized)
        """
        config = self.config
        radius = config.agent_radius
        restitution = config.wall_restitution
        
        # Left wall
        left_collision = self.agent_x - radius < 0
        self.agent_x[left_collision] = radius
        self.agent_vx[left_collision] *= -restitution
        self.agent_omega[left_collision] *= 0.8
        
        # Right wall
        right_collision = self.agent_x + radius > config.world_width
        self.agent_x[right_collision] = config.world_width - radius
        self.agent_vx[right_collision] *= -restitution
        self.agent_omega[right_collision] *= 0.8
        
        # Top wall
        top_collision = self.agent_y - radius < 0
        self.agent_y[top_collision] = radius
        self.agent_vy[top_collision] *= -restitution
        self.agent_omega[top_collision] *= 0.8
        
        # Bottom wall
        bottom_collision = self.agent_y + radius > config.world_height
        self.agent_y[bottom_collision] = config.world_height - radius
        self.agent_vy[bottom_collision] *= -restitution
        self.agent_omega[bottom_collision] *= 0.8
    
    def _handle_obstacle_collisions(self):
        """
        Handle collisions with obstacles (circles and segments) for all environments (vectorized)
        """
        config = self.config
        radius = config.agent_radius
        restitution = config.wall_restitution
        
        # Handle circle obstacle collisions
        for i in range(self.scene.Kc_max):
            # Skip inactive circles or food circles
            active_mask = self.scene.circles_active[:, i]
            food_mask = self.scene.circles_material[:, i] == 2  # MATERIAL_FOOD
            obstacle_mask = active_mask & ~food_mask
            
            if not np.any(obstacle_mask):
                continue
            
            # Compute distances to circle centers
            dx = self.agent_x - self.scene.circles_center[:, i, 0]
            dy = self.agent_y - self.scene.circles_center[:, i, 1]
            distances = np.sqrt(dx * dx + dy * dy)
            
            # Check collision (agent radius + obstacle radius)
            obstacle_radii = self.scene.circles_radius[:, i]
            collision_distance = radius + obstacle_radii
            collision_mask = obstacle_mask & (distances <= collision_distance)
            
            if not np.any(collision_mask):
                continue
            
            # Handle collisions - push agent out and reverse velocity component
            # Normalize collision direction
            safe_distances = np.where(distances > 1e-6, distances, 1e-6)
            nx = dx / safe_distances
            ny = dy / safe_distances
            
            # Push agent out to surface
            target_distance = collision_distance[collision_mask]
            current_distance = distances[collision_mask]
            push_distance = target_distance - current_distance
            
            self.agent_x[collision_mask] += nx[collision_mask] * push_distance
            self.agent_y[collision_mask] += ny[collision_mask] * push_distance
            
            # Reflect velocity component along collision normal
            vn = self.agent_vx[collision_mask] * nx[collision_mask] + self.agent_vy[collision_mask] * ny[collision_mask]
            self.agent_vx[collision_mask] -= (1 + restitution) * vn * nx[collision_mask]
            self.agent_vy[collision_mask] -= (1 + restitution) * vn * ny[collision_mask]
            
            # Reduce angular velocity
            self.agent_omega[collision_mask] *= 0.8
        
        # Handle segment obstacle collisions
        for i in range(self.scene.Ks_max):
            # Skip inactive segments or boundary walls (first 4 segments)
            active_mask = self.scene.segments_active[:, i]
            
            # Only process obstacle segments (material_id >= 3), not boundary walls
            obstacle_mask = active_mask & (self.scene.segments_material[:, i] >= 3)
            
            if not np.any(obstacle_mask):
                continue
            
            # Get segment endpoints
            p1 = self.scene.segments_p1[:, i]  # [B, 2]
            p2 = self.scene.segments_p2[:, i]  # [B, 2]
            
            # Agent positions
            agent_pos = np.stack([self.agent_x, self.agent_y], axis=1)  # [B, 2]
            
            # Find closest point on segment to agent
            seg_vec = p2 - p1  # [B, 2]
            agent_vec = agent_pos - p1  # [B, 2]
            
            # Project agent onto segment line
            seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)  # [B]
            safe_seg_len_sq = np.where(seg_len_sq > 1e-6, seg_len_sq, 1e-6)
            
            t = np.sum(agent_vec * seg_vec, axis=1) / safe_seg_len_sq  # [B]
            t = np.clip(t, 0, 1)  # Clamp to segment
            
            # Closest point on segment
            closest_point = p1 + t[:, np.newaxis] * seg_vec  # [B, 2]
            
            # Distance from agent to closest point
            diff = agent_pos - closest_point  # [B, 2]
            distances = np.sqrt(np.sum(diff * diff, axis=1))  # [B]
            
            # Check collision
            collision_mask = obstacle_mask & (distances <= radius)
            
            if not np.any(collision_mask):
                continue
            
            # Handle collisions - push agent out and reflect velocity
            safe_distances = np.where(distances > 1e-6, distances, 1e-6)
            nx = diff[:, 0] / safe_distances
            ny = diff[:, 1] / safe_distances
            
            # Push agent out
            push_distance = radius - distances[collision_mask]
            self.agent_x[collision_mask] += nx[collision_mask] * push_distance
            self.agent_y[collision_mask] += ny[collision_mask] * push_distance
            
            # Reflect velocity component along collision normal
            vn = self.agent_vx[collision_mask] * nx[collision_mask] + self.agent_vy[collision_mask] * ny[collision_mask]
            self.agent_vx[collision_mask] -= (1 + restitution) * vn * nx[collision_mask]
            self.agent_vy[collision_mask] -= (1 + restitution) * vn * ny[collision_mask]
            
            # Reduce angular velocity
            self.agent_omega[collision_mask] *= 0.8
    
    def _check_and_handle_food_collisions(self) -> np.ndarray:
        """
        Check food collisions and respawn food (vectorized)
        
        Returns:
            Boolean mask indicating which environments collected food this step
        """
        # Compute distances to food
        dx = self.agent_x - self.food_x
        dy = self.agent_y - self.food_y
        distances = np.sqrt(dx * dx + dy * dy)
        
        # Check collision
        collision_distance = self.config.agent_radius + self.config.food_radius
        food_collected_mask = distances <= collision_distance
        
        # Update food count
        self.food_collected += food_collected_mask.astype(np.int32)
        
        # Respawn food for environments that collected it
        if np.any(food_collected_mask):
            self._respawn_food_for_envs(food_collected_mask)
        
        return food_collected_mask
    
    def _respawn_food_for_envs(self, env_mask: np.ndarray):
        """
        Respawn food for specified environments using deterministic seeding
        
        DETERMINISTIC RESPAWN DESIGN:
        - Uses initial seed + food collection count for deterministic respawning
        - Same environment with same food collection history = same food positions
        - Cross-platform reproducible (no hash() dependency)
        
        Args:
            env_mask: Boolean mask indicating which environments need food respawn
        """
        margin = self.config.agent_radius + self.config.food_radius + 20
        min_coord = margin
        max_x = self.config.world_width - margin
        max_y = self.config.world_height - margin
        
        # Generate new food positions for environments that need it
        for i in np.where(env_mask)[0]:
            # Deterministic seed based on initial seed + food collection count
            # This ensures same food positions for same collection history
            respawn_seed = int(self.initial_seeds[i]) + int(self.food_collected[i]) * 1000000
            
            # Ensure seed is in valid range for numpy.random.RandomState
            respawn_seed = respawn_seed % (2**31 - 1)
            
            rng = np.random.RandomState(respawn_seed)
            self.food_x[i] = rng.uniform(min_coord, max_x)
            self.food_y[i] = rng.uniform(min_coord, max_y)
        
        # Update scene with new food positions
        food_positions = np.stack([self.food_x, self.food_y], axis=1)
        self.scene.set_food_positions(food_positions)
    
    def _compute_vision_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vision for all environments using batched ray casting
        
        Returns:
            Tuple of (distances, materials) each shape [B, R]
        """
        # Agent positions and headings
        origins = np.stack([self.agent_x, self.agent_y], axis=1)  # [B, 2]
        headings = self.agent_theta  # [B]
        
        # Cast rays
        distances, materials = self.raycaster.cast_rays(origins, headings, self.scene)
        
        return distances, materials
    
    def _get_agent_states(self) -> Dict[str, np.ndarray]:
        """Get agent states as dictionary of arrays"""
        return {
            'x': self.agent_x.copy(),
            'y': self.agent_y.copy(),
            'vx': self.agent_vx.copy(),
            'vy': self.agent_vy.copy(),
            'theta': self.agent_theta.copy(),
            'omega': self.agent_omega.copy(),
            'throttle': self.agent_throttle.copy(),
            'speed': np.sqrt(self.agent_vx**2 + self.agent_vy**2)
        }
    
    def get_observations(self, v_scale: float = 400.0, omega_scale: float = 10.0, 
                        profiler=None) -> np.ndarray:
        """
        Get neural network observations for all environments
        
        Args:
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            profiler: Optional profiler for timing measurements
            
        Returns:
            Observation batch [B, 388]
        """
        # Compute vision (this is the expensive part)
        profiler and profiler.start_timer('vision_raycast')
        distances, materials = self._compute_vision_batch()
        profiler and profiler.end_timer('vision_raycast')
        
        # Get agent states and build observations (feature extraction)
        profiler and profiler.start_timer('obs_build')
        agent_states = self._get_agent_states()
        observations = build_obs_batch(agent_states, distances, materials, v_scale, omega_scale)
        profiler and profiler.end_timer('obs_build')
        
        return observations
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get full simulation state for all environments
        
        Returns:
            Dict with state information for all environments
        """
        distances, materials = self._compute_vision_batch()
        
        return {
            'time': self.time.copy(),
            'step': self.step_count.copy(),
            'agent_states': self._get_agent_states(),
            'food_positions': np.stack([self.food_x, self.food_y], axis=1),
            'food_collected': self.food_collected.copy(),
            'vision_distances': distances,
            'vision_materials': materials,
            'done': self.done.copy()
        }