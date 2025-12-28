#!/usr/bin/env python3
"""
Batched scene representation with expandable object storage
Supports circles (food, obstacles) and segments (walls, obstacles) with material IDs
"""

import numpy as np
from typing import Tuple, Optional
from ..core import SimulationConfig


# Material IDs
MATERIAL_NONE = 0
MATERIAL_WALL = 1
MATERIAL_FOOD = 2
# 3+ reserved for future obstacles


class BatchedScene:
    """
    Batched scene storage with fixed max capacities and active masks
    Supports B environments with up to Ks_max segments and Kc_max circles each
    """
    
    def __init__(self, batch_size: int, config: SimulationConfig, 
                 ks_max: int = 50, kc_max: int = 20):
        """
        Initialize batched scene
        
        Args:
            batch_size: Number of environments (B)
            config: Simulation configuration
            ks_max: Maximum segments per environment (walls + obstacles)
            kc_max: Maximum circles per environment (food + obstacles)
        """
        self.B = batch_size
        self.config = config
        self.Ks_max = ks_max
        self.Kc_max = kc_max
        
        # Segments (walls + segment obstacles)
        # Shape: [B, Ks_max, 2] for p1 and p2 endpoints
        self.segments_p1 = np.zeros((batch_size, ks_max, 2), dtype=np.float32)
        self.segments_p2 = np.zeros((batch_size, ks_max, 2), dtype=np.float32)
        self.segments_active = np.zeros((batch_size, ks_max), dtype=bool)
        self.segments_material = np.zeros((batch_size, ks_max), dtype=np.int16)
        self.segments_color = np.zeros((batch_size, ks_max, 3), dtype=np.uint8)
        
        # Circles (food + circular obstacles)
        # Shape: [B, Kc_max, 2] for centers, [B, Kc_max] for radii
        self.circles_center = np.zeros((batch_size, kc_max, 2), dtype=np.float32)
        self.circles_radius = np.zeros((batch_size, kc_max), dtype=np.float32)
        self.circles_active = np.zeros((batch_size, kc_max), dtype=bool)
        self.circles_material = np.zeros((batch_size, kc_max), dtype=np.int16)
        self.circles_color = np.zeros((batch_size, kc_max, 3), dtype=np.uint8)
        
        # Initialize with boundary walls for all environments
        self.set_walls_bounds()
    
    def set_walls_bounds(self):
        """Set up boundary walls for all environments"""
        w = self.config.world_width
        h = self.config.world_height
        
        # Wall segments: top, right, bottom, left
        walls = [
            ([0, 0], [w, 0]),      # Top wall
            ([w, 0], [w, h]),      # Right wall  
            ([w, h], [0, h]),      # Bottom wall
            ([0, h], [0, 0])       # Left wall
        ]
        
        # Set walls for all environments
        for i, (p1, p2) in enumerate(walls):
            self.segments_p1[:, i] = p1
            self.segments_p2[:, i] = p2
            self.segments_active[:, i] = True
            self.segments_material[:, i] = MATERIAL_WALL
            
            # Wall colors (different for each wall for debugging)
            if i == 0:    # Top - red
                self.segments_color[:, i] = [255, 0, 0]
            elif i == 1:  # Right - green
                self.segments_color[:, i] = [0, 255, 0]
            elif i == 2:  # Bottom - blue
                self.segments_color[:, i] = [0, 0, 255]
            else:         # Left - yellow
                self.segments_color[:, i] = [255, 255, 0]
    
    def set_food_positions(self, food_xy: np.ndarray):
        """
        Set food positions for all environments
        
        Args:
            food_xy: Food positions shape [B, 2]
        """
        assert food_xy.shape == (self.B, 2), f"Expected shape ({self.B}, 2), got {food_xy.shape}"
        
        # Food goes in circle slot 0
        self.circles_center[:, 0] = food_xy
        self.circles_radius[:, 0] = self.config.food_radius
        self.circles_active[:, 0] = True
        self.circles_material[:, 0] = MATERIAL_FOOD
        self.circles_color[:, 0] = [255, 165, 0]  # Orange
    
    def clear_obstacles(self):
        """Clear all obstacles, keeping walls and food intact"""
        # Clear segment obstacles (keep first 4 slots for walls)
        if self.Ks_max > 4:
            self.segments_active[:, 4:] = False
            self.segments_material[:, 4:] = MATERIAL_NONE
        
        # Clear circle obstacles (keep first slot for food)
        if self.Kc_max > 1:
            self.circles_active[:, 1:] = False
            self.circles_material[:, 1:] = MATERIAL_NONE
    
    def add_circle_obstacle(self, slot_idx: int, centers: np.ndarray, radii: np.ndarray,
                           material_id: int = 3, color: Tuple[int, int, int] = (128, 128, 128)):
        """
        Add circular obstacles to specified slot
        
        Args:
            slot_idx: Circle slot index (must be >= 1, slot 0 reserved for food)
            centers: Centers shape [B, 2]
            radii: Radii shape [B] or scalar
            material_id: Material ID for obstacles
            color: RGB color tuple
        """
        assert 1 <= slot_idx < self.Kc_max, f"Invalid circle slot {slot_idx}, must be in [1, {self.Kc_max})"
        assert centers.shape == (self.B, 2), f"Expected centers shape ({self.B}, 2), got {centers.shape}"
        
        if np.isscalar(radii):
            radii = np.full(self.B, radii, dtype=np.float32)
        assert radii.shape == (self.B,), f"Expected radii shape ({self.B},), got {radii.shape}"
        
        self.circles_center[:, slot_idx] = centers
        self.circles_radius[:, slot_idx] = radii
        self.circles_active[:, slot_idx] = True
        self.circles_material[:, slot_idx] = material_id
        self.circles_color[:, slot_idx] = color
    
    def add_segment_obstacle(self, slot_idx: int, p1: np.ndarray, p2: np.ndarray,
                            material_id: int = 3, color: Tuple[int, int, int] = (128, 128, 128)):
        """
        Add segment obstacles to specified slot
        
        Args:
            slot_idx: Segment slot index (must be >= 4, slots 0-3 reserved for walls)
            p1: Start points shape [B, 2]
            p2: End points shape [B, 2]
            material_id: Material ID for obstacles
            color: RGB color tuple
        """
        assert 4 <= slot_idx < self.Ks_max, f"Invalid segment slot {slot_idx}, must be in [4, {self.Ks_max})"
        assert p1.shape == (self.B, 2), f"Expected p1 shape ({self.B}, 2), got {p1.shape}"
        assert p2.shape == (self.B, 2), f"Expected p2 shape ({self.B}, 2), got {p2.shape}"
        
        self.segments_p1[:, slot_idx] = p1
        self.segments_p2[:, slot_idx] = p2
        self.segments_active[:, slot_idx] = True
        self.segments_material[:, slot_idx] = material_id
        self.segments_color[:, slot_idx] = color
    
    def get_active_segments(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get active segments data
        
        Returns:
            Tuple of (p1, p2, materials, active_mask) all with active segments only
        """
        return (
            self.segments_p1,
            self.segments_p2, 
            self.segments_material,
            self.segments_active
        )
    
    def get_active_circles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get active circles data
        
        Returns:
            Tuple of (centers, radii, materials, active_mask) all with active circles only
        """
        return (
            self.circles_center,
            self.circles_radius,
            self.circles_material,
            self.circles_active
        )