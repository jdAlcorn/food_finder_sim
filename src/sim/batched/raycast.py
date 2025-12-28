#!/usr/bin/env python3
"""
Batched ray casting with true occlusion
Vectorized ray-circle and ray-segment intersections
"""

import numpy as np
from typing import Tuple
from .scene import BatchedScene, MATERIAL_NONE


class BatchedRaycaster:
    """
    Batched ray caster for multiple environments
    Computes ray intersections with circles and segments using vectorized operations
    """
    
    def __init__(self, config):
        """
        Initialize batched raycaster
        
        Args:
            config: SimulationConfig with fov_degrees, num_rays, max_range
        """
        self.config = config
        self.fov_rad = np.radians(config.fov_degrees)
        self.num_rays = config.num_rays
        self.max_range = config.max_range
        
        # Precompute base ray angles relative to agent heading
        self.base_angles = np.linspace(-self.fov_rad/2, self.fov_rad/2, self.num_rays)
    
    def cast_rays(self, origins: np.ndarray, headings: np.ndarray, 
                  scene: BatchedScene) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cast rays from multiple origins with given headings
        
        Args:
            origins: Ray origins shape [B, 2] (agent positions)
            headings: Agent headings shape [B] (in radians)
            scene: BatchedScene with objects to intersect
            
        Returns:
            Tuple of (distances, materials):
            - distances: shape [B, R] with distances to nearest hits (max_range if no hit)
            - materials: shape [B, R] with material IDs (0 if no hit)
        """
        B = origins.shape[0]
        R = self.num_rays
        
        # Compute ray directions for all environments and rays
        # Shape: [B, R]
        ray_angles = headings[:, None] + self.base_angles[None, :]
        
        # Ray directions: [B, R, 2]
        ray_dirs = np.stack([
            np.cos(ray_angles),
            np.sin(ray_angles)
        ], axis=2)
        
        # Initialize results
        min_distances = np.full((B, R), self.max_range, dtype=np.float32)
        hit_materials = np.zeros((B, R), dtype=np.int16)
        
        # Ray-segment intersections
        seg_p1, seg_p2, seg_materials, seg_active = scene.get_active_segments()
        seg_distances = self._intersect_rays_segments(origins, ray_dirs, seg_p1, seg_p2, seg_active)
        
        # Update minimum distances and materials for segments
        self._update_nearest_hits(min_distances, hit_materials, seg_distances, seg_materials, seg_active)
        
        # Ray-circle intersections
        circ_centers, circ_radii, circ_materials, circ_active = scene.get_active_circles()
        circ_distances = self._intersect_rays_circles(origins, ray_dirs, circ_centers, circ_radii, circ_active)
        
        # Update minimum distances and materials for circles
        self._update_nearest_hits(min_distances, hit_materials, circ_distances, circ_materials, circ_active)
        
        # Clamp distances to max_range and clear materials for no-hit
        no_hit_mask = min_distances >= self.max_range
        min_distances = np.clip(min_distances, 0, self.max_range)
        hit_materials[no_hit_mask] = MATERIAL_NONE
        
        return min_distances, hit_materials
    
    def _intersect_rays_segments(self, origins: np.ndarray, ray_dirs: np.ndarray,
                                seg_p1: np.ndarray, seg_p2: np.ndarray, 
                                seg_active: np.ndarray) -> np.ndarray:
        """
        Vectorized ray-segment intersections
        
        Args:
            origins: Ray origins [B, 2]
            ray_dirs: Ray directions [B, R, 2]
            seg_p1: Segment start points [B, Ks, 2]
            seg_p2: Segment end points [B, Ks, 2]
            seg_active: Active segment mask [B, Ks]
            
        Returns:
            Intersection distances [B, R, Ks] (inf for no intersection)
        """
        B, R = ray_dirs.shape[:2]
        Ks = seg_p1.shape[1]
        
        # Expand dimensions for broadcasting
        # origins: [B, 1, 1, 2] -> [B, R, Ks, 2]
        # ray_dirs: [B, R, 1, 2] -> [B, R, Ks, 2]
        # seg_p1: [B, 1, Ks, 2] -> [B, R, Ks, 2]
        # seg_p2: [B, 1, Ks, 2] -> [B, R, Ks, 2]
        
        origins_exp = origins[:, None, None, :]  # [B, 1, 1, 2]
        ray_dirs_exp = ray_dirs[:, :, None, :]   # [B, R, 1, 2]
        seg_p1_exp = seg_p1[:, None, :, :]       # [B, 1, Ks, 2]
        seg_p2_exp = seg_p2[:, None, :, :]       # [B, 1, Ks, 2]
        
        # Segment vectors
        seg_vec = seg_p2_exp - seg_p1_exp  # [B, R, Ks, 2]
        
        # Vector from ray origin to segment start
        h_vec = seg_p1_exp - origins_exp   # [B, R, Ks, 2]
        
        # Cross products for intersection calculation
        # a = ray_dir × seg_vec (2D cross product: dx*sy - dy*sx)
        a = ray_dirs_exp[..., 0] * seg_vec[..., 1] - ray_dirs_exp[..., 1] * seg_vec[..., 0]  # [B, R, Ks]
        
        # Avoid division by zero
        eps = 1e-10
        valid_a = np.abs(a) > eps
        
        # Safe division: replace small values of 'a' with 1.0 to avoid warnings
        # The result will be ignored anyway due to valid_a mask
        a_safe = np.where(valid_a, a, 1.0)
        
        # s = (h × seg_vec) / a  (parameter along ray)
        # t = (h × ray_dir) / a  (parameter along segment)
        h_cross_seg = h_vec[..., 0] * seg_vec[..., 1] - h_vec[..., 1] * seg_vec[..., 0]  # [B, R, Ks]
        h_cross_ray = h_vec[..., 0] * ray_dirs_exp[..., 1] - h_vec[..., 1] * ray_dirs_exp[..., 0]  # [B, R, Ks]
        
        s = np.where(valid_a, h_cross_seg / a_safe, np.inf)
        t = np.where(valid_a, h_cross_ray / a_safe, np.inf)
        
        # Valid intersection conditions:
        # s > eps (positive distance along ray)
        # 0 <= t <= 1 (intersection within segment)
        # segment is active
        valid_intersection = (
            valid_a & 
            (s > eps) & 
            (t >= 0) & 
            (t <= 1) & 
            seg_active[:, None, :]  # Broadcast active mask
        )
        
        # Return distances (inf for invalid intersections)
        distances = np.where(valid_intersection, s, np.inf)
        
        return distances
    
    def _intersect_rays_circles(self, origins: np.ndarray, ray_dirs: np.ndarray,
                               circ_centers: np.ndarray, circ_radii: np.ndarray,
                               circ_active: np.ndarray) -> np.ndarray:
        """
        Vectorized ray-circle intersections
        
        Args:
            origins: Ray origins [B, 2]
            ray_dirs: Ray directions [B, R, 2]
            circ_centers: Circle centers [B, Kc, 2]
            circ_radii: Circle radii [B, Kc]
            circ_active: Active circle mask [B, Kc]
            
        Returns:
            Intersection distances [B, R, Kc] (inf for no intersection)
        """
        B, R = ray_dirs.shape[:2]
        Kc = circ_centers.shape[1]
        
        # Expand dimensions for broadcasting
        origins_exp = origins[:, None, None, :]      # [B, 1, 1, 2]
        ray_dirs_exp = ray_dirs[:, :, None, :]       # [B, R, 1, 2]
        circ_centers_exp = circ_centers[:, None, :, :] # [B, 1, Kc, 2]
        circ_radii_exp = circ_radii[:, None, :]      # [B, 1, Kc]
        
        # Vector from ray origin to circle center
        oc = origins_exp - circ_centers_exp  # [B, R, Kc, 2]
        
        # Quadratic equation coefficients: at² + bt + c = 0
        a = np.sum(ray_dirs_exp * ray_dirs_exp, axis=3)  # [B, R, Kc]
        b = 2.0 * np.sum(oc * ray_dirs_exp, axis=3)     # [B, R, Kc]
        c = np.sum(oc * oc, axis=3) - circ_radii_exp * circ_radii_exp  # [B, R, Kc]
        
        # Discriminant
        discriminant = b * b - 4 * a * c  # [B, R, Kc]
        
        # Valid intersections have non-negative discriminant and non-zero 'a'
        eps = 1e-10
        valid_a = np.abs(a) > eps
        valid_discriminant = discriminant >= 0
        valid_intersection = valid_a & valid_discriminant
        
        # Safe division: replace small values of 'a' with 1.0 to avoid warnings
        a_safe = np.where(valid_a, a, 1.0)
        
        # Compute intersection parameters (only for valid cases)
        sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0))  # Clamp to avoid sqrt of negative
        t1 = np.where(valid_intersection, (-b - sqrt_discriminant) / (2 * a_safe), np.inf)
        t2 = np.where(valid_intersection, (-b + sqrt_discriminant) / (2 * a_safe), np.inf)
        
        # Choose nearest positive intersection
        ray_eps = 1e-6
        t1_valid = (t1 > ray_eps) & valid_intersection
        t2_valid = (t2 > ray_eps) & valid_intersection
        
        # Use t1 if valid, otherwise t2 if valid, otherwise inf
        t = np.where(t1_valid, t1, np.where(t2_valid, t2, np.inf))
        
        # Apply active mask
        distances = np.where(circ_active[:, None, :], t, np.inf)
        
        return distances
    
    def _update_nearest_hits(self, min_distances: np.ndarray, hit_materials: np.ndarray,
                            new_distances: np.ndarray, materials: np.ndarray, 
                            active_mask: np.ndarray):
        """
        Update nearest hit distances and materials
        
        Args:
            min_distances: Current minimum distances [B, R] (modified in-place)
            hit_materials: Current hit materials [B, R] (modified in-place)
            new_distances: New distances to check [B, R, K]
            materials: Material IDs for new objects [B, K]
            active_mask: Active object mask [B, K]
        """
        B, R, K = new_distances.shape
        
        # Find minimum distance for each ray across all objects
        min_new_distances = np.min(new_distances, axis=2)  # [B, R]
        min_indices = np.argmin(new_distances, axis=2)     # [B, R]
        
        # Create index arrays for advanced indexing
        b_indices = np.arange(B)[:, None]  # [B, 1]
        r_indices = np.arange(R)[None, :]  # [1, R]
        
        # Get materials for minimum distance hits
        min_materials = materials[b_indices, min_indices]  # [B, R]
        
        # Check if new minimum is better than current minimum
        is_better = min_new_distances < min_distances
        
        # Also check that the hit object is active
        min_active = active_mask[b_indices, min_indices]  # [B, R]
        is_valid_better = is_better & min_active
        
        # Update minimum distances and materials where better
        min_distances[is_valid_better] = min_new_distances[is_valid_better]
        hit_materials[is_valid_better] = min_materials[is_valid_better]