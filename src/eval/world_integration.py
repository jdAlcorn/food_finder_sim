#!/usr/bin/env python3
"""
Integration between world specs and simulation setup
Handles applying world geometry and physics to batched simulation
"""

import numpy as np
from typing import List, Tuple
from ..sim.core import SimulationConfig
from ..sim.batched import BatchedSimulation
from .worlds import WorldSpec, RectangleObstacle
from .testcases import TestCase
from .load_world import load_world


def apply_world_to_simulation(batched_sim: BatchedSimulation, world: WorldSpec, 
                             test_cases: List[TestCase]) -> SimulationConfig:
    """
    Apply world geometry and physics to batched simulation
    
    Args:
        batched_sim: Batched simulation instance
        world: World specification
        test_cases: List of test cases (for per-case overrides)
    
    Returns:
        Updated simulation config with world physics applied
    """
    # Update simulation config with world physics
    config = batched_sim.config
    
    # Apply world bounds
    config.world_width = world.bounds.width
    config.world_height = world.bounds.height
    
    # Apply world physics overrides
    if world.physics.agent_max_thrust is not None:
        config.agent_max_thrust = world.physics.agent_max_thrust
    if world.physics.agent_max_turn_accel is not None:
        config.agent_max_turn_accel = world.physics.agent_max_turn_accel
    if world.physics.agent_linear_drag is not None:
        config.agent_linear_drag = world.physics.agent_linear_drag
    if world.physics.agent_angular_drag is not None:
        config.agent_angular_drag = world.physics.agent_angular_drag
    if world.physics.wall_restitution is not None:
        config.wall_restitution = world.physics.wall_restitution
    
    # Update scene bounds (this will recreate boundary walls)
    batched_sim.scene.config = config
    batched_sim.scene.set_walls_bounds()
    
    # Clear existing obstacles
    batched_sim.scene.clear_obstacles()
    
    # Apply world geometry
    _apply_world_geometry(batched_sim, world, test_cases)
    
    return config


def _apply_world_geometry(batched_sim: BatchedSimulation, world: WorldSpec, 
                         test_cases: List[TestCase]):
    """Apply world geometry obstacles to simulation"""
    batch_size = batched_sim.B
    
    # Track obstacle slot usage
    circle_slot = 1  # Slot 0 reserved for food
    segment_slot = 4  # Slots 0-3 reserved for boundary walls
    
    # Apply rectangle obstacles (convert to segments)
    for rect in world.geometry.rectangles:
        if segment_slot + 4 > batched_sim.scene.Ks_max:
            print(f"Warning: Not enough segment slots for rectangle obstacle at ({rect.x}, {rect.y})")
            continue
        
        # Convert rectangle to 4 segments
        segments = _rectangle_to_segments(rect)
        
        for seg_data in segments:
            # Replicate segment across all environments
            p1 = np.tile([[seg_data['x1'], seg_data['y1']]], (batch_size, 1))
            p2 = np.tile([[seg_data['x2'], seg_data['y2']]], (batch_size, 1))
            
            batched_sim.scene.add_segment_obstacle(
                slot_idx=segment_slot,
                p1=p1,
                p2=p2,
                material_id=rect.material_id,
                color=(128, 128, 128)  # Gray for obstacles
            )
            segment_slot += 1
    
    # Apply circle obstacles
    for circle in world.geometry.circles:
        if circle_slot >= batched_sim.scene.Kc_max:
            print(f"Warning: Not enough circle slots for circle obstacle at ({circle.x}, {circle.y})")
            continue
        
        # Replicate circle across all environments
        centers = np.tile([[circle.x, circle.y]], (batch_size, 1))
        
        batched_sim.scene.add_circle_obstacle(
            slot_idx=circle_slot,
            centers=centers,
            radii=circle.radius,
            material_id=circle.material_id,
            color=(128, 128, 128)  # Gray for obstacles
        )
        circle_slot += 1
    
    # Apply segment obstacles
    for segment in world.geometry.segments:
        if segment_slot >= batched_sim.scene.Ks_max:
            print(f"Warning: Not enough segment slots for segment obstacle")
            continue
        
        # Replicate segment across all environments
        p1 = np.tile([[segment.x1, segment.y1]], (batch_size, 1))
        p2 = np.tile([[segment.x2, segment.y2]], (batch_size, 1))
        
        batched_sim.scene.add_segment_obstacle(
            slot_idx=segment_slot,
            p1=p1,
            p2=p2,
            material_id=segment.material_id,
            color=(128, 128, 128)  # Gray for obstacles
        )
        segment_slot += 1
    
    # Apply legacy per-test-case obstacles for backward compatibility
    _apply_legacy_obstacles(batched_sim, test_cases, circle_slot, segment_slot)


def _rectangle_to_segments(rect: RectangleObstacle) -> List[dict]:
    """Convert rectangle to 4 line segments"""
    half_w = rect.width / 2
    half_h = rect.height / 2
    
    # Rectangle corners
    x1, y1 = rect.x - half_w, rect.y - half_h  # Top-left
    x2, y2 = rect.x + half_w, rect.y - half_h  # Top-right
    x3, y3 = rect.x + half_w, rect.y + half_h  # Bottom-right
    x4, y4 = rect.x - half_w, rect.y + half_h  # Bottom-left
    
    return [
        {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},  # Top edge
        {'x1': x2, 'y1': y2, 'x2': x3, 'y2': y3},  # Right edge
        {'x1': x3, 'y1': y3, 'x2': x4, 'y2': y4},  # Bottom edge
        {'x1': x4, 'y1': y4, 'x2': x1, 'y2': y1},  # Left edge
    ]


def _apply_legacy_obstacles(batched_sim: BatchedSimulation, test_cases: List[TestCase],
                           start_circle_slot: int, start_segment_slot: int):
    """Apply legacy per-test-case obstacles for backward compatibility"""
    batch_size = len(test_cases)
    
    # Collect all unique obstacles across test cases
    all_circles = []
    all_segments = []
    
    for case in test_cases:
        for circle in case.obstacles.circles:
            if circle not in all_circles:
                all_circles.append(circle)
        
        for segment in case.obstacles.segments:
            if segment not in all_segments:
                all_segments.append(segment)
    
    # Apply circles
    circle_slot = start_circle_slot
    for circle in all_circles:
        if circle_slot >= batched_sim.scene.Kc_max:
            print(f"Warning: Not enough circle slots for legacy obstacle")
            break
        
        # Create per-environment positions (some may be inactive)
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        radii = np.zeros(batch_size, dtype=np.float32)
        
        for i, case in enumerate(test_cases):
            if circle in case.obstacles.circles:
                centers[i] = [circle.x, circle.y]
                radii[i] = circle.radius
        
        batched_sim.scene.add_circle_obstacle(
            slot_idx=circle_slot,
            centers=centers,
            radii=radii,
            material_id=circle.material_id,
            color=(128, 128, 128)
        )
        circle_slot += 1
    
    # Apply segments
    segment_slot = start_segment_slot
    for segment in all_segments:
        if segment_slot >= batched_sim.scene.Ks_max:
            print(f"Warning: Not enough segment slots for legacy obstacle")
            break
        
        # Create per-environment positions (some may be inactive)
        p1 = np.zeros((batch_size, 2), dtype=np.float32)
        p2 = np.zeros((batch_size, 2), dtype=np.float32)
        
        for i, case in enumerate(test_cases):
            if segment in case.obstacles.segments:
                p1[i] = [segment.x1, segment.y1]
                p2[i] = [segment.x2, segment.y2]
        
        batched_sim.scene.add_segment_obstacle(
            slot_idx=segment_slot,
            p1=p1,
            p2=p2,
            material_id=segment.material_id,
            color=(128, 128, 128)
        )
        segment_slot += 1


def resolve_test_case_world(test_case: TestCase) -> WorldSpec:
    """
    Resolve the world for a test case, applying overrides if specified
    
    Args:
        test_case: Test case that may reference a world
    
    Returns:
        WorldSpec with any overrides applied
    """
    # Load base world (or default if none specified)
    world_id = test_case.world_id or "default_empty"
    world = load_world(world_id)
    
    # Apply overrides if any
    if test_case.world_overrides:
        world = world.apply_overrides(test_case.world_overrides)
    
    return world


def setup_simulation_with_worlds(test_cases: List[TestCase], sim_config: SimulationConfig) -> Tuple[BatchedSimulation, SimulationConfig]:
    """
    Set up batched simulation with world geometry from test cases
    
    Args:
        test_cases: List of test cases
        sim_config: Base simulation configuration
    
    Returns:
        Tuple of (configured BatchedSimulation, updated SimulationConfig)
    """
    batch_size = len(test_cases)
    
    # Resolve worlds for all test cases
    worlds = [resolve_test_case_world(case) for case in test_cases]
    
    # For simplicity, use the first world's settings as base
    # In a more advanced implementation, you could merge compatible worlds
    primary_world = worlds[0]
    
    # Create batched simulation
    batched_sim = BatchedSimulation(batch_size, sim_config)
    
    # Apply world to simulation
    updated_config = apply_world_to_simulation(batched_sim, primary_world, test_cases)
    
    return batched_sim, updated_config