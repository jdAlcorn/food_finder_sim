#!/usr/bin/env python3
"""
World/Map schema for reusable environment layouts
Separates static geometry from test case initial conditions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import copy


@dataclass
class WorldBounds:
    """World boundary configuration"""
    width: float = 800.0
    height: float = 800.0


@dataclass
class WorldPhysics:
    """World physics parameters that can override simulation defaults"""
    agent_max_thrust: Optional[float] = None
    agent_max_turn_accel: Optional[float] = None
    agent_linear_drag: Optional[float] = None
    agent_angular_drag: Optional[float] = None
    wall_restitution: Optional[float] = None


@dataclass
class RectangleObstacle:
    """Rectangular obstacle (converted to 4 segments internally)"""
    x: float  # Center x
    y: float  # Center y
    width: float
    height: float
    material_id: int = 3  # Default obstacle material


@dataclass
class CircleObstacle:
    """Circular obstacle"""
    x: float
    y: float
    radius: float
    material_id: int = 3  # Default obstacle material


@dataclass
class SegmentObstacle:
    """Line segment obstacle"""
    x1: float
    y1: float
    x2: float
    y2: float
    material_id: int = 3  # Default obstacle material


@dataclass
class WorldGeometry:
    """Collection of static obstacles in the world"""
    rectangles: List[RectangleObstacle] = field(default_factory=list)
    circles: List[CircleObstacle] = field(default_factory=list)
    segments: List[SegmentObstacle] = field(default_factory=list)


@dataclass
class WorldSpec:
    """
    Complete world/map specification
    
    Defines static environment layout and physics parameters
    that can be reused across multiple test cases.
    """
    world_id: str
    version: str = "1.0"
    description: str = ""
    bounds: WorldBounds = field(default_factory=WorldBounds)
    physics: WorldPhysics = field(default_factory=WorldPhysics)
    geometry: WorldGeometry = field(default_factory=WorldGeometry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'world_id': self.world_id,
            'version': self.version,
            'description': self.description,
            'bounds': {
                'width': self.bounds.width,
                'height': self.bounds.height
            },
            'physics': {
                k: v for k, v in {
                    'agent_max_thrust': self.physics.agent_max_thrust,
                    'agent_max_turn_accel': self.physics.agent_max_turn_accel,
                    'agent_linear_drag': self.physics.agent_linear_drag,
                    'agent_angular_drag': self.physics.agent_angular_drag,
                    'wall_restitution': self.physics.wall_restitution
                }.items() if v is not None
            },
            'geometry': {
                'rectangles': [
                    {
                        'x': r.x, 'y': r.y, 'width': r.width, 'height': r.height,
                        'material_id': r.material_id
                    }
                    for r in self.geometry.rectangles
                ],
                'circles': [
                    {
                        'x': c.x, 'y': c.y, 'radius': c.radius,
                        'material_id': c.material_id
                    }
                    for c in self.geometry.circles
                ],
                'segments': [
                    {
                        'x1': s.x1, 'y1': s.y1, 'x2': s.x2, 'y2': s.y2,
                        'material_id': s.material_id
                    }
                    for s in self.geometry.segments
                ]
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldSpec':
        """Create WorldSpec from dictionary"""
        bounds_data = data.get('bounds', {})
        bounds = WorldBounds(
            width=bounds_data.get('width', 800.0),
            height=bounds_data.get('height', 800.0)
        )
        
        physics_data = data.get('physics', {})
        physics = WorldPhysics(
            agent_max_thrust=physics_data.get('agent_max_thrust'),
            agent_max_turn_accel=physics_data.get('agent_max_turn_accel'),
            agent_linear_drag=physics_data.get('agent_linear_drag'),
            agent_angular_drag=physics_data.get('agent_angular_drag'),
            wall_restitution=physics_data.get('wall_restitution')
        )
        
        geometry_data = data.get('geometry', {})
        geometry = WorldGeometry(
            rectangles=[
                RectangleObstacle(
                    x=r['x'], y=r['y'], width=r['width'], height=r['height'],
                    material_id=r.get('material_id', 3)
                )
                for r in geometry_data.get('rectangles', [])
            ],
            circles=[
                CircleObstacle(
                    x=c['x'], y=c['y'], radius=c['radius'],
                    material_id=c.get('material_id', 3)
                )
                for c in geometry_data.get('circles', [])
            ],
            segments=[
                SegmentObstacle(
                    x1=s['x1'], y1=s['y1'], x2=s['x2'], y2=s['y2'],
                    material_id=s.get('material_id', 3)
                )
                for s in geometry_data.get('segments', [])
            ]
        )
        
        return cls(
            world_id=data['world_id'],
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            bounds=bounds,
            physics=physics,
            geometry=geometry
        )
    
    def validate(self) -> List[str]:
        """
        Validate world specification
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check bounds are positive
        if self.bounds.width <= 0:
            errors.append("bounds width must be positive")
        if self.bounds.height <= 0:
            errors.append("bounds height must be positive")
        
        # Check physics parameters are positive if specified
        physics_params = [
            ('agent_max_thrust', self.physics.agent_max_thrust),
            ('agent_max_turn_accel', self.physics.agent_max_turn_accel),
            ('agent_linear_drag', self.physics.agent_linear_drag),
            ('agent_angular_drag', self.physics.agent_angular_drag),
            ('wall_restitution', self.physics.wall_restitution)
        ]
        
        for name, value in physics_params:
            if value is not None and value <= 0:
                errors.append(f"physics {name} must be positive")
        
        # Check obstacle geometry
        for i, rect in enumerate(self.geometry.rectangles):
            if rect.width <= 0:
                errors.append(f"rectangle {i} width must be positive")
            if rect.height <= 0:
                errors.append(f"rectangle {i} height must be positive")
        
        for i, circle in enumerate(self.geometry.circles):
            if circle.radius <= 0:
                errors.append(f"circle {i} radius must be positive")
        
        return errors
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> 'WorldSpec':
        """
        Apply overrides to create a modified world spec
        
        Args:
            overrides: Dictionary with override values using dot notation
                      e.g., {'bounds.width': 1000, 'geometry.circles': [...]}
        
        Returns:
            New WorldSpec with overrides applied
        """
        # Deep copy to avoid modifying original
        world_dict = copy.deepcopy(self.to_dict())
        
        # Apply overrides using dot notation
        for key, value in overrides.items():
            self._set_nested_value(world_dict, key, value)
        
        return WorldSpec.from_dict(world_dict)
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value


def create_default_world() -> WorldSpec:
    """Create default empty world for backward compatibility"""
    return WorldSpec(
        world_id="default_empty",
        version="1.0",
        description="Default empty world with just boundary walls",
        bounds=WorldBounds(width=800.0, height=800.0),
        physics=WorldPhysics(),
        geometry=WorldGeometry()
    )