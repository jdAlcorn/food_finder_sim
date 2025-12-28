#!/usr/bin/env python3
"""
Test case schema for deterministic fitness evaluation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class AgentState:
    """Initial agent state for test case"""
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    throttle: float = 0.0


@dataclass
class FoodState:
    """Food position and properties"""
    x: float
    y: float
    radius: float = 8.0  # Default food radius


@dataclass
class CircleObstacle:
    """Circular obstacle"""
    x: float
    y: float
    radius: float
    material_id: int = 1  # Default wall material


@dataclass
class SegmentObstacle:
    """Line segment obstacle"""
    x1: float
    y1: float
    x2: float
    y2: float
    material_id: int = 1  # Default wall material


@dataclass
class Obstacles:
    """Collection of obstacles for a test case"""
    circles: List[CircleObstacle] = field(default_factory=list)
    segments: List[SegmentObstacle] = field(default_factory=list)


@dataclass
class TestCase:
    """
    Single deterministic test case for fitness evaluation
    
    Defines initial conditions and constraints for evaluating a policy
    on a specific scenario without randomness.
    """
    id: str
    max_steps: int
    agent_start: AgentState
    food: FoodState
    dt: Optional[float] = None  # Use sim_config dt if None
    obstacles: Obstacles = field(default_factory=Obstacles)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'max_steps': self.max_steps,
            'dt': self.dt,
            'agent_start': {
                'x': self.agent_start.x,
                'y': self.agent_start.y,
                'theta': self.agent_start.theta,
                'vx': self.agent_start.vx,
                'vy': self.agent_start.vy,
                'omega': self.agent_start.omega,
                'throttle': self.agent_start.throttle
            },
            'food': {
                'x': self.food.x,
                'y': self.food.y,
                'radius': self.food.radius
            },
            'obstacles': {
                'circles': [
                    {'x': c.x, 'y': c.y, 'radius': c.radius, 'material_id': c.material_id}
                    for c in self.obstacles.circles
                ],
                'segments': [
                    {'x1': s.x1, 'y1': s.y1, 'x2': s.x2, 'y2': s.y2, 'material_id': s.material_id}
                    for s in self.obstacles.segments
                ]
            },
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create TestCase from dictionary"""
        agent_data = data['agent_start']
        agent_start = AgentState(
            x=agent_data['x'],
            y=agent_data['y'],
            theta=agent_data['theta'],
            vx=agent_data.get('vx', 0.0),
            vy=agent_data.get('vy', 0.0),
            omega=agent_data.get('omega', 0.0),
            throttle=agent_data.get('throttle', 0.0)
        )
        
        food_data = data['food']
        food = FoodState(
            x=food_data['x'],
            y=food_data['y'],
            radius=food_data.get('radius', 8.0)
        )
        
        obstacles_data = data.get('obstacles', {})
        obstacles = Obstacles(
            circles=[
                CircleObstacle(
                    x=c['x'], y=c['y'], radius=c['radius'],
                    material_id=c.get('material_id', 1)
                )
                for c in obstacles_data.get('circles', [])
            ],
            segments=[
                SegmentObstacle(
                    x1=s['x1'], y1=s['y1'], x2=s['x2'], y2=s['y2'],
                    material_id=s.get('material_id', 1)
                )
                for s in obstacles_data.get('segments', [])
            ]
        )
        
        return cls(
            id=data['id'],
            max_steps=data['max_steps'],
            dt=data.get('dt'),
            agent_start=agent_start,
            food=food,
            obstacles=obstacles,
            notes=data.get('notes', '')
        )


@dataclass
class TestSuite:
    """
    Collection of test cases for comprehensive evaluation
    """
    suite_id: str
    version: str
    test_cases: List[TestCase]
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __iter__(self):
        return iter(self.test_cases)
    
    def __getitem__(self, index):
        return self.test_cases[index]
    
    def get_case_by_id(self, case_id: str) -> Optional[TestCase]:
        """Get test case by ID"""
        for case in self.test_cases:
            if case.id == case_id:
                return case
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'suite_id': self.suite_id,
            'version': self.version,
            'description': self.description,
            'test_cases': [case.to_dict() for case in self.test_cases]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestSuite':
        """Create TestSuite from dictionary"""
        test_cases = [TestCase.from_dict(case_data) for case_data in data['test_cases']]
        
        return cls(
            suite_id=data['suite_id'],
            version=data['version'],
            description=data.get('description', ''),
            test_cases=test_cases
        )
    
    def validate(self) -> List[str]:
        """
        Validate test suite for common issues
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for unique IDs
        ids = [case.id for case in self.test_cases]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in set(ids) if ids.count(id) > 1]
            errors.append(f"Duplicate test case IDs: {duplicates}")
        
        # Validate individual test cases
        for case in self.test_cases:
            case_errors = self._validate_test_case(case)
            errors.extend([f"Case '{case.id}': {err}" for err in case_errors])
        
        return errors
    
    def _validate_test_case(self, case: TestCase) -> List[str]:
        """Validate individual test case"""
        errors = []
        
        # Check max_steps is positive
        if case.max_steps <= 0:
            errors.append("max_steps must be positive")
        
        # Check dt is positive if specified
        if case.dt is not None and case.dt <= 0:
            errors.append("dt must be positive")
        
        # Check food radius is positive
        if case.food.radius <= 0:
            errors.append("food radius must be positive")
        
        # Check obstacle radii are positive
        for i, circle in enumerate(case.obstacles.circles):
            if circle.radius <= 0:
                errors.append(f"circle obstacle {i} radius must be positive")
        
        return errors