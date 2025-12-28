"""Simulation core module"""
from .core import SimulationConfig
from .unified import SimulationSingle, Simulation, create_simulation
from .batched import BatchedSimulation

# Deprecated - use unified interface instead
from .core import Agent, Food, VisionSystem

__all__ = [
    'SimulationConfig', 
    'Simulation',  # Now points to unified SimulationSingle
    'SimulationSingle', 
    'BatchedSimulation',
    'create_simulation',
    # Deprecated
    'Agent', 'Food', 'VisionSystem'
]