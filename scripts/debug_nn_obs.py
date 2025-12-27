#!/usr/bin/env python3
"""
Debug script for neural network observation extraction
Uses manual control but shows NN observation data in real-time
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_debug_nn import run_debug_nn_visualization
from src.sim.core import SimulationConfig


def main():
    """Run debug visualization with manual control + NN observation monitoring"""
    print("NN Observation Debug Visualization")
    print("=" * 50)
    print("This runs manual keyboard control while showing NN observation data")
    print()
    print("Controls:")
    print("  W/A/S/D - Manual control (same as normal)")
    print("  V - Toggle vision display")
    print("  P - Toggle NN observation console printing")
    print("  ESC - Quit")
    print()
    print("The right side shows real-time NN observation statistics")
    print("Press P to see detailed observation data printed to console")
    print()
    
    # Create default config
    config = SimulationConfig()
    
    # Run debug visualization
    run_debug_nn_visualization(config, fps=60)


if __name__ == "__main__":
    main()