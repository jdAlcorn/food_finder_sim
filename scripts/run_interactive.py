#!/usr/bin/env python3
"""
Run interactive simulation with manual keyboard control
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viz.pygame_single import run_simulation_gui
from src.policy.manual import ManualPolicy
from src.sim.core import SimulationConfig


def main():
    """Run interactive simulation with manual control"""
    print("Starting interactive simulation with manual control...")
    print("Controls: W=thrust, A/D=steer, V=toggle vision, ESC=quit")
    
    # Create manual policy and default config
    policy = ManualPolicy()
    config = SimulationConfig()
    
    # Run GUI
    run_simulation_gui(policy, config, fps=60, policy_name="Manual")


if __name__ == "__main__":
    main()