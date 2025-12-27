# 2D Continuous Simulation

A physics-based 2D simulation environment with ray-casting vision for agent training and testing.

## Structure

```
src/
├── sim/           # Core simulation engine (pure logic, no graphics)
├── viz/           # Visualization modules (pygame GUI)
└── policy/        # Policy/controller interfaces and implementations
scripts/           # Runnable scripts
checkpoints/       # Saved policy checkpoints
```

## Quick Start

### Interactive Manual Control
Run the simulation with keyboard control:
```bash
python -m scripts.run_interactive
```

Controls:
- `W` - Forward thrust (hold)
- `A/D` - Steer left/right  
- `V` - Toggle vision display
- `ESC` - Quit

### Watch a Loaded Agent
Load and watch a saved policy:
```bash
python -m scripts.watch_agent --checkpoint checkpoints/example.json
```

Options:
- `--fps 30` - Set frame rate
- `--vision` - Start with vision display on

## Policy System

The simulation supports pluggable policies/controllers:

### Manual Policy
Keyboard-driven control for interactive use.

### Scripted Policy  
Simple food-seeking behavior for testing. Parameters:
- `seek_strength` - How aggressively to steer toward food
- `throttle_level` - Constant throttle level

### Custom Policies
Implement the `Policy` interface in `src/policy/base.py`:

```python
class MyPolicy(Policy):
    def reset(self):
        # Reset policy state
        pass
    
    def act(self, sim_state):
        # Return {'steer': float, 'throttle': float}
        return {'steer': 0.0, 'throttle': 1.0}
```

## Checkpoints

Save/load policies with configuration:

```python
from src.policy.checkpoint import save_policy, load_policy
from src.policy.scripted import ScriptedPolicy
from src.sim.core import SimulationConfig

# Save
policy = ScriptedPolicy(seek_strength=2.0)
config = SimulationConfig()
save_policy('my_agent.json', 'Scripted', policy.get_params(), config)

# Load  
policy, config, metadata = load_policy('my_agent.json')
```

## Simulation State

The simulation provides rich state information to policies:

```python
sim_state = {
    'agent_state': {
        'x': float, 'y': float,           # Position
        'vx': float, 'vy': float,         # Velocity  
        'theta': float,                   # Heading angle
        'omega': float,                   # Angular velocity
        'throttle': float, 'speed': float # Current throttle/speed
    },
    'food_position': {'x': float, 'y': float},
    'vision_distances': [float, ...],     # Ray distances (128 rays)
    'vision_hit_types': [str, ...],       # 'wall', 'food', or None
    'vision_hit_wall_ids': [int, ...],    # Wall IDs (0-3) or None
    'time': float,                        # Simulation time
    'step': int,                          # Step count
    'food_collected': int                 # Total food collected
}
```

## Vision System

- 120° field of view with 128 rays
- Ray-casting with occlusion (nearest hit only)
- Detects walls (4 boundaries) and food
- LiDAR-style depth strip visualization
- Wall colors: Top=White, Right=Cyan, Bottom=Magenta, Left=Orange

## Development

The core simulation (`src/sim/core.py`) is pure Python with no graphics dependencies, enabling:
- Fast batch processing
- Headless training
- Deterministic execution with seeds
- Easy integration with ML frameworks

The GUI (`src/viz/pygame_single.py`) is a thin visualization layer that uses the core simulation.