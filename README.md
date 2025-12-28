# 2D Continuous Simulation

A physics-based 2D simulation environment with ray-casting vision for agent training and testing. Features neural network policies, multi-agent visualization, and headless batch processing.

## Structure

```
src/
├── sim/           # Core simulation engine (pure logic, no graphics)
├── viz/           # Visualization modules (pygame GUI)
│   ├── pygame_single.py    # Single agent viewer
│   ├── pygame_multi.py     # Multi-agent grid viewer  
│   └── pygame_debug_nn.py  # NN observation debugging
├── policy/        # Policy/controller interfaces and implementations
│   ├── base.py            # Policy interface
│   ├── manual.py          # Keyboard control
│   ├── scripted.py        # Simple food-seeking
│   ├── nn_policy_stub.py  # NN observation extraction
│   ├── nn_torch_mlp.py    # PyTorch neural network
│   └── models/            # Neural network architectures
│       └── mlp.py         # Multi-layer perceptron
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
- `N` - Next test case (viewer only)
- `ESC` - Quit

### Watch a Loaded Agent
Load and watch a saved policy:
```bash
python -m scripts.watch_agent --checkpoint checkpoints/torch_mlp.json
```

### Multi-Agent Visualization
Watch multiple agents running in parallel:
```bash
python -m scripts.run_multi --num-sims 16 --policy scripted
```

### Headless Simulation
Run fast simulations without graphics:
```bash
python -m scripts.run_headless --policy torch_mlp --max-steps 5000
```

## Policy System

The simulation supports pluggable policies/controllers:

### Manual Policy
Keyboard-driven control for interactive use.

### Scripted Policy  
Simple food-seeking behavior for testing. Parameters:
- `seek_strength` - How aggressively to steer toward food
- `throttle_level` - Constant throttle level

### Neural Network Policies

#### Neural Policy Stub
Extracts 388-dimensional egocentric observations for ML training:
- **Vision channels** (384 features): proximity, food detection, wall detection
- **Proprioception** (4 features): forward/lateral velocity, angular velocity, throttle

#### PyTorch MLP Policy
Fully functional neural network policy using PyTorch:
- Maps 388-dim observations to (steer, throttle) actions
- Configurable architecture (default: 256→128→2)
- Supports CPU/CUDA execution
- Deterministic with seed control

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

## Scripts Reference

### Interactive Scripts
- `run_interactive.py` - Manual keyboard control
- `watch_agent.py` - Load and watch any policy from checkpoint
- `watch_random_mlp.py` - Quick test of random neural network
- `debug_nn_obs.py` - Manual control + NN observation monitoring

### Batch Scripts  
- `run_multi.py` - Multi-agent parallel visualization
- `run_headless.py` - Fast CLI-only simulation with detailed logging

### Usage Examples

**Single agent testing:**
```bash
# Manual control
python -m scripts.run_interactive

# Random neural network
python -m scripts.watch_random_mlp

# Loaded checkpoint
python -m scripts.watch_agent --checkpoint checkpoints/torch_mlp.json
```

**Multi-agent comparison:**
```bash
# 25 scripted agents
python -m scripts.run_multi --num-sims 25 --policy scripted

# 16 random neural networks  
python -m scripts.run_multi --num-sims 16 --policy torch_mlp

# Load from checkpoint
python -m scripts.run_multi --policy checkpoint --checkpoint my_agent.json
```

**Headless evaluation:**
```bash
# Quick test
python -m scripts.run_headless --policy torch_mlp --max-steps 1000 --quiet

# Detailed logging
python -m scripts.run_headless --policy scripted --max-steps 5000 --print-interval 200

# Fast simulation (3x speedup)
python -m scripts.run_headless --policy torch_mlp --dt 0.05 --max-steps 3000
```

**Debugging:**
```bash
# Monitor NN observations while driving manually
python -m scripts.debug_nn_obs
```

## Checkpoints

Save/load policies with configuration and model weights:

```python
from src.policy.checkpoint import save_policy, load_policy
from src.policy.nn_torch_mlp import TorchMLPPolicy
from src.sim.core import SimulationConfig

# Save neural network policy
policy = TorchMLPPolicy(hidden_dims=(256, 128))
config = SimulationConfig()
save_policy('my_nn.json', 'TorchMLP', policy.get_params(), config, policy_instance=policy)

# Load with automatic weight loading
policy, config, metadata = load_policy('my_nn.json')
```

### Available Checkpoints
- `checkpoints/example.json` - Scripted food-seeking policy
- `checkpoints/torch_mlp.json` - Random PyTorch MLP policy

## Neural Network Integration

### Observation Vector (388 dimensions)
The simulation provides rich egocentric observations:

```python
# Vision channels (128 rays each)
vision_close = [0.0, 0.3, 0.8, ...]    # Proximity (1 = close, 0 = far)
vision_food = [0, 0, 1, ...]           # Food detection (binary)
vision_wall = [1, 0, 0, ...]           # Wall detection (binary)

# Proprioception (agent's internal state)
proprioception = [
    v_forward_norm,   # Forward velocity in agent frame
    v_sideways_norm,  # Lateral velocity in agent frame  
    omega_norm,       # Angular velocity
    throttle_state    # Current throttle level
]

# Complete observation
obs = concat(vision_close, vision_food, vision_wall, proprioception)  # 388 floats
```

### Model Architecture
```python
# PyTorch MLP (src/policy/models/mlp.py)
SimpleMLP(
    input_dim=388,      # Observation vector
    hidden_dims=(256, 128),  # Hidden layers
    output_dim=2        # (steer, throttle)
)

# Output activations
steer = tanh(raw_output[0])      # [-1, 1]
throttle = sigmoid(raw_output[1]) # [0, 1]
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

- **120° field of view** with 128 rays
- **Ray-casting with occlusion** (nearest hit only)
- **Detects walls** (4 boundaries) and food targets
- **LiDAR-style depth strip** visualization
- **Wall colors**: Top=White, Right=Cyan, Bottom=Magenta, Left=Orange
- **Semi-transparent vision cone** showing visible area

## Performance

### Simulation Speed
- **Interactive GUI**: ~60 steps/second (real-time)
- **Multi-agent GUI**: ~30 steps/second (16 agents)
- **Headless**: 2000+ steps/second (30-50x speedup)

### Timestep Guidelines
- **Default**: `dt = 1/60s` (stable, matches GUI)
- **Fast**: `dt = 0.05s` (3x speedup, minimal risk)
- **Aggressive**: `dt = 0.1s` (6x speedup, may cause instability)

## Development

### Core Features
- **Pure simulation core** (`src/sim/core.py`) - no graphics dependencies
- **Deterministic execution** with seed control
- **Fast batch processing** for training/evaluation
- **Easy ML integration** with structured observations

### Visualization Layers
- **Single agent** (`pygame_single.py`) - detailed view with full vision system
- **Multi-agent** (`pygame_multi.py`) - grid view for comparing multiple agents
- **Debug mode** (`pygame_debug_nn.py`) - manual control + NN observation monitoring

### Requirements
```bash
pip install -r requirements.txt
# Installs: pygame, numpy, torch
```

## Training Integration

The simulation is designed for easy integration with ML training frameworks:

```python
# Example training loop structure
for episode in range(num_episodes):
    sim = Simulation(config, seed=episode)
    policy.reset()
    
    for step in range(max_steps):
        obs = extract_observation(sim.get_state())  # 388-dim vector
        action = policy.act(obs)                    # {'steer': float, 'throttle': float}
        step_info = sim.step(dt, action)
        
        # Training logic here
        reward = calculate_reward(step_info)
        # Update policy...
```

The observation extraction, action formatting, and simulation stepping are all handled by the existing policy framework.