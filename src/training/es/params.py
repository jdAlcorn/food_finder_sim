#!/usr/bin/env python3
"""
Parameter vectorization utilities for PyTorch models
Flatten/unflatten model parameters for Evolution Strategies
"""

import torch
import torch.nn as nn
from typing import List, Tuple, NamedTuple
from collections import OrderedDict


class ParamSpec(NamedTuple):
    """Parameter specification for consistent flattening/unflattening"""
    name: str
    shape: Tuple[int, ...]
    numel: int


def make_param_spec(model: nn.Module) -> List[ParamSpec]:
    """
    Create parameter specification from model
    
    Args:
        model: PyTorch model
        
    Returns:
        List of ParamSpec objects in consistent order
    """
    spec = []
    for name, param in model.named_parameters():
        spec.append(ParamSpec(
            name=name,
            shape=tuple(param.shape),
            numel=param.numel()
        ))
    return spec


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """
    Flatten model parameters to 1D tensor
    
    Args:
        model: PyTorch model
        
    Returns:
        Flattened parameters as CPU tensor of shape [P]
    """
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().flatten())
    
    if len(params) == 0:
        return torch.tensor([], dtype=torch.float32)
    
    return torch.cat(params)


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Load flattened parameters into model
    
    Args:
        model: PyTorch model to update
        flat_params: Flattened parameters tensor
    """
    if flat_params.numel() == 0:
        return
    
    # Ensure flat_params is on CPU and float32
    flat_params = flat_params.detach().cpu().float()
    
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        # Extract slice and reshape
        param_slice = flat_params[offset:offset + numel].view(param.shape)
        # Copy to parameter (preserving device)
        param.data.copy_(param_slice.to(param.device))
        offset += numel
    
    assert offset == flat_params.numel(), f"Parameter count mismatch: {offset} != {flat_params.numel()}"


def flatten_from_spec(model: nn.Module, spec: List[ParamSpec]) -> torch.Tensor:
    """
    Flatten model parameters using specification for consistent ordering
    
    Args:
        model: PyTorch model
        spec: Parameter specification
        
    Returns:
        Flattened parameters tensor
    """
    state_dict = model.state_dict()
    params = []
    
    for param_spec in spec:
        param = state_dict[param_spec.name]
        params.append(param.detach().cpu().flatten())
    
    if len(params) == 0:
        return torch.tensor([], dtype=torch.float32)
    
    return torch.cat(params)


def assign_from_spec(model: nn.Module, spec: List[ParamSpec], flat_params: torch.Tensor) -> None:
    """
    Assign flattened parameters to model using specification
    
    Args:
        model: PyTorch model to update
        spec: Parameter specification
        flat_params: Flattened parameters tensor
    """
    if flat_params.numel() == 0:
        return
    
    # Ensure flat_params is on CPU and float32
    flat_params = flat_params.detach().cpu().float()
    
    state_dict = model.state_dict()
    offset = 0
    
    for param_spec in spec:
        numel = param_spec.numel
        # Extract slice and reshape
        param_slice = flat_params[offset:offset + numel].view(param_spec.shape)
        # Update state dict
        state_dict[param_spec.name].copy_(param_slice)
        offset += numel
    
    assert offset == flat_params.numel(), f"Parameter count mismatch: {offset} != {flat_params.numel()}"


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in model"""
    return sum(p.numel() for p in model.parameters())


def test_param_vectorization():
    """Test parameter vectorization functions"""
    from src.policy.models.mlp import SimpleMLP
    
    # Create test model
    model = SimpleMLP(input_dim=10, hidden_dims=(8, 4), output_dim=2)
    original_params = get_flat_params(model)
    
    print(f"Model has {count_parameters(model)} parameters")
    print(f"Flattened shape: {original_params.shape}")
    
    # Test spec-based functions
    spec = make_param_spec(model)
    print(f"Parameter spec: {len(spec)} layers")
    for s in spec:
        print(f"  {s.name}: {s.shape} ({s.numel} params)")
    
    # Test round-trip
    flat_params = flatten_from_spec(model, spec)
    assert torch.allclose(original_params, flat_params), "Spec-based flattening mismatch"
    
    # Modify parameters
    new_params = torch.randn_like(flat_params)
    assign_from_spec(model, spec, new_params)
    
    # Verify assignment
    recovered_params = flatten_from_spec(model, spec)
    assert torch.allclose(new_params, recovered_params), "Parameter assignment failed"
    
    print("Parameter vectorization tests passed!")


if __name__ == "__main__":
    test_param_vectorization()