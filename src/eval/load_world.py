#!/usr/bin/env python3
"""
World loader and registry for reusable environment layouts
"""

import json
import os
from typing import Dict, List, Optional, Any
from .worlds import WorldSpec, create_default_world


class WorldRegistry:
    """Registry for loading and caching world specifications"""
    
    def __init__(self, world_dir: str = "data/worlds"):
        self.world_dir = world_dir
        self._cache: Dict[str, WorldSpec] = {}
        self._default_world = create_default_world()
    
    def load_world(self, world_id: str) -> WorldSpec:
        """
        Load world by ID
        
        Args:
            world_id: World identifier
            
        Returns:
            WorldSpec object
            
        Raises:
            FileNotFoundError: If world file doesn't exist
            ValueError: If world format is invalid or validation fails
        """
        # Return default world for backward compatibility
        if world_id == "default_empty" or world_id is None:
            return self._default_world
        
        # Check cache first
        if world_id in self._cache:
            return self._cache[world_id]
        
        # Load from file
        world_path = os.path.join(self.world_dir, f"{world_id}.json")
        
        if not os.path.exists(world_path):
            available = self.list_available_worlds()
            available_str = ", ".join(available) if available else "none"
            raise FileNotFoundError(
                f"World '{world_id}' not found at {world_path}. "
                f"Available worlds: {available_str}"
            )
        
        try:
            with open(world_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in world file {world_path}: {e}")
        
        try:
            world = WorldSpec.from_dict(data)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid world format in {world_path}: {e}")
        
        # Validate the loaded world
        errors = world.validate()
        if errors:
            error_msg = f"World '{world_id}' validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)
        
        # Cache and return
        self._cache[world_id] = world
        return world
    
    def save_world(self, world: WorldSpec, world_dir: Optional[str] = None) -> str:
        """
        Save world to JSON file
        
        Args:
            world: WorldSpec to save
            world_dir: Optional directory override
            
        Returns:
            Path to saved file
        """
        save_dir = world_dir or self.world_dir
        os.makedirs(save_dir, exist_ok=True)
        
        world_path = os.path.join(save_dir, f"{world.world_id}.json")
        
        # Atomic write
        temp_path = world_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(world.to_dict(), f, indent=2)
        os.rename(temp_path, world_path)
        
        # Update cache
        self._cache[world.world_id] = world
        
        return world_path
    
    def list_available_worlds(self) -> List[str]:
        """
        List available world IDs
        
        Returns:
            List of world IDs (without .json extension)
        """
        if not os.path.exists(self.world_dir):
            return ["default_empty"]
        
        worlds = ["default_empty"]  # Always include default
        
        for filename in os.listdir(self.world_dir):
            if filename.endswith('.json'):
                world_id = filename[:-5]  # Remove .json
                worlds.append(world_id)
        
        return sorted(worlds)
    
    def get_world_info(self, world_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic info about a world without fully loading it
        
        Args:
            world_id: World identifier
            
        Returns:
            Dict with world_id, version, description, or None if invalid
        """
        if world_id == "default_empty":
            return {
                'world_id': 'default_empty',
                'version': '1.0',
                'description': 'Default empty world with just boundary walls'
            }
        
        world_path = os.path.join(self.world_dir, f"{world_id}.json")
        
        try:
            with open(world_path, 'r') as f:
                data = json.load(f)
            
            return {
                'world_id': data.get('world_id', world_id),
                'version': data.get('version', 'unknown'),
                'description': data.get('description', '')
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
    def clear_cache(self):
        """Clear the world cache"""
        self._cache.clear()


# Global registry instance
_world_registry = WorldRegistry()


def load_world(world_id: str) -> WorldSpec:
    """Load world by ID using global registry"""
    return _world_registry.load_world(world_id)


def save_world(world: WorldSpec, world_dir: Optional[str] = None) -> str:
    """Save world using global registry"""
    return _world_registry.save_world(world, world_dir)


def list_available_worlds() -> List[str]:
    """List available worlds using global registry"""
    return _world_registry.list_available_worlds()


def get_world_info(world_id: str) -> Optional[Dict[str, Any]]:
    """Get world info using global registry"""
    return _world_registry.get_world_info(world_id)


def set_world_directory(world_dir: str):
    """Set the world directory for global registry"""
    global _world_registry
    _world_registry = WorldRegistry(world_dir)