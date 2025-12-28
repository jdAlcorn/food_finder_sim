#!/usr/bin/env python3
"""
CLI tool for managing world specifications
"""

import argparse
import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.worlds import WorldSpec, create_default_world
from src.eval.load_world import load_world, save_world, list_available_worlds, get_world_info


def list_worlds():
    """List all available worlds"""
    worlds = list_available_worlds()
    
    print("Available Worlds:")
    print("=" * 50)
    
    for world_id in worlds:
        info = get_world_info(world_id)
        if info:
            print(f"ID: {world_id}")
            print(f"  Version: {info['version']}")
            print(f"  Description: {info['description']}")
            print()
        else:
            print(f"ID: {world_id} (invalid)")


def show_world(world_id: str):
    """Show detailed information about a world"""
    try:
        world = load_world(world_id)
        
        print(f"World: {world.world_id}")
        print("=" * 50)
        print(f"Version: {world.version}")
        print(f"Description: {world.description}")
        print()
        
        print("Bounds:")
        print(f"  Width: {world.bounds.width}")
        print(f"  Height: {world.bounds.height}")
        print()
        
        print("Physics Overrides:")
        physics_attrs = [
            'agent_max_thrust', 'agent_max_turn_accel', 
            'agent_linear_drag', 'agent_angular_drag', 'wall_restitution'
        ]
        has_physics = False
        for attr in physics_attrs:
            value = getattr(world.physics, attr)
            if value is not None:
                print(f"  {attr}: {value}")
                has_physics = True
        if not has_physics:
            print("  None")
        print()
        
        print("Geometry:")
        print(f"  Rectangles: {len(world.geometry.rectangles)}")
        for i, rect in enumerate(world.geometry.rectangles):
            print(f"    {i}: ({rect.x}, {rect.y}) {rect.width}x{rect.height}")
        
        print(f"  Circles: {len(world.geometry.circles)}")
        for i, circle in enumerate(world.geometry.circles):
            print(f"    {i}: ({circle.x}, {circle.y}) r={circle.radius}")
        
        print(f"  Segments: {len(world.geometry.segments)}")
        for i, segment in enumerate(world.geometry.segments):
            print(f"    {i}: ({segment.x1}, {segment.y1}) -> ({segment.x2}, {segment.y2})")
        
    except Exception as e:
        print(f"Error loading world '{world_id}': {e}")


def create_world(world_id: str, description: str = ""):
    """Create a new empty world"""
    world = create_default_world()
    world.world_id = world_id
    world.description = description
    
    try:
        path = save_world(world)
        print(f"Created new world '{world_id}' at {path}")
    except Exception as e:
        print(f"Error creating world: {e}")


def validate_world(world_id: str):
    """Validate a world specification"""
    try:
        world = load_world(world_id)
        errors = world.validate()
        
        if errors:
            print(f"World '{world_id}' has validation errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"World '{world_id}' is valid ✓")
            
    except Exception as e:
        print(f"Error validating world '{world_id}': {e}")


def export_world(world_id: str, output_path: str):
    """Export world to JSON file"""
    try:
        world = load_world(world_id)
        
        with open(output_path, 'w') as f:
            json.dump(world.to_dict(), f, indent=2)
        
        print(f"Exported world '{world_id}' to {output_path}")
        
    except Exception as e:
        print(f"Error exporting world: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Manage world specifications')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all available worlds')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show detailed world information')
    show_parser.add_argument('world_id', help='World ID to show')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new empty world')
    create_parser.add_argument('world_id', help='World ID to create')
    create_parser.add_argument('--description', default='', help='World description')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a world specification')
    validate_parser.add_argument('world_id', help='World ID to validate')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export world to JSON file')
    export_parser.add_argument('world_id', help='World ID to export')
    export_parser.add_argument('output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_worlds()
    elif args.command == 'show':
        show_world(args.world_id)
    elif args.command == 'create':
        create_world(args.world_id, args.description)
    elif args.command == 'validate':
        validate_world(args.world_id)
    elif args.command == 'export':
        export_world(args.world_id, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()