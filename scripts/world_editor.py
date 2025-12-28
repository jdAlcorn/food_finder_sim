#!/usr/bin/env python3
"""
Interactive world editor
Create, modify, and delete rectangular obstacles visually
"""

import argparse
import os
import sys
import json
import math
from typing import Optional, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import pygame
try:
    import pygame
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)

from src.sim.core import SimulationConfig
from src.eval.worlds import WorldSpec, WorldBounds, WorldPhysics, WorldGeometry, RectangleObstacle
from src.eval.load_world import load_world, list_available_worlds
from src.eval.world_integration import apply_world_to_simulation
from src.sim.batched import BatchedSimulation
from src.sim.unified import SimulationSingle


class WorldEditor:
    """Interactive world editor for creating and modifying worlds"""
    
    def __init__(self, config: SimulationConfig, world_id: str = None):
        self.config = config
        self.world_id = world_id or "new_world"
        
        # Load existing world or create new one
        if world_id and world_id != "new_world":
            try:
                self.world = load_world(world_id)
                print(f"Loaded existing world: {world_id}")
                print(f"Description: {self.world.description}")
                print(f"Rectangles: {len(self.world.geometry.rectangles)}")
                    
            except Exception as e:
                print(f"Error loading world '{world_id}': {e}")
                print("Creating new world instead...")
                self.world = self._create_default_world()
        else:
            self.world = self._create_default_world()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.world_width, config.world_height))
        pygame.display.set_caption(f"World Editor - {self.world_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (100, 150, 255)  # Light blue for selected
        self.YELLOW = (255, 255, 0)
        self.GRAY = (160, 160, 160)  # Light gray for better visibility
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (64, 64, 64)
        
        # Editor state
        self.running = True
        self.mode = "select"  # "select", "create", "resize"
        self.selected_rect = None  # Index of selected rectangle
        self.drag_start = None  # (x, y) where drag started
        self.creating_rect = None  # Rectangle being created
        self.resize_handle = None  # Which handle is being dragged
        
        # Handle size for resize grips
        self.handle_size = 8
        
        # Grid settings
        self.show_grid = True
        self.grid_size = 20
        self.snap_to_grid = True
    
    def _create_default_world(self) -> WorldSpec:
        """Create a default empty world"""
        return WorldSpec(
            world_id=self.world_id,
            version="1.0",
            description="World created with visual editor",
            bounds=WorldBounds(
                width=float(self.config.world_width),
                height=float(self.config.world_height)
            ),
            physics=WorldPhysics(),
            geometry=WorldGeometry(
                rectangles=[],
                circles=[],
                segments=[]
            )
        )
    
    def snap_coords_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid if enabled"""
        if self.snap_to_grid:
            x = round(x / self.grid_size) * self.grid_size
            y = round(y / self.grid_size) * self.grid_size
        return x, y
    
    def get_rect_bounds(self, rect: RectangleObstacle) -> Tuple[float, float, float, float]:
        """Get rectangle bounds as (x1, y1, x2, y2)"""
        half_w = rect.width / 2
        half_h = rect.height / 2
        x1 = rect.x - half_w
        y1 = rect.y - half_h
        x2 = rect.x + half_w
        y2 = rect.y + half_h
        return x1, y1, x2, y2
    
    def point_in_rect(self, px: float, py: float, rect: RectangleObstacle) -> bool:
        """Check if point is inside rectangle"""
        x1, y1, x2, y2 = self.get_rect_bounds(rect)
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def get_resize_handle(self, px: float, py: float, rect: RectangleObstacle) -> Optional[str]:
        """Get which resize handle the point is over"""
        x1, y1, x2, y2 = self.get_rect_bounds(rect)
        h = self.handle_size
        
        # Check each corner and edge handle
        handles = {
            'nw': (x1 - h//2, y1 - h//2, x1 + h//2, y1 + h//2),  # Top-left
            'ne': (x2 - h//2, y1 - h//2, x2 + h//2, y1 + h//2),  # Top-right
            'sw': (x1 - h//2, y2 - h//2, x1 + h//2, y2 + h//2),  # Bottom-left
            'se': (x2 - h//2, y2 - h//2, x2 + h//2, y2 + h//2),  # Bottom-right
            'n': ((x1 + x2)//2 - h//2, y1 - h//2, (x1 + x2)//2 + h//2, y1 + h//2),  # Top
            's': ((x1 + x2)//2 - h//2, y2 - h//2, (x1 + x2)//2 + h//2, y2 + h//2),  # Bottom
            'w': (x1 - h//2, (y1 + y2)//2 - h//2, x1 + h//2, (y1 + y2)//2 + h//2),  # Left
            'e': (x2 - h//2, (y1 + y2)//2 - h//2, x2 + h//2, (y1 + y2)//2 + h//2),  # Right
        }
        
        for handle, (hx1, hy1, hx2, hy2) in handles.items():
            if hx1 <= px <= hx2 and hy1 <= py <= hy2:
                return handle
        
        return None
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mode == "create" and self.creating_rect:
                        # Cancel creation
                        self.creating_rect = None
                        self.mode = "select"
                    else:
                        self.running = False
                
                elif event.key == pygame.K_s and (pygame.key.get_pressed()[pygame.K_LCTRL] or 
                                                 pygame.key.get_pressed()[pygame.K_RCTRL]):
                    # Ctrl+S to save
                    self.save_world()
                
                elif event.key == pygame.K_n and (pygame.key.get_pressed()[pygame.K_LCTRL] or 
                                                 pygame.key.get_pressed()[pygame.K_RCTRL]):
                    # Ctrl+N for new rectangle
                    self.mode = "create"
                    self.selected_rect = None
                    self.creating_rect = None
                
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    # Delete selected rectangle
                    if self.selected_rect is not None:
                        del self.world.geometry.rectangles[self.selected_rect]
                        self.selected_rect = None
                        print(f"Deleted rectangle. Total: {len(self.world.geometry.rectangles)}")
                
                elif event.key == pygame.K_g:
                    # Toggle grid
                    self.show_grid = not self.show_grid
                
                elif event.key == pygame.K_t:
                    # Toggle snap to grid
                    self.snap_to_grid = not self.snap_to_grid
                    print(f"Snap to grid: {'ON' if self.snap_to_grid else 'OFF'}")
                
                elif event.key == pygame.K_TAB:
                    # Test world in simulation
                    self.test_world()
                
                elif event.key == pygame.K_F1:
                    # Debug: Add a simple test rectangle
                    test_rect = RectangleObstacle(
                        x=100, y=100, width=50, height=50, material_id=3
                    )
                    self.world.geometry.rectangles.append(test_rect)
                    print(f"Debug: Added test rectangle. Total: {len(self.world.geometry.rectangles)}")
                
                elif event.key == pygame.K_F2:
                    # Debug: Clear all rectangles
                    self.world.geometry.rectangles.clear()
                    self.selected_rect = None
                    print(f"Debug: Cleared all rectangles")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.handle_left_click(mouse_x, mouse_y)
                
                elif event.button == 3:  # Right click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.handle_right_click(mouse_x, mouse_y)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    self.handle_left_release()
            
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.handle_mouse_motion(mouse_x, mouse_y)
    
    def handle_left_click(self, x: float, y: float):
        """Handle left mouse click"""
        if self.mode == "create":
            # Start creating new rectangle
            x, y = self.snap_coords_to_grid(x, y)
            self.drag_start = (x, y)
            self.creating_rect = RectangleObstacle(
                x=x, y=y, width=1, height=1, material_id=3
            )
        
        elif self.mode == "select":
            # Check if clicking on a resize handle of selected rectangle
            if self.selected_rect is not None:
                rect = self.world.geometry.rectangles[self.selected_rect]
                handle = self.get_resize_handle(x, y, rect)
                if handle:
                    self.mode = "resize"
                    self.resize_handle = handle
                    self.drag_start = (x, y)
                    return
            
            # Check if clicking on any rectangle
            clicked_rect = None
            for i, rect in enumerate(self.world.geometry.rectangles):
                if self.point_in_rect(x, y, rect):
                    clicked_rect = i
                    break
            
            if clicked_rect is not None:
                # Select rectangle and start dragging
                self.selected_rect = clicked_rect
                self.drag_start = (x, y)
                print(f"Selected rectangle {clicked_rect}")
            else:
                # Deselect
                self.selected_rect = None
    
    def handle_right_click(self, x: float, y: float):
        """Handle right mouse click"""
        # Right click to create new rectangle at cursor
        x, y = self.snap_coords_to_grid(x, y)
        
        # Ensure rectangle fits within bounds
        rect_size = 40
        x = max(rect_size/2, min(self.config.world_width - rect_size/2, x))
        y = max(rect_size/2, min(self.config.world_height - rect_size/2, y))
        
        new_rect = RectangleObstacle(
            x=x, y=y, width=rect_size, height=rect_size, material_id=3
        )
        self.world.geometry.rectangles.append(new_rect)
        self.selected_rect = len(self.world.geometry.rectangles) - 1
        print(f"Created rectangle at ({x}, {y}). Total: {len(self.world.geometry.rectangles)}")
    
    def handle_left_release(self):
        """Handle left mouse button release"""
        if self.mode == "create" and self.creating_rect:
            # Finish creating rectangle
            if self.creating_rect.width > 5 and self.creating_rect.height > 5:
                self.world.geometry.rectangles.append(self.creating_rect)
                self.selected_rect = len(self.world.geometry.rectangles) - 1
                print(f"Created rectangle. Total: {len(self.world.geometry.rectangles)}")
            self.creating_rect = None
            self.mode = "select"
        
        elif self.mode == "resize":
            # Finish resizing
            self.mode = "select"
            self.resize_handle = None
        
        self.drag_start = None
    
    def handle_mouse_motion(self, x: float, y: float):
        """Handle mouse motion"""
        if self.drag_start is None:
            return
        
        x, y = self.snap_coords_to_grid(x, y)
        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]
        
        if self.mode == "create" and self.creating_rect:
            # Update creating rectangle size
            start_x, start_y = self.drag_start
            self.creating_rect.x = (start_x + x) / 2
            self.creating_rect.y = (start_y + y) / 2
            self.creating_rect.width = abs(x - start_x)
            self.creating_rect.height = abs(y - start_y)
        
        elif self.mode == "select" and self.selected_rect is not None:
            # Move selected rectangle
            rect = self.world.geometry.rectangles[self.selected_rect]
            new_x = rect.x + dx
            new_y = rect.y + dy
            
            # Constrain to world bounds
            half_w = rect.width / 2
            half_h = rect.height / 2
            new_x = max(half_w, min(self.config.world_width - half_w, new_x))
            new_y = max(half_h, min(self.config.world_height - half_h, new_y))
            
            rect.x = new_x
            rect.y = new_y
            self.drag_start = (x, y)
        
        elif self.mode == "resize" and self.selected_rect is not None:
            # Resize selected rectangle
            rect = self.world.geometry.rectangles[self.selected_rect]
            self.resize_rectangle(rect, self.resize_handle, dx, dy)
            self.drag_start = (x, y)
    
    def resize_rectangle(self, rect: RectangleObstacle, handle: str, dx: float, dy: float):
        """Resize rectangle based on handle and delta"""
        x1, y1, x2, y2 = self.get_rect_bounds(rect)
        
        # Adjust bounds based on handle
        if 'n' in handle:  # North (top)
            y1 += dy
        if 's' in handle:  # South (bottom)
            y2 += dy
        if 'w' in handle:  # West (left)
            x1 += dx
        if 'e' in handle:  # East (right)
            x2 += dx
        
        # Ensure minimum size
        if x2 - x1 < 10:
            if 'w' in handle:
                x1 = x2 - 10
            else:
                x2 = x1 + 10
        
        if y2 - y1 < 10:
            if 'n' in handle:
                y1 = y2 - 10
            else:
                y2 = y1 + 10
        
        # Update rectangle
        rect.x = (x1 + x2) / 2
        rect.y = (y1 + y2) / 2
        rect.width = x2 - x1
        rect.height = y2 - y1
    
    def render(self):
        """Render the current state"""
        self.screen.fill(self.BLACK)
        
        # Draw grid
        if self.show_grid:
            for x in range(0, self.config.world_width, self.grid_size):
                pygame.draw.line(self.screen, self.DARK_GRAY, (x, 0), (x, self.config.world_height))
            for y in range(0, self.config.world_height, self.grid_size):
                pygame.draw.line(self.screen, self.DARK_GRAY, (0, y), (self.config.world_width, y))
        
        # Draw border
        pygame.draw.rect(self.screen, self.WHITE, 
                        (0, 0, self.config.world_width, self.config.world_height), 2)
        
        # Draw rectangles
        for i, rect in enumerate(self.world.geometry.rectangles):
            x1, y1, x2, y2 = self.get_rect_bounds(rect)
            
            # Convert to pygame rect format
            pygame_x = int(x1)
            pygame_y = int(y1)
            pygame_w = int(x2 - x1)
            pygame_h = int(y2 - y1)
            
            # Skip invalid rectangles
            if pygame_w <= 0 or pygame_h <= 0:
                continue
            
            # Choose colors - make them more visible
            if i == self.selected_rect:
                fill_color = self.BLUE  # Light blue
                border_color = self.YELLOW  # Yellow
            else:
                fill_color = self.GRAY  # Light gray (more visible than dark gray)
                border_color = self.WHITE  # White border
            
            # Draw filled rectangle
            pygame.draw.rect(self.screen, fill_color, (pygame_x, pygame_y, pygame_w, pygame_h))
            # Draw border
            pygame.draw.rect(self.screen, border_color, (pygame_x, pygame_y, pygame_w, pygame_h), 2)
            
            # Draw resize handles for selected rectangle
            if i == self.selected_rect:
                self.draw_resize_handles(rect)
        
        # Draw creating rectangle
        if self.creating_rect:
            x1, y1, x2, y2 = self.get_rect_bounds(self.creating_rect)
            width = x2 - x1
            height = y2 - y1
            pygame.draw.rect(self.screen, self.GREEN, (x1, y1, width, height))
            pygame.draw.rect(self.screen, self.WHITE, (x1, y1, width, height), 2)
        
        # Draw UI
        ui_lines = [
            f"World Editor - {self.world_id}",
            f"Mode: {self.mode.upper()}",
            f"Rectangles: {len(self.world.geometry.rectangles)}",
            f"Selected: {self.selected_rect if self.selected_rect is not None else 'None'}",
            f"Grid: {'ON' if self.show_grid else 'OFF'} (G to toggle)",
            f"Snap: {'ON' if self.snap_to_grid else 'OFF'} (T to toggle)",
            "",
            "Controls:",
            "  Right-click: Create rectangle at cursor",
            "  Left-click: Select/move rectangle",
            "  Drag handles: Resize selected rectangle",
            "  Ctrl+N: Start creating rectangle",
            "  Delete: Remove selected rectangle",
            "  Ctrl+S: Save world",
            "  TAB: Test world in simulation",
            "  G: Toggle grid, T: Toggle snap",
            "  F1: Add test rectangle, F2: Clear all",
            "  ESC: Exit"
        ]
        
        for i, line in enumerate(ui_lines):
            color = self.YELLOW if i == 0 else self.WHITE
            text = self.font.render(line, True, color)
            self.screen.blit(text, (10, 10 + i * 22))
        
        pygame.display.flip()
    
    def draw_resize_handles(self, rect: RectangleObstacle):
        """Draw resize handles for a rectangle"""
        x1, y1, x2, y2 = self.get_rect_bounds(rect)
        h = self.handle_size
        
        # Handle positions
        handles = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x1, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
            ((x1 + x2) / 2, y1),  # Top
            ((x1 + x2) / 2, y2),  # Bottom
            (x1, (y1 + y2) / 2),  # Left
            (x2, (y1 + y2) / 2),  # Right
        ]
        
        for hx, hy in handles:
            pygame.draw.rect(self.screen, self.YELLOW, 
                           (hx - h//2, hy - h//2, h, h))
            pygame.draw.rect(self.screen, self.BLACK, 
                           (hx - h//2, hy - h//2, h, h), 1)
    
    def save_world(self):
        """Save the current world to file"""
        world_dir = "data/worlds"
        os.makedirs(world_dir, exist_ok=True)
        
        world_path = os.path.join(world_dir, f"{self.world_id}.json")
        
        try:
            # Convert to dict for JSON serialization
            world_dict = {
                "world_id": self.world.world_id,
                "version": self.world.version,
                "description": self.world.description,
                "bounds": {
                    "width": self.world.bounds.width,
                    "height": self.world.bounds.height
                },
                "physics": {
                    "agent_max_thrust": self.world.physics.agent_max_thrust,
                    "agent_max_turn_accel": self.world.physics.agent_max_turn_accel,
                    "agent_linear_drag": self.world.physics.agent_linear_drag,
                    "agent_angular_drag": self.world.physics.agent_angular_drag,
                    "wall_restitution": self.world.physics.wall_restitution
                },
                "geometry": {
                    "rectangles": [
                        {
                            "x": rect.x,
                            "y": rect.y,
                            "width": rect.width,
                            "height": rect.height,
                            "material_id": rect.material_id
                        }
                        for rect in self.world.geometry.rectangles
                    ],
                    "circles": [],
                    "segments": []
                }
            }
            
            with open(world_path, 'w') as f:
                json.dump(world_dict, f, indent=2)
            
            print(f"Saved world to {world_path}")
            
        except Exception as e:
            print(f"Error saving world: {e}")
    
    def test_world(self):
        """Test the current world in a simulation"""
        try:
            print("Testing world in simulation...")
            
            # Create temporary simulation to test world
            batched_sim = BatchedSimulation(1, self.config)
            updated_config = apply_world_to_simulation(batched_sim, self.world, [])
            
            sim = SimulationSingle(updated_config)
            sim.batched_sim = batched_sim
            
            # Position agent in center
            sim.batched_sim.agent_x[0] = self.config.world_width / 2
            sim.batched_sim.agent_y[0] = self.config.world_height / 2
            sim.batched_sim.agent_theta[0] = 0.0
            
            # Get vision data to test raycasting
            state = sim.get_state()
            distances = state['vision_distances']
            hit_types = state['vision_hit_types']
            
            wall_hits = sum(1 for ht in hit_types if ht == 'wall')
            print(f"World test: {wall_hits}/{len(hit_types)} rays hit walls")
            
        except Exception as e:
            print(f"Error testing world: {e}")
    
    def run(self):
        """Main loop"""
        print("World Editor")
        print("=" * 50)
        print(f"Editing world: {self.world_id}")
        print(f"Current rectangles: {len(self.world.geometry.rectangles)}")
        print("Right-click to create rectangles, left-click to select/move")
        print("Use Ctrl+S to save, TAB to test in simulation")
        print()
        
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visual world editor for creating obstacle layouts')
    
    parser.add_argument('--world', type=str, default="new_world",
                       help='World ID to edit (default: new_world)')
    parser.add_argument('--list-worlds', action='store_true',
                       help='List available worlds and exit')
    
    args = parser.parse_args()
    
    # List worlds if requested
    if args.list_worlds:
        worlds = list_available_worlds()
        print("Available worlds:")
        for world_id in worlds:
            if world_id == "default_empty":
                print(f"  - {world_id} (default)")
            else:
                try:
                    world = load_world(world_id)
                    rect_count = len(world.geometry.rectangles)
                    print(f"  - {world_id}: {world.description} ({rect_count} rectangles)")
                except Exception as e:
                    print(f"  - {world_id}: (error loading: {e})")
        return
    
    # Create simulation config
    config = SimulationConfig()
    
    # Create and run editor
    editor = WorldEditor(config, args.world)
    editor.run()


if __name__ == "__main__":
    main()