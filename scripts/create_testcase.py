#!/usr/bin/env python3
"""
Interactive test case creator
Move agent and food positions, then save as a test case
"""

import argparse
import os
import sys
import json
import math
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import pygame
try:
    import pygame
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)

from src.sim.core import SimulationConfig
from src.eval.testcases import TestCase, AgentState, FoodState, Obstacles
from src.eval.load_suite import load_suite, save_suite, TestSuite
from src.eval.load_world import list_available_worlds, load_world
from src.viz.vision_rendering import draw_vision_cone, create_mock_vision_data


class TestCaseCreator:
    """Interactive test case creation tool"""
    
    def __init__(self, config: SimulationConfig, suite_path: str = None):
        self.config = config
        self.suite_path = suite_path
        
        # Load existing suite or create new one
        if suite_path and os.path.exists(suite_path):
            try:
                self.suite = load_suite(suite_path)
                print(f"Loaded existing test suite: {suite_path}")
                print(f"Suite ID: {self.suite.suite_id}")
                print(f"Existing test cases: {len(self.suite.test_cases)}")
            except Exception as e:
                print(f"Error loading suite {suite_path}: {e}")
                print("Creating new suite instead...")
                self.suite = None
        else:
            self.suite = None
            if suite_path:
                print(f"Suite file {suite_path} doesn't exist, will create new one")
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.world_width, config.world_height))
        pygame.display.set_caption("Test Case Creator - Position Agent & Food")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (50, 150, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # State
        self.running = True
        self.mode = "agent"  # "agent" or "food"
        self.show_vision = True  # Toggle vision cone display
        
        # Agent state (start in center)
        self.agent_x = config.world_width // 2
        self.agent_y = config.world_height // 2
        self.agent_theta = 0.0  # Facing right
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.agent_omega = 0.0
        self.agent_throttle = 0.0
        
        # Food state (start in upper right)
        self.food_x = config.world_width * 0.75
        self.food_y = config.world_height * 0.25
        self.food_radius = 8.0
        
        # Test case parameters
        self.test_case_id = "custom_test_case"
        self.max_steps = 600
        self.notes = "Custom test case created interactively"
        self.world_id = None  # No world selected by default
        
        # Available worlds
        self.available_worlds = list_available_worlds()
        self.current_world_idx = 0  # Index into available_worlds (0 = default_empty)
        
        # World geometry cache
        self.current_world = None
        self.world_obstacles = []  # List of obstacle shapes for rendering
        
        # Movement speed
        self.move_speed = 5.0
        self.rotation_speed = 0.1
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_TAB:
                    # Switch between agent and food positioning
                    self.mode = "food" if self.mode == "agent" else "agent"
                
                elif event.key == pygame.K_s and (pygame.key.get_pressed()[pygame.K_LCTRL] or 
                                                 pygame.key.get_pressed()[pygame.K_RCTRL]):
                    # Ctrl+S to save
                    self.save_test_case()
                
                elif event.key == pygame.K_r:
                    # Reset to defaults
                    self.reset_positions()
                
                elif event.key == pygame.K_v:
                    # Toggle vision display
                    self.show_vision = not self.show_vision
                
                elif event.key == pygame.K_w:
                    # Cycle through available worlds
                    self.current_world_idx = (self.current_world_idx + 1) % len(self.available_worlds)
                    self.world_id = self.available_worlds[self.current_world_idx] if self.current_world_idx > 0 else None
                    self.load_world_geometry()  # Load new world geometry
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    if self.mode == "agent":
                        # Move agent to mouse position
                        self.agent_x = mouse_x
                        self.agent_y = mouse_y
                    elif self.mode == "food":
                        # Move food to mouse position
                        self.food_x = mouse_x
                        self.food_y = mouse_y
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        
        if self.mode == "agent":
            # Move agent with arrow keys
            if keys[pygame.K_LEFT]:
                self.agent_x -= self.move_speed
            if keys[pygame.K_RIGHT]:
                self.agent_x += self.move_speed
            if keys[pygame.K_UP]:
                self.agent_y -= self.move_speed
            if keys[pygame.K_DOWN]:
                self.agent_y += self.move_speed
            
            # Rotate agent with A/D
            if keys[pygame.K_a]:
                self.agent_theta -= self.rotation_speed
            if keys[pygame.K_d]:
                self.agent_theta += self.rotation_speed
            
            # Clamp agent position
            self.agent_x = max(self.config.agent_radius, 
                              min(self.config.world_width - self.config.agent_radius, self.agent_x))
            self.agent_y = max(self.config.agent_radius, 
                              min(self.config.world_height - self.config.agent_radius, self.agent_y))
        
        elif self.mode == "food":
            # Move food with arrow keys
            if keys[pygame.K_LEFT]:
                self.food_x -= self.move_speed
            if keys[pygame.K_RIGHT]:
                self.food_x += self.move_speed
            if keys[pygame.K_UP]:
                self.food_y -= self.move_speed
            if keys[pygame.K_DOWN]:
                self.food_y += self.move_speed
            
            # Clamp food position
            self.food_x = max(self.food_radius, 
                             min(self.config.world_width - self.food_radius, self.food_x))
            self.food_y = max(self.food_radius, 
                             min(self.config.world_height - self.food_radius, self.food_y))
    
    def reset_positions(self):
        """Reset to default positions"""
        self.agent_x = self.config.world_width // 2
        self.agent_y = self.config.world_height // 2
        self.agent_theta = 0.0
        self.food_x = self.config.world_width * 0.75
        self.food_y = self.config.world_height * 0.25
    
    def load_world_geometry(self):
        """Load and cache world geometry for rendering"""
        self.world_obstacles = []
        
        if self.world_id and self.world_id != "default_empty":
            try:
                self.current_world = load_world(self.world_id)
                
                # Convert world geometry to renderable obstacles
                # Rectangle obstacles
                for rect in self.current_world.geometry.rectangles:
                    # Convert center+size to corner coordinates
                    half_w = rect.width / 2
                    half_h = rect.height / 2
                    x1, y1 = rect.x - half_w, rect.y - half_h
                    x2, y2 = rect.x + half_w, rect.y + half_h
                    
                    self.world_obstacles.append({
                        'type': 'rectangle',
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'color': (128, 128, 128)  # Gray
                    })
                
                # Circle obstacles
                for circle in self.current_world.geometry.circles:
                    self.world_obstacles.append({
                        'type': 'circle',
                        'x': circle.x, 'y': circle.y, 'radius': circle.radius,
                        'color': (128, 128, 128)  # Gray
                    })
                
                # Segment obstacles
                for segment in self.current_world.geometry.segments:
                    self.world_obstacles.append({
                        'type': 'segment',
                        'x1': segment.x1, 'y1': segment.y1,
                        'x2': segment.x2, 'y2': segment.y2,
                        'color': (128, 128, 128)  # Gray
                    })
                
                print(f"Loaded world '{self.world_id}' with {len(self.world_obstacles)} obstacles")
                
            except Exception as e:
                print(f"Error loading world '{self.world_id}': {e}")
                self.current_world = None
                self.world_obstacles = []
        else:
            self.current_world = None
    
    def render(self):
        """Render the current state"""
        self.screen.fill(self.BLACK)
        
        # Draw border
        pygame.draw.rect(self.screen, self.WHITE, 
                        (0, 0, self.config.world_width, self.config.world_height), 2)
        
        # Draw world obstacles
        for obstacle in self.world_obstacles:
            if obstacle['type'] == 'rectangle':
                x1, y1, x2, y2 = obstacle['x1'], obstacle['y1'], obstacle['x2'], obstacle['y2']
                width = x2 - x1
                height = y2 - y1
                pygame.draw.rect(self.screen, obstacle['color'], (x1, y1, width, height))
                pygame.draw.rect(self.screen, self.WHITE, (x1, y1, width, height), 2)  # Outline
            
            elif obstacle['type'] == 'circle':
                pygame.draw.circle(self.screen, obstacle['color'], 
                                 (int(obstacle['x']), int(obstacle['y'])), int(obstacle['radius']))
                pygame.draw.circle(self.screen, self.WHITE, 
                                 (int(obstacle['x']), int(obstacle['y'])), int(obstacle['radius']), 2)  # Outline
            
            elif obstacle['type'] == 'segment':
                pygame.draw.line(self.screen, obstacle['color'], 
                               (obstacle['x1'], obstacle['y1']), (obstacle['x2'], obstacle['y2']), 3)
        
        # Draw food
        food_color = self.YELLOW if self.mode == "food" else self.GREEN
        pygame.draw.circle(self.screen, food_color, 
                          (int(self.food_x), int(self.food_y)), int(self.food_radius))
        
        # Draw agent
        agent_color = self.YELLOW if self.mode == "agent" else self.BLUE
        pygame.draw.circle(self.screen, agent_color, 
                          (int(self.agent_x), int(self.agent_y)), self.config.agent_radius)
        
        # Draw agent direction indicator
        dir_length = self.config.agent_radius * 1.5
        end_x = self.agent_x + dir_length * math.cos(self.agent_theta)
        end_y = self.agent_y + dir_length * math.sin(self.agent_theta)
        pygame.draw.line(self.screen, self.WHITE, 
                        (self.agent_x, self.agent_y), (end_x, end_y), 2)
        
        # Draw vision cone if enabled
        if self.show_vision:
            distances, hit_types = create_mock_vision_data(
                self.agent_x, self.agent_y, self.agent_theta,
                self.food_x, self.food_y, self.config
            )
            
            draw_vision_cone(
                screen=self.screen,
                agent_x=self.agent_x,
                agent_y=self.agent_y,
                agent_theta=self.agent_theta,
                distances=distances,
                hit_types=hit_types,
                config=self.config,
                show_rays=True,
                show_polygon=True,
                show_cone_outline=True
            )
        
        # Draw UI
        ui_lines = [
            f"Mode: {self.mode.upper()} (TAB to switch)",
            f"Agent: ({self.agent_x:.0f}, {self.agent_y:.0f}) θ={math.degrees(self.agent_theta):.0f}°",
            f"Food: ({self.food_x:.0f}, {self.food_y:.0f})",
            f"Distance: {math.sqrt((self.agent_x - self.food_x)**2 + (self.agent_y - self.food_y)**2):.1f}",
            f"Suite: {os.path.basename(self.suite_path) if self.suite_path else 'Individual file'}",
            f"Case ID: {self.test_case_id}",
            f"World: {self.world_id or 'default_empty'} (W to cycle)",
            f"Vision: {'ON' if self.show_vision else 'OFF'} (V to toggle)",
            "",
            "Controls:",
            "  Arrow keys: Move selected object",
            "  A/D: Rotate agent (agent mode only)",
            "  Click: Move selected object to mouse",
            "  TAB: Switch between agent/food",
            "  W: Cycle through available worlds",
            "  V: Toggle vision cone display",
            "  Ctrl+S: Save test case",
            "  R: Reset positions",
            "  ESC: Exit"
        ]
        
        for i, line in enumerate(ui_lines):
            color = self.YELLOW if i == 0 else self.WHITE
            text = self.font.render(line, True, color)
            self.screen.blit(text, (10, 10 + i * 25))
        
        pygame.display.flip()
    
    def save_test_case(self):
        """Save current configuration as a test case to the suite"""
        # Create test case
        agent_state = AgentState(
            x=float(self.agent_x),
            y=float(self.agent_y),
            theta=float(self.agent_theta),
            vx=self.agent_vx,
            vy=self.agent_vy,
            omega=self.agent_omega,
            throttle=self.agent_throttle
        )
        
        food_state = FoodState(
            x=float(self.food_x),
            y=float(self.food_y),
            radius=self.food_radius
        )
        
        obstacles = Obstacles(circles=[], segments=[])
        
        test_case = TestCase(
            id=self.test_case_id,
            max_steps=self.max_steps,
            dt=None,  # Use default
            agent_start=agent_state,
            food=food_state,
            world_id=self.world_id,  # Reference to world
            world_overrides={},  # No overrides in creator
            obstacles=obstacles,  # Keep for backward compatibility
            notes=self.notes
        )
        
        try:
            if self.suite_path:
                # Save to test suite
                self.save_to_suite(test_case)
            else:
                # Save as individual file
                self.save_individual_file(test_case)
            
            # Show success message on screen briefly
            self.show_save_message()
            
        except Exception as e:
            print(f"Error saving test case: {e}")
    
    def save_to_suite(self, test_case: TestCase):
        """Save test case to a test suite"""
        # Create or update suite
        if self.suite is None:
            # Create new suite
            suite_id = os.path.splitext(os.path.basename(self.suite_path))[0]
            self.suite = TestSuite(
                suite_id=suite_id,
                version="1.0",
                description=f"Custom test suite created interactively",
                test_cases=[]
            )
            print(f"Created new test suite: {suite_id}")
        
        # Check if test case ID already exists
        existing_case = None
        for i, case in enumerate(self.suite.test_cases):
            if case.id == test_case.id:
                existing_case = i
                break
        
        if existing_case is not None:
            # Overwrite existing case
            print(f"WARNING: Overwriting existing test case '{test_case.id}' in suite")
            self.suite.test_cases[existing_case] = test_case
        else:
            # Add new case
            self.suite.test_cases.append(test_case)
            print(f"Added new test case '{test_case.id}' to suite")
        
        # Save suite to file
        save_suite(self.suite, self.suite_path)
        
        print(f"Test suite saved to: {self.suite_path}")
        print(f"Total test cases in suite: {len(self.suite.test_cases)}")
        print(f"Test case details:")
        print(f"  Agent: ({self.agent_x:.1f}, {self.agent_y:.1f}) θ={math.degrees(self.agent_theta):.1f}°")
        print(f"  Food: ({self.food_x:.1f}, {self.food_y:.1f})")
        
        distance = math.sqrt((self.agent_x - self.food_x)**2 + (self.agent_y - self.food_y)**2)
        print(f"  Distance: {distance:.1f}")
    
    def save_individual_file(self, test_case: TestCase):
        """Save test case as individual JSON file"""
        output_file = f"{self.test_case_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(test_case.to_dict(), f, indent=2)
        
        print(f"Test case saved to: {output_file}")
        print(f"Agent position: ({self.agent_x:.1f}, {self.agent_y:.1f})")
        print(f"Agent angle: {math.degrees(self.agent_theta):.1f}°")
        print(f"Food position: ({self.food_x:.1f}, {self.food_y:.1f})")
        
        distance = math.sqrt((self.agent_x - self.food_x)**2 + (self.agent_y - self.food_y)**2)
        print(f"Distance: {distance:.1f}")
    
    def show_save_message(self):
        """Show save confirmation message"""
        # Create overlay
        overlay = pygame.Surface((500, 120))
        overlay.set_alpha(200)
        overlay.fill(self.BLACK)
        pygame.draw.rect(overlay, self.GREEN, (0, 0, 500, 120), 2)
        
        # Add text
        if self.suite_path:
            text1 = self.font.render("Test case saved to suite!", True, self.GREEN)
            text2 = self.font.render(f"Suite: {os.path.basename(self.suite_path)}", True, self.WHITE)
            text3 = self.font.render(f"Case ID: {self.test_case_id}", True, self.WHITE)
        else:
            text1 = self.font.render("Test case saved!", True, self.GREEN)
            text2 = self.font.render(f"File: {self.test_case_id}.json", True, self.WHITE)
            text3 = None
        
        overlay.blit(text1, (10, 20))
        overlay.blit(text2, (10, 50))
        if text3:
            overlay.blit(text3, (10, 80))
        
        # Show overlay
        x = (self.config.world_width - 500) // 2
        y = (self.config.world_height - 120) // 2
        self.screen.blit(overlay, (x, y))
        pygame.display.flip()
        
        # Wait briefly
        pygame.time.wait(1500)
    
    def run(self):
        """Main loop"""
        print("Test Case Creator")
        print("=" * 50)
        print("Position the agent and food, then save as a test case")
        print("Use TAB to switch between positioning agent and food")
        print("Use Ctrl+S to save the current configuration")
        print()
        
        # Load initial world geometry
        self.load_world_geometry()
        
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create test cases interactively')
    
    parser.add_argument('--suite', type=str, default=None,
                       help='Test suite file path (creates new if doesn\'t exist)')
    parser.add_argument('--id', type=str, default='custom_test_case',
                       help='Test case ID (default: custom_test_case)')
    parser.add_argument('--max-steps', type=int, default=600,
                       help='Maximum steps for test case (default: 600)')
    parser.add_argument('--notes', type=str, default='Custom test case created interactively',
                       help='Description/notes for test case')
    parser.add_argument('--world', type=str, default=None,
                       help='Initial world to load (default: default_empty, use --list-worlds to see options)')
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
                    print(f"  - {world_id}: {world.description}")
                except Exception as e:
                    print(f"  - {world_id}: (error loading: {e})")
        return
    
    # Create simulation config
    config = SimulationConfig()
    
    # Create and run creator
    creator = TestCaseCreator(config, args.suite)
    creator.test_case_id = args.id
    creator.max_steps = args.max_steps
    creator.notes = args.notes
    
    # Set initial world if specified
    if args.world:
        available_worlds = list_available_worlds()
        if args.world in available_worlds:
            creator.world_id = args.world
            creator.current_world_idx = available_worlds.index(args.world)
            print(f"Starting with world: {args.world}")
        else:
            print(f"Warning: World '{args.world}' not found. Available worlds:")
            for world_id in available_worlds:
                print(f"  - {world_id}")
            print("Starting with default world.")
    
    creator.run()


if __name__ == "__main__":
    main()