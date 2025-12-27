#!/usr/bin/env python3
"""
Minimal 2D Continuous Simulation
- Agent with physics-based movement (steer + throttle)
- Food collection
- Wall collisions
- Manual keyboard controls
- Ray-casting vision system with LiDAR strip visualization
"""

import pygame
import math
import random
import sys
import numpy as np

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60

# Agent constants
AGENT_RADIUS = 12
AGENT_MAX_THRUST = 600.0  # pixels/s²
AGENT_MAX_TURN_ACCEL = 12.0  # rad/s²
AGENT_LINEAR_DRAG = 2.0
AGENT_ANGULAR_DRAG = 5.0
WALL_RESTITUTION = 0.6

# Food constants
FOOD_RADIUS = 8

# Vision constants
FOV_DEGREES = 120
NUM_RAYS = 128
MAX_RANGE = 300
STRIP_HEIGHT = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0  # heading angle in radians
        self.omega = 0.0  # angular velocity
        self.throttle = 0.0  # 0 to 1
        
    def update(self, dt, steer_input, throttle_input):
        # Direct throttle control - only apply force while key is held
        self.throttle = max(0.0, throttle_input)  # throttle_input is 0 or 1
        
        # Forward direction
        fx = math.cos(self.theta)
        fy = math.sin(self.theta)
        
        # Apply thrust
        thrust_accel = self.throttle * AGENT_MAX_THRUST
        ax = thrust_accel * fx
        ay = thrust_accel * fy
        
        # Update velocity
        self.vx += ax * dt
        self.vy += ay * dt
        
        # Apply linear drag
        drag_factor = max(0.0, 1.0 - AGENT_LINEAR_DRAG * dt)
        self.vx *= drag_factor
        self.vy *= drag_factor
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Update angular motion
        self.omega += steer_input * AGENT_MAX_TURN_ACCEL * dt
        
        # Apply angular drag
        angular_drag_factor = max(0.0, 1.0 - AGENT_ANGULAR_DRAG * dt)
        self.omega *= angular_drag_factor
        
        # Update heading
        self.theta += self.omega * dt
        
        # Normalize theta to [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
    
    def handle_wall_collisions(self):
        # Left wall
        if self.x - AGENT_RADIUS < 0:
            self.x = AGENT_RADIUS
            self.vx *= -WALL_RESTITUTION
            self.omega *= 0.8
        
        # Right wall
        if self.x + AGENT_RADIUS > WINDOW_WIDTH:
            self.x = WINDOW_WIDTH - AGENT_RADIUS
            self.vx *= -WALL_RESTITUTION
            self.omega *= 0.8
        
        # Top wall
        if self.y - AGENT_RADIUS < 0:
            self.y = AGENT_RADIUS
            self.vy *= -WALL_RESTITUTION
            self.omega *= 0.8
        
        # Bottom wall
        if self.y + AGENT_RADIUS > WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT - AGENT_RADIUS
            self.vy *= -WALL_RESTITUTION
            self.omega *= 0.8
    
    def get_speed(self):
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)


class Food:
    def __init__(self):
        self.respawn()
    
    def respawn(self):
        # Spawn food away from edges to avoid overlap with agent spawn
        margin = AGENT_RADIUS + FOOD_RADIUS + 20
        self.x = random.uniform(margin, WINDOW_WIDTH - margin)
        self.y = random.uniform(margin, WINDOW_HEIGHT - margin)


# Ray-casting helper functions
def intersect_ray_circle(origin_x, origin_y, dir_x, dir_y, circle_x, circle_y, radius):
    """
    Ray-circle intersection. Returns distance to nearest intersection or None.
    """
    # Vector from ray origin to circle center
    oc_x = origin_x - circle_x
    oc_y = origin_y - circle_y
    
    # Quadratic equation coefficients: at² + bt + c = 0
    a = dir_x * dir_x + dir_y * dir_y
    b = 2.0 * (oc_x * dir_x + oc_y * dir_y)
    c = oc_x * oc_x + oc_y * oc_y - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return None  # No intersection
    
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    # We want the nearest positive intersection
    if t1 > 0:
        return t1
    elif t2 > 0:
        return t2
    else:
        return None  # Circle is behind ray origin


def intersect_ray_segment(origin_x, origin_y, dir_x, dir_y, p1_x, p1_y, p2_x, p2_y):
    """
    Ray-line segment intersection. Returns distance to intersection or None.
    """
    # Line segment vector
    seg_x = p2_x - p1_x
    seg_y = p2_y - p1_y
    
    # Vector from segment start to ray origin
    h_x = p1_x - origin_x
    h_y = p1_y - origin_y
    
    # Cross products for 2D
    a = dir_x * seg_y - dir_y * seg_x
    
    if abs(a) < 1e-10:  # Ray parallel to segment
        return None
    
    f = 1.0 / a
    s = f * (h_x * seg_y - h_y * seg_x)
    t = f * (h_x * dir_y - h_y * dir_x)
    
    # More robust checks: s must be positive (forward ray) and t must be in [0,1] (on segment)
    if s > 1e-6 and 0 <= t <= 1:  # Small epsilon to avoid numerical issues
        return s
    
    return None


def cast_ray(origin_x, origin_y, dir_x, dir_y, food):
    """
    Cast a single ray and find the nearest intersection.
    Returns (distance, hit_type, hit_color) or (None, None, None) if no hit.
    """
    nearest_distance = float('inf')
    hit_type = None
    hit_color = None
    
    # Check wall intersections (4 segments) with different colors for debugging
    walls = [
        (0, 0, WINDOW_WIDTH, 0, (255, 255, 255)),  # Top wall - White
        (WINDOW_WIDTH, 0, WINDOW_WIDTH, WINDOW_HEIGHT, (0, 255, 255)),  # Right wall - Cyan
        (WINDOW_WIDTH, WINDOW_HEIGHT, 0, WINDOW_HEIGHT, (255, 0, 255)),  # Bottom wall - Magenta
        (0, WINDOW_HEIGHT, 0, 0, (255, 165, 0))  # Left wall - Orange
    ]
    
    for wall_idx, (p1_x, p1_y, p2_x, p2_y, wall_color) in enumerate(walls):
        distance = intersect_ray_segment(origin_x, origin_y, dir_x, dir_y, p1_x, p1_y, p2_x, p2_y)
        if distance is not None and distance < nearest_distance and distance > 1e-6:
            nearest_distance = distance
            hit_type = f'wall_{wall_idx}'
            hit_color = wall_color
    
    # Check food intersection
    food_distance = intersect_ray_circle(origin_x, origin_y, dir_x, dir_y, food.x, food.y, FOOD_RADIUS)
    if food_distance is not None and food_distance < nearest_distance and food_distance > 1e-6:
        nearest_distance = food_distance
        hit_type = 'food'
        hit_color = RED
    
    if nearest_distance == float('inf'):
        return None, None, None
    
    return nearest_distance, hit_type, hit_color


def compute_vision(agent, food):
    """
    Compute ray-casting vision for the agent.
    Returns arrays of distances, hit_points, and colors.
    """
    fov_rad = math.radians(FOV_DEGREES)
    angles = np.linspace(-fov_rad/2, fov_rad/2, NUM_RAYS)
    
    distances = []
    hit_points = []
    colors = []
    
    for angle in angles:
        ray_angle = agent.theta + angle
        dir_x = math.cos(ray_angle)
        dir_y = math.sin(ray_angle)
        
        distance, hit_type, hit_color = cast_ray(agent.x, agent.y, dir_x, dir_y, food)
        
        if distance is not None and distance <= MAX_RANGE:
            hit_x = agent.x + distance * dir_x
            hit_y = agent.y + distance * dir_y
            distances.append(distance)
            hit_points.append((hit_x, hit_y))
            colors.append(hit_color)
        else:
            distances.append(MAX_RANGE)
            hit_x = agent.x + MAX_RANGE * dir_x
            hit_y = agent.y + MAX_RANGE * dir_y
            hit_points.append((hit_x, hit_y))
            colors.append(BLACK)
    
    return distances, hit_points, colors


def build_depth_strip(distances, colors):
    """
    Build the 1D depth image strip as a pygame surface.
    """
    strip = pygame.Surface((NUM_RAYS, STRIP_HEIGHT))
    strip.fill(BLACK)
    
    for i, (distance, color) in enumerate(zip(distances, colors)):
        if color != BLACK:  # Hit something
            # Brightness based on distance (closer = brighter)
            brightness = max(0.0, 1.0 - distance / MAX_RANGE)
            
            # Apply brightness to color
            bright_color = (
                int(color[0] * brightness),
                int(color[1] * brightness),
                int(color[2] * brightness)
            )
            
            # Fill the column
            pygame.draw.rect(strip, bright_color, (i, 0, 1, STRIP_HEIGHT))
    
    return strip


def check_food_collision(agent, food):
    dx = agent.x - food.x
    dy = agent.y - food.y
    distance = math.sqrt(dx * dx + dy * dy)
    return distance <= (AGENT_RADIUS + FOOD_RADIUS)


def draw_agent(screen, agent):
    # Draw agent circle
    pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), AGENT_RADIUS)
    
    # Draw heading indicator
    head_length = AGENT_RADIUS + 8
    head_x = agent.x + head_length * math.cos(agent.theta)
    head_y = agent.y + head_length * math.sin(agent.theta)
    pygame.draw.line(screen, WHITE, (agent.x, agent.y), (head_x, head_y), 3)


def draw_food(screen, food):
    pygame.draw.circle(screen, RED, (int(food.x), int(food.y)), FOOD_RADIUS)


def draw_vision(screen, agent, distances, hit_points, show_vision):
    """
    Draw the FOV cone and ray hits visualization.
    """
    if not show_vision:
        return
    
    fov_rad = math.radians(FOV_DEGREES)
    
    # Draw visible region polygon
    if len(hit_points) > 0:
        # Create polygon vertices: agent position + hit points
        polygon_points = [(agent.x, agent.y)]
        for hit_x, hit_y in hit_points:
            # Clamp to screen bounds for drawing
            hit_x = max(0, min(WINDOW_WIDTH, hit_x))
            hit_y = max(0, min(WINDOW_HEIGHT, hit_y))
            polygon_points.append((hit_x, hit_y))
        
        if len(polygon_points) > 2:
            # Draw semi-transparent filled polygon
            temp_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            temp_surface.set_alpha(30)
            temp_surface.fill(BLACK)
            pygame.draw.polygon(temp_surface, CYAN, polygon_points)
            screen.blit(temp_surface, (0, 0))
            
            # Draw polygon outline
            pygame.draw.polygon(screen, CYAN, polygon_points, 1)
    
    # Draw hit points with debugging info
    for i, ((hit_x, hit_y), distance) in enumerate(zip(hit_points, distances)):
        if distance < MAX_RANGE:
            # Check if hit point is outside the visible window bounds
            outside_bounds = (hit_x < -5 or hit_x > WINDOW_WIDTH + 5 or hit_y < -5 or hit_y > WINDOW_HEIGHT + 5)
            
            if outside_bounds:
                # Draw red circle for hits outside bounds - this might be the bug!
                pygame.draw.circle(screen, (255, 0, 0), (int(max(0, min(WINDOW_WIDTH, hit_x))), int(max(0, min(WINDOW_HEIGHT, hit_y)))), 4)
                # Print debug info for the first few out-of-bounds hits
                if i < 5:
                    print(f"Ray {i}: Hit at ({hit_x:.1f}, {hit_y:.1f}), distance {distance:.1f} - OUTSIDE BOUNDS")
            else:
                pygame.draw.circle(screen, YELLOW, (int(hit_x), int(hit_y)), 2)
        
        # Draw every 8th ray for clarity, with different colors for debugging
        if i % 8 == 0:
            color = (100, 100, 100)
            if distance < MAX_RANGE:
                outside_bounds = (hit_x < -5 or hit_x > WINDOW_WIDTH + 5 or hit_y < -5 or hit_y > WINDOW_HEIGHT + 5)
                if outside_bounds:
                    color = (255, 100, 100)  # Red for out-of-bounds hits
            
            # Clamp ray endpoint for drawing
            draw_hit_x = max(0, min(WINDOW_WIDTH, hit_x))
            draw_hit_y = max(0, min(WINDOW_HEIGHT, hit_y))
            pygame.draw.line(screen, color, (agent.x, agent.y), (draw_hit_x, draw_hit_y), 1)

    # Draw FOV cone outline
    left_angle = agent.theta - fov_rad/2
    right_angle = agent.theta + fov_rad/2
    
    left_x = agent.x + MAX_RANGE * math.cos(left_angle)
    left_y = agent.y + MAX_RANGE * math.sin(left_angle)
    right_x = agent.x + MAX_RANGE * math.cos(right_angle)
    right_y = agent.y + MAX_RANGE * math.sin(right_angle)

    # Draw FOV cone lines
    pygame.draw.line(screen, YELLOW, (agent.x, agent.y), (left_x, left_y), 1)
    pygame.draw.line(screen, YELLOW, (agent.x, agent.y), (right_x, right_y), 1)

def draw_ui(screen, font, agent, fps, show_vision):
    # Throttle display
    throttle_text = font.render(f"Throttle: {agent.throttle:.2f}", True, WHITE)
    screen.blit(throttle_text, (10, 10))
    
    # Speed display
    speed_text = font.render(f"Speed: {agent.get_speed():.1f}", True, WHITE)
    screen.blit(speed_text, (10, 35))
    
    # FPS display
    fps_text = font.render(f"FPS: {fps:.1f}", True, WHITE)
    screen.blit(fps_text, (10, 60))
    
    # Agent position and heading
    pos_text = font.render(f"Position: ({agent.x:.1f}, {agent.y:.1f})", True, WHITE)
    screen.blit(pos_text, (10, 85))
    
    heading_degrees = math.degrees(agent.theta)
    heading_text = font.render(f"Heading: {heading_degrees:.1f}°", True, WHITE)
    screen.blit(heading_text, (10, 110))
    
    # Vision toggle status
    vision_text = font.render(f"Vision: {'ON' if show_vision else 'OFF'}", True, WHITE)
    screen.blit(vision_text, (10, 135))
    
    # Controls
    controls = [
        "Controls:",
        "W - Forward thrust (hold)",
        "A/D - Steer left/right",
        "V - Toggle vision display",
        "ESC - Quit"
    ]
    for i, text in enumerate(controls):
        color = GRAY if i == 0 else WHITE
        control_text = font.render(text, True, color)
        screen.blit(control_text, (WINDOW_WIDTH - 220, 10 + i * 25))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Continuous Simulation with Ray-Cast Vision")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Initialize game objects
    agent = Agent(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    food = Food()
    
    # Game state
    running = True
    food_collected = 0
    show_vision = True
    
    while running:
        dt = clock.tick(FPS) / 1000.0  # Convert to seconds
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    show_vision = not show_vision
        
        # Handle continuous input
        keys = pygame.key.get_pressed()
        
        # Throttle input - direct control (0 or 1)
        throttle_input = 0
        if keys[pygame.K_w]:
            throttle_input = 1
        
        # Steering input
        steer_input = 0
        if keys[pygame.K_a]:
            steer_input = -1
        elif keys[pygame.K_d]:
            steer_input = 1
        
        # Update agent
        agent.update(dt, steer_input, throttle_input)
        agent.handle_wall_collisions()
        
        # Check food collision
        if check_food_collision(agent, food):
            food_collected += 1
            print(f"FOOD REACHED! Total collected: {food_collected}")
            food.respawn()
        
        # Compute vision
        distances, hit_points, colors = compute_vision(agent, food)
        depth_strip = build_depth_strip(distances, colors)
        
        # Render
        screen.fill(BLACK)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 2)
        
        # Draw game objects
        draw_food(screen, food)
        draw_agent(screen, agent)
        
        # Draw vision system
        draw_vision(screen, agent, distances, hit_points, show_vision)
        
        # Draw depth strip (LiDAR visualization)
        strip_x = 10
        strip_y = WINDOW_HEIGHT - STRIP_HEIGHT - 10
        # Scale up the strip for better visibility
        scaled_strip = pygame.transform.scale(depth_strip, (NUM_RAYS * 3, STRIP_HEIGHT))
        screen.blit(scaled_strip, (strip_x, strip_y))
        
        # Draw strip border
        pygame.draw.rect(screen, WHITE, (strip_x - 1, strip_y - 1, NUM_RAYS * 3 + 2, STRIP_HEIGHT + 2), 1)
        
        # Strip label
        strip_label = font.render("LiDAR Depth Strip", True, WHITE)
        screen.blit(strip_label, (strip_x, strip_y - 25))
        
        # Draw UI
        draw_ui(screen, font, agent, clock.get_fps(), show_vision)
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()