#!/usr/bin/env python3
"""
Minimal 2D Continuous Simulation
- Agent with physics-based movement (steer + throttle)
- Food collection
- Wall collisions
- Manual keyboard controls
"""

import pygame
import math
import random
import sys

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60

# Agent constants
AGENT_RADIUS = 12
AGENT_MAX_THRUST = 300.0  # pixels/s²
AGENT_MAX_TURN_ACCEL = 8.0  # rad/s²
AGENT_LINEAR_DRAG = 2.0
AGENT_ANGULAR_DRAG = 5.0
WALL_RESTITUTION = 0.6

# Food constants
FOOD_RADIUS = 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
GRAY = (128, 128, 128)


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
        # Update throttle based on input
        if throttle_input > 0:
            self.throttle = min(1.0, self.throttle + 3.0 * dt)
        elif throttle_input < 0:
            self.throttle = max(0.0, self.throttle - 3.0 * dt)
        
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


def draw_ui(screen, font, agent, fps):
    # Throttle display
    throttle_text = font.render(f"Throttle: {agent.throttle:.2f}", True, WHITE)
    screen.blit(throttle_text, (10, 10))
    
    # Speed display
    speed_text = font.render(f"Speed: {agent.get_speed():.1f}", True, WHITE)
    screen.blit(speed_text, (10, 35))
    
    # FPS display
    fps_text = font.render(f"FPS: {fps:.1f}", True, WHITE)
    screen.blit(fps_text, (10, 60))
    
    # Controls
    controls = [
        "Controls:",
        "W/S - Throttle up/down",
        "A/D - Steer left/right",
        "ESC - Quit"
    ]
    for i, text in enumerate(controls):
        color = GRAY if i == 0 else WHITE
        control_text = font.render(text, True, color)
        screen.blit(control_text, (WINDOW_WIDTH - 200, 10 + i * 25))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Continuous Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Initialize game objects
    agent = Agent(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    food = Food()
    
    # Game state
    running = True
    food_collected = 0
    
    while running:
        dt = clock.tick(FPS) / 1000.0  # Convert to seconds
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Handle continuous input
        keys = pygame.key.get_pressed()
        
        # Throttle input
        throttle_input = 0
        if keys[pygame.K_w]:
            throttle_input = 1
        elif keys[pygame.K_s]:
            throttle_input = -1
        
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
        
        # Render
        screen.fill(BLACK)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 2)
        
        # Draw game objects
        draw_food(screen, food)
        draw_agent(screen, agent)
        
        # Draw UI
        draw_ui(screen, font, agent, clock.get_fps())
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()