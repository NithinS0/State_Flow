import pygame
import math
import random

class Enemy:
    def __init__(self, x, y, speed, hp):
        self.x = float(x)
        self.y = float(y)
        self.speed = speed
        self.hp = hp
        self.alive = True
        self.size = 20
        self.rect = pygame.Rect(x - self.size//2, y - self.size//2, self.size, self.size)

    def update(self, dt):
        """Move straight down."""
        self.y += self.speed * dt
        self.rect.center = (int(self.x), int(self.y))

    def draw(self, screen):
        """Draw a red diamond shape."""
        pts = [
            (self.x, self.y - 12),
            (self.x + 12, self.y),
            (self.x, self.y + 12),
            (self.x - 12, self.y)
        ]
        pygame.draw.polygon(screen, (255, 60, 60), pts)
        pygame.draw.polygon(screen, (255, 255, 255), pts, 1)

class EnemyManager:
    """Handles spawning and mass update/draw."""
    def __init__(self, sw, sh):
        self.sw = sw
        self.sh = sh
        self.enemies = []
        self.max_enemies = 30
        self.spawn_timer = 0.0

    def spawn(self, speed, hp):
        """Spawn at a random X at the top."""
        x = random.randint(20, self.sw - 20)
        y = -30
        self.enemies.append(Enemy(x, y, speed, hp))

    def update(self, dt, diff_params):
        """Spawn and move all enemies. Returns number of newly escaped enemies."""
        # Spawning logic
        self.spawn_timer += dt
        spawn_rate = diff_params.get("spawn_rate", 0.8)
        if self.spawn_timer >= (1.0 / spawn_rate):
            if len(self.enemies) < self.max_enemies:
                self.spawn(diff_params.get("enemy_speed", 100.0), diff_params.get("enemy_hp", 2))
            self.spawn_timer = 0
            
        escaped_count = 0
        # Update each
        for e in self.enemies:
            e.update(dt)
            # Check bottom boundary (passed the player line)
            if e.y > self.sh - 40:
                e.alive = False
                escaped_count += 1
                
        # Efficient list cleanup
        self.enemies = [e for e in self.enemies if e.alive]
        return escaped_count

    def draw(self, screen):
        for e in self.enemies:
            e.draw(screen)
