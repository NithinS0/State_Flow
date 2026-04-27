import pygame
import math

class Player:
    def __init__(self, x, y, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = float(x)
        self.y = float(screen_h - 60) # Locked to bottom
        
        self.hp = 100
        self.max_hp = 100
        self.speed = 450.0 # Faster as it's only 1D
        self.size = 28
        
        self.kills = 0
        self.deaths = 0
        self.score = 0
        
        self.angle = -math.pi / 2 # Facing UP
        self.invincibility_time = 0.0
        self.rect = pygame.Rect(self.x - self.size//2, self.y - self.size//2, self.size, self.size)

    def update(self, dt, keys):
        """Handle horizontal-only movement."""
        # A/D or Left/Right
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: self.x -= self.speed * dt
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.x += self.speed * dt
        
        # Clamp to screen
        self.x = max(20, min(self.screen_w - 20, self.x))
        
        self.rect.center = (int(self.x), int(self.y))
        
        # Angle remains UP
        self.angle = -math.pi / 2
        
        # Cooldown
        if self.invincibility_time > 0:
            self.invincibility_time -= dt

    def take_damage(self, amount):
        """Apply damage with cooldown."""
        if self.invincibility_time <= 0:
            self.hp -= amount
            self.invincibility_time = 0.5
            return True
        return False

    def draw(self, screen):
        """Draw a triangle ship pointing toward the angle."""
        # Triangle points relative to center
        # Tip (forward), Left back, Right back
        points = [
            (math.cos(self.angle) * 15, math.sin(self.angle) * 15),
            (math.cos(self.angle + 2.5) * 12, math.sin(self.angle + 2.5) * 12),
            (math.cos(self.angle - 2.5) * 12, math.sin(self.angle - 2.5) * 12),
        ]
        # Offset to current position
        points = [(self.x + p[0], self.y + p[1]) for p in points]
        
        # Flicker if invincible
        if self.invincibility_time > 0 and int(pygame.time.get_ticks() / 100) % 2 == 0:
            return
            
        pygame.draw.polygon(screen, (0, 255, 255), points)
        # Outline
        pygame.draw.polygon(screen, (255, 255, 255), points, 1)
