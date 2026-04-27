import pygame
import math

class Bullet:
    def __init__(self, x, y, tx=0, ty=0, speed=800):
        self.x = float(x)
        self.y = float(y)
        
        # Always straight UP
        self.vx = 0
        self.vy = -speed
        
        self.rect = pygame.Rect(self.x - 3, self.y - 3, 6, 6)
        self.alive = True

    def update(self, dt):
        """Update position and rect."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.rect.center = (int(self.x), int(self.y))
        
        # Kill if off-screen (with buffer)
        if not (-50 < self.x < 850 and -50 < self.y < 650):
            self.alive = False

    def draw(self, screen):
        """Draw a small circle."""
        pygame.draw.circle(screen, (230, 230, 230), self.rect.center, 3)
