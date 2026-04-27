import pygame
import sys
import os
from pathlib import Path

# Local imports
from flowstate_rl.game.player import Player
from flowstate_rl.game.enemy import EnemyManager
from flowstate_rl.game.bullet import Bullet
from flowstate_rl.game.metrics import MetricsCollector
from flowstate_rl.game.difficulty_controller import DifficultyController

# --- Config ---
WIDTH, HEIGHT = 800, 600
FPS = 60
TITLE = "FlowState RL Adaptive Difficulty"

# Colours
COL_WHITE = (230, 230, 230)
COL_CYAN  = (0, 255, 255)
COL_RED   = (255, 60, 60)
COL_YELLOW = (255, 200, 50)
COL_GRAY  = (40, 40, 50)
COL_BG    = (10, 10, 20)

class GameEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        
        self.running = True
        self.init_session()

    def init_session(self):
        """Reset everything for a new session."""
        self.player = Player(WIDTH//2, HEIGHT//2, WIDTH, HEIGHT)
        self.enemies = EnemyManager(WIDTH, HEIGHT)
        self.bullets = []
        self.metrics = MetricsCollector()
        self.ctrl = DifficultyController()
        self.escaped_enemies = 0
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Shoot toward mouse
                if len(self.bullets) < 50:
                    mx, my = pygame.mouse.get_pos()
                    self.bullets.append(Bullet(self.player.x, self.player.y, mx, my))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: self.init_session()
                if event.key == pygame.K_ESCAPE: self.running = False

    def update(self, dt):
        # 1. Player
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys)
        
        # 2. Bullets
        for b in self.bullets:
            b.update(dt)
        self.bullets = [b for b in self.bullets if b.alive]

        # 3. Enemies (Internal spawns and movement)
        escaped = self.enemies.update(dt, self.ctrl.params)
        self.escaped_enemies += escaped
        
        # 4. Collisions & Scoring
        self.handle_collisions()
        
        # 5. Clean up dead entities immediately
        self.enemies.enemies = [e for e in self.enemies.enemies if e.alive]
        self.bullets = [b for b in self.bullets if b.alive]
        
        # 6. Metrics & RL
        self.metrics.update(dt, self.player, self.enemies.enemies, self.ctrl.params, self.ctrl.last_action, self.escaped_enemies)
        self.ctrl.update(dt, self.metrics.current_metrics)

        # 7. Auto-Reset if too many escaped
        if self.escaped_enemies >= 20:
            self.init_session()

    def handle_collisions(self):
        # Bullet vs Enemy
        for b in self.bullets:
            if not b.alive: continue
            for e in self.enemies.enemies:
                if not e.alive: continue
                
                # Check collision using rects
                if b.rect.colliderect(e.rect):
                    e.hp -= 1
                    b.alive = False
                    if e.hp <= 0:
                        e.alive = False
                        self.player.kills += 1
                        self.player.score += 10
                    break # Bullet is consumed

        # Enemy vs Player (Damage)
        for e in self.enemies.enemies:
            if not e.alive: continue
            if e.rect.colliderect(self.player.rect):
                if self.player.take_damage(20):
                    # Damage visual feedback could go here
                    pass

    def draw(self):
        self.screen.fill(COL_BG)
        
        # Grid for visuals
        for x in range(0, WIDTH, 50): pygame.draw.line(self.screen, (20, 20, 35), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 50): pygame.draw.line(self.screen, (20, 20, 35), (0, y), (WIDTH, y))
        
        # Entities
        for b in self.bullets: b.draw(self.screen)
        self.enemies.draw(self.screen)
        self.player.draw(self.screen)
        
        # HUD
        self.draw_hud()
        
        pygame.display.flip()

    def draw_hud(self):
        # HP Bar
        pygame.draw.rect(self.screen, (60, 60, 70), (20, 20, 200, 20))
        hp_ratio = self.player.hp / self.player.max_hp
        hp_col = COL_CYAN if hp_ratio > 0.3 else COL_RED
        pygame.draw.rect(self.screen, hp_col, (20, 20, int(200 * hp_ratio), 20))
        
        # Stats
        items = [
            f"Score: {self.player.score}",
            f"Kills: {self.player.kills}",
            f"Escaped: {self.escaped_enemies} / 20",
            f"Enemies: {len(self.enemies.enemies)}",
            f"Speed: {self.ctrl.params['enemy_speed']:.0f}",
            f"Spawn: {self.ctrl.params['spawn_rate']:.1f}",
            f"E-HP: {self.ctrl.params['enemy_hp']}"
        ]
        for i, text in enumerate(items):
            surf = self.font.render(text, True, COL_WHITE)
            self.screen.blit(surf, (20, 50 + i * 25))
            
        # Flow State Badge
        state = self.metrics.current_metrics.get("current_state", "FLOW")
        state_col = COL_CYAN if state == "FLOW" else (COL_YELLOW if state == "BORED" else COL_RED)
        
        badge_txt = self.font.render(state, True, state_col)
        pygame.draw.rect(self.screen, state_col, (WIDTH - 150, 20, 130, 40), 3, border_radius=6)
        self.screen.blit(badge_txt, (WIDTH - 142, 28))

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameEngine()
    game.run()