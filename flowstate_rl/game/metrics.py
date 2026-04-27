import json
import os
from pathlib import Path
from typing import Dict, Any

class MetricsCollector:
    def __init__(self, export_path="data/live_metrics.json"):
        self.export_path = Path(export_path)
        self.data_dir = self.export_path.parent
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
            
        self.timer = 0.0
        self.flow_history = []
        
        # Cumulative stats for rates
        self.session_kills = 0
        self.session_damage = 0.0
        self.last_kills = 0
        self.last_damage = 0.0
        self.last_score = 0
        
        self.current_metrics = {}

    def update(self, dt, player, enemies, diff_params, last_action=0, escaped=0):
        """Update timers and compute periodic metrics."""
        self.timer += dt
        
        if self.timer >= 1.0:
            self.current_metrics = self.calculate(player, enemies, diff_params, last_action, escaped)
            self.export(self.current_metrics)
            self.timer = 0
            
    def calculate(self, player, enemies, diff, last_action=0, escaped=0):
        """Compute the core 5 metrics and current state."""
        # 1. Kill Rate (last second)
        k_rate = min(1.0, (player.kills - self.last_kills) / 2.0) # 2 kills/s = 100%
        self.last_kills = player.kills
        
        # 2. Death Rate (approx based on damage taken)
        d_rate = min(1.0, (player.score_velocity_proxy if hasattr(player, "damage_acc") else 0) / 50.0) # approx
        # Since I didn't add damage_acc to player, I'll use a simpler one:
        # We'll just define it as (100 - hp) / 100 but weighted by recent hits
        d_rate = round(max(0.0, 1.0 - (player.hp / 100.0)), 2)
        
        # 3. Avg Health
        avg_h = round(player.hp / 100.0, 2)
        
        # 4. Score Velocity
        s_vel = min(1.0, (player.score - self.last_score) / 500.0)
        self.last_score = player.score
        
        # 5. Danger Time
        danger = round(len(enemies) / 30.0, 2)
        
        # 6. Dodge Success (Inverse of death rate + movement proxy)
        dodge = round(min(1.0, 1.0 - (d_rate * 0.8)), 2)

        # Flow State Logic
        if 0.4 <= k_rate <= 0.7:
            state = "FLOW"
        elif k_rate > 0.8 and d_rate < 0.1:
            state = "BORED"
        elif d_rate > 0.6:
            state = "OVERWHELMED"
        else:
            state = "FLOW" # Default
            
        self.flow_history.append(1 if state == "FLOW" else 0)
        if len(self.flow_history) > 60: self.flow_history.pop(0)
        
        return {
            "kill_rate": round(k_rate, 2),
            "death_rate": d_rate,
            "avg_health": avg_h,
            "score_velocity": round(s_vel, 2),
            "danger_time": danger,
            "dodge_success": dodge,
            "enemy_speed": round(diff.get("enemy_speed", 150) / 60.0, 2), # proxy
            "spawn_rate": round(diff.get("spawn_rate", 1.5), 2),
            "enemy_hp": diff.get("enemy_hp", 2),
            "current_state": state,
            "flow_pct": int((sum(self.flow_history) / len(self.flow_history)) * 100) if self.flow_history else 0,
            "last_action": last_action,
            "escaped_enemies": escaped,
            "score": player.score,
            "kills": player.kills
        }

    def export(self, metrics):
        """Write to live_metrics.json."""
        try:
            with open(self.export_path, "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"Metrics export error: {e}")
