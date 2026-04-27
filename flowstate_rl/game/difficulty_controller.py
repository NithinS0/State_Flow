import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

class DifficultyController:
    """
    Bridge between the RL agent and the game engine.
    Loads PPO model and provides updated difficulty parameters.
    """
    def __init__(self, model_path="agent/model.zip"):
        self.model_path = Path(model_path)
        self.model = None
        self.params = {
            "enemy_speed": 100.0,
            "spawn_rate": 0.8,
            "enemy_hp": 2
        }
        self.last_action = 0
        self.timer = 0.0
        
        # Load model if it exists
        if PPO and self.model_path.exists():
            try:
                self.model = PPO.load(str(self.model_path))
                print(f"[RL] Model loaded from {self.model_path}")
            except Exception as e:
                print(f"[RL] Error loading model: {e}")
        else:
            print("[RL] Model not found or SB3 missing. Using static difficulty.")

    def update(self, dt, metrics_dict: Dict[str, Any]):
        """Runs evaluation every 1 second."""
        self.timer += dt
        if self.timer >= 1.0:
            self.timer = 0
            if self.model:
                # Prepare state vector
                state = np.array([
                    metrics_dict.get("kill_rate", 0.0),
                    metrics_dict.get("death_rate", 0.0),
                    metrics_dict.get("avg_health", 0.0),
                    metrics_dict.get("dodge_success", 0.0),
                    metrics_dict.get("score_velocity", 0.0),
                    metrics_dict.get("danger_time", 0.0)
                ], dtype=np.float32)
                
                # Predict
                action, _ = self.model.predict(state, deterministic=True)
                self.last_action = int(action)
                self._apply_action(self.last_action)
            else:
                # Fallback / Demo logic
                pass

    def _apply_action(self, action: int):
        """
        Maps action ID to difficulty increments.
        Simplified 9-action mapping:
        0: Spd+ | 1: Spd- | 2: Spd=
        3: Spwn+ | 4: Spwn- | 5: Spwn=
        6: HP+ | 7: HP- | 8: HP=
        """
        # Clamping logic inside update
        if action == 0: self.params["enemy_speed"] += 20
        elif action == 1: self.params["enemy_speed"] -= 20
        
        elif action == 3: self.params["spawn_rate"] += 0.2
        elif action == 4: self.params["spawn_rate"] -= 0.2
        
        elif action == 6: self.params["enemy_hp"] += 1
        elif action == 7: self.params["enemy_hp"] -= 1

        # Clamping
        self.params["enemy_speed"] = max(100.0, min(500.0, self.params["enemy_speed"]))
        self.params["spawn_rate"]  = max(0.5,   min(3.0,   self.params["spawn_rate"]))
        self.params["enemy_hp"]    = max(1,     min(5,     self.params["enemy_hp"]))
