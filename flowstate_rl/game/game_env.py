"""
game_env.py
-----------
ShooterEnv — a Gymnasium environment wrapping the FlowState-RL Pygame shooter.

The environment runs the game simulation headlessly each step: it advances a
Player and EnemyManager for SIM_TICKS_PER_STEP internal ticks, collects
metrics via MetricsCollector, and computes the Flow-zone reward.

Observation space : Box(6,) float32
    [kill_rate, death_rate, avg_health, dodge_success,
     score_velocity, danger_time]

Action space      : Discrete(9)
    ┌─ enemy_speed ──┐  ┌─ spawn_rate ──┐  ┌─ enemy_hp ────┐
    0 decrease        3 decrease          6 decrease
    1 maintain        4 maintain          7 maintain
    2 increase        5 increase          8 increase

Reward
------
    +2.0   if  0.5 ≤ performance ≤ 0.7          (Flow zone)
    +0.5   bonus when flow streak ≥ 30 steps
    -1.0   if  death_rate > 0.7                  (overwhelmed)
    -0.5   if  kill_rate > 0.8 and death_rate < 0.1  (bored)
    -0.1   otherwise

Episode length : 200 agent steps

Usage
-----
    env = ShooterEnv(persona_name="erratic")
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

Run random-policy test:
    python -m flowstate_rl.game.game_env
"""

from __future__ import annotations

import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Force headless SDL before importing anything Pygame
# ---------------------------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

# ---------------------------------------------------------------------------
# FlowState-RL game modules
# ---------------------------------------------------------------------------
from flowstate_rl.game.player  import Player
from flowstate_rl.game.enemy   import EnemyManager
from flowstate_rl.game.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

SW, SH          = 800, 600          # virtual screen dimensions
MAX_STEPS       = 200               # agent steps per episode
SIM_TICKS_PER_STEP = 10            # pygame ticks simulated per agent step
SIM_DT          = 1.0 / 60.0       # fixed simulation timestep (s)

# Flow zone
FLOW_LOW  = 0.50
FLOW_HIGH = 0.70
FLOW_STREAK_BONUS_AT = 30

# Performance weights (same as flow_env.py)
PERF_WEIGHTS = np.array(
    [0.25, -0.25, 0.15, 0.20, 0.10, -0.15], dtype=np.float32
)

# Difficulty parameter bounds & step sizes
_DIFF_CFG: Dict[str, Dict] = {
    "enemy_speed": {"lo": 1.0, "hi": 5.0, "step": 0.20},
    "spawn_rate":  {"lo": 0.5, "hi": 3.0, "step": 0.125},
    "enemy_hp":    {"lo": 1.0, "hi": 5.0, "step": 0.20},
}
_DIFF_INIT = {k: (v["lo"] + v["hi"]) / 2 for k, v in _DIFF_CFG.items()}

# Action → (parameter, direction)
_ACTION_MAP: Dict[int, Tuple[str, int]] = {
    0: ("enemy_speed", -1),
    1: ("enemy_speed",  0),
    2: ("enemy_speed", +1),
    3: ("spawn_rate",  -1),
    4: ("spawn_rate",   0),
    5: ("spawn_rate",  +1),
    6: ("enemy_hp",    -1),
    7: ("enemy_hp",     0),
    8: ("enemy_hp",    +1),
}

# State vector keys (must match MetricsCollector / flow_env)
_STATE_KEYS = (
    "kill_rate", "death_rate", "avg_health",
    "dodge_success", "score_velocity", "danger_time",
)


# ===========================================================================
# Headless simulated player (auto-pilots the game for RL training)
# ===========================================================================


class _AutoPlayer:
    """
    A thin wrapper around Player that provides deterministic auto-pilot
    behaviour for headless simulation:
        • Moves toward the nearest enemy (or wanders randomly)
        • Shoots every N ticks toward the nearest enemy
        • No Pygame event handling required
    """

    SHOOT_EVERY = 8          # ticks between auto-shots

    def __init__(self, screen_w: int, screen_h: int, rng: np.random.Generator) -> None:
        self._player = Player(screen_w // 2, screen_h // 2, screen_w, screen_h)
        self._rng    = rng
        self._ticks  = 0
        self._wander_angle: float = 0.0

    # ── Delegation ─────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate attribute lookups to the underlying Player."""
        return getattr(self._player, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Own attributes are stored on self; everything else on _player
        if name.startswith("_") or name in (
            "SHOOT_EVERY",
        ):
            object.__setattr__(self, name, value)
        else:
            try:
                # If _player not yet created, store on self
                object.__getattribute__(self, "_player")
                setattr(self._player, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)

    # ── Auto-pilot ─────────────────────────────────────────────────────

    def auto_update(
        self,
        dt:      float,
        enemies: List[Any],
    ) -> None:
        """
        Compute movement & shoot direction automatically, then call
        the real player's internals.
        """
        self._ticks += 1
        p = self._player

        # Find nearest alive enemy
        alive    = [e for e in enemies if getattr(e, "alive", True)]
        nearest  = self._nearest(alive)

        # ── Aim at nearest enemy ─────────────────────────────────────
        if nearest is not None:
            dx = nearest.x - p.x
            dy = nearest.y - p.y
            p._angle = math.atan2(dy, dx)
        else:
            # Wander aim
            self._wander_angle += self._rng.uniform(-0.15, 0.15)
            p._angle = self._wander_angle

        # ── Move toward nearest or wander ────────────────────────────
        if nearest is not None:
            raw_dx = nearest.x - p.x
            raw_dy = nearest.y - p.y
            dist   = math.hypot(raw_dx, raw_dy)
            if dist > 80:                        # only chase if not too close
                step = p.speed * dt
                p.x += (raw_dx / dist) * step
                p.y += (raw_dy / dist) * step
            # else hold position
        else:
            wangle = self._wander_angle
            p.x += math.cos(wangle) * p.speed * 0.3 * dt
            p.y += math.sin(wangle) * p.speed * 0.3 * dt

        # Boundary clamp
        hw, hh  = 14, 14
        p.x = float(np.clip(p.x, hw, SW - hw))
        p.y = float(np.clip(p.y, hh, SH - hh))
        p._sync_rect()

        # ── Shoot ────────────────────────────────────────────────────
        if nearest is not None and self._ticks % self.SHOOT_EVERY == 0:
            p._can_shoot = True          # bypass cooldown for auto
            p.shoot()

        # ── Tick player internals ────────────────────────────────────
        p._update_shoot_timer(dt)
        p._update_invincibility()
        p._update_bullets(dt)

    def _nearest(self, enemies: List[Any]) -> Optional[Any]:
        if not enemies:
            return None
        p = self._player
        return min(enemies, key=lambda e: (e.x - p.x) ** 2 + (e.y - p.y) ** 2)


# ===========================================================================
# ShooterEnv
# ===========================================================================


class ShooterEnv(gym.Env):
    """
    Gymnasium environment wrapping the FlowState-RL Pygame shooter.

    Parameters
    ----------
    persona_name : str
        Player persona simulation type (beginner | average | expert | erratic).
    sim_ticks    : int
        Number of internal 60-Hz game ticks simulated per agent step.
    render_mode  : str or None
        Currently only None is supported (headless).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        persona_name: str           = "average",
        sim_ticks:    int           = SIM_TICKS_PER_STEP,
        render_mode:  Optional[str] = None,
    ) -> None:
        super().__init__()

        self.persona_name = persona_name
        self.sim_ticks    = sim_ticks
        self.render_mode  = render_mode

        # ── Spaces ──────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        # ── Pygame init (dummy display) ──────────────────────────────────
        if not pygame.get_init():
            pygame.init()
        # Create a minimal dummy surface (required by Player.draw internals)
        self._surface = pygame.Surface((SW, SH))

        # ── Internal state (populated in reset) ─────────────────────────
        self._player:  Optional[_AutoPlayer]       = None
        self._enemies: Optional[EnemyManager]      = None
        self._metrics: Optional[MetricsCollector]  = None

        self._difficulty: Dict[str, float] = dict(_DIFF_INIT)
        self._step_count: int  = 0
        self._flow_streak: int = 0
        self._score:  float   = 0.0
        self._obs:    np.ndarray = np.zeros(6, dtype=np.float32)

        # Shared RNG (seeded in reset)
        self._rng = np.random.default_rng()

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed:    Optional[int]           = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns
        -------
        obs  : np.ndarray  shape (6,)
        info : dict
        """
        super().reset(seed=seed)

        # Seed numpy RNG for reproducibility
        self._rng = np.random.default_rng(seed)

        # Reset difficulty to midpoints
        self._difficulty = dict(_DIFF_INIT)
        self._step_count  = 0
        self._flow_streak = 0
        self._score       = 0.0

        # Recreate subsystems
        pyg_rng_seed = int(self._rng.integers(0, 2**31))
        self._player  = _AutoPlayer(SW, SH, rng=self._rng)
        self._enemies = EnemyManager(SW, SH, max_enemies=18)
        self._enemies.set_difficulty(self._difficulty)

        self._metrics = MetricsCollector(
            window_secs=10.0, debug=False, screen_w=SW, screen_h=SH
        )

        # Warm up simulation for a few ticks to get initial metrics
        for _ in range(self.sim_ticks):
            self._sim_tick(SIM_DT)

        self._metrics.update(self._player, self._enemies.alive_enemies)
        self._obs = self._metrics.get_state()

        return self._obs.copy(), self._build_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply a difficulty action, simulate the game, collect metrics,
        and compute the reward.

        Parameters
        ----------
        action : int  (0–8)

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Apply difficulty action ─────────────────────────────────────
        self._apply_action(action)
        self._enemies.set_difficulty(self._difficulty)

        # 2. Run game simulation ─────────────────────────────────────────
        kills_this_step = 0
        for _ in range(self.sim_ticks):
            kills_this_step += self._sim_tick(SIM_DT)

        # Update score & player kill counter
        self._player.kills += kills_this_step          # type: ignore[union-attr]
        self._score        += kills_this_step * 100

        # 3. Collect metrics ──────────────────────────────────────────────
        self._metrics.update(             # type: ignore[union-attr]
            self._player,
            self._enemies.alive_enemies,  # type: ignore[union-attr]
        )
        self._obs = self._metrics.get_state()

        # 4. Compute reward ───────────────────────────────────────────────
        reward = self._compute_reward(self._obs)

        # 5. Bookkeeping ──────────────────────────────────────────────────
        self._step_count += 1
        truncated  = self._step_count >= MAX_STEPS
        terminated = False   # no hard terminal condition (player auto-respawns)

        return self._obs.copy(), reward, terminated, truncated, self._build_info(reward)

    def render(self) -> None:
        """Rendering is headless-only; this is a no-op."""
        pass

    def close(self) -> None:
        """Release Pygame resources."""
        pass   # pygame.quit() left to caller to avoid breaking other envs

    # ──────────────────────────────────────────────────────────────────────
    # Private: game simulation
    # ──────────────────────────────────────────────────────────────────────

    def _sim_tick(self, dt: float) -> int:
        """
        Advance the game by one 60-Hz tick.

        Returns
        -------
        int  Number of enemy kills this tick.
        """
        # Auto-pilot player movement & shooting
        self._player.auto_update(dt, self._enemies.alive_enemies)  # type: ignore

        # Enemies chase and deal damage
        kills = self._enemies.update(dt, self._player)             # type: ignore

        # Auto-respawn if dead (simplify training — deaths tracked in metrics)
        if not self._player.is_alive:                              # type: ignore
            self._player.respawn(SW // 2, SH // 2)                # type: ignore

        return kills

    # ──────────────────────────────────────────────────────────────────────
    # Private: difficulty
    # ──────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """Clip-and-update one difficulty parameter based on the action."""
        param, sign = _ACTION_MAP[action]
        cfg  = _DIFF_CFG[param]
        curr = self._difficulty[param]
        self._difficulty[param] = float(
            np.clip(curr + sign * cfg["step"], cfg["lo"], cfg["hi"])
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private: reward
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Map the current 6-dim observation to a scalar reward.

        Parameters
        ----------
        obs : np.ndarray, shape (6,)

        Returns
        -------
        float
        """
        performance = float(np.clip(np.dot(PERF_WEIGHTS, obs) + 0.5, 0.0, 1.0))
        kill_rate   = float(obs[0])
        death_rate  = float(obs[1])

        in_flow = FLOW_LOW <= performance <= FLOW_HIGH

        if in_flow:
            reward = 2.0
            self._flow_streak += 1
        elif death_rate > 0.7:
            reward = -1.0          # overwhelmed
            self._flow_streak = 0
        elif kill_rate > 0.8 and death_rate < 0.1:
            reward = -0.5          # bored / too easy
            self._flow_streak = 0
        else:
            reward = -0.1          # transitioning / off-target
            self._flow_streak = 0

        # Sustained flow bonus
        if self._flow_streak >= FLOW_STREAK_BONUS_AT and in_flow:
            reward += 0.5

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Private: info dict
    # ──────────────────────────────────────────────────────────────────────

    def _build_info(self, reward: Optional[float] = None) -> Dict[str, Any]:
        return {
            "step":         self._step_count,
            "flow_streak":  self._flow_streak,
            "score":        self._score,
            "difficulty":   dict(self._difficulty),
            "reward":       reward,
            "kills":        getattr(self._player, "kills", 0),
            "deaths":       getattr(self._player, "deaths", 0),
            "persona":      self.persona_name,
        }


# ===========================================================================
# __main__ — random policy episode test
# ===========================================================================


if __name__ == "__main__":
    import time

    PERSONAS = ["beginner", "average", "expert", "erratic"]
    N_EPISODES = 2

    print("=" * 65)
    print("  ShooterEnv — Random Policy Episode Test")
    print(f"  {N_EPISODES} episodes × {MAX_STEPS} steps × {SIM_TICKS_PER_STEP} sim-ticks")
    print("=" * 65)

    for persona in PERSONAS:
        env = ShooterEnv(persona_name=persona, sim_ticks=SIM_TICKS_PER_STEP)
        ep_rewards = []

        for ep in range(N_EPISODES):
            obs, info = env.reset(seed=ep * 17)
            total_reward = 0.0
            flow_steps   = 0
            done         = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if info["flow_streak"] > 0:
                    flow_steps += 1
                done = terminated or truncated

            ep_rewards.append(total_reward)

            print(
                f"  [{persona.upper():<10}] ep={ep+1}  "
                f"reward={total_reward:+7.2f}  "
                f"flow_steps={flow_steps:>3}/{MAX_STEPS}  "
                f"kills={info['kills']:>3}  "
                f"deaths={info['deaths']:>2}  "
                f"streak={info['flow_streak']:>3}"
            )

        env.close()
        print(
            f"         ↳ mean reward = {float(np.mean(ep_rewards)):+.3f}\n"
        )

    # ── Gym API validation ────────────────────────────────────────────────
    print("Running gymnasium env_checker …")
    try:
        from gymnasium.utils.env_checker import check_env
        test_env = ShooterEnv(persona_name="average", sim_ticks=3)
        check_env(test_env, warn=True)
        print("[PASS] check_env passed\n")
        test_env.close()
    except Exception as exc:
        print(f"[WARN] check_env raised: {exc}\n")

    print("Done.")
