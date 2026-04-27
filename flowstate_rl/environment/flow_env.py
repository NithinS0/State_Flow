"""
flow_env.py
-----------
FlowEnv — a Gymnasium environment for adaptive difficulty control in FlowState-RL.

The agent controls three game difficulty parameters (enemy_speed, spawn_rate,
enemy_hp) by choosing from 9 discrete actions.  At each step a simulated player
persona generates a 6-dim state vector:

    [kill_rate, death_rate, avg_health, dodge_success, score_velocity, danger_time]

The reward is shaped to keep the player inside the "Flow zone" — challenged
but not overwhelmed.

Reward structure
----------------
  +2.0   when 0.5 ≤ performance ≤ 0.7             (in Flow zone)
  +0.5   bonus when Flow has been maintained ≥ 30 consecutive steps
  -1.0   when death_rate > 0.7                     (overwhelmed)
  -0.5   when kill_rate > 0.8 and death_rate < 0.1 (bored / too easy)
  -0.1   otherwise                                  (neutral / off-target)

Action space  : Discrete(9)  — 3 actions × 3 parameters
Observation   : Box(6,) float32
Episode length: 200 steps
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from flowstate_rl.environment.personas import PersonaFactory, BasePlayer

# ---------------------------------------------------------------------------
# Difficulty parameter bounds
# ---------------------------------------------------------------------------

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "enemy_speed": (1.0, 5.0),
    "spawn_rate":  (0.5, 3.0),
    "enemy_hp":    (1.0, 5.0),
}

# Step sizes for each parameter (one unit = this fraction of the full range)
PARAM_STEP: Dict[str, float] = {
    "enemy_speed": 0.20,   # 4.0 range / 20 steps
    "spawn_rate":  0.125,  # 2.5 range / 20 steps
    "enemy_hp":    0.20,   # 4.0 range / 20 steps
}

# Flow zone boundaries
FLOW_LOW  = 0.50
FLOW_HIGH = 0.70

# Sustained-flow bonus threshold (consecutive steps)
FLOW_STREAK_BONUS_AT = 30

# Episode length
MAX_STEPS = 200

# Performance is derived as a weighted combination of state components
# weights: [kill_rate, -death_rate, avg_health, dodge_success, score_velocity, -danger_time]
PERF_WEIGHTS = np.array([0.25, -0.25, 0.15, 0.20, 0.10, -0.15], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action → (parameter, direction) lookup
# ---------------------------------------------------------------------------

# action_id -> (param_name, delta_sign)   delta_sign: -1=decrease, 0=keep, +1=increase
ACTION_MAP: Dict[int, Tuple[str, int]] = {
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


# ---------------------------------------------------------------------------
# FlowEnv
# ---------------------------------------------------------------------------


class FlowEnv(gym.Env):
    """
    Adaptive difficulty Gymnasium environment.

    Parameters
    ----------
    persona_name : str
        Player persona to use: "beginner" | "average" | "expert" | "erratic".
    rng_seed : int or None
        Seed passed to the persona generator and Gymnasium RNG.
    render_mode : str or None
        Supported: "human" (prints to stdout), None.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        persona_name: str = "average",
        rng_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.persona_name = persona_name
        self.rng_seed = rng_seed
        self.render_mode = render_mode

        # ── Spaces ──────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        # ── Internal state (populated by reset) ─────────────────────────────
        self._player: BasePlayer = PersonaFactory.get_persona(persona_name, rng_seed)
        self._difficulty: Dict[str, float] = {}
        self._obs: np.ndarray = np.zeros(6, dtype=np.float32)
        self._step_count: int = 0
        self._flow_streak: int = 0          # consecutive steps in Flow zone

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns
        -------
        obs : np.ndarray, shape (6,)
        info : dict
        """
        super().reset(seed=seed)

        # Re-create persona so its internal step counter resets cleanly
        effective_seed = seed if seed is not None else self.rng_seed
        self._player = PersonaFactory.get_persona(self.persona_name, effective_seed)

        # Initialise difficulty parameters to midpoints of their ranges
        self._difficulty = {
            "enemy_speed": 3.0,   # mid of [1.0, 5.0]
            "spawn_rate":  1.75,  # mid of [0.5, 3.0]
            "enemy_hp":    3.0,   # mid of [1.0, 5.0]
        }

        self._step_count = 0
        self._flow_streak = 0
        self._obs = self._player.step()

        return self._obs.copy(), self._build_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply an action, advance the player simulation by one step.

        Parameters
        ----------
        action : int
            Must be in [0, 8].

        Returns
        -------
        obs         : np.ndarray, shape (6,)
        reward      : float
        terminated  : bool
        truncated   : bool
        info        : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Apply difficulty action ─────────────────────────────────────
        self._apply_action(action)

        # 2. Advance player simulation ───────────────────────────────────
        self._obs = self._player.step()

        # 3. Compute performance scalar ──────────────────────────────────
        performance = float(np.dot(PERF_WEIGHTS, self._obs) + 0.5)
        performance = float(np.clip(performance, 0.0, 1.0))

        # 4. Compute reward ──────────────────────────────────────────────
        reward = self._compute_reward(self._obs, performance)

        # 5. Bookkeeping ─────────────────────────────────────────────────
        self._step_count += 1
        truncated  = self._step_count >= MAX_STEPS
        terminated = False          # no terminal condition beyond max steps

        if self.render_mode == "human":
            self.render()

        return self._obs.copy(), reward, terminated, truncated, self._build_info(reward, performance)

    def render(self) -> None:
        """Print a compact one-line summary of current environment state."""
        d = self._difficulty
        obs = self._obs
        print(
            f"Step {self._step_count:>3} | "
            f"spd={d['enemy_speed']:.2f}  "
            f"spwn={d['spawn_rate']:.2f}  "
            f"hp={d['enemy_hp']:.2f} | "
            f"kill={obs[0]:.2f}  death={obs[1]:.2f}  "
            f"hlth={obs[2]:.2f}  dodge={obs[3]:.2f}  "
            f"scvel={obs[4]:.2f}  dngr={obs[5]:.2f} | "
            f"streak={self._flow_streak}"
        )

    def close(self) -> None:
        pass

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """
        Update a difficulty parameter based on the chosen action.

        Actions 0-2 control enemy_speed, 3-5 control spawn_rate,
        6-8 control enemy_hp.  The parameter is clipped to its bounds
        after the update.
        """
        param, sign = ACTION_MAP[action]
        lo, hi = PARAM_BOUNDS[param]
        step = PARAM_STEP[param]

        self._difficulty[param] = float(
            np.clip(self._difficulty[param] + sign * step, lo, hi)
        )

    def _compute_reward(self, obs: np.ndarray, performance: float) -> float:
        """
        Map the current observation and performance to a scalar reward.

        Parameters
        ----------
        obs : np.ndarray
            Raw 6-dim player state.
        performance : float
            Scalar in [0, 1] derived from obs.

        Returns
        -------
        float
        """
        kill_rate  = float(obs[IDX_KILL_RATE])
        death_rate = float(obs[IDX_DEATH_RATE])

        in_flow = FLOW_LOW <= performance <= FLOW_HIGH

        # ── Primary reward signal ────────────────────────────────────────
        if in_flow:
            reward = 2.0
            self._flow_streak += 1
        elif death_rate > 0.7:
            reward = -1.0           # overwhelmed
            self._flow_streak = 0
        elif kill_rate > 0.8 and death_rate < 0.1:
            reward = -0.5           # bored / too easy
            self._flow_streak = 0
        else:
            reward = -0.1           # transitioning / off-target
            self._flow_streak = 0

        # ── Sustained-flow bonus ─────────────────────────────────────────
        if self._flow_streak >= FLOW_STREAK_BONUS_AT and in_flow:
            reward += 0.5

        return float(reward)

    def _build_info(
        self,
        reward: Optional[float] = None,
        performance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build an info dict for the current state."""
        return {
            "step":         self._step_count,
            "flow_streak":  self._flow_streak,
            "difficulty":   dict(self._difficulty),
            "reward":       reward,
            "performance":  performance,
            "persona":      self.persona_name,
        }


# ---------------------------------------------------------------------------
# Index aliases (mirrors personas.py for convenience)
# ---------------------------------------------------------------------------

IDX_KILL_RATE      = 0
IDX_DEATH_RATE     = 1
IDX_AVG_HEALTH     = 2
IDX_DODGE_SUCCESS  = 3
IDX_SCORE_VELOCITY = 4
IDX_DANGER_TIME    = 5


# ---------------------------------------------------------------------------
# __main__ — run one random episode and print summary
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import random

    PERSONA = "erratic"          # change to test other personas
    SEED    = 0
    VERBOSE = True               # set False to suppress per-step render

    print("=" * 70)
    print(f"  FlowEnv — random episode  |  persona={PERSONA}  seed={SEED}")
    print("=" * 70)

    env = FlowEnv(
        persona_name=PERSONA,
        rng_seed=SEED,
        render_mode="human" if VERBOSE else None,
    )

    obs, info = env.reset(seed=SEED)
    print(f"\nInitial obs: {obs}\n")

    total_reward = 0.0
    flow_steps   = 0
    done         = False
    step         = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["flow_streak"] > 0:
            flow_steps += 1
        done = terminated or truncated
        step += 1

    print("\n" + "=" * 70)
    print(f"  Episode complete after {step} steps")
    print(f"  Total reward   : {total_reward:.3f}")
    print(f"  Steps in flow  : {flow_steps} / {step}  "
          f"({100 * flow_steps / step:.1f}%)")
    print(f"  Max flow streak: {info['flow_streak']}")
    print(f"  Final difficulty: {info['difficulty']}")
    print("=" * 70)

    env.close()
