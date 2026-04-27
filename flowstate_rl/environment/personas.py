"""
personas.py
-----------
Simulated player personas for FlowState-RL.

Each persona generates a 6-dimensional normalised state vector per step:

    [kill_rate, death_rate, avg_health, dodge_success, score_velocity, danger_time]

All values are in [0, 1].

Personas
--------
- BeginnerPlayer  : low skill, struggles to survive
- AveragePlayer   : moderate, balanced behaviour
- ExpertPlayer    : high skill, dominates encounters
- ErraticPlayer   : switches between Expert and Beginner every 20 steps

Usage
-----
    from flowstate_rl.environment.personas import PersonaFactory

    player = PersonaFactory.get_persona("erratic")
    state  = player.step()          # np.ndarray shape (6,)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type


# ---------------------------------------------------------------------------
# State vector index constants
# ---------------------------------------------------------------------------

IDX_KILL_RATE      = 0
IDX_DEATH_RATE     = 1
IDX_AVG_HEALTH     = 2
IDX_DODGE_SUCCESS  = 3
IDX_SCORE_VELOCITY = 4
IDX_DANGER_TIME    = 5

STATE_DIM = 6


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BasePlayer(ABC):
    """
    Abstract base for all player personas.

    Subclasses implement `_base_state()` which returns the *mean* values
    for each component of the state vector.  `step()` then adds small
    Gaussian noise, clips to [0, 1], and returns the result.
    """

    # Noise standard deviation applied to every component each step
    NOISE_STD: float = 0.03

    def __init__(self, rng_seed: int | None = None) -> None:
        self.rng = np.random.default_rng(rng_seed)
        self._step_count: int = 0

    @abstractmethod
    def _base_state(self) -> np.ndarray:
        """
        Return the mean state vector (shape (6,)) for this persona.
        Values should be in [0, 1].
        """

    def step(self) -> np.ndarray:
        """
        Produce one noisy observation of the player state.

        Returns
        -------
        np.ndarray, shape (6,), dtype float32
            [kill_rate, death_rate, avg_health,
             dodge_success, score_velocity, danger_time]
            all clipped to [0, 1].
        """
        base = self._base_state().astype(np.float32)
        noise = self.rng.normal(loc=0.0, scale=self.NOISE_STD, size=STATE_DIM).astype(np.float32)
        state = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
        self._step_count += 1
        return state

    def reset(self) -> None:
        """Reset step counter (and any internal state in subclasses)."""
        self._step_count = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steps={self._step_count})"


# ---------------------------------------------------------------------------
# Concrete personas
# ---------------------------------------------------------------------------


class BeginnerPlayer(BasePlayer):
    """
    Beginner — low kill rate, high death rate, poor dodging.

    Typical profile:
        kill_rate      ≈ 0.15   (rarely eliminates opponents)
        death_rate     ≈ 0.75   (dies often)
        avg_health     ≈ 0.25   (tends to be low HP)
        dodge_success  ≈ 0.20   (misses most dodge windows)
        score_velocity ≈ 0.15   (accumulates score slowly)
        danger_time    ≈ 0.70   (spends most time in danger)
    """

    NOISE_STD = 0.04   # beginners are noisier / less consistent

    def _base_state(self) -> np.ndarray:
        return np.array(
            [0.15, 0.75, 0.25, 0.20, 0.15, 0.70],
            dtype=np.float32,
        )


class AveragePlayer(BasePlayer):
    """
    Average — moderate, well-balanced gameplay.

    Typical profile:
        kill_rate      ≈ 0.50
        death_rate     ≈ 0.45
        avg_health     ≈ 0.55
        dodge_success  ≈ 0.50
        score_velocity ≈ 0.50
        danger_time    ≈ 0.45
    """

    NOISE_STD = 0.035

    def _base_state(self) -> np.ndarray:
        return np.array(
            [0.50, 0.45, 0.55, 0.50, 0.50, 0.45],
            dtype=np.float32,
        )


class ExpertPlayer(BasePlayer):
    """
    Expert — dominant performance with minimal deaths.

    Typical profile:
        kill_rate      ≈ 0.85   (eliminates most opponents)
        death_rate     ≈ 0.08   (almost never dies)
        avg_health     ≈ 0.88   (maintains high HP)
        dodge_success  ≈ 0.90   (evades nearly all attacks)
        score_velocity ≈ 0.88   (rapid score accumulation)
        danger_time    ≈ 0.10   (rarely in danger)
    """

    NOISE_STD = 0.02   # experts are very consistent

    def _base_state(self) -> np.ndarray:
        return np.array(
            [0.85, 0.08, 0.88, 0.90, 0.88, 0.10],
            dtype=np.float32,
        )


class ErraticPlayer(BasePlayer):
    """
    Erratic — switches between Expert-level and Beginner-level
    behaviour every 20 steps.

    This simulates a player who alternates between focus and distraction,
    making it the most challenging persona for the RL controller to track.
    """

    SWITCH_EVERY: int = 20
    NOISE_STD = 0.05   # highest noise — deliberately unpredictable

    def __init__(self, rng_seed: int | None = None) -> None:
        super().__init__(rng_seed)
        self._expert_base  = ExpertPlayer._base_state(self)   # type: ignore[arg-type]
        self._beginner_base = BeginnerPlayer._base_state(self) # type: ignore[arg-type]
        self._in_expert_phase: bool = True

    def _base_state(self) -> np.ndarray:
        """Alternate persona every SWITCH_EVERY steps."""
        phase = (self._step_count // self.SWITCH_EVERY) % 2
        self._in_expert_phase = phase == 0

        if self._in_expert_phase:
            return np.array([0.85, 0.08, 0.88, 0.90, 0.88, 0.10], dtype=np.float32)
        else:
            return np.array([0.15, 0.75, 0.25, 0.20, 0.15, 0.70], dtype=np.float32)

    @property
    def current_phase(self) -> str:
        """Return a human-readable label for the current behaviour phase."""
        return "Expert" if self._in_expert_phase else "Beginner"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class PersonaFactory:
    """
    Registry and factory for player personas.

    Supported keys (case-insensitive):
        "beginner", "average", "expert", "erratic"
    """

    _REGISTRY: Dict[str, Type[BasePlayer]] = {
        "beginner": BeginnerPlayer,
        "average":  AveragePlayer,
        "expert":   ExpertPlayer,
        "erratic":  ErraticPlayer,
    }

    @classmethod
    def get_persona(cls, name: str, rng_seed: int | None = None) -> BasePlayer:
        """
        Instantiate and return a player persona by name.

        Parameters
        ----------
        name : str
            One of "beginner", "average", "expert", "erratic".
        rng_seed : int or None
            Optional seed for reproducible noise.

        Returns
        -------
        BasePlayer subclass instance.

        Raises
        ------
        ValueError
            If `name` is not a registered persona key.
        """
        key = name.strip().lower()
        if key not in cls._REGISTRY:
            available = ", ".join(f'"{k}"' for k in cls._REGISTRY)
            raise ValueError(
                f"Unknown persona '{name}'. Available: {available}"
            )
        return cls._REGISTRY[key](rng_seed=rng_seed)

    @classmethod
    def list_personas(cls) -> list[str]:
        """Return all registered persona names."""
        return list(cls._REGISTRY.keys())


# ---------------------------------------------------------------------------
# __main__ — print 10 sample states from each persona
# ---------------------------------------------------------------------------


STATE_LABELS = [
    "kill_rate    ",
    "death_rate   ",
    "avg_health   ",
    "dodge_success",
    "score_vel    ",
    "danger_time  ",
]


def _print_persona_samples(name: str, n: int = 10) -> None:
    """Print `n` step outputs from a named persona in a formatted table."""
    player = PersonaFactory.get_persona(name, rng_seed=42)
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  Persona: {name.upper():<12}  {player.__class__.__name__}")
    print(f"{'═' * 72}")
    header = f"  {'Step':>4}  " + "  ".join(f"{lbl}" for lbl in STATE_LABELS)
    print(header)
    print(sep)

    for i in range(n):
        state = player.step()

        # For ErraticPlayer, tag which phase we're in
        phase_tag = ""
        if isinstance(player, ErraticPlayer):
            phase_tag = f"  [{player.current_phase}]"

        row = f"  {i + 1:>4}  " + "  ".join(f"{v:>13.4f}" for v in state)
        print(row + phase_tag)

    print(sep)


if __name__ == "__main__":
    print("\nFlowState-RL — Persona State Samples")
    print(f"State vector: {STATE_LABELS}\n")

    for persona_name in PersonaFactory.list_personas():
        _print_persona_samples(persona_name, n=10)

    print("\nDone.")
