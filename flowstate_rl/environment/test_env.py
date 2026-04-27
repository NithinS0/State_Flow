"""
test_env.py
-----------
Validation and statistical test suite for FlowEnv.

Tests every registered persona ("beginner", "average", "expert", "erratic")
using a random policy over 5 episodes each.

Per-persona metrics reported
-----------------------------
  mean_reward           : average total episode reward
  flow_percentage       : % of steps inside the Flow zone (reward == +2.0)
  overwhelmed_percentage: % of steps where death_rate > 0.7
  bored_percentage      : % of steps where kill_rate > 0.8 and death_rate < 0.1

Validations (assert-based, fail-fast)
--------------------------------------
  ✔ observation shape  == (6,)
  ✔ all values in      [0, 1]
  ✔ episode length     == 200

Run directly:
    python -m flowstate_rl.environment.test_env

Or via pytest:
    pytest flowstate_rl/environment/test_env.py -v
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from flowstate_rl.environment.flow_env import FlowEnv, FLOW_LOW, FLOW_HIGH, PERF_WEIGHTS, MAX_STEPS
from flowstate_rl.environment.personas import PersonaFactory

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PERSONAS       = PersonaFactory.list_personas()   # ["beginner", "average", "expert", "erratic"]
N_EPISODES     = 5
BASE_SEED      = 42

# State vector index aliases (mirrors flow_env.py / personas.py)
IDX_KILL_RATE  = 0
IDX_DEATH_RATE = 1


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Raw results from a single episode."""
    persona:       str
    episode_idx:   int
    total_reward:  float
    n_steps:       int
    flow_steps:    int          # steps in Flow zone
    overwhelmed_steps: int      # steps where death_rate > 0.7
    bored_steps:   int          # steps where kill_rate>0.8 and death_rate<0.1


@dataclass
class PersonaSummary:
    """Aggregated statistics across N_EPISODES for one persona."""
    persona:                 str
    mean_reward:             float
    std_reward:              float
    flow_percentage:         float
    overwhelmed_percentage:  float
    bored_percentage:        float
    episodes:                int


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_observation(obs: np.ndarray, step: int, persona: str) -> None:
    """
    Assert shape and value-range constraints on a single observation.

    Parameters
    ----------
    obs : np.ndarray
    step : int
        Current step index (for error messages).
    persona : str
        Current persona name (for error messages).

    Raises
    ------
    AssertionError
        If any constraint is violated.
    """
    assert obs.shape == (6,), (
        f"[{persona}] step {step}: expected shape (6,), got {obs.shape}"
    )
    assert obs.dtype == np.float32, (
        f"[{persona}] step {step}: expected float32, got {obs.dtype}"
    )
    assert float(obs.min()) >= 0.0 - 1e-6, (
        f"[{persona}] step {step}: obs contains value < 0: {obs}"
    )
    assert float(obs.max()) <= 1.0 + 1e-6, (
        f"[{persona}] step {step}: obs contains value > 1: {obs}"
    )


def validate_episode_length(n_steps: int, persona: str, ep: int) -> None:
    """Assert the episode ran for exactly MAX_STEPS steps."""
    assert n_steps == MAX_STEPS, (
        f"[{persona}] ep {ep}: expected {MAX_STEPS} steps, got {n_steps}"
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    persona_name: str,
    episode_idx:  int,
    seed:         int,
) -> EpisodeResult:
    """
    Execute one random-policy episode and collect step-level statistics.

    Parameters
    ----------
    persona_name : str
    episode_idx  : int
    seed         : int

    Returns
    -------
    EpisodeResult
    """
    env = FlowEnv(persona_name=persona_name, rng_seed=seed)
    obs, _ = env.reset(seed=seed)

    validate_observation(obs, step=0, persona=persona_name)

    total_reward      = 0.0
    flow_steps        = 0
    overwhelmed_steps = 0
    bored_steps       = 0
    step              = 0
    done              = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Per-step validations
        validate_observation(obs, step=step + 1, persona=persona_name)

        # Classify this step
        performance = float(np.dot(PERF_WEIGHTS, obs) + 0.5)
        performance = float(np.clip(performance, 0.0, 1.0))

        kill_rate  = float(obs[IDX_KILL_RATE])
        death_rate = float(obs[IDX_DEATH_RATE])

        if FLOW_LOW <= performance <= FLOW_HIGH:
            flow_steps += 1
        if death_rate > 0.7:
            overwhelmed_steps += 1
        if kill_rate > 0.8 and death_rate < 0.1:
            bored_steps += 1

        total_reward += reward
        step         += 1
        done          = terminated or truncated

    env.close()

    # Episode-level validation
    validate_episode_length(step, persona_name, episode_idx)

    return EpisodeResult(
        persona=persona_name,
        episode_idx=episode_idx,
        total_reward=total_reward,
        n_steps=step,
        flow_steps=flow_steps,
        overwhelmed_steps=overwhelmed_steps,
        bored_steps=bored_steps,
    )


# ---------------------------------------------------------------------------
# Persona aggregation
# ---------------------------------------------------------------------------


def run_persona(persona_name: str, n_episodes: int = N_EPISODES) -> PersonaSummary:
    """
    Run `n_episodes` random-policy episodes for one persona and aggregate.

    Parameters
    ----------
    persona_name : str
    n_episodes   : int

    Returns
    -------
    PersonaSummary
    """
    results: List[EpisodeResult] = []

    for ep in range(n_episodes):
        seed = BASE_SEED + ep
        result = run_episode(persona_name, episode_idx=ep, seed=seed)
        results.append(result)

    rewards      = [r.total_reward for r in results]
    total_steps  = sum(r.n_steps for r in results)

    flow_pct         = 100.0 * sum(r.flow_steps        for r in results) / total_steps
    overwhelmed_pct  = 100.0 * sum(r.overwhelmed_steps for r in results) / total_steps
    bored_pct        = 100.0 * sum(r.bored_steps       for r in results) / total_steps

    mean_r = float(np.mean(rewards))
    std_r  = float(np.std(rewards))

    return PersonaSummary(
        persona=persona_name,
        mean_reward=mean_r,
        std_reward=std_r,
        flow_percentage=flow_pct,
        overwhelmed_percentage=overwhelmed_pct,
        bored_percentage=bored_pct,
        episodes=n_episodes,
    )


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------


def print_summary_table(summaries: List[PersonaSummary]) -> None:
    """Render a formatted summary table to stdout."""
    col_w = {
        "persona":    12,
        "reward":     18,
        "flow":       12,
        "overwhelm":  16,
        "bored":      12,
    }
    total_w = sum(col_w.values()) + len(col_w) * 3 + 1

    border = "═" * total_w
    thin   = "─" * total_w

    header = (
        f"  {'Persona':<{col_w['persona']}}"
        f"  {'Mean Reward ± Std':>{col_w['reward']}}"
        f"  {'Flow %':>{col_w['flow']}}"
        f"  {'Overwhelmed %':>{col_w['overwhelm']}}"
        f"  {'Bored %':>{col_w['bored']}}"
    )

    print(f"\n{border}")
    print(f"  FlowEnv — Random-Policy Evaluation  "
          f"({N_EPISODES} episodes / persona, {MAX_STEPS} steps / episode)")
    print(border)
    print(header)
    print(thin)

    for s in summaries:
        reward_str = f"{s.mean_reward:+.2f} ± {s.std_reward:.2f}"
        row = (
            f"  {s.persona.capitalize():<{col_w['persona']}}"
            f"  {reward_str:>{col_w['reward']}}"
            f"  {s.flow_percentage:>{col_w['flow']}.1f}%"
            f"  {s.overwhelmed_percentage:>{col_w['overwhelm']}.1f}%"
            f"  {s.bored_percentage:>{col_w['bored']}.1f}%"
        )
        print(row)

    print(border)


# ---------------------------------------------------------------------------
# Pytest-compatible test functions
# ---------------------------------------------------------------------------


def test_observation_shape_and_range() -> None:
    """Pytest: obs shape is (6,) and all values in [0, 1]."""
    env = FlowEnv(persona_name="average")
    obs, _ = env.reset(seed=0)
    validate_observation(obs, step=0, persona="average")
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
        validate_observation(obs, step=1, persona="average")
    env.close()
    print("[PASS] test_observation_shape_and_range")


def test_episode_length() -> None:
    """Pytest: every episode terminates after exactly MAX_STEPS steps."""
    for persona in PERSONAS:
        env = FlowEnv(persona_name=persona)
        env.reset(seed=0)
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            steps += 1
            done = term or trunc
        validate_episode_length(steps, persona, ep=0)
        env.close()
    print("[PASS] test_episode_length")


def test_all_personas_run() -> None:
    """Pytest: all personas can run a full episode without exceptions."""
    for persona in PERSONAS:
        result = run_episode(persona, episode_idx=0, seed=0)
        assert result.n_steps == MAX_STEPS, f"Short episode for {persona}"
    print("[PASS] test_all_personas_run")


def test_action_space_coverage() -> None:
    """Pytest: each of the 9 actions can be stepped without error."""
    env = FlowEnv(persona_name="average")
    env.reset(seed=0)
    for action in range(9):
        env.reset(seed=action)
        obs, reward, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs), f"Action {action} produced out-of-bounds obs"
    env.close()
    print("[PASS] test_action_space_coverage")


# ---------------------------------------------------------------------------
# Main — full evaluation + summary table
# ---------------------------------------------------------------------------


def main() -> None:
    summaries: List[PersonaSummary] = []

    print("\nRunning validation tests...")
    test_observation_shape_and_range()
    test_episode_length()
    test_all_personas_run()
    test_action_space_coverage()
    print("All validation tests passed.\n")

    print(f"Evaluating {len(PERSONAS)} personas × {N_EPISODES} episodes "
          f"× {MAX_STEPS} steps = "
          f"{len(PERSONAS) * N_EPISODES * MAX_STEPS:,} total env steps...\n")

    for persona in PERSONAS:
        print(f"  [{persona.upper():<12}] ", end="", flush=True)
        summary = run_persona(persona, n_episodes=N_EPISODES)
        summaries.append(summary)
        print(
            f"reward={summary.mean_reward:+.2f}  "
            f"flow={summary.flow_percentage:.1f}%  "
            f"overwhelmed={summary.overwhelmed_percentage:.1f}%  "
            f"bored={summary.bored_percentage:.1f}%"
        )

    print_summary_table(summaries)
    print()


if __name__ == "__main__":
    main()
