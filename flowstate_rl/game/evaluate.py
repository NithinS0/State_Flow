"""
evaluate.py
-----------
Evaluate and compare policies on ShooterEnv.

Policies
--------
  PPO         — trained model loaded from models/ppo_flowstate*.zip
  Random      — uniform random action from Discrete(9)
  StaticEasy  — always action 0  (decrease enemy_speed)
  StaticHard  — always action 8  (increase enemy_hp)

Metrics (per policy, per persona)
-----------------------------------
  mean_reward            mean total episode reward
  std_reward             standard deviation of episode rewards
  flow_percentage        % of steps in the Flow zone (0.5 ≤ perf ≤ 0.7)
  overwhelmed_percentage % of steps where death_rate > 0.7
  bored_percentage       % of steps where kill_rate > 0.8 & death_rate < 0.1
  most_common_action     mode action across all steps

Output
------
  data/evaluation_results.json   — merged with any existing records
  Console: formatted comparison table + global leaderboard

Usage
-----
  python -m flowstate_rl.game.evaluate
  python -m flowstate_rl.game.evaluate --persona expert --episodes 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flowstate_rl.game.game_env import (
    ShooterEnv,
    PERF_WEIGHTS,
    FLOW_LOW,
    FLOW_HIGH,
    MAX_STEPS,
    _ACTION_MAP,
)
from flowstate_rl.environment.personas import PersonaFactory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent
_PROJECT     = _HERE.parents[1]
MODELS_DIR   = _PROJECT / "models"
DATA_DIR     = _PROJECT / "data"
RESULTS_FILE = DATA_DIR / "evaluation_results.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONAS   = PersonaFactory.list_personas()
N_EPISODES = 10
BASE_SEED  = 31

# Human-readable action labels
_ACTION_LABELS: Dict[int, str] = {
    0: "spd↓", 1: "spd=", 2: "spd↑",
    3: "spwn↓", 4: "spwn=", 5: "spwn↑",
    6: "hp↓",  7: "hp=",  8: "hp↑",
}

# Performance weights (same as game_env / flow_env)
IDX_KILL_RATE  = 0
IDX_DEATH_RATE = 1

# Policy type alias
Policy = Callable[[np.ndarray], int]


# ===========================================================================
# Policies
# ===========================================================================

def random_policy(obs: np.ndarray) -> int:
    """Uniform random action ∈ [0, 8]."""
    return int(np.random.randint(0, 9))


def static_easy_policy(obs: np.ndarray) -> int:
    """Always action 0 — decrease enemy_speed (pull toward easy)."""
    return 0


def static_hard_policy(obs: np.ndarray) -> int:
    """Always action 8 — increase enemy_hp (push toward hard)."""
    return 8


def make_ppo_policy(model) -> Policy:
    """Wrap a loaded SB3 model as a deterministic policy callable."""
    def _policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _policy


# ===========================================================================
# Data containers
# ===========================================================================


class EpisodeStats:
    """Accumulates per-step data within a single episode."""

    __slots__ = (
        "total_reward", "flow_steps", "overwhelmed_steps",
        "bored_steps", "n_steps", "actions",
    )

    def __init__(self) -> None:
        self.total_reward:      float      = 0.0
        self.flow_steps:        int        = 0
        self.overwhelmed_steps: int        = 0
        self.bored_steps:       int        = 0
        self.n_steps:           int        = 0
        self.actions:           List[int]  = []

    def record(self, obs: np.ndarray, reward: float, action: int) -> None:
        performance = float(np.clip(np.dot(PERF_WEIGHTS, obs) + 0.5, 0.0, 1.0))
        kill_rate   = float(obs[IDX_KILL_RATE])
        death_rate  = float(obs[IDX_DEATH_RATE])

        self.total_reward += reward
        self.n_steps      += 1
        self.actions.append(action)

        if FLOW_LOW <= performance <= FLOW_HIGH:
            self.flow_steps += 1
        if death_rate > 0.7:
            self.overwhelmed_steps += 1
        if kill_rate > 0.8 and death_rate < 0.1:
            self.bored_steps += 1


class PolicyResult:
    """Aggregated metrics across N_EPISODES for one (persona × policy)."""

    def __init__(
        self,
        persona:     str,
        policy_name: str,
        episodes:    List[EpisodeStats],
    ) -> None:
        self.persona     = persona
        self.policy_name = policy_name
        self.n_episodes  = len(episodes)

        rewards     = [e.total_reward for e in episodes]
        total_steps = sum(e.n_steps   for e in episodes)

        self.mean_reward            = float(np.mean(rewards))
        self.std_reward             = float(np.std(rewards))
        self.flow_percentage        = 100.0 * sum(e.flow_steps        for e in episodes) / max(total_steps, 1)
        self.overwhelmed_percentage = 100.0 * sum(e.overwhelmed_steps for e in episodes) / max(total_steps, 1)
        self.bored_percentage       = 100.0 * sum(e.bored_steps       for e in episodes) / max(total_steps, 1)

        all_actions          = [a for e in episodes for a in e.actions]
        top_id               = Counter(all_actions).most_common(1)[0][0]
        self.most_common_action    = _ACTION_LABELS.get(top_id, str(top_id))
        self.most_common_action_id = top_id

    def to_dict(self) -> Dict:
        return {
            "persona":               self.persona,
            "policy":                self.policy_name,
            "mean_reward":           round(self.mean_reward,            4),
            "std_reward":            round(self.std_reward,             4),
            "flow_percentage":       round(self.flow_percentage,        2),
            "overwhelmed_percentage":round(self.overwhelmed_percentage, 2),
            "bored_percentage":      round(self.bored_percentage,       2),
            "most_common_action":    self.most_common_action,
            "n_episodes":            self.n_episodes,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }


# ===========================================================================
# Episode runner
# ===========================================================================


def run_episode(env: ShooterEnv, policy: Policy, seed: int) -> EpisodeStats:
    """Execute one episode under `policy` and return per-step stats."""
    obs, _ = env.reset(seed=seed)
    stats   = EpisodeStats()
    done    = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        stats.record(obs, reward, action)
        done = terminated or truncated

    return stats


def evaluate_policy(
    persona_name: str,
    policy_name:  str,
    policy_fn:    Policy,
    n_episodes:   int = N_EPISODES,
    sim_ticks:    int = 10,
) -> PolicyResult:
    """Run `n_episodes` and return a PolicyResult."""
    env = ShooterEnv(persona_name=persona_name, sim_ticks=sim_ticks)
    episodes: List[EpisodeStats] = []

    for ep in range(n_episodes):
        stats = run_episode(env, policy_fn, seed=BASE_SEED + ep)
        episodes.append(stats)

    env.close()
    return PolicyResult(persona_name, policy_name, episodes)


# ===========================================================================
# Model loading
# ===========================================================================


def load_ppo_model(persona_name: str):
    """
    Try to load the PPO model for `persona_name`.  Falls back to the
    generic ppo_flowstate.zip, then the best_model checkpoint.
    Returns None with a warning if nothing is found.
    """
    candidates = [
        MODELS_DIR / f"ppo_flowstate_{persona_name}.zip",
        MODELS_DIR / "ppo_flowstate.zip",
        MODELS_DIR / f"ppo_flowstate_{persona_name}_best" / "best_model.zip",
        MODELS_DIR / "ppo_flowstate_best" / "best_model.zip",
    ]
    for path in candidates:
        if path.exists():
            from stable_baselines3 import PPO
            model = PPO.load(str(path))
            print(f"  [PPO] Loaded: {path.name}")
            return model

    print(
        f"  [WARN] No trained model found for persona '{persona_name}'.\n"
        f"         Run: python -m flowstate_rl.game.train --persona {persona_name}\n"
        f"         Skipping PPO policy.\n"
    )
    return None


# ===========================================================================
# Per-persona evaluation
# ===========================================================================


def evaluate_persona(
    persona_name: str,
    n_episodes:   int = N_EPISODES,
    sim_ticks:    int = 10,
) -> List[PolicyResult]:
    """
    Evaluate Random, StaticEasy, StaticHard, and PPO for one persona.

    Returns a list of PolicyResult (length 3 or 4).
    """
    np.random.seed(BASE_SEED)   # reproducible random policy

    policies: List[Tuple[str, Policy]] = [
        ("Random",     random_policy),
        ("StaticEasy", static_easy_policy),
        ("StaticHard", static_hard_policy),
    ]

    ppo = load_ppo_model(persona_name)
    if ppo is not None:
        policies.append(("PPO", make_ppo_policy(ppo)))

    results: List[PolicyResult] = []

    for name, fn in policies:
        print(f"    {name:<12}", end="", flush=True)
        result = evaluate_policy(persona_name, name, fn, n_episodes, sim_ticks)
        results.append(result)
        print(
            f"  reward={result.mean_reward:+7.2f}  "
            f"flow={result.flow_percentage:5.1f}%  "
            f"overwhelm={result.overwhelmed_percentage:5.1f}%  "
            f"bored={result.bored_percentage:5.1f}%  "
            f"top={result.most_common_action}"
        )

    return results


# ===========================================================================
# Persist results
# ===========================================================================


def save_results(all_results: List[PolicyResult], path: Path = RESULTS_FILE) -> None:
    """Merge results into JSON keyed by (persona, policy)."""
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    new_keys = {(r.persona, r.policy_name) for r in all_results}
    merged   = [
        e for e in existing
        if (e.get("persona"), e.get("policy")) not in new_keys
    ]
    merged.extend(r.to_dict() for r in all_results)

    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n  Results saved → {path}")


# ===========================================================================
# Comparison table
# ===========================================================================


_C = {"policy": 12, "reward": 22, "flow": 10, "overwhelm": 14, "bored": 10, "action": 12}
_TW = sum(_C.values()) + len(_C) * 3


def _border(ch: str = "═") -> str:
    return ch * _TW


def _header() -> str:
    return (
        f"  {'Policy':<{_C['policy']}}"
        f"  {'Mean Reward ± Std':>{_C['reward']}}"
        f"  {'Flow %':>{_C['flow']}}"
        f"  {'Overwhelm %':>{_C['overwhelm']}}"
        f"  {'Bored %':>{_C['bored']}}"
        f"  {'Top Action':>{_C['action']}}"
    )


def _row(r: PolicyResult, best: bool = False) -> str:
    reward_str = f"{r.mean_reward:+.2f} +/- {r.std_reward:.2f}"
    mark = " *" if best else "  "
    return (
        f"{mark}{r.policy_name:<{_C['policy']}}"
        f"  {reward_str:>{_C['reward']}}"
        f"  {r.flow_percentage:>{_C['flow']}.1f}%"
        f"  {r.overwhelmed_percentage:>{_C['overwhelm']}.1f}%"
        f"  {r.bored_percentage:>{_C['bored']}.1f}%"
        f"  {r.most_common_action:>{_C['action']}}"
    )


def print_persona_table(persona: str, results: List[PolicyResult]) -> None:
    """Print a comparison table for one persona."""
    best_r = max(r.mean_reward for r in results)

    print(f"\n{_border()}")
    print(
        f"  Persona: {persona.upper():<10}  "
        f"({results[0].n_episodes} eps × {MAX_STEPS} steps  |  * = best)"
    )
    print(_border())
    print(_header())
    print(_border("─"))
    for r in results:
        print(_row(r, best=abs(r.mean_reward - best_r) < 1e-6))
    print(_border())


def print_global_leaderboard(all_results: List[PolicyResult]) -> None:
    """Cross-persona leaderboard ranked by mean reward."""
    from collections import defaultdict
    by_policy: Dict[str, List[float]] = defaultdict(list)
    for r in all_results:
        by_policy[r.policy_name].append(r.mean_reward)

    ranked = sorted(
        by_policy.items(),
        key=lambda kv: float(np.mean(kv[1])),
        reverse=True,
    )

    tw = 52
    print(f"\n{'═' * tw}")
    print("  Global Leaderboard — Mean Reward across All Personas")
    print(f"{'═' * tw}")
    print(f"  {'Rank':<6}{'Policy':<14}{'Avg Reward':>12}{'Best':>10}{'Worst':>10}")
    print(f"{'─' * tw}")
    for rank, (policy_name, rewards) in enumerate(ranked, 1):
        print(
            f"  #{rank:<5}{policy_name:<14}"
            f"{float(np.mean(rewards)):>+12.3f}"
            f"{float(np.max(rewards)):>+10.3f}"
            f"{float(np.min(rewards)):>+10.3f}"
        )
    print(f"{'═' * tw}")


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShooterEnv policies")
    parser.add_argument(
        "--persona",
        default=None,
        choices=PERSONAS + ["all"],
        help="Persona to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=N_EPISODES,
        help=f"Episodes per policy (default: {N_EPISODES})",
    )
    parser.add_argument(
        "--sim-ticks",
        type=int,
        default=10,
        help="ShooterEnv sim ticks per agent step (default: 10)",
    )
    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    args = _parse_args()

    target_personas = (
        PERSONAS
        if args.persona is None or args.persona == "all"
        else [args.persona]
    )

    print("\nFlowState-RL — ShooterEnv Policy Evaluation")
    print(f"Personas  : {', '.join(target_personas)}")
    print(f"Episodes  : {args.episodes} per policy")
    print(f"Policies  : Random | StaticEasy | StaticHard | PPO (if trained)\n")

    all_results: List[PolicyResult] = []

    for persona in target_personas:
        print(f"\n[{persona.upper()}]")
        results = evaluate_persona(
            persona, n_episodes=args.episodes, sim_ticks=args.sim_ticks
        )
        all_results.extend(results)
        print_persona_table(persona, results)

    print_global_leaderboard(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
