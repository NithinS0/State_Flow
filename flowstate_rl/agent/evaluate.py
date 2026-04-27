"""
evaluate.py
-----------
Evaluate trained PPO models against three baseline policies on FlowEnv.

Policies compared (per persona)
---------------------------------
  PPO           — trained Stable-Baselines3 model
  Random        — uniform random action from Discrete(9)
  StaticEasy    — always action 0 (decrease enemy_speed, easiest bias)
  StaticHard    — always action 8 (increase enemy_hp, hardest bias)

Metrics (per policy, per persona)
-----------------------------------
  mean_reward           : mean total episode reward across N episodes
  flow_percentage       : % of steps in the Flow zone  (0.5 ≤ perf ≤ 0.7)
  overwhelmed_percentage: % of steps where death_rate > 0.7
  bored_percentage      : % of steps where kill_rate > 0.8 & death_rate < 0.1
  most_common_action    : mode action chosen across all steps

Results saved to
-----------------
  data/evaluation_results.json

Usage
-----
  # Evaluate all personas
  python -m flowstate_rl.agent.evaluate

  # Evaluate a single persona
  python -m flowstate_rl.agent.evaluate --persona expert

  # Quick smoke-test (3 episodes)
  python -m flowstate_rl.agent.evaluate --episodes 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from flowstate_rl.environment.flow_env import (
    FlowEnv,
    FLOW_LOW,
    FLOW_HIGH,
    PERF_WEIGHTS,
    MAX_STEPS,
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

PERSONAS      = PersonaFactory.list_personas()
N_EPISODES    = 10
BASE_SEED     = 7

IDX_KILL_RATE  = 0
IDX_DEATH_RATE = 1

# Action labels for readability
ACTION_LABELS: Dict[int, str] = {
    0: "spd↓", 1: "spd=", 2: "spd↑",
    3: "spwn↓", 4: "spwn=", 5: "spwn↑",
    6: "hp↓", 7: "hp=", 8: "hp↑",
}


# ---------------------------------------------------------------------------
# Policy callables
# ---------------------------------------------------------------------------

# A policy is any callable:  (obs: np.ndarray) -> int
Policy = Callable[[np.ndarray], int]


def random_policy(obs: np.ndarray) -> int:
    """Uniform random action in [0, 8]."""
    return int(np.random.randint(0, 9))


def static_easy_policy(obs: np.ndarray) -> int:
    """Always choose action 0 — decrease enemy_speed (easiest bias)."""
    return 0


def static_hard_policy(obs: np.ndarray) -> int:
    """Always choose action 8 — increase enemy_hp (hardest bias)."""
    return 8


def make_ppo_policy(model: PPO) -> Policy:
    """Wrap a loaded PPO model as a deterministic policy callable."""
    def _policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _policy


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class EpisodeStats:
    """Accumulates per-step data for a single episode."""

    def __init__(self) -> None:
        self.total_reward:       float      = 0.0
        self.flow_steps:         int        = 0
        self.overwhelmed_steps:  int        = 0
        self.bored_steps:        int        = 0
        self.n_steps:            int        = 0
        self.actions:            List[int]  = []

    def record_step(
        self,
        obs:    np.ndarray,
        reward: float,
        action: int,
    ) -> None:
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
    """Aggregated metrics across N_EPISODES for one (persona, policy) pair."""

    def __init__(
        self,
        persona:      str,
        policy_name:  str,
        episodes:     List[EpisodeStats],
    ) -> None:
        self.persona     = persona
        self.policy_name = policy_name
        self.n_episodes  = len(episodes)

        rewards      = [e.total_reward for e in episodes]
        total_steps  = sum(e.n_steps for e in episodes)

        self.mean_reward            = float(np.mean(rewards))
        self.std_reward             = float(np.std(rewards))
        self.flow_percentage        = 100.0 * sum(e.flow_steps        for e in episodes) / total_steps
        self.overwhelmed_percentage = 100.0 * sum(e.overwhelmed_steps for e in episodes) / total_steps
        self.bored_percentage       = 100.0 * sum(e.bored_steps       for e in episodes) / total_steps

        all_actions         = [a for e in episodes for a in e.actions]
        most_common_id      = Counter(all_actions).most_common(1)[0][0]
        self.most_common_action     = ACTION_LABELS.get(most_common_id, str(most_common_id))
        self.most_common_action_id  = most_common_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona":               self.persona,
            "policy":                self.policy_name,
            "mean_reward":           round(self.mean_reward, 4),
            "std_reward":            round(self.std_reward, 4),
            "flow_percentage":       round(self.flow_percentage, 2),
            "overwhelmed_percentage":round(self.overwhelmed_percentage, 2),
            "bored_percentage":      round(self.bored_percentage, 2),
            "most_common_action":    self.most_common_action,
            "n_episodes":            self.n_episodes,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env:    FlowEnv,
    policy: Policy,
    seed:   int,
) -> EpisodeStats:
    """
    Execute one episode under the given policy and return step statistics.

    Parameters
    ----------
    env    : FlowEnv (already constructed)
    policy : callable (obs) -> int
    seed   : int

    Returns
    -------
    EpisodeStats
    """
    obs, _ = env.reset(seed=seed)
    stats   = EpisodeStats()
    done    = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        stats.record_step(obs, reward, action)
        done = terminated or truncated

    return stats


# ---------------------------------------------------------------------------
# Evaluate one (persona, policy_name, policy_fn) triple
# ---------------------------------------------------------------------------


def evaluate_policy(
    persona_name: str,
    policy_name:  str,
    policy_fn:    Policy,
    n_episodes:   int = N_EPISODES,
) -> PolicyResult:
    """
    Run `n_episodes` episodes and return a PolicyResult.

    Parameters
    ----------
    persona_name : str
    policy_name  : str
    policy_fn    : Policy
    n_episodes   : int

    Returns
    -------
    PolicyResult
    """
    env     = FlowEnv(persona_name=persona_name, rng_seed=BASE_SEED)
    episodes: List[EpisodeStats] = []

    for ep in range(n_episodes):
        stats = run_episode(env, policy_fn, seed=BASE_SEED + ep)
        episodes.append(stats)

    env.close()
    return PolicyResult(persona_name, policy_name, episodes)


# ---------------------------------------------------------------------------
# Load PPO model (graceful fallback if not yet trained)
# ---------------------------------------------------------------------------


def load_ppo(persona_name: str) -> Optional[PPO]:
    """
    Attempt to load the trained PPO model for a persona.

    Returns None (with a warning) if the model file does not exist.
    """
    model_path = MODELS_DIR / f"ppo_flowstate_{persona_name}.zip"
    if not model_path.exists():
        # Also try the best-model path
        best_path = MODELS_DIR / f"ppo_flowstate_{persona_name}_best" / "best_model.zip"
        if best_path.exists():
            model_path = best_path
        else:
            print(
                f"  [WARN] No trained model found for '{persona_name}'. "
                f"Skipping PPO policy.\n"
                f"         Run: python -m flowstate_rl.agent.train --persona {persona_name}"
            )
            return None
    return PPO.load(str(model_path))


# ---------------------------------------------------------------------------
# Evaluate all policies for one persona
# ---------------------------------------------------------------------------


def evaluate_persona(
    persona_name: str,
    n_episodes:   int = N_EPISODES,
) -> List[PolicyResult]:
    """
    Evaluate Random, StaticEasy, StaticHard, and PPO (if available)
    for a single persona.

    Returns
    -------
    list of PolicyResult (length 3 or 4)
    """
    np.random.seed(BASE_SEED)   # fix random policy seed for reproducibility

    # Always-present baselines
    policies: List[Tuple[str, Policy]] = [
        ("Random",     random_policy),
        ("StaticEasy", static_easy_policy),
        ("StaticHard", static_hard_policy),
    ]

    # PPO (optional — only if model exists)
    ppo_model = load_ppo(persona_name)
    if ppo_model is not None:
        policies.append(("PPO", make_ppo_policy(ppo_model)))

    results: List[PolicyResult] = []

    for policy_name, policy_fn in policies:
        print(f"    {policy_name:<12}", end="", flush=True)
        result = evaluate_policy(persona_name, policy_name, policy_fn, n_episodes)
        results.append(result)
        print(
            f"  reward={result.mean_reward:+.2f}  "
            f"flow={result.flow_percentage:.1f}%  "
            f"overwhelm={result.overwhelmed_percentage:.1f}%  "
            f"bored={result.bored_percentage:.1f}%  "
            f"common_action={result.most_common_action}"
        )

    return results


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------


def save_results(all_results: List[PolicyResult], path: Path = RESULTS_FILE) -> None:
    """
    Write evaluation results to JSON, merging with any existing file.
    Entries are keyed by (persona, policy) — duplicates are overwritten.
    """
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path, "r") as f:
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

    print(f"\n  Results saved to {path}")


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

_COL = {
    "policy":   12,
    "reward":   20,
    "flow":     10,
    "overwhelm":15,
    "bored":    10,
    "action":   14,
}
_TOTAL_W = sum(_COL.values()) + len(_COL) * 3


def _table_border(char: str = "═") -> str:
    return char * _TOTAL_W


def _table_header() -> str:
    return (
        f"  {'Policy':<{_COL['policy']}}"
        f"  {'Mean Reward ± Std':>{_COL['reward']}}"
        f"  {'Flow %':>{_COL['flow']}}"
        f"  {'Overwhelm %':>{_COL['overwhelm']}}"
        f"  {'Bored %':>{_COL['bored']}}"
        f"  {'Top Action':>{_COL['action']}}"
    )


def _table_row(r: PolicyResult, is_best: bool = False) -> str:
    reward_str = f"{r.mean_reward:+.2f} +/- {r.std_reward:.2f}"
    marker = " *" if is_best else "  "
    return (
        f"{marker}{r.policy_name:<{_COL['policy']}}"
        f"  {reward_str:>{_COL['reward']}}"
        f"  {r.flow_percentage:>{_COL['flow']}.1f}%"
        f"  {r.overwhelmed_percentage:>{_COL['overwhelm']}.1f}%"
        f"  {r.bored_percentage:>{_COL['bored']}.1f}%"
        f"  {r.most_common_action:>{_COL['action']}}"
    )


def print_persona_table(persona_name: str, results: List[PolicyResult]) -> None:
    """Render the comparison table for one persona."""
    best_reward = max(r.mean_reward for r in results)

    n_ep = results[0].n_episodes
    print(f"\n{_table_border('═')}")
    print(
        f"  Persona: {persona_name.upper():<10}  "
        f"({n_ep} episodes per policy  |  * = best reward)"
    )
    print(_table_border("═"))
    print(_table_header())
    print(_table_border("─"))

    for r in results:
        is_best = abs(r.mean_reward - best_reward) < 1e-6
        print(_table_row(r, is_best))

    print(_table_border("═"))


def print_global_summary(all_results: List[PolicyResult]) -> None:
    """Print a global leaderboard across all personas."""
    from collections import defaultdict

    by_policy: Dict[str, List[float]] = defaultdict(list)
    for r in all_results:
        by_policy[r.policy_name].append(r.mean_reward)

    print(f"\n{'═' * 50}")
    print("  Global Leaderboard — Mean Reward across All Personas")
    print(f"{'═' * 50}")
    print(f"  {'Policy':<14}  {'Avg Reward':>12}  {'Best':>8}")
    print(f"{'─' * 50}")

    rankings = sorted(
        by_policy.items(),
        key=lambda kv: float(np.mean(kv[1])),
        reverse=True,
    )
    for rank, (policy_name, rewards) in enumerate(rankings, 1):
        print(
            f"  #{rank}  {policy_name:<12}  "
            f"{float(np.mean(rewards)):>+12.3f}  "
            f"{float(np.max(rewards)):>+8.3f}"
        )

    print(f"{'═' * 50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FlowState-RL policies"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=PERSONAS + ["all"],
        help="Persona to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
        help=f"Episodes per policy (default: {N_EPISODES})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    target_personas = (
        PERSONAS
        if args.persona is None or args.persona == "all"
        else [args.persona]
    )

    print("\nFlowState-RL — Policy Evaluation")
    print(f"Personas : {', '.join(target_personas)}")
    print(f"Episodes : {args.episodes} per policy\n")
    print("Policies : Random | StaticEasy | StaticHard | PPO (if trained)\n")

    all_results: List[PolicyResult] = []

    for persona in target_personas:
        print(f"\n[{persona.upper()}]")
        persona_results = evaluate_persona(persona, n_episodes=args.episodes)
        all_results.extend(persona_results)
        print_persona_table(persona, persona_results)

    print_global_summary(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
