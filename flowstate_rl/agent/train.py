"""
train.py
--------
Train a PPO agent on FlowEnv for each of the four player personas.

For every persona a dedicated model is trained, saved, and its evaluation
reward logged.  All results are persisted to JSON for downstream analysis.

Saved artefacts
---------------
  models/ppo_flowstate_{persona}.zip      — trained SB3 model
  data/logs/{persona}/                    — TensorBoard event files
  data/training_results.json              — mean rewards + metadata

Usage
-----
  # Train all personas (default)
  python -m flowstate_rl.agent.train

  # Train a single persona
  python -m flowstate_rl.agent.train --persona expert

  # Quick smoke-test (1000 timesteps)
  python -m flowstate_rl.agent.train --timesteps 1000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from flowstate_rl.environment.flow_env import FlowEnv
from flowstate_rl.environment.personas import PersonaFactory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent          # flowstate_rl/agent/
_PROJECT     = _HERE.parents[1]                         # project root
MODELS_DIR   = _PROJECT / "models"
DATA_DIR     = _PROJECT / "data"
LOGS_DIR     = DATA_DIR / "logs"
RESULTS_FILE = DATA_DIR / "training_results.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

PERSONAS: List[str] = PersonaFactory.list_personas()   # fixed order

TOTAL_TIMESTEPS = 50_000
N_EVAL_EPISODES = 10          # episodes used inside EvalCallback

PPO_KWARGS = dict(
    policy       = "MlpPolicy",
    learning_rate= 3e-4,
    n_steps      = 2048,
    batch_size   = 64,
    n_epochs     = 10,
    gamma        = 0.99,
    clip_range   = 0.2,
    ent_coef     = 0.01,
    verbose      = 1,
    policy_kwargs= dict(net_arch=[128, 64]),
)


# ---------------------------------------------------------------------------
# Custom callback: track mean rewards per rollout
# ---------------------------------------------------------------------------


class RewardLoggerCallback(BaseCallback):
    """
    Collect mean episode rewards from the Monitor wrapper after each
    rollout and store them for later retrieval.

    Parameters
    ----------
    verbose : int
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self._ep_rewards_buf: List[float] = []

    def _on_step(self) -> bool:
        """Accumulate episode rewards reported in the infos dict."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards_buf.append(info["episode"]["r"])
        return True   # continue training

    def _on_rollout_end(self) -> None:
        """Flush buffer into the permanent list at the end of each rollout."""
        self.episode_rewards.extend(self._ep_rewards_buf)
        self._ep_rewards_buf.clear()

    @property
    def mean_reward(self) -> float:
        """Mean reward across all completed episodes so far."""
        if not self.episode_rewards:
            return float("nan")
        return float(np.mean(self.episode_rewards))


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _make_env(persona_name: str, seed: int = 0):
    """Return a Monitor-wrapped FlowEnv factory (for make_vec_env)."""
    def _factory():
        env = FlowEnv(persona_name=persona_name, rng_seed=seed)
        return Monitor(env)
    return _factory


# ---------------------------------------------------------------------------
# Single-persona training
# ---------------------------------------------------------------------------


def train_persona(
    persona_name:     str,
    total_timesteps:  int = TOTAL_TIMESTEPS,
    seed:             int = 0,
) -> Dict:
    """
    Train a PPO agent on FlowEnv for one persona.

    Parameters
    ----------
    persona_name : str
    total_timesteps : int
    seed : int

    Returns
    -------
    dict
        Summary dictionary with keys:
        persona, model_path, mean_reward, best_mean_reward,
        total_timesteps, wall_time_s, timestamp.
    """
    print(f"\n{'═' * 65}")
    print(f"  Training  : PPO  |  persona = {persona_name.upper()}")
    print(f"  Timesteps : {total_timesteps:,}  |  seed = {seed}")
    print(f"{'═' * 65}\n")

    # ── Paths ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"ppo_flowstate_{persona_name}"
    log_path   = LOGS_DIR   / persona_name
    best_path  = MODELS_DIR / f"ppo_flowstate_{persona_name}_best"
    log_path.mkdir(parents=True, exist_ok=True)
    best_path.mkdir(parents=True, exist_ok=True)

    # ── Environments ─────────────────────────────────────────────────────
    train_env = make_vec_env(_make_env(persona_name, seed=seed), n_envs=1)
    eval_env  = make_vec_env(_make_env(persona_name, seed=seed + 99), n_envs=1)

    # ── Callbacks ────────────────────────────────────────────────────────
    reward_logger = RewardLoggerCallback(verbose=0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(best_path),
        log_path             = str(log_path / "eval"),
        eval_freq            = max(total_timesteps // 10, 1),
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PPO(
        env              = train_env,
        tensorboard_log  = str(LOGS_DIR),
        seed             = seed,
        **PPO_KWARGS,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    model.learn(
        total_timesteps  = total_timesteps,
        callback         = [reward_logger, eval_callback],
        tb_log_name      = persona_name,
        progress_bar     = False,
        reset_num_timesteps = True,
    )

    wall_time = time.perf_counter() - t_start

    # ── Save final model ──────────────────────────────────────────────────
    model.save(str(model_path))
    print(f"\n  ✔ Final model  → {model_path}.zip")

    # ── Compute best mean reward from eval callback ───────────────────────
    best_mean_reward = getattr(eval_callback, "best_mean_reward", float("nan"))

    # ── Cleanup ───────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()

    summary = {
        "persona":          persona_name,
        "model_path":       str(model_path) + ".zip",
        "best_model_path":  str(best_path / "best_model.zip"),
        "mean_reward":      reward_logger.mean_reward,
        "best_mean_reward": float(best_mean_reward),
        "total_timesteps":  total_timesteps,
        "wall_time_s":      round(wall_time, 2),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    print(
        f"\n  mean_reward      = {summary['mean_reward']:+.3f}\n"
        f"  best_mean_reward = {summary['best_mean_reward']:+.3f}\n"
        f"  wall_time        = {wall_time:.1f}s\n"
    )
    return summary


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------


def save_results(results: List[Dict], path: Path = RESULTS_FILE) -> None:
    """
    Persist training results to JSON.  Merges with any existing records
    so multiple runs accumulate over time.

    Parameters
    ----------
    results : list of dict
    path : Path
    """
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    # Replace any existing record with the same persona name
    existing_personas = {r["persona"] for r in existing}
    merged = [r for r in existing if r["persona"] not in {n["persona"] for n in results}]
    merged.extend(results)

    with open(path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n  ✔ Results saved → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_training_summary(results: List[Dict]) -> None:
    """Print a compact table of training results for all personas."""
    col = {"persona": 12, "mean_r": 16, "best_r": 16, "time": 10}
    total_w = sum(col.values()) + len(col) * 3 + 1
    border  = "═" * total_w
    thin    = "─" * total_w

    print(f"\n{border}")
    print("  FlowState-RL — Training Summary")
    print(border)
    print(
        f"  {'Persona':<{col['persona']}}"
        f"  {'Mean Reward':>{col['mean_r']}}"
        f"  {'Best Reward':>{col['best_r']}}"
        f"  {'Time (s)':>{col['time']}}"
    )
    print(thin)

    for r in results:
        print(
            f"  {r['persona'].capitalize():<{col['persona']}}"
            f"  {r['mean_reward']:>+{col['mean_r']}.3f}"
            f"  {r['best_mean_reward']:>+{col['best_r']}.3f}"
            f"  {r['wall_time_s']:>{col['time']}.1f}"
        )

    total_time = sum(r["wall_time_s"] for r in results)
    print(border)
    print(f"  Total training time: {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Models dir : {MODELS_DIR}")
    print(f"  Logs dir   : {LOGS_DIR}")
    print(f"  Results    : {RESULTS_FILE}")
    print(border)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO agent(s) on FlowEnv"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        choices=PERSONAS + ["all"],
        help="Which persona to train (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Training timesteps per persona (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for training (default: 0)",
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

    print("\nFlowState-RL — PPO Training")
    print(f"Personas  : {', '.join(target_personas)}")
    print(f"Timesteps : {args.timesteps:,} per persona")
    print(f"Network   : MlpPolicy  net_arch=[128, 64]")
    print(f"TensorBoard : tensorboard --logdir {LOGS_DIR}")

    results: List[Dict] = []

    for persona in target_personas:
        summary = train_persona(
            persona_name    = persona,
            total_timesteps = args.timesteps,
            seed            = args.seed,
        )
        results.append(summary)

    save_results(results)
    print_training_summary(results)


if __name__ == "__main__":
    main()
