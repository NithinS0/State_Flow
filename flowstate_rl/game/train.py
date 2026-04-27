"""
train.py
--------
Train a PPO agent on ShooterEnv using Stable-Baselines3.

Produces
--------
  models/ppo_flowstate.zip           — final trained model
  models/ppo_flowstate_best/         — best checkpoint (by eval reward)
  data/logs/ShooterEnv/              — TensorBoard event file
  data/shooter_training_results.json — reward history + metadata

Hyperparameters
---------------
  Policy          : MlpPolicy   net_arch = [128, 64]
  learning_rate   : 3e-4
  gamma           : 0.99
  clip_range      : 0.2
  n_steps         : 2048
  batch_size      : 64
  n_epochs        : 10
  ent_coef        : 0.01
  total_timesteps : 100 000

Usage
-----
  # Full 100 k-step run (all personas)
  python -m flowstate_rl.game.train

  # Single persona
  python -m flowstate_rl.game.train --persona expert

  # Quick smoke-test
  python -m flowstate_rl.game.train --timesteps 2048
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

from flowstate_rl.game.game_env import ShooterEnv
from flowstate_rl.environment.personas import PersonaFactory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent          # flowstate_rl/game/
_PROJECT     = _HERE.parents[1]                         # project root
MODELS_DIR   = _PROJECT / "models"
DATA_DIR     = _PROJECT / "data"
LOGS_DIR     = DATA_DIR / "logs"
RESULTS_FILE = DATA_DIR / "shooter_training_results.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PERSONAS             = PersonaFactory.list_personas()
TOTAL_TIMESTEPS      = 100_000
PRINT_REWARD_EVERY   = 5_000       # timesteps between avg-reward console prints
N_EVAL_EPISODES      = 5           # episodes used by EvalCallback

PPO_HYPERPARAMS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 1,
    policy_kwargs = dict(net_arch=[128, 64]),
)


# ===========================================================================
# Callbacks
# ===========================================================================


class RewardPrinterCallback(BaseCallback):
    """
    Prints the rolling mean episode reward to the console every
    PRINT_REWARD_EVERY environment steps.

    Also stores full reward history for JSON export.
    """

    def __init__(self, print_every: int = PRINT_REWARD_EVERY, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.print_every      = print_every
        self.episode_rewards: List[float] = []
        self._buf:            List[float] = []
        self._last_print_at:  int         = 0

    # ── SB3 callback hooks ─────────────────────────────────────────────

    def _on_step(self) -> bool:
        """Collect episode rewards from Monitor infos."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                self._buf.append(r)
                self.episode_rewards.append(r)
        return True

    def _on_rollout_end(self) -> None:
        """Print summary when enough steps have accumulated."""
        steps = self.num_timesteps
        if steps - self._last_print_at >= self.print_every and self.episode_rewards:
            window_rewards = self.episode_rewards[-max(1, len(self.episode_rewards) // 5):]
            mean_r  = float(np.mean(window_rewards))
            std_r   = float(np.std(window_rewards))
            min_r   = float(np.min(self.episode_rewards))
            max_r   = float(np.max(self.episode_rewards))

            print(
                f"\n  ┌─ Step {steps:>8,} "
                f"{'─' * 40}\n"
                f"  │  Episodes completed : {len(self.episode_rewards):>6}\n"
                f"  │  Mean reward (recent): {mean_r:>+9.3f} ± {std_r:.3f}\n"
                f"  │  All-time  min / max : {min_r:>+9.3f} / {max_r:>+.3f}\n"
                f"  └{'─' * 50}"
            )
            self._last_print_at = steps
        self._buf.clear()

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else float("nan")

    @property
    def recent_mean_reward(self) -> float:
        tail = self.episode_rewards[-N_EVAL_EPISODES:] if self.episode_rewards else []
        return float(np.mean(tail)) if tail else float("nan")


# ===========================================================================
# Environment factory
# ===========================================================================


def _make_env(persona_name: str, seed: int = 0, sim_ticks: int = 10):
    """Return a Monitor-wrapped ShooterEnv factory for make_vec_env."""
    def _factory():
        env = ShooterEnv(persona_name=persona_name, sim_ticks=sim_ticks)
        return Monitor(env)
    return _factory


# ===========================================================================
# Training
# ===========================================================================


def train_shooter(
    persona_name:    str  = "average",
    total_timesteps: int  = TOTAL_TIMESTEPS,
    seed:            int  = 0,
    model_tag:       str  = "ppo_flowstate",
    sim_ticks:       int  = 10,
) -> Dict:
    """
    Train a PPO agent on ShooterEnv for one persona.

    Parameters
    ----------
    persona_name    : str   persona to train against
    total_timesteps : int   total env steps
    seed            : int   RNG seed for reproducibility
    model_tag       : str   filename prefix (without .zip)
    sim_ticks       : int   ShooterEnv sim ticks per agent step

    Returns
    -------
    dict — training summary
    """
    run_name   = f"{model_tag}_{persona_name}" if persona_name != "average" else model_tag
    model_path = MODELS_DIR / run_name
    best_path  = MODELS_DIR / f"{run_name}_best"
    log_path   = LOGS_DIR / "ShooterEnv"

    log_path.mkdir(parents=True, exist_ok=True)
    best_path.mkdir(parents=True, exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  FlowState-RL — PPO on ShooterEnv")
    print(f"  Persona     : {persona_name.upper()}")
    print(f"  Timesteps   : {total_timesteps:,}")
    print(f"  Network     : MlpPolicy  [128, 64]")
    print(f"  Sim ticks   : {sim_ticks} per agent step")
    print(f"  Model → {model_path}.zip")
    print(f"  TensorBoard : tensorboard --logdir {log_path}")
    print(f"{'═' * 65}\n")

    # ── Environments ─────────────────────────────────────────────────────
    train_env = make_vec_env(
        _make_env(persona_name, seed=seed, sim_ticks=sim_ticks),
        n_envs=1,
    )
    eval_env = make_vec_env(
        _make_env(persona_name, seed=seed + 999, sim_ticks=sim_ticks),
        n_envs=1,
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    reward_printer = RewardPrinterCallback(print_every=PRINT_REWARD_EVERY)

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
        env             = train_env,
        tensorboard_log = str(log_path),
        seed            = seed,
        **PPO_HYPERPARAMS,
    )

    print(f"  Model params: {sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [reward_printer, eval_callback],
        tb_log_name         = f"PPO_{persona_name}",
        reset_num_timesteps = True,
        progress_bar        = False,
    )

    wall_time = time.perf_counter() - t_start

    # ── Save ─────────────────────────────────────────────────────────────
    model.save(str(model_path))
    print(f"\n  ✔ Final model  → {model_path}.zip")

    # ── Episode reward summary ────────────────────────────────────────────
    all_rewards = reward_printer.episode_rewards
    mean_r      = float(np.mean(all_rewards))  if all_rewards else float("nan")
    std_r       = float(np.std(all_rewards))   if all_rewards else float("nan")
    best_mean_r = float(getattr(eval_callback, "best_mean_reward", float("nan")))

    # ── Cleanup ───────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()

    summary = {
        "persona":          persona_name,
        "model_path":       str(model_path) + ".zip",
        "best_model_path":  str(best_path / "best_model.zip"),
        "total_timesteps":  total_timesteps,
        "n_episodes":       len(all_rewards),
        "mean_reward":      round(mean_r,  4),
        "std_reward":       round(std_r,   4),
        "best_mean_reward": round(best_mean_r, 4),
        "reward_history":   [round(r, 3) for r in all_rewards],
        "wall_time_s":      round(wall_time, 2),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "hyperparams": {
            "learning_rate": PPO_HYPERPARAMS["learning_rate"],
            "gamma":         PPO_HYPERPARAMS["gamma"],
            "clip_range":    PPO_HYPERPARAMS["clip_range"],
            "n_steps":       PPO_HYPERPARAMS["n_steps"],
            "batch_size":    PPO_HYPERPARAMS["batch_size"],
            "ent_coef":      PPO_HYPERPARAMS["ent_coef"],
            "net_arch":      PPO_HYPERPARAMS["policy_kwargs"]["net_arch"],
            "total_timesteps": total_timesteps,
        },
    }

    print(f"\n  mean_reward      = {mean_r:+.4f}")
    print(f"  std_reward       = {std_r:.4f}")
    print(f"  best_mean_reward = {best_mean_r:+.4f}")
    print(f"  n_episodes       = {len(all_rewards)}")
    print(f"  wall_time        = {wall_time:.1f}s  ({wall_time/60:.1f} min)\n")

    return summary


# ===========================================================================
# Persist results
# ===========================================================================


def save_results(results: List[Dict], path: Path = RESULTS_FILE) -> None:
    """Merge new results into the JSON file (keyed by persona)."""
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    updated_personas = {r["persona"] for r in results}
    merged = [e for e in existing if e.get("persona") not in updated_personas]
    merged.extend(results)

    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  ✔ Results saved → {path}\n")


# ===========================================================================
# Summary table
# ===========================================================================


def print_summary(results: List[Dict]) -> None:
    """Print a neatly formatted training summary."""
    col = {"persona": 12, "eps": 8, "mean": 16, "best": 14, "time": 10}
    tw  = sum(col.values()) + len(col) * 3
    div = "═" * tw
    sep = "─" * tw

    print(f"\n{div}")
    print("  ShooterEnv — PPO Training Summary")
    print(div)
    print(
        f"  {'Persona':<{col['persona']}}"
        f"  {'Episodes':>{col['eps']}}"
        f"  {'Mean Reward ± Std':>{col['mean']}}"
        f"  {'Best Reward':>{col['best']}}"
        f"  {'Time (s)':>{col['time']}}"
    )
    print(sep)

    for r in results:
        reward_str = f"{r['mean_reward']:+.3f} ± {r['std_reward']:.3f}"
        print(
            f"  {r['persona'].capitalize():<{col['persona']}}"
            f"  {r['n_episodes']:>{col['eps']}}"
            f"  {reward_str:>{col['mean']}}"
            f"  {r['best_mean_reward']:>+{col['best']}.3f}"
            f"  {r['wall_time_s']:>{col['time']}.1f}"
        )

    total_t = sum(r["wall_time_s"] for r in results)
    print(div)
    print(
        f"  Total time : {total_t:.0f}s  ({total_t/60:.1f} min)\n"
        f"  Models dir : {MODELS_DIR}\n"
        f"  Logs dir   : {LOGS_DIR}\n"
        f"  Results    : {RESULTS_FILE}"
    )
    print(div)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on ShooterEnv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--persona",
        default=None,
        choices=PERSONAS + ["all"],
        help="Persona to train (default: all)",
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Training timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (default: 0)",
    )
    parser.add_argument(
        "--sim-ticks",
        type=int,
        default=10,
        help="ShooterEnv sim ticks per agent step (default: 10)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ppo_flowstate",
        help="Model filename prefix (default: ppo_flowstate)",
    )
    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    args = _parse_args()

    personas = (
        PERSONAS
        if args.persona is None or args.persona == "all"
        else [args.persona]
    )

    print("\n" + "═" * 65)
    print("  FlowState-RL — ShooterEnv PPO Training")
    print(f"  Persona(s)  : {', '.join(personas)}")
    print(f"  Timesteps   : {args.timesteps:,} per persona")
    print(f"  Sim ticks   : {args.sim_ticks} per agent step")
    print(f"  TensorBoard : tensorboard --logdir {LOGS_DIR}")
    print("═" * 65)

    results: List[Dict] = []

    for persona in personas:
        summary = train_shooter(
            persona_name    = persona,
            total_timesteps = args.timesteps,
            seed            = args.seed,
            model_tag       = args.tag,
            sim_ticks       = args.sim_ticks,
        )
        results.append(summary)

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
