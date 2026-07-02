#!/usr/bin/env python3
"""Evaluate MAPPO checkpoints and write fixed-policy return scalars.

This script launches `srb agent eval` for each checkpoint, reads the eval
TensorBoard scalars, and mirrors a compact set of `Eval / ...` metrics into the
training run directory at the checkpoint step.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


STEP_RE = re.compile(r"agent_(\d+)\.pt$")
DEFAULT_ENV = "marl_waypoint_navigation"
DEFAULT_ALGO = "skrl_mappo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-policy eval for MAPPO checkpoints and log mean test returns."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory, e.g. logs/marl_waypoint_navigation/skrl_mappo/<timestamp>",
    )
    parser.add_argument("--env", default=DEFAULT_ENV, help="SRB env id")
    parser.add_argument("--algo", default=DEFAULT_ALGO, help="SRB algorithm id")
    parser.add_argument("--srb-bin", default="srb", help="SRB executable")
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of vectorized eval environments to run",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help=(
            "Number of agents sharing the team reward. Used to convert skrl "
            "summed return to team return."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help=(
            "Eval timesteps per checkpoint. 0 infers from training logs, "
            "then falls back to 2500."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum checkpoints to evaluate after stride/filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth numbered checkpoint after sorting by step.",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        help="Also evaluate best_agent.pt, logging it at --best-step.",
    )
    parser.add_argument(
        "--best-step",
        type=int,
        default=0,
        help=(
            "TensorBoard step to use for best_agent.pt. 0 uses the latest "
            "numbered checkpoint step."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use mean actions in eval instead of stochastic policy samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints whose Eval / Mean team return already exists at that step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eval commands without launching Isaac Sim.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override to append to each eval command. Repeat as needed.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    match = STEP_RE.match(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(
    run_dir: Path, include_best: bool, best_step: int
) -> list[tuple[int, Path]]:
    checkpoint_dir = run_dir / "checkpoints"
    numbered = sorted(
        (
            (step, path)
            for path in checkpoint_dir.glob("agent_*.pt")
            if (step := checkpoint_step(path)) is not None
        ),
        key=lambda item: item[0],
    )

    if include_best:
        best = checkpoint_dir / "best_agent.pt"
        if best.exists():
            step = best_step or (numbered[-1][0] if numbered else 0)
            numbered.append((step, best))
    return numbered


def scalar_points(
    event_root: Path, tag: str, *, recursive: bool = True
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    pattern = "**/events.out.tfevents.*" if recursive else "events.out.tfevents.*"
    for event_file in event_root.glob(pattern):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            points.extend(
                (event.step, float(event.value)) for event in accumulator.Scalars(tag)
            )
        except Exception:
            continue
    return points


def latest_scalar(event_root: Path, tag: str) -> float | None:
    points = scalar_points(event_root, tag)
    if not points:
        return None
    return sorted(points, key=lambda item: item[0])[-1][1]


def existing_steps(run_dir: Path, tag: str) -> set[int]:
    return {step for step, _ in scalar_points(run_dir, tag, recursive=False)}


def infer_timesteps(run_dir: Path) -> int:
    values = [
        value
        for _, value in scalar_points(
            run_dir, "Episode / Total timesteps (max)", recursive=False
        )
        if value > 0
    ]
    return int(max(values)) if values else 2500


def latest_eval_event_root(run_dir: Path, previous_events: set[Path]) -> Path | None:
    eval_root = run_dir / "eval"
    event_files = set(eval_root.glob("**/events.out.tfevents.*"))
    new_events = event_files - previous_events
    candidates = new_events or event_files
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def command_for_checkpoint(
    *,
    srb_bin: str,
    env: str,
    algo: str,
    checkpoint: Path,
    episodes: int,
    timesteps: int,
    stochastic: bool,
    extra_overrides: Iterable[str],
) -> list[str]:
    return [
        srb_bin,
        "agent",
        "eval",
        "--algo",
        algo,
        "--env",
        env,
        "--model",
        checkpoint.as_posix(),
        "--headless",
        f"env.num_envs={episodes}",
        f"agent.trainer.timesteps={timesteps}",
        f"agent.trainer.stochastic_evaluation={str(stochastic).lower()}",
        f"agent.agent.experiment.write_interval={timesteps}",
        "agent.agent.experiment.checkpoint_interval=0",
        "agent.agent.experiment.wandb=false",
        *extra_overrides,
    ]


def write_eval_scalars(
    *,
    writer: SummaryWriter,
    step: int,
    eval_event_root: Path,
    num_agents: int,
) -> dict[str, float]:
    raw_return = latest_scalar(eval_event_root, "Reward / Total reward (mean)")
    if raw_return is None:
        raise RuntimeError(f"No eval total return scalar found in {eval_event_root}")

    metrics = {
        "Eval / Mean skrl total return": raw_return,
        "Eval / Mean team return": raw_return / num_agents,
    }

    optional_tags = {
        "Episode / Total timesteps (mean)": "Eval / Mean episode length",
        "Info / Termination / success rate current": "Eval / Success rate current",
        "Info / Termination / rollover rate current": "Eval / Rollover rate current",
        "Info / Termination / timeout rate current": "Eval / Timeout rate current",
        "Info / Debug / Goal reached rate / explorers": "Eval / Goal reached rate explorers",
    }
    for source, target in optional_tags.items():
        value = latest_scalar(eval_event_root, source)
        if value is not None:
            metrics[target] = value

    for tag, value in metrics.items():
        writer.add_scalar(tag, value, step)
    writer.flush()
    return metrics


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_agents <= 0:
        raise ValueError("--num-agents must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    timesteps = args.timesteps or infer_timesteps(run_dir)
    checkpoints = discover_checkpoints(run_dir, args.include_best, args.best_step)
    checkpoints = checkpoints[:: args.stride]
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise FileNotFoundError(
            f"No numbered checkpoints found under {run_dir / 'checkpoints'}"
        )

    already_logged = (
        existing_steps(run_dir, "Eval / Mean team return")
        if args.skip_existing
        else set()
    )
    writer = SummaryWriter(log_dir=run_dir)
    try:
        for step, checkpoint in checkpoints:
            if step in already_logged:
                print(f"[skip] step={step} checkpoint={checkpoint}")
                continue

            previous_events = set((run_dir / "eval").glob("**/events.out.tfevents.*"))
            command = command_for_checkpoint(
                srb_bin=args.srb_bin,
                env=args.env,
                algo=args.algo,
                checkpoint=checkpoint,
                episodes=args.episodes,
                timesteps=timesteps,
                stochastic=not args.deterministic,
                extra_overrides=args.extra_override,
            )
            print("[eval]", " ".join(command))
            if args.dry_run:
                continue

            subprocess.run(command, check=True)
            eval_event_root = latest_eval_event_root(run_dir, previous_events)
            if eval_event_root is None:
                raise RuntimeError(
                    f"No eval TensorBoard event file was produced for {checkpoint}"
                )
            metrics = write_eval_scalars(
                writer=writer,
                step=step,
                eval_event_root=eval_event_root,
                num_agents=args.num_agents,
            )
            print(
                f"[logged] step={step} team_return={metrics['Eval / Mean team return']:.6g} "
                f"eval_dir={eval_event_root}"
            )
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
