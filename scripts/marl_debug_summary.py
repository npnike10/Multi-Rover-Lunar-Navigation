#!/usr/bin/env python3
"""Summarize MARL waypoint TensorBoard scalars and checkpoint inventory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAG_PATTERNS = (
    r"^Reward /",
    r"^Episode /",
    r"^Policy / Standard deviation",
    r"^Loss / (Policy|Value|Entropy)",
    r"^Info / Debug /",
    r"^Info / Episode /",
    r"^Info / RewardComponents /",
    r"^Info / Termination /",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print scalar samples from a MARL waypoint run directory. "
            "Use this after training/eval to inspect learning trends quickly."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory, e.g. logs/marl_waypoint_navigation/skrl_mappo/<timestamp>",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Also summarize event files under eval/ subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Additional regex for scalar tags to print. Can be passed multiple times. "
            "Defaults already include rewards, losses, std, and MARL debug metrics."
        ),
    )
    return parser.parse_args()


def event_files(run_dir: Path, include_eval: bool) -> list[Path]:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    if include_eval:
        files.extend(sorted(run_dir.glob("eval/**/events.out.tfevents*")))
    return files


def matching_tags(tags: list[str], patterns: list[str]) -> list[str]:
    regexes = [re.compile(pattern) for pattern in patterns]
    return [tag for tag in tags if any(regex.search(tag) for regex in regexes)]


def sample_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    return sorted({0, length // 4, length // 2, 3 * length // 4, length - 1})


def print_scalar_summary(event_file: Path, patterns: list[str]) -> None:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = matching_tags(accumulator.Tags().get("scalars", []), patterns)
    if not tags:
        print(f"\n{event_file}: no matching scalar tags")
        return

    print(f"\n{event_file}")
    for tag in tags:
        values = accumulator.Scalars(tag)
        samples = []
        for index in sample_indices(len(values)):
            event = values[index]
            samples.append(f"{event.step}:{event.value:.6g}")
        print(f"  {tag} [{len(values)}] " + ", ".join(samples))


def print_checkpoint_summary(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print("\ncheckpoints: none")
        return

    numbered = []
    for path in checkpoint_dir.glob("agent_*.pt"):
        match = re.fullmatch(r"agent_(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()

    print("\ncheckpoints:")
    if (checkpoint_dir / "best_agent.pt").exists():
        print(f"  best: {checkpoint_dir / 'best_agent.pt'}")
    if numbered:
        print(f"  count: {len(numbered)}")
        print(f"  first: {numbered[0][1]}")
        print(f"  latest: {numbered[-1][1]}")
        if len(numbered) > 2:
            mid = numbered[len(numbered) // 2][1]
            print(f"  middle: {mid}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    patterns = [*DEFAULT_TAG_PATTERNS, *args.pattern]

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    files = event_files(run_dir, args.include_eval)
    if not files:
        print(f"No event files found under {run_dir}")
    for event_file in files:
        print_scalar_summary(event_file, patterns)

    print_checkpoint_summary(run_dir)


if __name__ == "__main__":
    main()
