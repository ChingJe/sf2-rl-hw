from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sf2-rl-hw",
        description="Utilities for the Street Fighter II RL course project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    eval_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override.",
    )

    record_parser = subparsers.add_parser("record", help="Record a policy video.")
    record_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    record_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Single checkpoint to record.",
    )
    record_parser.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Record the latest N checkpoints for this experiment.",
    )
    record_parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default=None,
        help="Glob pattern for batch checkpoint selection.",
    )

    capture_parser = subparsers.add_parser(
        "capture-state",
        help="Run a checkpoint until the next round starts and save a new .state file.",
    )
    capture_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baseline.yaml"),
        help="Path to the experiment config.",
    )
    capture_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override.",
    )
    capture_parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output state name without extension. Defaults to '<current_state>.Round2Start'.",
    )
    capture_parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum agent steps to wait before giving up.",
    )
    capture_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game window while capturing the next-round state.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from .train import run_training

        run_training(args.config)
        return

    if args.command == "evaluate":
        from .evaluate import run_evaluation

        run_evaluation(args.config, args.checkpoint)
        return

    if args.command == "record":
        from .record import run_recording

        run_recording(args.config, args.checkpoint, args.latest, args.glob_pattern)
        return

    if args.command == "capture-state":
        from .capture_state import run_capture_state

        run_capture_state(
            args.config,
            checkpoint_override=args.checkpoint,
            output_name=args.output_name,
            max_steps=args.max_steps,
            render=args.render,
        )
        return

    parser.error(f"Unsupported command: {args.command}")
