#!/usr/bin/env python
"""CLI for running head-to-head series between policy and/or negamax engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
LEARNING_POLICY_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(LEARNING_POLICY_DIR))
sys.path.insert(0, str(REPO_ROOT))

from interface.match_runner import (  # noqa: E402
    ConfigError,
    EngineFactory,
    default_vocab_path,
    normalize_engine_config,
    play_match,
    run_series,
)


def _engine_from_args(args: argparse.Namespace, side: str) -> dict[str, object]:
    kind = getattr(args, f"{side}_kind")
    temp = float(getattr(args, f"{side}_temp"))

    if kind == "policy":
        checkpoint = getattr(args, f"{side}_checkpoint")
        if not checkpoint:
            raise ConfigError(f"--{side}-checkpoint is required when --{side}-kind=policy")
        return {
            "kind": "policy",
            "checkpoint": checkpoint,
            "temperature": temp,
            "name": getattr(args, f"{side}_name") or Path(checkpoint).stem,
        }

    return {
        "kind": "negamax",
        "eval_name": getattr(args, f"{side}_eval"),
        "depth": int(getattr(args, f"{side}_depth")),
        "temperature": temp,
        "name": (
            getattr(args, f"{side}_name")
            or f"negamax-{getattr(args, f'{side}_eval')}-d{getattr(args, f'{side}_depth')}"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Head-to-head chess win-rate CLI")
    default_checkpoint = "/Users/erik/code/chess_ai/learning_policy/checkpoints/scale-40M.pt"

    parser.add_argument("--vocab", type=Path, default=default_vocab_path())
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--swap-colors", action="store_true", default=True)
    parser.add_argument("--no-swap-colors", action="store_false", dest="swap_colors")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--start-fen", type=str, default="startpos")
    parser.add_argument(
        "--min-arrow-probability",
        type=float,
        default=0.2,
        help="Only used when --save-trace is set",
    )
    parser.add_argument(
        "--save-trace",
        type=Path,
        default=None,
        help="Optional path to write a JSON trace for one game",
    )

    for side in ["white", "black"]:
        parser.add_argument(f"--{side}-kind", choices=["policy", "negamax"], default="policy")
        parser.add_argument(f"--{side}-name", type=str, default="")
        parser.add_argument(f"--{side}-temp", type=float, default=0.0)

        parser.add_argument(f"--{side}-checkpoint", type=str, default=default_checkpoint)
        parser.add_argument(f"--{side}-eval", choices=["piece_value", "piece_position"], default="piece_value")
        parser.add_argument(f"--{side}-depth", type=int, default=2)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    vocab_path = args.vocab.expanduser().resolve()
    if not vocab_path.exists():
        print(f"Error: vocab not found at {vocab_path}")
        return 1

    start_fen = args.start_fen
    if start_fen == "startpos":
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    rng = np.random.default_rng(args.seed)

    try:
        white_cfg = normalize_engine_config(_engine_from_args(args, "white"), "white")
        black_cfg = normalize_engine_config(_engine_from_args(args, "black"), "black")

        factory = EngineFactory(vocab_path=vocab_path, device=args.device)
        white = factory.create(white_cfg)
        black = factory.create(black_cfg)

        summary = run_series(
            white,
            black,
            games=args.games,
            swap_colors=args.swap_colors,
            max_plies=args.max_plies,
            min_arrow_probability=args.min_arrow_probability,
            start_fen=start_fen,
            rng=rng,
        )

        print("=" * 72)
        print("Head-to-Head Summary")
        print("=" * 72)
        print(f"White config: {white.kind} ({white.name}), temp={white.temperature}")
        print(f"Black config: {black.kind} ({black.name}), temp={black.temperature}")
        print(f"Games: {args.games} | Swap colors: {args.swap_colors}")
        print()

        for name, stats in summary["by_model"].items():
            print(
                f"{name}: W={stats['wins']} L={stats['losses']} D={stats['draws']} "
                f"| win_rate={stats['win_rate'] * 100:.1f}%"
            )

        if args.save_trace:
            trace_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
            trace = play_match(
                white,
                black,
                max_plies=args.max_plies,
                min_arrow_probability=args.min_arrow_probability,
                start_fen=start_fen,
                rng=trace_rng,
                collect_trace=True,
            )
            path = args.save_trace.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trace, indent=2))
            print()
            print(f"Saved one match trace to: {path}")

    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
