"""
Analyze the deepest session per FEN: how many moves does it return?

Usage:
    python analyze_deepest_session.py
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

IMG_DIR = Path(__file__).parent / "img"


def parse_num(s: str) -> int:
    s = s.lower()
    if s.endswith("m"):
        return int(s[:-1]) * 1_000_000
    if s.endswith("k"):
        return int(s[:-1]) * 1_000
    return int(s)


def analyze(num_rows: str = "500k"):
    num_rows = parse_num(num_rows)
    print(f"Streaming {num_rows:,} rows...")
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    fen_data = defaultdict(list)
    t0 = time.perf_counter()
    for i, row in enumerate(ds):
        if i >= num_rows:
            break
        if i % 100_000 == 0 and i > 0:
            print(f"  {i:,}...")
        fen_data[row["fen"]].append({
            "depth": row["depth"] or 1,
            "knodes": row["knodes"] or 0,
            "cp": row["cp"],
            "mate": row["mate"],
            "move": row["line"].partition(" ")[0] if row["line"] else None,
        })
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    n_unique = len(fen_data)
    print(f"{num_rows:,} rows -> {n_unique:,} unique FENs\n")

    # For each FEN: find deepest session, count its moves
    deepest_moves = []        # moves in deepest session
    all_session_moves = []    # moves across ALL sessions
    deepest_depth = []
    deepest_is_single = []    # True if deepest session has 1 move

    for fen, rows in fen_data.items():
        # Group by session
        sessions = defaultdict(list)
        for r in rows:
            sessions[(r["depth"], r["knodes"])].append(r)

        # Find deepest session
        best_key = max(sessions, key=lambda k: k[0])
        best_rows = sessions[best_key]
        unique_moves_deepest = set(r["move"] for r in best_rows if r["move"])
        unique_moves_all = set(r["move"] for r in rows if r["move"])

        deepest_moves.append(len(unique_moves_deepest))
        all_session_moves.append(len(unique_moves_all))
        deepest_depth.append(best_key[0])
        deepest_is_single.append(len(unique_moves_deepest) == 1)

    deepest_moves = np.array(deepest_moves)
    all_session_moves = np.array(all_session_moves)
    deepest_depth = np.array(deepest_depth)

    # Print distributions
    print("--- Moves in DEEPEST session per FEN ---")
    counts = Counter(deepest_moves)
    for val in sorted(counts.keys())[:10]:
        pct = counts[val] / n_unique * 100
        print(f"  {val:2d} moves: {counts[val]:>7,} FENs ({pct:5.1f}%)")

    print(f"\n--- Moves across ALL sessions per FEN ---")
    counts = Counter(all_session_moves)
    for val in sorted(counts.keys())[:10]:
        pct = counts[val] / n_unique * 100
        print(f"  {val:2d} moves: {counts[val]:>7,} FENs ({pct:5.1f}%)")

    # How much signal do you lose by keeping only deepest?
    kept = deepest_moves.sum()
    total = all_session_moves.sum()
    print(f"\n--- Signal retention ---")
    print(f"  Total move entries (all sessions): {total:,}")
    print(f"  Kept (deepest only):               {kept:,} ({kept/total*100:.1f}%)")
    print(f"  FENs where deepest has 1 move:     {deepest_is_single.count(True):,} ({deepest_is_single.count(True)/n_unique*100:.1f}%)")

    # Does depth correlate with multi-PV count?
    print(f"\n--- Avg moves in deepest session by depth bucket ---")
    buckets = [(1, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 100), (100, 300)]
    for lo, hi in buckets:
        mask = (deepest_depth >= lo) & (deepest_depth < hi)
        if mask.sum() == 0:
            continue
        avg = deepest_moves[mask].mean()
        n = mask.sum()
        pct_single = (deepest_moves[mask] == 1).sum() / n * 100
        print(f"  depth {lo:3d}-{hi:3d}: avg {avg:.2f} moves, {pct_single:.0f}% single-move  (n={n:,})")

    # Figure
    IMG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    clipped = np.clip(deepest_moves, 1, 10)
    ax.hist(clipped, bins=range(1, 12), edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Moves in deepest session")
    ax.set_ylabel("Count")
    ax.set_title(f"Moves per FEN (deepest session only)")

    ax = axes[1]
    clipped = np.clip(all_session_moves, 1, 15)
    ax.hist(clipped, bins=range(1, 17), edgecolor="black", alpha=0.7, color="coral")
    ax.set_xlabel("Moves across all sessions")
    ax.set_ylabel("Count")
    ax.set_title(f"Moves per FEN (all sessions)")

    ax = axes[2]
    # scatter: depth vs moves in deepest session (jittered)
    jitter = np.random.default_rng(42).normal(0, 0.15, len(deepest_moves))
    ax.scatter(deepest_depth, deepest_moves + jitter, alpha=0.05, s=3, color="steelblue")
    ax.set_xlabel("Depth of deepest session")
    ax.set_ylabel("Moves in that session")
    ax.set_title("Depth vs multi-PV count")
    ax.set_ylim(0, 12)

    fig.suptitle(f"Deepest session analysis ({n_unique:,} FENs from {num_rows:,} rows)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = IMG_DIR / "deepest_session.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    import fire
    fire.Fire(analyze)
