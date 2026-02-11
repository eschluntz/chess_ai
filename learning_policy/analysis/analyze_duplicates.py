"""
Analyze duplicate FEN distribution in the raw Lichess dataset.

Streams N rows, groups by FEN, and reports:
- Rows per FEN distribution
- Unique depths and sessions per FEN
- Move agreement/disagreement across sessions
- Depth distribution and spread

Usage:
    python analyze_duplicates.py                    # 500k rows
    python analyze_duplicates.py --num-rows 2M
"""

import math
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

    # Group by FEN
    fen_data = defaultdict(list)
    t0 = time.perf_counter()
    for i, row in enumerate(ds):
        if i >= num_rows:
            break
        if i % 100_000 == 0 and i > 0:
            print(f"  {i:,}...")
        fen_data[row["fen"]].append({
            "depth": row["depth"],
            "knodes": row["knodes"],
            "line": row["line"],
            "cp": row["cp"],
            "mate": row["mate"],
        })

    n_unique = len(fen_data)
    elapsed = time.perf_counter() - t0
    print(f"Loaded in {elapsed:.1f}s\n")

    # ---- Basic stats ----
    print(f"Total rows:   {num_rows:,}")
    print(f"Unique FENs:  {n_unique:,}")
    print(f"Rows per FEN: {num_rows / n_unique:.2f} avg")

    rows_per_fen = np.array([len(v) for v in fen_data.values()])
    depths_per_fen = np.array([len(set(r["depth"] for r in rows)) for rows in fen_data.values()])
    sessions_per_fen = np.array([len(set((r["depth"], r["knodes"]) for r in rows)) for rows in fen_data.values()])
    moves_per_fen = np.array([len(set(r["line"].partition(" ")[0] for r in rows if r["line"])) for rows in fen_data.values()])

    # ---- Print distributions ----
    for name, arr in [
        ("Rows per FEN", rows_per_fen),
        ("Unique depths per FEN", depths_per_fen),
        ("Unique sessions per FEN", sessions_per_fen),
        ("Unique best-moves per FEN", moves_per_fen),
    ]:
        counts = Counter(arr)
        print(f"\n--- {name} ---")
        for val in sorted(counts.keys())[:12]:
            pct = counts[val] / n_unique * 100
            print(f"  {val:3d}: {counts[val]:>7,} FENs ({pct:5.1f}%)")
        rest = sum(v for k, v in counts.items() if k > 12)
        if rest:
            print(f"  >12: {rest:>7,} FENs ({rest / n_unique * 100:5.1f}%)")

    # ---- Multi-session agreement ----
    multi_fens = {fen: rows for fen, rows in fen_data.items()
                  if len(set((r["depth"], r["knodes"]) for r in rows)) > 1}
    print(f"\n--- Agreement among {len(multi_fens):,} multi-session FENs ---")

    agree = disagree = 0
    for rows in multi_fens.values():
        sessions = defaultdict(list)
        for r in rows:
            sessions[(r["depth"], r["knodes"])].append(r)

        best_moves = set()
        for srows in sessions.values():
            def score(r):
                if r["mate"] is not None:
                    return 10000 if r["mate"] > 0 else -10000
                return r["cp"] if r["cp"] is not None else 0
            best = max(srows, key=score)
            move = best["line"].partition(" ")[0] if best["line"] else ""
            best_moves.add(move)

        if len(best_moves) == 1:
            agree += 1
        else:
            disagree += 1

    print(f"  Agree:    {agree:,} ({agree / len(multi_fens) * 100:.1f}%)")
    print(f"  Disagree: {disagree:,} ({disagree / len(multi_fens) * 100:.1f}%)")

    # ---- Depth distribution ----
    all_depths = np.array([r["depth"] for rows in fen_data.values() for r in rows if r["depth"]])
    print(f"\n--- Depth distribution (all rows) ---")
    print(f"  min={all_depths.min()}, max={all_depths.max()}, mean={all_depths.mean():.1f}, median={np.median(all_depths):.0f}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(all_depths, p):.0f}")

    # ---- Max depth per FEN ----
    max_depth_per_fen = np.array([max(r["depth"] for r in rows if r["depth"]) for rows in fen_data.values()
                                  if any(r["depth"] for r in rows)])
    print(f"\n--- Max depth per FEN ---")
    print(f"  mean={max_depth_per_fen.mean():.1f}, median={np.median(max_depth_per_fen):.0f}")
    for p in [25, 50, 75, 90]:
        print(f"  p{p}: {np.percentile(max_depth_per_fen, p):.0f}")

    # ---- Depth spread in multi-session FENs ----
    depth_spreads = np.array([
        max(r["depth"] for r in rows if r["depth"]) - min(r["depth"] for r in rows if r["depth"])
        for rows in multi_fens.values()
        if any(r["depth"] for r in rows)
    ])
    print(f"\n--- Depth spread (max-min) in multi-session FENs ---")
    print(f"  mean={depth_spreads.mean():.1f}, median={np.median(depth_spreads):.0f}")
    for p in [25, 50, 75, 90]:
        print(f"  p{p}: {np.percentile(depth_spreads, p):.0f}")

    # ---- Figures ----
    IMG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Rows per FEN histogram
    ax = axes[0, 0]
    clipped = np.clip(rows_per_fen, 1, 20)
    ax.hist(clipped, bins=range(1, 22), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Rows per FEN")
    ax.set_ylabel("Count")
    ax.set_title(f"Rows per FEN (n={n_unique:,})")
    ax.set_xticks(range(1, 21, 2))

    # 2. Unique depths per FEN
    ax = axes[0, 1]
    clipped = np.clip(depths_per_fen, 1, 15)
    ax.hist(clipped, bins=range(1, 17), edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Unique depths per FEN")
    ax.set_ylabel("Count")
    ax.set_title("Unique depths per FEN")

    # 3. Unique sessions per FEN
    ax = axes[0, 2]
    clipped = np.clip(sessions_per_fen, 1, 15)
    ax.hist(clipped, bins=range(1, 17), edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("Sessions per FEN")
    ax.set_ylabel("Count")
    ax.set_title("Sessions (depth, knodes) per FEN")

    # 4. Unique best moves per FEN
    ax = axes[1, 0]
    clipped = np.clip(moves_per_fen, 1, 10)
    ax.hist(clipped, bins=range(1, 12), edgecolor="black", alpha=0.7, color="red")
    ax.set_xlabel("Unique best moves per FEN")
    ax.set_ylabel("Count")
    ax.set_title("Distinct recommended moves per FEN")

    # 5. Depth distribution (all rows)
    ax = axes[1, 1]
    ax.hist(all_depths, bins=50, edgecolor="black", alpha=0.7, color="purple")
    ax.set_xlabel("Search depth")
    ax.set_ylabel("Count")
    ax.set_title("Depth distribution (all rows)")

    # 6. Depth spread in multi-session FENs
    ax = axes[1, 2]
    clipped = np.clip(depth_spreads, 0, 30)
    ax.hist(clipped, bins=30, edgecolor="black", alpha=0.7, color="teal")
    ax.set_xlabel("Depth spread (max - min)")
    ax.set_ylabel("Count")
    ax.set_title(f"Depth spread in multi-session FENs (n={len(multi_fens):,})")

    fig.suptitle(f"Duplicate FEN analysis ({num_rows:,} rows)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = IMG_DIR / "duplicate_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    import fire
    fire.Fire(analyze)
