"""
Scatter plots of cp vs depth for individual FENs.

Each point is one row from the dataset. Colored by session (depth, knodes),
labeled with the recommended move. Helps visualize how analyses interact.

Usage:
    python analyze_cp_depth.py
    python analyze_cp_depth.py --num-rows 1M --min-sessions 3
"""

import time
from collections import defaultdict
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


def analyze(num_rows: str = "500k", min_sessions: int = 3, num_plots: int = 12):
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
        if row["cp"] is None:
            continue
        fen_data[row["fen"]].append({
            "depth": row["depth"],
            "knodes": row["knodes"],
            "cp": row["cp"],
            "move": row["line"].partition(" ")[0] if row["line"] else "?",
        })
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    # Find FENs with enough sessions and data points to be interesting
    candidates = []
    for fen, rows in fen_data.items():
        sessions = set((r["depth"], r["knodes"]) for r in rows)
        unique_moves = set(r["move"] for r in rows)
        if len(sessions) >= min_sessions and len(unique_moves) >= 2 and len(rows) >= 5:
            # Score by number of data points and session diversity
            cp_spread = max(r["cp"] for r in rows) - min(r["cp"] for r in rows)
            candidates.append((len(rows) * len(sessions) * cp_spread, fen))

    candidates.sort(reverse=True)
    print(f"Found {len(candidates):,} FENs with {min_sessions}+ sessions and 2+ moves")

    # Pick evenly spaced candidates for variety
    step = max(1, len(candidates) // num_plots)
    selected = [candidates[i * step] for i in range(min(num_plots, len(candidates)))]

    cols = 3
    rows_grid = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(7 * cols, 5.5 * rows_grid))
    axes = np.array(axes).flatten()

    for idx, (_, fen) in enumerate(selected):
        ax = axes[idx]
        rows = fen_data[fen]

        # Assign colors by session
        sessions = sorted(set((r["depth"], r["knodes"]) for r in rows))
        session_colors = {}
        cmap = plt.cm.tab10
        for si, sess in enumerate(sessions):
            session_colors[sess] = cmap(si % 10)

        # Plot each point
        for r in rows:
            sess = (r["depth"], r["knodes"])
            color = session_colors[sess]
            ax.scatter(r["depth"], r["cp"], color=color, s=50, zorder=3, edgecolors="black", linewidths=0.5)
            ax.annotate(
                r["move"], (r["depth"], r["cp"]),
                fontsize=6, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points",
                color=color,
            )

        # Legend: one entry per session
        for sess in sessions:
            color = session_colors[sess]
            ax.scatter([], [], color=color, label=f"d={sess[0]} kn={sess[1]}", edgecolors="black", linewidths=0.5)
        ax.legend(fontsize=6, loc="best", title="session", title_fontsize=7)

        ax.set_xlabel("depth")
        ax.set_ylabel("cp")
        # Truncate FEN for title
        fen_short = fen.split(" ")[0]
        if len(fen_short) > 40:
            fen_short = fen_short[:40] + "..."
        ax.set_title(fen_short, fontsize=8, fontfamily="monospace")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"cp vs depth per FEN — colored by session (depth, knodes)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    IMG_DIR.mkdir(exist_ok=True)
    out = IMG_DIR / "cp_vs_depth.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    import fire
    fire.Fire(analyze)
