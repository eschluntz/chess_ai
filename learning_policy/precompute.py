"""
Precompute chess dataset with soft labels from aggregated Stockfish analyses.

Groups positions by FEN, then by analysis session (depth, knodes), and creates
probability distributions over moves using max-normalized cp scores.

See readme.md for full documentation of the weighting strategy.

Storage format:
    planes.npy:        (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta.npy:          (N, 5) int8 - [turn, K, Q, k, q]
    label_indices.npy: (total_nonzero,) uint16 - move vocab indices
    label_probs.npy:   (total_nonzero,) float16 - probabilities
    label_offsets.npy: (N + 1,) uint32 - position boundaries

Usage:
    python precompute.py --num-samples 1M
    modal run --detach precompute.py::precompute_modal --num-samples 50M
"""

import math
import time
from collections import defaultdict
from pathlib import Path

import modal
import numpy as np
from tqdm import tqdm

app = modal.App("chess-precompute")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "datasets", "tqdm", "torch")
    .add_local_file("board_repr.py", "/root/board_repr.py")
)

data_volume = modal.Volume.from_name("chess-policy-data", create_if_missing=True)

STREAMING_THRESHOLD = 5_000_000
TEMPERATURE = 50  # centipawns - controls spread within session


def format_count(n: int) -> str:
    """Format sample count for folder name: 1000000 -> '1M'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    elif n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def parse_num_samples(ns: str) -> int | str:
    """Parse num_samples string: '1M' -> 1000000, 'full' -> 'full'."""
    ns = ns.lower()
    if ns == "full":
        return "full"
    elif ns.endswith("m"):
        return int(ns[:-1]) * 1_000_000
    elif ns.endswith("k"):
        return int(ns[:-1]) * 1_000
    return int(ns)


def mate_to_score(mate_in_n: int) -> float:
    """Convert mate-in-N to a score. Mate in 1 = 10k, mate in 20+ = 1k."""
    n = abs(mate_in_n)
    if n >= 20:
        score = 1000
    else:
        score = 10000 - 9000 * (n - 1) / 19
    return score if mate_in_n > 0 else -score


def analysis_to_score(cp: int | None, mate: int | None) -> float | None:
    """Convert cp/mate to comparable score."""
    if mate is not None:
        return mate_to_score(mate)
    elif cp is not None:
        return float(cp)
    return None


def compute_soft_labels(analyses: list[tuple], is_white_to_move: bool, uci_to_move_tuple) -> dict[tuple, float]:
    """
    Compute soft label probabilities from multiple analyses.

    Args:
        analyses: List of (move, depth, knodes, cp, mate) tuples
        is_white_to_move: True if white to move
        uci_to_move_tuple: Function to convert UCI to (from_sq, to_sq, promo)

    Returns:
        Dict mapping move tuple (from_sq, to_sq, promo) to probability
    """
    # Group by session (depth, knodes)
    sessions = defaultdict(list)
    for move_uci, depth, knodes, cp, mate in analyses:
        score = analysis_to_score(cp, mate)
        if score is None:
            continue
        if not is_white_to_move:
            score = -score
        key = (depth, knodes)
        move = uci_to_move_tuple(move_uci)
        sessions[key].append((move, score, depth))

    if not sessions:
        return {}

    # For each move, find max(depth * session_weight) across sessions
    move_weights = {}

    for (depth, knodes), moves_scores in sessions.items():
        # Max-normalize within session
        max_score = max(s for _, s, _ in moves_scores)

        for move, score, d in moves_scores:
            delta = max_score - score
            session_weight = math.exp(-delta / TEMPERATURE)
            weight = d * session_weight

            if move not in move_weights or weight > move_weights[move]:
                move_weights[move] = weight

    # Normalize to probabilities
    total = sum(move_weights.values())
    return {move: w / total for move, w in move_weights.items()}


def _precompute(num_samples: str | int, output_dir: Path, eval_frac: float = 0.0002):
    """Core precompute logic with soft labels."""
    from datasets import load_dataset

    from board_repr import fen_to_planes, uci_to_move_tuple

    parsed = parse_num_samples(num_samples) if isinstance(num_samples, str) else num_samples
    use_full = parsed == "full"

    if use_full:
        print("Loading full dataset...")
        ds = load_dataset("Lichess/chess-position-evaluations", split="train")
        num_samples = len(ds)
        folder_name = "full"
    elif parsed < STREAMING_THRESHOLD:
        num_samples = parsed
        folder_name = format_count(num_samples)
        print(f"Loading dataset (streaming {num_samples:,} samples)...")
        ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
    else:
        num_samples = parsed
        folder_name = format_count(num_samples)
        print(f"Loading dataset ({num_samples:,} samples)...")
        ds = load_dataset("Lichess/chess-position-evaluations", split="train")
        total = len(ds)
        if num_samples < total:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(total, size=num_samples, replace=False)
            indices = np.sort(indices)
            ds = ds.select(indices)

    # First pass: group by FEN (batched for speed, tuples for memory)
    print("Grouping by FEN...")
    fen_analyses = defaultdict(list)

    t0 = time.perf_counter()
    batch_size = 10000
    rows_processed = 0

    for batch in tqdm(ds.iter(batch_size=batch_size), total=(num_samples + batch_size - 1) // batch_size):
        fens = batch["fen"]
        lines = batch["line"]
        depths = batch["depth"]
        knodess = batch["knodes"]
        cps = batch["cp"]
        mates = batch["mate"]

        for i in range(len(fens)):
            if rows_processed >= num_samples:
                break
            rows_processed += 1

            line = lines[i]
            if not line:
                continue

            fen_analyses[fens[i]].append((
                line.partition(' ')[0],
                depths[i] or 1,
                knodess[i] or 0,
                cps[i],
                mates[i],
            ))

        if rows_processed >= num_samples:
            break

    n_unique = len(fen_analyses)
    elapsed = time.perf_counter() - t0
    print(f"Grouped {num_samples:,} rows into {n_unique:,} unique positions in {elapsed:.1f}s")
    print(f"Dedup ratio: {num_samples / n_unique:.2f}x")

    # Load existing vocab
    vocab_path = Path(__file__).parent / "cache" / "planes" / "vocab.npy"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found at {vocab_path}. Run build_vocab.py first.")
    vocab_moves = np.load(vocab_path)
    move_to_idx = {tuple(m): i for i, m in enumerate(vocab_moves)}
    print(f"Vocab size: {len(move_to_idx)}")

    # Second pass: compute soft labels and convert to arrays
    print("Computing soft labels...")

    # Pre-allocate arrays
    planes = np.zeros((n_unique, 13, 8, 8), dtype=np.uint8)
    meta = np.zeros((n_unique, 5), dtype=np.int8)

    # Sparse label storage
    all_indices = []
    all_probs = []
    offsets = [0]

    fens = list(fen_analyses.keys())
    rng = np.random.default_rng(seed=43)
    rng.shuffle(fens)

    for i, fen in enumerate(tqdm(fens)):
        # Convert board
        p, m = fen_to_planes(fen)
        planes[i] = p
        meta[i] = m

        # Compute soft labels
        is_white = m[0] == 1  # meta[0] is turn
        probs = compute_soft_labels(fen_analyses[fen], is_white, uci_to_move_tuple)

        # Convert to sparse format
        for move, prob in sorted(probs.items()):
            idx = move_to_idx.get(move)
            if idx is not None:
                all_indices.append(idx)
                all_probs.append(prob)

        offsets.append(len(all_indices))

    # Convert to numpy
    label_indices = np.array(all_indices, dtype=np.uint16)
    label_probs = np.array(all_probs, dtype=np.float16)
    label_offsets = np.array(offsets, dtype=np.uint32)

    print(f"Sparse labels: {len(label_indices):,} total entries")
    print(f"Average moves per position: {len(label_indices) / n_unique:.2f}")

    # Split train/eval
    eval_size = int(n_unique * eval_frac)
    train_size = n_unique - eval_size
    print(f"Train: {train_size:,}, Eval: {eval_size:,}")

    # Create output directory
    output_dir = output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_split(prefix: str, start: int, end: int):
        np.save(output_dir / f"{prefix}_planes.npy", planes[start:end])
        np.save(output_dir / f"{prefix}_meta.npy", meta[start:end])

        # Sparse labels
        label_start = label_offsets[start]
        label_end = label_offsets[end]
        np.save(output_dir / f"{prefix}_label_indices.npy", label_indices[label_start:label_end])
        np.save(output_dir / f"{prefix}_label_probs.npy", label_probs[label_start:label_end])
        new_offsets = label_offsets[start:end + 1] - label_start
        np.save(output_dir / f"{prefix}_label_offsets.npy", new_offsets)

    print("Saving eval...")
    save_split("eval", 0, eval_size)

    print("Saving train...")
    save_split("train", eval_size, n_unique)

    # Save stats
    stats = {
        "raw_samples": num_samples,
        "unique_positions": n_unique,
        "dedup_ratio": num_samples / n_unique,
        "train_size": train_size,
        "eval_size": eval_size,
        "total_label_entries": len(label_indices),
        "avg_moves_per_position": len(label_indices) / n_unique,
        "temperature": TEMPERATURE,
    }
    np.save(output_dir / "stats.npy", stats)

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npy"))
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print(f"Stats: {stats}")
    print("Done!")


def precompute_local(num_samples: str = "1M", output_dir: str = "cache/planes"):
    """Run precompute locally."""
    _precompute(num_samples, Path(output_dir))


@app.function(
    image=image,
    volumes={"/root/cache": data_volume},
    timeout=86400,
    memory=128000,
)
def precompute_modal(num_samples: str = "50M"):
    """Run precompute on Modal."""
    import sys
    sys.path.insert(0, "/root")
    _precompute(num_samples, Path("/root/cache/planes"))
    data_volume.commit()


if __name__ == "__main__":
    import fire
    fire.Fire(precompute_local)
