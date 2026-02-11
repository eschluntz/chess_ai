"""
Precompute chess dataset: board planes + raw Stockfish analyses per position.

Groups positions by FEN and stores all analysis rows. Label computation
(soft labels, deepest-only, etc.) happens at train time in data.py.

Storage format:
    planes.npy:           (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta.npy:             (N, 5) int8 - [turn, K, Q, k, q]
    analysis_data.npy:    (total_analyses, 5) int32 - [move_idx, depth, knodes, cp_or_val, is_mate]
    analysis_offsets.npy: (N + 1,) uint32 - position boundaries

Usage:
    python precompute.py --num-samples 1M
    modal run --detach precompute.py::precompute_modal --num-samples 50M
"""

import pickle
import shutil
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


def _precompute(num_samples: str | int, output_dir: Path, eval_frac: float = 0.0002, volume=None):
    """Core precompute logic: board planes + raw analyses."""
    from datasets import load_dataset

    from board_repr import fen_to_planes, uci_to_move_tuple

    parsed = parse_num_samples(num_samples) if isinstance(num_samples, str) else num_samples
    use_full = parsed == "full"

    if use_full:
        folder_name = "full"
    else:
        num_samples = parsed
        folder_name = format_count(num_samples)

    checkpoint_dir = output_dir / folder_name / "_checkpoint"
    fen_checkpoint = checkpoint_dir / "fen_analyses.pkl"

    # Phase 1: group by FEN (or resume from checkpoint)
    if fen_checkpoint.exists():
        print(f"Resuming from phase 1 checkpoint...")
        with open(fen_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        fen_analyses = checkpoint['fen_analyses']
        num_samples = checkpoint['num_samples']
        n_unique = len(fen_analyses)
        print(f"Loaded {n_unique:,} unique positions ({num_samples:,} raw samples)")
    else:
        if use_full:
            print("Loading full dataset...")
            ds = load_dataset("Lichess/chess-position-evaluations", split="train")
            num_samples = len(ds)
        elif parsed < STREAMING_THRESHOLD:
            num_samples = parsed
            print(f"Loading dataset (streaming {num_samples:,} samples)...")
            ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
        else:
            num_samples = parsed
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

        # Checkpoint phase 1 so preemption doesn't lose this work
        print("Saving phase 1 checkpoint...")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(fen_checkpoint, 'wb') as f:
            pickle.dump({'fen_analyses': dict(fen_analyses), 'num_samples': num_samples}, f)
        if volume:
            volume.commit()
        print("Phase 1 checkpoint saved.")

    # Load existing vocab
    vocab_path = Path(__file__).parent / "cache" / "planes" / "vocab.npy"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found at {vocab_path}. Run build_vocab.py first.")
    vocab_moves = np.load(vocab_path)
    move_to_idx = {tuple(m): i for i, m in enumerate(vocab_moves)}
    print(f"Vocab size: {len(move_to_idx)}")

    # Second pass: convert boards and store raw analyses
    print("Converting boards and storing raw analyses...")

    planes = np.zeros((n_unique, 13, 8, 8), dtype=np.uint8)
    meta = np.zeros((n_unique, 5), dtype=np.int8)

    # Raw analysis storage: [move_idx, depth, knodes, cp_or_val, is_mate]
    all_analyses = []
    offsets = [0]

    fens = list(fen_analyses.keys())
    rng = np.random.default_rng(seed=43)
    rng.shuffle(fens)

    skipped_moves = 0

    for i, fen in enumerate(tqdm(fens)):
        p, m = fen_to_planes(fen)
        planes[i] = p
        meta[i] = m

        for move_uci, depth, knodes, cp, mate in fen_analyses[fen]:
            move = uci_to_move_tuple(move_uci)
            idx = move_to_idx.get(move)
            if idx is None:
                skipped_moves += 1
                continue

            if mate is not None:
                all_analyses.append((idx, depth, knodes, mate, 1))
            elif cp is not None:
                all_analyses.append((idx, depth, knodes, cp, 0))
            else:
                skipped_moves += 1

        offsets.append(len(all_analyses))

    analysis_data = np.array(all_analyses, dtype=np.int32)
    analysis_offsets = np.array(offsets, dtype=np.uint32)

    total_analyses = len(analysis_data)
    print(f"Total analyses: {total_analyses:,} ({skipped_moves:,} skipped)")
    print(f"Average analyses per position: {total_analyses / n_unique:.2f}")

    # Split train/eval (minimum 2048 eval samples so we get at least a couple batches)
    eval_size = max(int(n_unique * eval_frac), min(2048, n_unique // 2))
    train_size = n_unique - eval_size
    print(f"Train: {train_size:,}, Eval: {eval_size:,}")

    # Create output directory
    output_dir = output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_split(prefix: str, start: int, end: int):
        np.save(output_dir / f"{prefix}_planes.npy", planes[start:end])
        np.save(output_dir / f"{prefix}_meta.npy", meta[start:end])

        a_start = analysis_offsets[start]
        a_end = analysis_offsets[end]
        np.save(output_dir / f"{prefix}_analysis_data.npy", analysis_data[a_start:a_end])
        new_offsets = analysis_offsets[start:end + 1] - a_start
        np.save(output_dir / f"{prefix}_analysis_offsets.npy", new_offsets)

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
        "total_analyses": total_analyses,
        "avg_analyses_per_position": total_analyses / n_unique,
    }
    np.save(output_dir / "stats.npy", stats)

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npy"))
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print(f"Stats: {stats}")

    # Commit final output before cleaning up checkpoint
    if volume:
        volume.commit()

    # Clean up checkpoint now that final output is saved
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("Cleaned up checkpoint.")

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
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/root/cache/hf"
    _precompute(num_samples, Path("/root/cache/planes"), volume=data_volume)
    data_volume.commit()


if __name__ == "__main__":
    import fire
    fire.Fire(precompute_local)
