"""
Precompute compact format locally or on Modal.

Usage:
    python precompute.py                     # 1M samples locally (default)
    python precompute.py --num-samples 50M  # 50M samples locally
    modal run precompute.py                  # 50M samples on Modal
    modal run precompute.py --num-samples 784M  # full dataset on Modal
"""

import time
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


def format_count(n: int) -> str:
    """Format sample count for folder name: 1000000 -> '1M', 50000000 -> '50M'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    elif n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def parse_num_samples(ns: str) -> int:
    """Parse num_samples string: '1M' -> 1000000, '50M' -> 50000000, 'full' -> 784M."""
    ns = ns.lower()
    if ns == "full":
        return 784_000_000
    elif ns.endswith("m"):
        return int(ns[:-1]) * 1_000_000
    elif ns.endswith("k"):
        return int(ns[:-1]) * 1_000
    return int(ns)


@app.function(
    image=image,
    volumes={"/root/cache": data_volume},
    timeout=86400,  # 24 hours for full dataset
)
def precompute_modal(num_samples: str = "50M", eval_size: int = 10_000):
    """Run precompute on Modal with volume storage."""
    import sys

    sys.path.insert(0, "/root")
    _precompute(num_samples, Path("/root/cache/compact"), eval_size)
    data_volume.commit()


def precompute_local(
    num_samples: str = "1M", eval_size: int = 10_000, output_dir: str = "cache/compact"
):
    """Run precompute locally."""
    _precompute(num_samples, Path(output_dir), eval_size)


def _precompute(num_samples: str, output_dir: Path, eval_size: int):
    """Core precompute logic."""
    from board_repr import fen_to_compact, uci_to_compact
    from datasets import load_dataset

    num_samples = (
        parse_num_samples(num_samples) if isinstance(num_samples, str) else num_samples
    )

    # Create subfolder based on sample count
    output_dir = output_dir / format_count(num_samples)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing {num_samples:,} samples to {output_dir}")
    print(f"  Train: {num_samples - eval_size:,}, Eval: {eval_size:,}")

    # Pre-allocate arrays
    boards = np.zeros((num_samples, 8, 8), dtype=np.int8)
    turn = np.zeros(num_samples, dtype=np.int8)
    castling = np.zeros((num_samples, 4), dtype=np.uint8)
    en_passant = np.zeros((num_samples, 8, 8), dtype=np.uint8)
    from_sq = np.zeros(num_samples, dtype=np.uint8)
    to_sq = np.zeros(num_samples, dtype=np.uint8)
    promotion = np.zeros(num_samples, dtype=np.uint8)

    # Stream dataset
    print("Loading dataset (streaming)...")
    ds = load_dataset(
        "Lichess/chess-position-evaluations", split="train", streaming=True
    )
    ds = ds.shuffle(seed=42, buffer_size=100_000)

    print("Processing samples...")
    t0 = time.perf_counter()

    for i, example in enumerate(tqdm(ds, total=num_samples)):
        if i >= num_samples:
            break

        # Parse FEN
        b, t, c, ep = fen_to_compact(example["fen"])
        boards[i] = b
        turn[i] = t
        castling[i] = c
        en_passant[i] = ep

        # Parse move
        uci = example["line"].split()[0]
        f, s, p = uci_to_compact(uci)
        from_sq[i] = f
        to_sq[i] = s
        promotion[i] = p

    elapsed = time.perf_counter() - t0
    rate = num_samples / elapsed
    print(
        f"Processed {num_samples:,} samples in {elapsed:.1f}s ({rate:.0f} samples/sec)"
    )

    # Shuffle indices for uniform train/eval split
    print("Shuffling indices...")
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(num_samples)
    eval_idx = perm[:eval_size]
    train_idx = perm[eval_size:]

    # Sort indices within each split for sequential disk reads during training
    eval_idx.sort()
    train_idx.sort()

    def save_indexed(arr, idx, path, chunk_size=1_000_000):
        """Save arr[idx] to path without loading full result into memory."""
        shape = (len(idx),) + arr.shape[1:]
        out = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=shape)
        for i in range(0, len(idx), chunk_size):
            out[i : i + chunk_size] = arr[idx[i : i + chunk_size]]
        del out  # Flush to disk

    # Save arrays by gathering shuffled indices in chunks
    print("Saving eval arrays...")
    save_indexed(boards, eval_idx, output_dir / "eval_boards.npy")
    save_indexed(turn, eval_idx, output_dir / "eval_turn.npy")
    save_indexed(castling, eval_idx, output_dir / "eval_castling.npy")
    save_indexed(en_passant, eval_idx, output_dir / "eval_en_passant.npy")
    save_indexed(from_sq, eval_idx, output_dir / "eval_from_sq.npy")
    save_indexed(to_sq, eval_idx, output_dir / "eval_to_sq.npy")
    save_indexed(promotion, eval_idx, output_dir / "eval_promotion.npy")

    print("Saving train arrays...")
    save_indexed(boards, train_idx, output_dir / "train_boards.npy")
    save_indexed(turn, train_idx, output_dir / "train_turn.npy")
    save_indexed(castling, train_idx, output_dir / "train_castling.npy")
    save_indexed(en_passant, train_idx, output_dir / "train_en_passant.npy")
    save_indexed(from_sq, train_idx, output_dir / "train_from_sq.npy")
    save_indexed(to_sq, train_idx, output_dir / "train_to_sq.npy")
    save_indexed(promotion, train_idx, output_dir / "train_promotion.npy")

    # Calculate size
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npy"))
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print("Done!")


@app.local_entrypoint()
def main(num_samples: str = "50M", eval_size: int = 10_000):
    """Modal entrypoint - runs precompute on Modal."""
    precompute_modal.remote(num_samples=num_samples, eval_size=eval_size)


if __name__ == "__main__":
    import fire

    fire.Fire(precompute_local)
