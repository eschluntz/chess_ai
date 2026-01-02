"""
Precompute features for chess policy training.

Usage:
    modal run precompute.py
"""
import modal

app = modal.App("chess-precompute")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "python-chess", "datasets", "tqdm")
    .add_local_file("features.py", "/root/features.py")
)

data_volume = modal.Volume.from_name("chess-policy-data", create_if_missing=True)


@app.function(image=image, volumes={"/root/cache": data_volume}, timeout=86400)
def precompute_shard(shard_idx: int, num_shards: int, train_size: int, eval_size: int):
    """Process one shard of the training set."""
    import pickle
    import sys
    sys.path.insert(0, "/root")

    import chess
    import numpy as np
    from datasets import load_dataset
    from tqdm import tqdm

    from features import extract_features_piece_square, TOTAL_FEATURES

    cache_dir = "/root/cache"
    precomputed_dir = f"{cache_dir}/precomputed"
    shards_dir = f"{precomputed_dir}/shards"

    # Calculate this shard's range
    shard_size = train_size // num_shards
    start = shard_idx * shard_size
    end = start + shard_size if shard_idx < num_shards - 1 else train_size
    actual_size = end - start

    print(f"[Shard {shard_idx}] Processing indices {start:,} to {end:,} ({actual_size:,} samples)")

    # Load vocab (saved by coordinator)
    with open(f"{precomputed_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load and shuffle dataset (same seed = same order across all workers)
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        cache_dir=cache_dir,
    )
    ds = ds.shuffle(seed=42)

    # Create shard arrays as memory-mapped files (doesn't use RAM)
    shard_X = np.lib.format.open_memmap(
        f"{shards_dir}/train_features_{shard_idx}.npy",
        mode='w+', dtype=np.float32, shape=(actual_size, TOTAL_FEATURES)
    )
    shard_y = np.lib.format.open_memmap(
        f"{shards_dir}/train_labels_{shard_idx}.npy",
        mode='w+', dtype=np.int64, shape=(actual_size,)
    )

    # Process this shard (offset by eval_size since eval comes first in shuffled dataset)
    for i in tqdm(range(actual_size), desc=f"Shard {shard_idx}"):
        example = ds[eval_size + start + i]
        board = chess.Board(example["fen"])
        shard_X[i] = extract_features_piece_square(board)
        shard_y[i] = vocab[example["line"].split()[0]]

    # Flush memmap
    del shard_X, shard_y

    print(f"[Shard {shard_idx}] Done!")
    return shard_idx, actual_size


@app.function(image=image, volumes={"/root/cache": data_volume}, timeout=86400)
def precompute(num_samples: int = 50_000_000, eval_size: int = 10_000, num_shards: int = 20):
    """Coordinate parallel precomputation of features."""
    import pickle
    import os
    import sys
    sys.path.insert(0, "/root")

    import chess
    import numpy as np
    from datasets import load_dataset
    from tqdm import tqdm

    from features import extract_features_piece_square, TOTAL_FEATURES

    cache_dir = "/root/cache"
    precomputed_dir = f"{cache_dir}/precomputed"
    shards_dir = f"{precomputed_dir}/shards"

    os.makedirs(precomputed_dir, exist_ok=True)
    os.makedirs(shards_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        cache_dir=cache_dir,
    )
    print(f"Dataset size: {len(ds):,}")

    # Build vocabulary
    print("Building vocabulary...")
    moves = set()
    sample_size = min(1_000_000, len(ds))
    for i in range(0, sample_size, 10_000):
        batch = ds.select(range(i, min(i + 10_000, sample_size)))
        moves.update(line.split()[0] for line in batch["line"])

    # Add all possible promotions
    files = "abcdefgh"
    promo_pieces = "qrbn"
    for from_rank, to_rank in [("7", "8"), ("2", "1")]:
        for f_idx, from_file in enumerate(files):
            from_sq = f"{from_file}{from_rank}"
            for to_f_idx in [f_idx - 1, f_idx, f_idx + 1]:
                if 0 <= to_f_idx <= 7:
                    to_sq = f"{files[to_f_idx]}{to_rank}"
                    moves.update(f"{from_sq}{to_sq}{p}" for p in promo_pieces)

    vocab = {move: idx for idx, move in enumerate(sorted(moves))}
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocab for workers
    with open(f"{precomputed_dir}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Shuffle and calculate sizes
    print("Shuffling dataset...")
    ds = ds.shuffle(seed=42)
    total_samples = min(num_samples, len(ds))
    train_size = total_samples - eval_size
    print(f"Using {total_samples:,} samples: {train_size:,} train, {eval_size:,} eval")

    # Process eval set in coordinator (small, not worth parallelizing)
    print("Processing eval set...")
    eval_X = np.zeros((eval_size, TOTAL_FEATURES), dtype=np.float32)
    eval_y = np.zeros(eval_size, dtype=np.int64)
    for i in tqdm(range(eval_size)):
        example = ds[i]
        board = chess.Board(example["fen"])
        eval_X[i] = extract_features_piece_square(board)
        eval_y[i] = vocab[example["line"].split()[0]]

    np.save(f"{precomputed_dir}/eval_features.npy", eval_X)
    np.save(f"{precomputed_dir}/eval_labels.npy", eval_y)
    print("Eval set saved.")

    # Commit so workers can see vocab and eval files
    data_volume.commit()

    # Fan out to workers
    print(f"Launching {num_shards} parallel workers...")
    results = list(precompute_shard.map(
        range(num_shards),
        kwargs={"num_shards": num_shards, "train_size": train_size, "eval_size": eval_size},
    ))
    print(f"All workers complete: {results}")

    # Reload to see workers' files
    data_volume.reload()

    # Concatenate shards
    print("Concatenating shards...")
    train_X = np.lib.format.open_memmap(
        f"{precomputed_dir}/train_features.npy",
        mode='w+',
        dtype=np.float32,
        shape=(train_size, TOTAL_FEATURES),
    )
    train_y = np.lib.format.open_memmap(
        f"{precomputed_dir}/train_labels.npy",
        mode='w+',
        dtype=np.int64,
        shape=(train_size,),
    )

    offset = 0
    for shard_idx in tqdm(range(num_shards)):
        shard_X = np.load(f"{shards_dir}/train_features_{shard_idx}.npy")
        shard_y = np.load(f"{shards_dir}/train_labels_{shard_idx}.npy")
        shard_len = len(shard_y)
        train_X[offset:offset + shard_len] = shard_X
        train_y[offset:offset + shard_len] = shard_y
        offset += shard_len

    del train_X, train_y

    # Clean up shards
    print("Cleaning up shards...")
    import shutil
    shutil.rmtree(shards_dir)

    # Final commit
    print("Committing final files...")
    data_volume.commit()

    print("Done!")
    print(f"  Train: {precomputed_dir}/train_features.npy ({train_size:,} samples)")
    print(f"  Eval: {precomputed_dir}/eval_features.npy ({eval_size:,} samples)")
    print(f"  Vocab: {precomputed_dir}/vocab.pkl")


@app.local_entrypoint()
def main():
    precompute.remote()
