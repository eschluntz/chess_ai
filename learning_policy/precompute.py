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
def precompute_shard(shard_idx: int, start_idx: int, end_idx: int, eval_size: int):
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

    shard_size = end_idx - start_idx
    print(f"[Shard {shard_idx}] Processing indices {start_idx:,} to {end_idx:,} ({shard_size:,} samples)")

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
        mode='w+', dtype=np.float32, shape=(shard_size, TOTAL_FEATURES)
    )
    shard_y = np.lib.format.open_memmap(
        f"{shards_dir}/train_labels_{shard_idx}.npy",
        mode='w+', dtype=np.int64, shape=(shard_size,)
    )

    # Process this shard (offset by eval_size since eval comes first in shuffled dataset)
    for i in tqdm(range(shard_size), desc=f"Shard {shard_idx}"):
        example = ds[eval_size + start_idx + i]
        board = chess.Board(example["fen"])
        shard_X[i] = extract_features_piece_square(board)
        shard_y[i] = vocab[example["line"].split()[0]]

    # Flush memmap
    del shard_X, shard_y

    print(f"[Shard {shard_idx}] Done!")
    return shard_idx, shard_size


@app.function(image=image, volumes={"/root/cache": data_volume}, timeout=86400)
def precompute(num_samples: int = 50_000_000, eval_size: int = 10_000, num_shards: int = 20):
    """Coordinate parallel precomputation of features."""
    import os
    import pickle
    import shutil
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

    # Load dataset first to get accurate sizes
    print("Loading dataset...")
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        cache_dir=cache_dir,
    )
    print(f"Dataset size: {len(ds):,}")

    total_samples = min(num_samples, len(ds))
    train_size = total_samples - eval_size
    shard_size = train_size // num_shards

    # Check for existing shards (for resume) - verify they're complete
    existing_shards = set()
    for i in range(num_shards):
        features_path = f"{shards_dir}/train_features_{i}.npy"
        labels_path = f"{shards_dir}/train_labels_{i}.npy"
        if os.path.exists(features_path) and os.path.exists(labels_path):
            # Verify shard is complete by checking size
            labels = np.load(labels_path, mmap_mode='r')
            expected_size = shard_size if i < num_shards - 1 else train_size - (num_shards - 1) * shard_size
            if len(labels) >= expected_size * 0.99:  # Allow 1% tolerance
                existing_shards.add(i)
            else:
                print(f"Shard {i} incomplete: {len(labels):,} < {expected_size:,}, will re-run")
            del labels

    if existing_shards:
        print(f"Found {len(existing_shards)} complete shards: {sorted(existing_shards)}")

    missing_shards = [i for i in range(num_shards) if i not in existing_shards]

    # Check if vocab exists (for resume)
    vocab_path = f"{precomputed_dir}/vocab.pkl"
    if os.path.exists(vocab_path):
        print("Loading existing vocab...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocabulary size: {len(vocab)}")
    else:
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
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    # Shuffle dataset
    print("Shuffling dataset...")
    ds = ds.shuffle(seed=42)
    print(f"Using {total_samples:,} samples: {train_size:,} train, {eval_size:,} eval")

    # Check if eval exists (for resume)
    eval_features_path = f"{precomputed_dir}/eval_features.npy"
    if os.path.exists(eval_features_path):
        print("Eval set already exists, skipping.")
    else:
        # Process eval set in coordinator (small, not worth parallelizing)
        print("Processing eval set...")
        eval_X = np.zeros((eval_size, TOTAL_FEATURES), dtype=np.float32)
        eval_y = np.zeros(eval_size, dtype=np.int64)
        for i in tqdm(range(eval_size)):
            example = ds[i]
            board = chess.Board(example["fen"])
            eval_X[i] = extract_features_piece_square(board)
            eval_y[i] = vocab[example["line"].split()[0]]

        np.save(eval_features_path, eval_X)
        np.save(f"{precomputed_dir}/eval_labels.npy", eval_y)
        print("Eval set saved.")

    # Commit so workers can see vocab and eval files
    data_volume.commit()

    # Fan out to workers (only missing shards)
    if missing_shards:
        print(f"Launching {len(missing_shards)} workers for shards: {missing_shards}")

        # Calculate start/end for each missing shard
        def shard_bounds(i):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else train_size
            return start, end

        results = list(precompute_shard.starmap(
            [(i, *shard_bounds(i), eval_size) for i in missing_shards]
        ))
        print(f"All workers complete: {results}")
        # Delete dataset to release file handles before reload
        del ds
        data_volume.reload()
    else:
        print("All shards already exist, skipping to concatenation.")
        del ds

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
