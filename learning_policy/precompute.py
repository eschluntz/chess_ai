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
    print(f"[Shard {shard_idx}] Processing {shard_size:,} samples (indices {start_idx:,} to {end_idx:,})")

    with open(f"{precomputed_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        cache_dir=cache_dir,
    )
    ds = ds.shuffle(seed=42)

    # Create memmap (pre-allocates with zeros)
    features_path = f"{shards_dir}/train_features_{shard_idx}.npy"
    labels_path = f"{shards_dir}/train_labels_{shard_idx}.npy"

    shard_X = np.lib.format.open_memmap(
        features_path, mode='w+', dtype=np.float32, shape=(shard_size, TOTAL_FEATURES)
    )
    shard_y = np.lib.format.open_memmap(
        labels_path, mode='w+', dtype=np.int64, shape=(shard_size,)
    )

    for i in tqdm(range(shard_size), desc=f"Shard {shard_idx}"):
        example = ds[eval_size + start_idx + i]
        board = chess.Board(example["fen"])
        shard_X[i] = extract_features_piece_square(board)
        shard_y[i] = vocab[example["line"].split()[0]]

    del shard_X, shard_y

    # Validate: reload and check last row isn't zeros (would indicate incomplete write)
    check_X = np.load(features_path, mmap_mode='r')
    if np.all(check_X[-1] == 0):
        raise RuntimeError(f"Shard {shard_idx} failed validation: last row is zeros")
    del check_X

    print(f"[Shard {shard_idx}] Done and validated!")
    return shard_idx, shard_size


@app.function(image=image, volumes={"/root/cache": data_volume}, timeout=86400)
def precompute(num_samples: int = 50_000_000, eval_size: int = 10_000, num_shards: int = 20):
    """Precompute features from chess positions."""
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

    # Clean slate - remove any existing precomputed data
    if os.path.exists(precomputed_dir):
        print("Removing existing precomputed data...")
        shutil.rmtree(precomputed_dir)

    os.makedirs(precomputed_dir)
    os.makedirs(shards_dir)

    # Load dataset
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

    print(f"Using {total_samples:,} samples: {eval_size:,} eval, {train_size:,} train")
    print(f"  {num_shards} shards of ~{shard_size:,} samples each")

    # Build vocabulary
    print("Building vocabulary...")
    moves = set()
    sample_size = min(1_000_000, len(ds))
    for i in range(0, sample_size, 10_000):
        batch = ds.select(range(i, min(i + 10_000, sample_size)))
        moves.update(line.split()[0] for line in batch["line"])

    # Add all possible promotions
    files = "abcdefgh"
    for from_rank, to_rank in [("7", "8"), ("2", "1")]:
        for f_idx, from_file in enumerate(files):
            from_sq = f"{from_file}{from_rank}"
            for to_f_idx in [f_idx - 1, f_idx, f_idx + 1]:
                if 0 <= to_f_idx <= 7:
                    to_sq = f"{files[to_f_idx]}{to_rank}"
                    moves.update(f"{from_sq}{to_sq}{p}" for p in "qrbn")

    vocab = {move: idx for idx, move in enumerate(sorted(moves))}
    print(f"Vocabulary size: {len(vocab)}")

    with open(f"{precomputed_dir}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Shuffle dataset
    print("Shuffling dataset...")
    ds = ds.shuffle(seed=42)

    # Process eval set
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
    del eval_X, eval_y
    print("Eval set saved.")

    # Commit so workers can see vocab
    data_volume.commit()

    # Launch parallel workers
    print(f"Launching {num_shards} parallel workers...")

    def shard_bounds(i):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else train_size
        return start, end

    del ds
    results = list(precompute_shard.starmap(
        [(i, *shard_bounds(i), eval_size) for i in range(num_shards)]
    ))
    print(f"All {len(results)} workers complete")

    data_volume.reload()

    # Validate all shards before concatenating
    print("Validating shards...")
    for shard_idx in range(num_shards):
        features_path = f"{shards_dir}/train_features_{shard_idx}.npy"
        labels_path = f"{shards_dir}/train_labels_{shard_idx}.npy"

        features = np.load(features_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')

        expected = shard_size if shard_idx < num_shards - 1 else train_size - (num_shards - 1) * shard_size

        if len(labels) != expected:
            raise RuntimeError(f"Shard {shard_idx} wrong size: {len(labels):,} != {expected:,}")
        if np.all(features[-1] == 0):
            raise RuntimeError(f"Shard {shard_idx} corrupted: last row is zeros")
        if np.all(features[len(features) // 2] == 0):
            raise RuntimeError(f"Shard {shard_idx} corrupted: middle row is zeros")

        del features, labels

    print("All shards validated.")

    # Concatenate shards
    print("Concatenating shards...")
    train_X = np.lib.format.open_memmap(
        f"{precomputed_dir}/train_features.npy",
        mode='w+', dtype=np.float32, shape=(train_size, TOTAL_FEATURES),
    )
    train_y = np.lib.format.open_memmap(
        f"{precomputed_dir}/train_labels.npy",
        mode='w+', dtype=np.int64, shape=(train_size,),
    )

    offset = 0
    for shard_idx in tqdm(range(num_shards)):
        shard_X = np.load(f"{shards_dir}/train_features_{shard_idx}.npy", mmap_mode='r')
        shard_y = np.load(f"{shards_dir}/train_labels_{shard_idx}.npy", mmap_mode='r')
        shard_len = len(shard_y)
        train_X[offset:offset + shard_len] = shard_X
        train_y[offset:offset + shard_len] = shard_y
        offset += shard_len
        del shard_X, shard_y

    del train_X, train_y

    # Validate final file
    print("Validating final file...")
    final_X = np.load(f"{precomputed_dir}/train_features.npy", mmap_mode='r')
    if len(final_X) != train_size:
        raise RuntimeError(f"Final file wrong size: {len(final_X):,} != {train_size:,}")
    if np.all(final_X[-1] == 0):
        raise RuntimeError("Final file corrupted: last row is zeros")
    if np.all(final_X[train_size // 2] == 0):
        raise RuntimeError("Final file corrupted: middle row is zeros")
    del final_X

    # Clean up shards
    print("Cleaning up shards...")
    shutil.rmtree(shards_dir)

    data_volume.commit()

    print("Done!")
    print(f"  Train: {train_size:,} samples")
    print(f"  Eval: {eval_size:,} samples")
    print(f"  Vocab: {len(vocab)} moves")


@app.local_entrypoint()
def main():
    precompute.remote()
