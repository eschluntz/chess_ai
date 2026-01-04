"""
Precompute features for chess policy training.

Usage:
    modal run precompute.py
    modal run precompute.py --fresh  # Force fresh start, ignore existing shards
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
    done_path = f"{shards_dir}/shard_{shard_idx}.done"

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

    # Validate: reload and check rows aren't zeros
    check_X = np.load(features_path, mmap_mode='r')
    if np.all(check_X[-1] == 0):
        raise RuntimeError(f"Shard {shard_idx} failed validation: last row is zeros")
    if np.all(check_X[len(check_X) // 2] == 0):
        raise RuntimeError(f"Shard {shard_idx} failed validation: middle row is zeros")
    del check_X

    # Write completion marker AFTER validation passes
    # This is the key to safe resume - no marker means incomplete
    with open(done_path, "w") as f:
        f.write(f"{shard_size}\n")

    # Commit shard to volume so coordinator can see it
    data_volume.commit()

    print(f"[Shard {shard_idx}] Done and validated!")
    return shard_idx, shard_size


@app.function(image=image, volumes={"/root/cache": data_volume}, timeout=86400)
def precompute(num_samples: int = 50_000_000, eval_size: int = 10_000, num_shards: int = 20, fresh: bool = False):
    """Precompute features from chess positions. Supports safe resume."""
    import gc
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

    # Fresh start if requested or if final files already exist (completed previous run)
    final_features_path = f"{precomputed_dir}/train_features.npy"
    if fresh:
        print("Fresh start requested, removing existing data...")
        if os.path.exists(precomputed_dir):
            shutil.rmtree(precomputed_dir)
    elif os.path.exists(final_features_path):
        print("Final files already exist. Use --fresh to regenerate.")
        return

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

    total_samples = min(num_samples, len(ds))
    train_size = total_samples - eval_size
    shard_size = train_size // num_shards

    print(f"Using {total_samples:,} samples: {eval_size:,} eval, {train_size:,} train")
    print(f"  {num_shards} shards of ~{shard_size:,} samples each")

    def shard_expected_size(i):
        return shard_size if i < num_shards - 1 else train_size - (num_shards - 1) * shard_size

    # Check for existing vocab or build new one
    vocab_path = f"{precomputed_dir}/vocab.pkl"
    if os.path.exists(vocab_path):
        print("Loading existing vocab...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocabulary size: {len(vocab)}")
    else:
        print("Building vocabulary...")
        moves = set()
        sample_size = min(1_000_000, len(ds))
        for i in range(0, sample_size, 10_000):
            batch = ds.select(range(i, min(i + 10_000, sample_size)))
            moves.update(line.split()[0] for line in batch["line"])

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

        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    # Shuffle dataset
    print("Shuffling dataset...")
    ds = ds.shuffle(seed=42)

    # Check for existing eval set or create new one
    eval_features_path = f"{precomputed_dir}/eval_features.npy"
    if os.path.exists(eval_features_path):
        print("Eval set already exists, skipping.")
    else:
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
        del eval_X, eval_y
        print("Eval set saved.")

    # Commit so workers can see vocab and eval
    data_volume.commit()

    # Check for completed shards (safe resume)
    # A shard is complete IFF: .done marker exists AND content validates
    print("Checking for completed shards...")
    existing_shards = set()
    for i in range(num_shards):
        done_path = f"{shards_dir}/shard_{i}.done"
        features_path = f"{shards_dir}/train_features_{i}.npy"
        labels_path = f"{shards_dir}/train_labels_{i}.npy"

        # Must have completion marker
        if not os.path.exists(done_path):
            continue

        # Must have data files
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            continue

        # Validate content
        features = np.load(features_path, mmap_mode='r')
        expected = shard_expected_size(i)

        if len(features) != expected:
            print(f"  Shard {i}: wrong size {len(features):,} != {expected:,}, will re-run")
            del features
            continue

        if np.all(features[-1] == 0):
            print(f"  Shard {i}: last row is zeros, will re-run")
            del features
            continue

        if np.all(features[len(features) // 2] == 0):
            print(f"  Shard {i}: middle row is zeros, will re-run")
            del features
            continue

        existing_shards.add(i)
        del features

    missing_shards = [i for i in range(num_shards) if i not in existing_shards]

    if existing_shards:
        print(f"Found {len(existing_shards)} complete shards: {sorted(existing_shards)}")
    if not missing_shards:
        print("All shards complete, skipping to concatenation.")
    else:
        # Clean up any partial files for missing shards before re-running
        for i in missing_shards:
            for path in [
                f"{shards_dir}/train_features_{i}.npy",
                f"{shards_dir}/train_labels_{i}.npy",
                f"{shards_dir}/shard_{i}.done",
            ]:
                if os.path.exists(path):
                    os.remove(path)

        print(f"Launching {len(missing_shards)} workers for shards: {missing_shards}")

        def shard_bounds(i):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else train_size
            return start, end

        del ds
        gc.collect()

        results = list(precompute_shard.starmap(
            [(i, *shard_bounds(i), eval_size) for i in missing_shards]
        ))
        print(f"All {len(results)} workers complete")

        gc.collect()
        data_volume.reload()

    # Validate all shards before concatenating
    print("Validating all shards...")
    for shard_idx in range(num_shards):
        features_path = f"{shards_dir}/train_features_{shard_idx}.npy"
        labels_path = f"{shards_dir}/train_labels_{shard_idx}.npy"
        done_path = f"{shards_dir}/shard_{shard_idx}.done"

        if not os.path.exists(done_path):
            raise RuntimeError(f"Shard {shard_idx} missing completion marker")

        features = np.load(features_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')

        expected = shard_expected_size(shard_idx)

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
def main(fresh: bool = False):
    precompute.remote(fresh=fresh)
