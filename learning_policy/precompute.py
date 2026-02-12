"""
Sharded precompute pipeline for the Lichess chess dataset.

Splits the 844M-row dataset across parallel Modal workers, each grouping
by FEN locally and converting to board planes. A coordinator concatenates
the results into a single output on a separate volume.

Storage format (per train_*.npy):
    planes:           (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta:             (N, 5) int8 - [turn, K, Q, k, q]
    analysis_data:    (total_analyses, 5) int32 - [move_idx, depth, knodes, cp_or_val, is_mate]
    analysis_offsets: (N + 1,) uint32 - position boundaries

Usage:
    modal run --detach precompute.py::precompute_sharded
    modal run --detach precompute.py::precompute_sharded --num-samples 1M --num-shards 2
"""

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
output_volume = modal.Volume.from_name("chess-policy-output", create_if_missing=True)

CHUNK_SIZE = 50_000


def _convert_fen_chunk(fen_chunk):
    """Convert a chunk of FENs to planes and meta arrays. Runs in worker process."""
    from board_repr import fen_to_planes

    n = len(fen_chunk)
    planes = np.empty((n, 13, 8, 8), dtype=np.uint8)
    meta = np.empty((n, 5), dtype=np.int8)
    for i, fen in enumerate(fen_chunk):
        planes[i], meta[i] = fen_to_planes(fen)
    return planes, meta


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



@app.function(
    image=image,
    volumes={"/root/cache": data_volume},
    timeout=86400,
    memory=64000,
    cpu=8,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def precompute_shard(shard_idx: int, num_shards: int, num_samples: str = "full"):
    """Process one shard: group by FEN, convert to planes, save to volume."""
    import os
    import sys
    from multiprocessing import Pool

    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/root/cache/hf"

    from board_repr import uci_to_move_tuple
    from datasets import load_dataset

    parsed = parse_num_samples(num_samples) if isinstance(num_samples, str) else num_samples
    folder_name = "full" if parsed == "full" else format_count(parsed)
    output_dir = Path(f"/root/cache/planes/{folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / f"shard_{shard_idx}_done"
    data_volume.reload()
    if marker.exists():
        print(f"Shard {shard_idx} already done, skipping")
        return

    # Load dataset (HF caches on volume)
    print(f"Shard {shard_idx}/{num_shards}: Loading dataset...")
    ds = load_dataset("Lichess/chess-position-evaluations", split="train")
    total_ds = len(ds)
    total_rows = total_ds if parsed == "full" else min(parsed, total_ds)
    data_volume.commit()

    # Compute row range
    shard_size = (total_rows + num_shards - 1) // num_shards
    start = shard_idx * shard_size
    end = min(start + shard_size, total_rows)
    num_rows = end - start

    if num_rows <= 0:
        marker.write_text("0,0")
        data_volume.commit()
        return

    print(f"Shard {shard_idx}: rows {start:,} to {end:,} ({num_rows:,} rows)")
    ds_shard = ds.select(range(start, end))

    # Group by FEN
    print(f"Shard {shard_idx}: Grouping by FEN...")
    fen_analyses = defaultdict(list)
    batch_size = 10000

    for batch in tqdm(
        ds_shard.iter(batch_size=batch_size),
        total=(num_rows + batch_size - 1) // batch_size,
    ):
        fens = batch["fen"]
        lines_batch = batch["line"]
        depths = batch["depth"]
        knodess = batch["knodes"]
        cps = batch["cp"]
        mates = batch["mate"]

        for i in range(len(fens)):
            line = lines_batch[i]
            if not line:
                continue
            fen_analyses[fens[i]].append(
                (
                    line.partition(" ")[0],
                    depths[i] or 1,
                    knodess[i] or 0,
                    cps[i],
                    mates[i],
                )
            )

    n_unique = len(fen_analyses)
    print(
        f"Shard {shard_idx}: {num_rows:,} rows -> {n_unique:,} unique "
        f"(dedup {num_rows / n_unique:.2f}x)"
    )

    # Load vocab
    vocab_moves = np.load(Path("/root/cache/planes/vocab.npy"))
    move_to_idx = {tuple(m): i for i, m in enumerate(vocab_moves)}

    # Convert FENs to planes (parallel)
    fens_list = list(fen_analyses.keys())
    planes = np.empty((n_unique, 13, 8, 8), dtype=np.uint8)
    meta_arr = np.empty((n_unique, 5), dtype=np.int8)

    chunks = [fens_list[i : i + CHUNK_SIZE] for i in range(0, len(fens_list), CHUNK_SIZE)]
    num_workers = os.cpu_count()
    print(f"Shard {shard_idx}: Converting {n_unique:,} positions with {num_workers} workers...")

    with Pool(num_workers) as pool:
        offset = 0
        for chunk_planes, chunk_meta in tqdm(
            pool.imap(_convert_fen_chunk, chunks), total=len(chunks)
        ):
            sz = len(chunk_planes)
            planes[offset : offset + sz] = chunk_planes
            meta_arr[offset : offset + sz] = chunk_meta
            offset += sz

    # Build analysis arrays
    print(f"Shard {shard_idx}: Building analysis arrays...")
    all_analyses = []
    offsets_list = [0]
    skipped = 0

    for fen in fens_list:
        for move_uci, depth, knodes, cp, mate in fen_analyses[fen]:
            move = uci_to_move_tuple(move_uci)
            idx = move_to_idx.get(move)
            if idx is None:
                skipped += 1
                continue
            if mate is not None:
                all_analyses.append((idx, depth, knodes, mate, 1))
            elif cp is not None:
                all_analyses.append((idx, depth, knodes, cp, 0))
            else:
                skipped += 1
        offsets_list.append(len(all_analyses))

    analysis_data = np.array(all_analyses, dtype=np.int32).reshape(-1, 5)
    analysis_offsets = np.array(offsets_list, dtype=np.uint32)

    print(f"Shard {shard_idx}: {len(analysis_data):,} analyses, {skipped:,} skipped")

    # Save shard output
    np.save(output_dir / f"shard_{shard_idx}_planes.npy", planes)
    np.save(output_dir / f"shard_{shard_idx}_meta.npy", meta_arr)
    np.save(output_dir / f"shard_{shard_idx}_analysis_data.npy", analysis_data)
    np.save(output_dir / f"shard_{shard_idx}_analysis_offsets.npy", analysis_offsets)

    marker.write_text(f"{n_unique},{num_rows}")
    data_volume.commit()
    print(f"Shard {shard_idx}: Done! ({n_unique:,} positions saved)")


@app.function(
    image=image,
    volumes={"/root/cache": data_volume, "/root/output": output_volume},
    timeout=86400,
    memory=64000,
    cpu=4,
    ephemeral_disk=1_048_576,
)
def concat_shards(num_shards: int, folder_name: str = "full"):
    """Concatenate shard outputs into a single train set.

    Reads shards from data_volume, writes output to output_volume.
    Uses separate volumes so we never need to hold both (~300 GB each)
    on the same volume simultaneously.
    """
    import os
    import shutil

    shard_dir = Path(f"/root/cache/planes/{folder_name}")
    out_dir = Path(f"/root/output/planes/{folder_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy vocab to output volume so training only needs one volume
    shutil.copy2("/root/cache/planes/vocab.npy", "/root/output/planes/vocab.npy")

    # Read shard sizes from markers (format: "unique_count,raw_count")
    shard_sizes = []
    total_raw = 0
    for i in range(num_shards):
        parts = (shard_dir / f"shard_{i}_done").read_text().split(",")
        shard_sizes.append(int(parts[0]))
        total_raw += int(parts[1])
    total = sum(shard_sizes)
    print(f"Total: {total:,} positions across {num_shards} shards ({total_raw:,} raw rows)")

    # Create mmap'd output on output volume
    out_planes = np.lib.format.open_memmap(
        str(out_dir / "train_planes.npy"),
        mode="w+", dtype=np.uint8, shape=(total, 13, 8, 8),
    )
    out_meta = np.lib.format.open_memmap(
        str(out_dir / "train_meta.npy"),
        mode="w+", dtype=np.int8, shape=(total, 5),
    )

    # Process shards one at a time (read from data volume, write to output volume)
    cursor = 0
    total_analyses = 0

    for s in range(num_shards):
        print(f"Processing shard {s}/{num_shards}...")
        s_planes = np.load(shard_dir / f"shard_{s}_planes.npy", mmap_mode="r")
        s_meta = np.load(shard_dir / f"shard_{s}_meta.npy", mmap_mode="r")
        s_ad = np.load(shard_dir / f"shard_{s}_analysis_data.npy", mmap_mode="r")
        s_ao = np.load(shard_dir / f"shard_{s}_analysis_offsets.npy", mmap_mode="r")
        n = shard_sizes[s]

        out_planes[cursor : cursor + n] = s_planes
        out_meta[cursor : cursor + n] = s_meta
        cursor += n

        # Save per-shard analysis to temp files on output volume
        np.save(out_dir / f"_tmp_ad_{s}.npy", s_ad)
        np.save(out_dir / f"_tmp_ao_{s}.npy", s_ao)
        total_analyses += len(s_ad)

        del s_planes, s_meta, s_ad, s_ao

    out_planes.flush()
    out_meta.flush()
    del out_planes, out_meta
    print(f"Planes/meta written. Total analyses: {total_analyses:,}")

    # Concatenate analysis data from temp files
    print("Concatenating analysis data...")
    shard_ao_list = []
    shard_ad_sizes = []
    for s in range(num_shards):
        ao = np.load(out_dir / f"_tmp_ao_{s}.npy")
        shard_ao_list.append(ao)
        shard_ad_sizes.append(int(ao[-1]))

    global_offsets_parts = [np.array([0], dtype=np.uint32)]
    running = np.uint64(0)
    for s in range(num_shards):
        local_offsets = shard_ao_list[s][1:].astype(np.uint64)
        global_offsets_parts.append((local_offsets + running).astype(np.uint32))
        running += np.uint64(shard_ad_sizes[s])
    global_offsets = np.concatenate(global_offsets_parts)

    total_ad = sum(shard_ad_sizes)
    out_ad = np.lib.format.open_memmap(
        str(out_dir / "train_analysis_data.npy"),
        mode="w+", dtype=np.int32, shape=(total_ad, 5) if total_ad > 0 else (0, 5),
    )
    ad_cursor = 0
    for s in range(num_shards):
        chunk = np.load(out_dir / f"_tmp_ad_{s}.npy")
        if len(chunk) > 0:
            out_ad[ad_cursor : ad_cursor + len(chunk)] = chunk
            ad_cursor += len(chunk)
        os.unlink(out_dir / f"_tmp_ad_{s}.npy")
        os.unlink(out_dir / f"_tmp_ao_{s}.npy")

    out_ad.flush()
    del out_ad
    np.save(out_dir / "train_analysis_offsets.npy", global_offsets)
    print(f"  {total_ad:,} analyses, {len(global_offsets) - 1:,} positions")

    # Save stats
    stats = {
        "raw_samples": total_raw,
        "unique_positions": total,
        "dedup_ratio": total_raw / total,
        "total_analyses": total_analyses,
        "avg_analyses_per_position": total_analyses / total,
        "num_shards": num_shards,
    }
    np.save(out_dir / "stats.npy", stats)
    print(f"Stats: {stats}")

    output_volume.commit()
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.npy"))
    print(f"Total output size: {total_bytes / 1e9:.2f} GB")
    print("Done!")


@app.function(
    image=image,
    volumes={"/root/cache": data_volume},
    timeout=86400,
)
def precompute_sharded(num_samples: str = "full", num_shards: int = 17):
    """Coordinator: spawn shard workers, wait, then concat.

    Re-runnable: completed shards (with marker files) are skipped.
    """
    parsed = parse_num_samples(num_samples) if isinstance(num_samples, str) else num_samples
    folder_name = "full" if parsed == "full" else format_count(parsed)
    output_dir = Path(f"/root/cache/planes/{folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which shards are already done
    data_volume.reload()
    done = {i for i in range(num_shards) if (output_dir / f"shard_{i}_done").exists()}
    missing = [i for i in range(num_shards) if i not in done]
    print(f"Shards done: {sorted(done)}")
    print(f"Shards missing: {missing}")

    if missing:
        print(f"Spawning {len(missing)} shard workers...")
        list(
            precompute_shard.starmap(
                [(i, num_shards, num_samples) for i in missing]
            )
        )
        print("All shards done!")
    else:
        print("All shards already complete.")

    # Reload volume to see shard outputs
    data_volume.reload()

    # Run concat
    print("Starting concat...")
    concat_shards.remote(num_shards, folder_name)
    print("Pipeline complete!")


