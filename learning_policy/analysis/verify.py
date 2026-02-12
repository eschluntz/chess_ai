"""Diagnostic tools for inspecting precomputed chess data on Modal volumes.

Usage:
    modal run verify.py::verify_output
    modal run verify.py::check_duplicates
    modal run verify.py::check_depths
"""

from pathlib import Path

import modal
import numpy as np

app = modal.App("chess-verify")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

output_volume = modal.Volume.from_name("chess-policy-output", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/output": output_volume},
    timeout=3600,
)
def verify_output(folder_name: str = "full"):
    """Check integrity of precomputed output on the volume."""
    d = Path(f"/root/output/planes/{folder_name}")

    planes = np.load(d / "train_planes.npy", mmap_mode="r")
    meta = np.load(d / "train_meta.npy", mmap_mode="r")
    ad = np.load(d / "train_analysis_data.npy", mmap_mode="r")
    ao = np.load(d / "train_analysis_offsets.npy")
    n = len(planes)

    print(f"Positions: {n:,}")
    print(f"Shapes: planes={planes.shape}, meta={meta.shape}, ad={ad.shape}, ao={ao.shape}")

    BLOCK = 100_000
    zero_boards = 0
    bad_turns = 0
    for i in range(0, n, BLOCK):
        end = min(i + BLOCK, n)
        p = np.array(planes[i:end])
        m = np.array(meta[i:end])
        zero_boards += (p.reshape(end - i, -1).sum(axis=1) == 0).sum()
        bad_turns += ((m[:, 0] != 1) & (m[:, 0] != -1)).sum()
    print(f"All-zero boards: {zero_boards} / {n}")
    print(f"Bad turn values: {bad_turns} / {n}")

    zero_blocks = 0
    for i in range(0, n - BLOCK, BLOCK):
        if np.array(planes[i:i + BLOCK]).sum() == 0:
            zero_blocks += 1
    print(f"All-zero blocks (of {BLOCK:,}): {zero_blocks}")

    assert ao[0] == 0, f"offsets[0] = {ao[0]}"
    assert ao[-1] == len(ad), f"offsets[-1]={ao[-1]} != len(ad)={len(ad)}"
    diffs = np.diff(ao.astype(np.int64))
    assert (diffs >= 0).all(), "offsets not monotonic!"
    print(f"Analyses per position: min={diffs.min()}, max={diffs.max()}, mean={diffs.mean():.2f}")

    print(f"Move idx range: [{ad[:, 0].min()}, {ad[:, 0].max()}]")
    print(f"Depth range: [{ad[:, 1].min()}, {ad[:, 1].max()}]")

    stats = np.load(d / "stats.npy", allow_pickle=True).item()
    print(f"Stats: {stats}")

    print("\nAll checks passed!")


@app.function(
    image=image,
    volumes={"/root/output": output_volume},
    timeout=7200,
    memory=32000,
)
def check_duplicates(folder_name: str = "full"):
    """Check cross-shard FEN duplication by hashing all board positions."""
    import hashlib

    d = Path(f"/root/output/planes/{folder_name}")
    planes = np.load(d / "train_planes.npy", mmap_mode="r")
    meta = np.load(d / "train_meta.npy", mmap_mode="r")
    n = len(planes)
    print(f"Checking {n:,} positions for duplicates...")

    BLOCK = 100_000
    seen = set()
    duplicates = 0

    for i in range(0, n, BLOCK):
        end = min(i + BLOCK, n)
        p_block = np.array(planes[i:end])
        m_block = np.array(meta[i:end])
        for j in range(end - i):
            h = hashlib.md5(p_block[j].tobytes() + m_block[j].tobytes()).digest()
            if h in seen:
                duplicates += 1
            else:
                seen.add(h)
        if i % (BLOCK * 100) == 0:
            print(f"  {i:,}/{n:,} — {duplicates:,} duplicates so far, {len(seen):,} unique")

    unique = len(seen)
    print(f"\nResults:")
    print(f"  Total positions: {n:,}")
    print(f"  Unique positions: {unique:,}")
    print(f"  Duplicates: {duplicates:,} ({100 * duplicates / n:.2f}%)")
    print(f"  Cross-shard dedup ratio: {n / unique:.4f}x")


@app.function(
    image=image,
    volumes={"/root/output": output_volume},
    timeout=3600,
)
def check_depths(folder_name: str = "full"):
    """Analyze depth distribution and soft vs hard label stats."""
    d = Path(f"/root/output/planes/{folder_name}")
    ad = np.load(d / "train_analysis_data.npy", mmap_mode="r")
    ao = np.load(d / "train_analysis_offsets.npy")
    n = len(ao) - 1

    print(f"Positions: {n:,}, Analyses: {len(ad):,}")

    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(n, size=min(1_000_000, n), replace=False)
    sample_idx.sort()

    max_depths = []
    num_moves_at_max = []
    analyses_per_pos = []

    for i in sample_idx:
        a_start, a_end = int(ao[i]), int(ao[i + 1])
        if a_start == a_end:
            continue
        analyses = ad[a_start:a_end]
        depths = analyses[:, 1]
        max_d = depths.max()
        max_depths.append(max_d)
        num_moves_at_max.append((depths == max_d).sum())
        analyses_per_pos.append(a_end - a_start)

    max_depths = np.array(max_depths)
    num_moves = np.array(num_moves_at_max)
    analyses_per = np.array(analyses_per_pos)

    print(f"\nSampled {len(max_depths):,} positions:")
    print(f"  Analyses per position: mean={analyses_per.mean():.2f}, median={np.median(analyses_per):.0f}")
    print(f"  Max depth per position: mean={max_depths.mean():.1f}, median={np.median(max_depths):.0f}")
    print(f"  Max depth distribution:")
    for threshold in [1, 5, 10, 20, 30, 40, 50, 100]:
        frac = (max_depths >= threshold).mean()
        print(f"    depth >= {threshold:>3}: {100*frac:.1f}%")
    print(f"\n  Moves at max depth (soft vs hard):")
    print(f"    1 move (hard label): {100*(num_moves == 1).mean():.1f}%")
    print(f"    2 moves: {100*(num_moves == 2).mean():.1f}%")
    print(f"    3+ moves: {100*(num_moves >= 3).mean():.1f}%")
    print(f"    mean moves at max depth: {num_moves.mean():.2f}")
