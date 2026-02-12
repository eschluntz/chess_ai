"""Local correctness test for sharded concat logic."""
import os
import tempfile
from pathlib import Path

import numpy as np


def make_fake_shard(shard_dir: Path, shard_idx: int, n_positions: int, n_analyses_per: int):
    """Create fake shard output files."""
    rng = np.random.default_rng(seed=shard_idx)
    planes = rng.integers(0, 2, size=(n_positions, 13, 8, 8), dtype=np.uint8)
    meta = rng.integers(-1, 2, size=(n_positions, 5), dtype=np.int8)

    analysis_data_parts = []
    offsets = [0]
    for i in range(n_positions):
        n_a = rng.integers(1, n_analyses_per + 1)
        rows = rng.integers(-1000, 1000, size=(n_a, 5), dtype=np.int32)
        rows[:, 0] = rng.integers(0, 1000, size=n_a)
        rows[:, 1] = rng.integers(1, 40, size=n_a)
        analysis_data_parts.append(rows)
        offsets.append(offsets[-1] + n_a)

    analysis_data = np.concatenate(analysis_data_parts)
    analysis_offsets = np.array(offsets, dtype=np.uint32)

    np.save(shard_dir / f"shard_{shard_idx}_planes.npy", planes)
    np.save(shard_dir / f"shard_{shard_idx}_meta.npy", meta)
    np.save(shard_dir / f"shard_{shard_idx}_analysis_data.npy", analysis_data)
    np.save(shard_dir / f"shard_{shard_idx}_analysis_offsets.npy", analysis_offsets)
    (shard_dir / f"shard_{shard_idx}_done").write_text(f"{n_positions},{n_positions * 2}")

    return planes, meta, analysis_data, analysis_offsets


def run_concat_locally(shard_dir: Path, num_shards: int):
    """Reproduce concat_shards logic locally (no Modal)."""
    shard_sizes = []
    total_raw = 0
    for i in range(num_shards):
        parts = (shard_dir / f"shard_{i}_done").read_text().split(",")
        shard_sizes.append(int(parts[0]))
        total_raw += int(parts[1])
    total = sum(shard_sizes)

    out_planes = np.lib.format.open_memmap(
        str(shard_dir / "train_planes.npy"),
        mode="w+", dtype=np.uint8, shape=(total, 13, 8, 8),
    )
    out_meta = np.lib.format.open_memmap(
        str(shard_dir / "train_meta.npy"),
        mode="w+", dtype=np.int8, shape=(total, 5),
    )

    cursor = 0
    total_analyses = 0

    for s in range(num_shards):
        s_planes = np.load(shard_dir / f"shard_{s}_planes.npy")
        s_meta = np.load(shard_dir / f"shard_{s}_meta.npy")
        s_ad = np.load(shard_dir / f"shard_{s}_analysis_data.npy")
        s_ao = np.load(shard_dir / f"shard_{s}_analysis_offsets.npy")
        n = shard_sizes[s]

        out_planes[cursor : cursor + n] = s_planes
        out_meta[cursor : cursor + n] = s_meta
        cursor += n

        np.save(shard_dir / f"_tmp_ad_{s}.npy", s_ad)
        np.save(shard_dir / f"_tmp_ao_{s}.npy", s_ao)
        total_analyses += len(s_ad)

        del s_planes, s_meta, s_ad, s_ao

    out_planes.flush()
    out_meta.flush()
    del out_planes, out_meta

    # Concatenate analysis data
    shard_ao_list = []
    shard_ad_sizes = []
    for s in range(num_shards):
        ao = np.load(shard_dir / f"_tmp_ao_{s}.npy")
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
        str(shard_dir / "train_analysis_data.npy"),
        mode="w+", dtype=np.int32, shape=(total_ad, 5) if total_ad > 0 else (0, 5),
    )
    ad_cursor = 0
    for s in range(num_shards):
        chunk = np.load(shard_dir / f"_tmp_ad_{s}.npy")
        if len(chunk) > 0:
            out_ad[ad_cursor : ad_cursor + len(chunk)] = chunk
            ad_cursor += len(chunk)
        os.unlink(shard_dir / f"_tmp_ad_{s}.npy")
        os.unlink(shard_dir / f"_tmp_ao_{s}.npy")

    out_ad.flush()
    del out_ad
    np.save(shard_dir / "train_analysis_offsets.npy", global_offsets)

    return total, total_raw


def test_concat():
    num_shards = 3
    shard_positions = [100, 150, 80]

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)

        # Create fake shards and keep reference data
        all_planes = []
        all_meta = []
        all_ad = []
        all_ao = []
        for i, n in enumerate(shard_positions):
            p, m, ad, ao = make_fake_shard(shard_dir, i, n, n_analyses_per=5)
            all_planes.append(p)
            all_meta.append(m)
            all_ad.append(ad)
            all_ao.append(ao)

        total, total_raw = run_concat_locally(shard_dir, num_shards)
        expected_total = sum(shard_positions)

        # 1. Total matches
        assert total == expected_total, f"{total} != {expected_total}"

        # 2. Check output shapes
        train_p = np.load(shard_dir / "train_planes.npy", mmap_mode="r")
        train_m = np.load(shard_dir / "train_meta.npy", mmap_mode="r")
        assert train_p.shape == (total, 13, 8, 8), f"planes shape: {train_p.shape}"
        assert train_m.shape == (total, 5), f"meta shape: {train_m.shape}"

        # 3. Analysis offsets consistency
        ao = np.load(shard_dir / "train_analysis_offsets.npy")
        ad = np.load(shard_dir / "train_analysis_data.npy")
        assert len(ao) == total + 1, f"offsets length: {len(ao)} != {total + 1}"
        assert ao[0] == 0
        assert ao[-1] == len(ad), f"offsets[-1]={ao[-1]} != len(ad)={len(ad)}"
        assert np.all(np.diff(ao.astype(np.int64)) >= 0), "offsets not monotonic"

        # 4. Verify all data matches shard-by-shard
        cursor = 0
        for s in range(num_shards):
            n = shard_positions[s]
            assert np.array_equal(train_p[cursor:cursor + n], all_planes[s]), \
                f"Shard {s} planes mismatch"
            assert np.array_equal(train_m[cursor:cursor + n], all_meta[s]), \
                f"Shard {s} meta mismatch"

            for li in range(n):
                out_a_start = int(ao[cursor + li])
                out_a_end = int(ao[cursor + li + 1])
                src_a_start = int(all_ao[s][li])
                src_a_end = int(all_ao[s][li + 1])
                assert np.array_equal(
                    ad[out_a_start:out_a_end],
                    all_ad[s][src_a_start:src_a_end],
                ), f"Shard {s}, position {li} analysis mismatch"

            cursor += n

        # 5. raw_samples
        assert total_raw == sum(n * 2 for n in shard_positions)

        print(f"All checks passed! {total} positions, {len(ad):,} analyses")


if __name__ == "__main__":
    test_concat()
