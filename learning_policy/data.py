"""
Data loading for precomputed planes format with raw analyses.

Stores raw Stockfish analysis rows per position; soft labels are computed
at train time so different labeling strategies can be tested without
re-running precompute.

Storage format:
    planes.npy:           (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
    meta.npy:             (N, 5) int8 - [turn, K, Q, k, q]
    analysis_data.npy:    (total_analyses, 5) int32 - [move_idx, depth, knodes, cp_or_val, is_mate]
    analysis_offsets.npy: (N + 1,) uint32 - position boundaries
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore', message='.*not writable.*')

CACHE_DIR = Path(__file__).parent / "cache" / "planes"
TEMPERATURE = 50  # centipawns - controls spread within session


def mate_to_score(mate_in_n: int) -> float:
    """Convert mate-in-N to a score. Mate in 1 = 10k, mate in 20+ = 1k."""
    n = abs(mate_in_n)
    if n >= 20:
        score = 1000
    else:
        score = 10000 - 9000 * (n - 1) / 19
    return score if mate_in_n > 0 else -score


def analysis_to_score(cp_or_val: int, is_mate: int) -> float:
    """Convert stored cp/mate value to comparable score."""
    if is_mate:
        return mate_to_score(cp_or_val)
    return float(cp_or_val)


def compute_soft_labels(
    analyses: np.ndarray,
    is_white_to_move: bool,
) -> list[tuple[int, float]]:
    """
    Compute soft label probabilities from the deepest analysis session.

    Keeps only analyses at the maximum depth, discarding shallower searches.
    Within the deepest session(s), uses Boltzmann weighting on cp scores.

    Args:
        analyses: (K, 5) int32 array - [move_idx, depth, knodes, cp_or_val, is_mate]
        is_white_to_move: True if white to move

    Returns:
        List of (move_idx, probability) pairs
    """
    if len(analyses) == 0:
        return []

    # Find the maximum depth
    max_depth = analyses[:, 1].max()

    # Keep only analyses at the deepest depth, collecting best score per move
    move_scores: dict[int, float] = {}

    for row in analyses:
        move_idx, depth, knodes, cp_or_val, is_mate = row
        if depth < max_depth:
            continue
        score = analysis_to_score(int(cp_or_val), int(is_mate))
        if not is_white_to_move:
            score = -score

        move_idx = int(move_idx)
        if move_idx not in move_scores or score > move_scores[move_idx]:
            move_scores[move_idx] = score

    if not move_scores:
        return []

    # Convert scores to Boltzmann weights
    max_score = max(move_scores.values())
    move_weights: dict[int, float] = {}
    for move_idx, score in move_scores.items():
        delta = max_score - score
        move_weights[move_idx] = math.exp(-delta / TEMPERATURE)

    # Normalize to probabilities
    total = sum(move_weights.values())
    return [(idx, w / total) for idx, w in move_weights.items()]


class MoveVocab:
    """Bidirectional mapping between (from_sq, to_sq, promotion) and vocab indices."""

    def __init__(self, vocab_path: Path):
        self.moves = np.load(vocab_path)  # (num_moves, 3)
        self.num_moves = len(self.moves)

        # Build tensor lookup table: packed_key -> index
        self._lookup = torch.zeros(64 * 320, dtype=torch.long)
        for idx, (f, t, p) in enumerate(self.moves):
            key = int(f) * 320 + int(t) * 5 + int(p)
            self._lookup[key] = idx

    def to_indices(self, moves: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 3) moves to vocab indices. Vectorized tensor operation."""
        keys = moves[:, 0].long() * 320 + moves[:, 1].long() * 5 + moves[:, 2].long()
        return self._lookup[keys]

    def to_uci(self, idx: int) -> str:
        """Convert vocab index to UCI string."""
        f, t, p = self.moves[idx]
        from_file, from_rank = f % 8, f // 8
        to_file, to_rank = t % 8, t // 8
        uci = chr(ord('a') + from_file) + str(from_rank + 1)
        uci += chr(ord('a') + to_file) + str(to_rank + 1)
        if p > 0:
            uci += ['', 'n', 'b', 'r', 'q'][p]
        return uci


class SoftLabelDataset:
    """Iterator that yields (planes, meta, soft_target) batches, computing labels from raw analyses."""

    def __init__(self, data_dir: Path, prefix: str, batch_size: int, num_classes: int,
                 min_depth: int = 0, indices: np.ndarray = None, preload: bool = False):
        self.batch_size = batch_size
        self.num_classes = num_classes

        t0 = time.perf_counter()
        # Memory-map position arrays
        self.planes = np.load(data_dir / f"{prefix}_planes.npy", mmap_mode='r')
        self.meta = np.load(data_dir / f"{prefix}_meta.npy", mmap_mode='r')
        print(f"  [{prefix}] mmap planes/meta: {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        # Load raw analyses
        self.analysis_data = np.load(data_dir / f"{prefix}_analysis_data.npy")
        print(f"  [{prefix}] load analysis_data ({self.analysis_data.nbytes / 1e9:.1f} GB): {time.perf_counter() - t0:.1f}s", flush=True)
        t0 = time.perf_counter()
        self.analysis_offsets = np.load(data_dir / f"{prefix}_analysis_offsets.npy")
        print(f"  [{prefix}] load analysis_offsets ({self.analysis_offsets.nbytes / 1e9:.1f} GB): {time.perf_counter() - t0:.1f}s", flush=True)

        candidates = indices if indices is not None else np.arange(len(self.planes))

        # Filter to positions where max analysis depth >= min_depth
        if min_depth > 0:
            depths = self.analysis_data[:, 1]  # depth is column 1
            mask = np.array([
                depths[self.analysis_offsets[i]:self.analysis_offsets[i + 1]].max()
                if self.analysis_offsets[i] < self.analysis_offsets[i + 1] else 0
                for i in candidates
            ]) >= min_depth
            candidates = candidates[mask]

        self.indices = candidates
        self.num_samples = len(self.indices)
        self.num_batches = self.num_samples // batch_size

        # Block shuffle: shuffle blocks of contiguous positions so mmap reads
        # stay sequential within each batch, while randomizing across blocks.
        BLOCK = batch_size
        rng = np.random.default_rng(seed=42)
        n_blocks = len(self.indices) // BLOCK
        blocks = self.indices[:n_blocks * BLOCK].reshape(n_blocks, BLOCK)
        rng.shuffle(blocks)
        self.indices = blocks.reshape(-1)

        if preload:
            # Load planes/meta into memory so reads don't hit mmap
            t0 = time.perf_counter()
            self.planes = np.array(self.planes[self.indices])
            self.meta = np.array(self.meta[self.indices])
            self.indices = np.arange(len(self.planes))
            print(f"  [{prefix}] preloaded ({self.planes.nbytes / 1e6:.0f} MB): {time.perf_counter() - t0:.1f}s", flush=True)

    def __iter__(self):
        self.time_mmap = 0.0
        self.time_tensor = 0.0
        self.time_labels = 0.0
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_indices = self.indices[start:end]

            t0 = time.perf_counter()
            planes = self.planes[batch_indices]
            meta = self.meta[batch_indices]
            self.time_mmap += time.perf_counter() - t0

            t0 = time.perf_counter()
            planes_t = torch.as_tensor(np.array(planes))
            meta_t = torch.as_tensor(np.array(meta))
            self.time_tensor += time.perf_counter() - t0

            # Compute soft labels from raw analyses
            t0 = time.perf_counter()

            soft_target = torch.zeros(self.batch_size, self.num_classes)

            for j in range(self.batch_size):
                pos_idx = batch_indices[j]
                a_start = self.analysis_offsets[pos_idx]
                a_end = self.analysis_offsets[pos_idx + 1]
                analyses = self.analysis_data[a_start:a_end]

                is_white = meta[j][0] == 1
                labels = compute_soft_labels(analyses, is_white)

                for move_idx, prob in labels:
                    soft_target[j, move_idx] = prob

            self.time_labels += time.perf_counter() - t0

            yield planes_t, meta_t, soft_target

    def __len__(self):
        return self.num_batches


def get_dataloaders(batch_size: int, num_samples: str = "full", min_depth: int = 0, eval_min_depth: int = None):
    """Load precomputed data with raw analyses.

    Args:
        min_depth: Filter training positions to those with max analysis depth >= this value.
        eval_min_depth: Filter eval positions separately. Defaults to min_depth if not set.

    Returns:
        train_loader: Iterator yielding (planes, meta, soft_target) batches
        eval_loader: Iterator yielding (planes, meta, soft_target) batches
        num_classes: Number of move classes (vocab size)
    """
    if eval_min_depth is None:
        eval_min_depth = min_depth

    data_dir = CACHE_DIR / num_samples

    # Get num_classes from vocab
    vocab_path = CACHE_DIR / "vocab.npy"
    vocab = np.load(vocab_path)
    num_classes = len(vocab)

    # Load stats if available
    stats_path = data_dir / "stats.npy"
    if stats_path.exists():
        stats = np.load(stats_path, allow_pickle=True).item()
        print(f"Loaded: {stats['unique_positions']:,} unique positions")
        print(f"  (from {stats['raw_samples']:,} raw samples, {stats['dedup_ratio']:.2f}x dedup)")
        print(f"  Avg analyses per position: {stats['avg_analyses_per_position']:.2f}")

    if min_depth > 0:
        print(f"Train filter: min_depth >= {min_depth}")
    if eval_min_depth > 0:
        print(f"Eval filter: min_depth >= {eval_min_depth}")

    # Split via shuffled indices so eval is sampled randomly across all positions
    train_planes = np.load(data_dir / "train_planes.npy", mmap_mode='r')
    n_total = len(train_planes)
    eval_size = min(n_total // 100, 100_000)

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(n_total)
    eval_indices = np.sort(all_indices[:eval_size])
    train_indices = np.sort(all_indices[eval_size:])

    train_loader = SoftLabelDataset(data_dir, "train", batch_size, num_classes,
                                    min_depth=min_depth, indices=train_indices)
    eval_loader = SoftLabelDataset(data_dir, "train", batch_size, num_classes,
                                   min_depth=eval_min_depth, indices=eval_indices, preload=True)

    print(f"Train: {train_loader.num_samples:,} positions ({train_loader.num_batches:,} batches)")
    print(f"Eval: {eval_loader.num_samples:,} positions ({eval_loader.num_batches:,} batches)")
    print(f"Vocab: {num_classes:,} unique moves")

    return train_loader, eval_loader, num_classes
