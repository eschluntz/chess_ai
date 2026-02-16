# Learning a Chess Policy

Learn a policy model to directly predict next moves from chess positions.

## Files

| File                  | Description                                            |
|-----------------------|--------------------------------------------------------|
| `board_repr.py`       | Board representation: FEN↔planes conversion, PlanesToFlat encoder |
| `build_vocab.py`      | Build move vocabulary from dataset                     |
| `precompute.py`       | Sharded precompute pipeline (Modal): shard workers + concat |
| `data.py`             | Data loading + soft label computation at train time    |
| `cnn_model.py`        | ResNet-style CNN policy network                        |
| `mlp_model.py`        | Simple MLP policy network (baseline)                   |
| `train_modal.py`      | Training script (Modal)                                |
| `analysis/verify.py`  | Data integrity checks + depth/duplicate analysis (Modal) |
| `interface/server.py` | Web UI to visualize model predictions                  |
| `interface/data_server.py` | Web UI to explore training data                   |

## Data Format

**Precomputed storage format** (per split, e.g. `train_*.npy`):
```
planes.npy:           (N, 13, 8, 8) uint8 - 12 piece planes + en_passant
meta.npy:             (N, 5) int8 - [turn, K, Q, k, q]
analysis_data.npy:    (total_analyses, 5) int32 - [move_idx, depth, knodes, cp_or_val, is_mate]
analysis_offsets.npy: (N + 1,) uint32 - position boundaries
```

Raw analyses are stored so that label computation (soft labels, deepest-only, etc.)
can be changed at train time without re-running precompute.

**Encoder modules** (run on GPU, part of model):
- `PlanesToFlat(planes, meta)` → `(batch, 837)` for MLPs

## Usage

**Precompute dataset (Modal):**
```bash
modal run --detach precompute.py::precompute_sharded                          # full dataset, 17 shards
modal run --detach precompute.py::precompute_sharded --num-samples 1M --num-shards 2  # small test
```

Shards are written to Modal volume `chess-policy-data`, final output to `chess-policy-output`.
Re-runnable: completed shards are skipped on retry.

**Train (Modal):**
```bash
modal run --detach train_modal.py::train --max-seconds 3600
modal run --detach train_modal.py::train --min-depth 30 --eval-min-depth 50 --max-seconds 14400
```

**Verify data integrity:**
```bash
modal run analysis/verify.py::verify_output
modal run analysis/verify.py::check_depths
```

**Explore training data locally:**
```bash
python interface/data_server.py --data cache/planes/full --port 5001
```
Then open http://localhost:5001. Arrow keys to navigate, R for random, hover arrows for probabilities.

**Run policy head-to-head arena locally:**
```bash
python interface/server.py --vocab cache/planes/vocab.npy --port 5000
```
Then open http://localhost:5000.

See `interface/README.md` for CLI series mode and checkpoint download examples.

## Dataset

Source: [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- 844M rows, 342M unique positions (~2.5 analyses per position on average)
- Crowdsourced from user browsers running Stockfish on Lichess analysis board
- Updated monthly, CC0 license

| Sample count | Unique positions | Output size | Precompute time |
|--------------|------------------|-------------|-----------------|
| 1M           | 173k             | 0.2 GB      | ~2 min (Modal)  |
| 844M (full)  | 342M             | 305 GB      | ~1.5 hours (17 shards on Modal) |

**IMPORTANT NOTES ON DATASET! PAY ATTENTION TO THIS WHEN WRITING RELATED CODE**
- The dataset is NOT uniformly distributed. There's distribution drift over time. Therefore to do accurate science it's very important that we shuffle the test/train split across whatever max data size we're working with.
- I want to be able to do small tests locally, but larger runs will always be on Modal.

**Modal volumes:**
- `chess-policy-data`: HF cache + shard intermediate files (~400 GB). Used by precompute workers.
- `chess-policy-output`: Final `train_*.npy` output + `vocab.npy` (~305 GB). Used by training.
- Two volumes are needed because a single volume can't hold both shards and output simultaneously (~700 GB total).

### Dataset Structure and Soft Labels

The raw dataset has one row per (position, PV) pair:

| Field | Description |
|-------|-------------|
| `fen` | Position in FEN notation |
| `line` | Principal variation (first move is the "best move") |
| `depth` | Search depth |
| `knodes` | Kilonodes searched |
| `cp` | Centipawn evaluation (None if mate) |
| `mate` | Mate in N (None if no forced mate) |

**Key insight: Many positions have multiple analyses with conflicting "best" moves.**

With the full dataset (2.47 analyses per position on average), 41.5% of positions have soft labels (2+ moves at max depth) and 58.5% have hard labels (single move). This motivates using soft labels instead of picking a single "correct" move.

### Analysis Sessions

Rows with identical `(fen, depth, knodes)` came from the **same analysis session** (one Stockfish run with Multi-PV enabled). Within a session:
- Multiple rows = multiple principal variations (1st best, 2nd best, etc.)
- `cp` scores are directly comparable
- The move with best `cp` is the engine's top choice

Different `(depth, knodes)` = different sessions (different users, times, or Stockfish versions).

### Soft Label Weighting Strategy

Label computation happens at train time in `data.py` (not during precompute),
so different strategies can be tested without re-running precompute.

Current strategy — to convert multiple analyses into a probability distribution:

1. **Group by FEN**, keep only rows at the **maximum depth** (discard all shallower analyses)

2. **Convert cp/mate to scores**:
   ```python
   if mate is not None:
       # Mate in 1 = 10k, mate in 20+ = 1k, linear between
       n = abs(mate)
       score = 10000 - 9000 * (min(n, 20) - 1) / 19 if n < 20 else 1000
       score = score if mate > 0 else -score
   else:
       score = cp
   # Flip sign for black to move
   ```

3. **Boltzmann weighting** on the deepest-depth rows:
   ```python
   weight = exp(-(max_score - score) / temperature)  # best move = 1.0
   ```

4. **Per move, take max weight** (handles multiple deepest-depth sessions with same depth but different knodes)

5. **Normalize** to get final probabilities

**Why keep only the deepest depth?**
- A depth-40 analysis subsumes a depth-20 analysis (same engine, deeper search tree)
- The deeper search likely discovered why shallower evaluations were optimistic/pessimistic
- Simpler than weighting across sessions with depth multipliers

**Temperature**: Controls how much cp differences matter within the deepest session. ~50cp (0.5 pawns) is reasonable.


## Architecture

**Training loop pattern:**
```python
for planes, meta, soft_target in dataloader:
    planes, meta, target = planes.to(device), meta.to(device), soft_target.to(device)
    logits = model(planes, meta)
    loss = criterion(logits, target)
```

**Model structure:**
```python
class PolicyCNN(nn.Module):
    def __init__(self):
        self.backbone = nn.Sequential(...)  # conv layers on (batch, 13, 8, 8) planes
        self.head = nn.Linear(...)

    def forward(self, planes, meta):
        x = self.backbone(planes.float())
        return self.head(torch.cat([x.flatten(1), meta.float()], dim=1))
```
