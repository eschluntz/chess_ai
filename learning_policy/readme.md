# Learning a Chess Policy

Learn a policy model to directly predict next moves from chess positions.

## Files

| File                  | Description                                            |
|-----------------------|--------------------------------------------------------|
| `board_repr.py`       | Board representation: FEN↔planes conversion, PlanesToFlat encoder |
| `precompute.py` | Precompute board planes + raw analyses from HuggingFace |
| `data.py`             | Data loading + soft label computation at train time    |
| `mlp_model.py`        | Simple MLP policy network                              |
| `train_modal.py`      | Training script (local or Modal)                       |
| `test_board_repr.py`  | Unit tests for board representation                    |
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

**Precompute dataset:**
```bash
python precompute.py --num-samples 1M    # -> cache/planes/1M/
python precompute.py --num-samples 50M   # -> cache/planes/50M/
python precompute.py --num-samples full  # -> cache/planes/full/
```

**Run tests:**
```bash
python -m pytest test_board_repr.py -v
python board_repr.py  # visualize a FEN
```

**Explore training data:**
```bash
python interface/data_server.py                    # default: cache/planes/1M
python interface/data_server.py --data cache/planes/50M --port 5002
```
Then open http://localhost:5001. Arrow keys to navigate, R for random, hover arrows for probabilities.

## Dataset

Source: [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- 844M rows, 342M unique positions (~2.5 analyses per position on average)
- Crowdsourced from user browsers running Stockfish on Lichess analysis board
- Updated monthly, CC0 license

| Sample count | Storage size | Precompute time |
|--------------|--------------|-----------------|
| 1M           | 140 MB       | ~30 sec         |
| 50M          | 6.8 GB       | ~25 min         |
| 844M (full)  | 115 GB       | ~8 hours        |

**IMPORTANT NOTES ON DATASET! PAY ATTENTION TO THIS WHEN WRITING RELATED CODE**
- The dataset is NOT uniformly distributed. There's distribution drift over time. Therefore to do accurate science it's very important that we shuffle the test/train split across whatever max data size we're working with.
- I want to be able to do small tests locally, but larger runs will always be on Modal.

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

Exploring the data with the analysis scripts and the data explorer UI shows ~90%+ of positions have multiple analyses, often at similar depths but recommending different moves. This motivates using soft labels instead of picking a single "correct" move.

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