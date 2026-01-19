# Learning a Chess Policy

Learn a policy model to directly predict next moves from chess positions.

## Files

| File                  | Description                                            |
|-----------------------|--------------------------------------------------------|
| `board_repr.py`       | Compact board format, encoders, BoardState/MoveLabel   |
| `precompute.py` | Precompute dataset to compact numpy format             |
| `data.py`             | Data loading for compact format                        |
| `mlp_model.py`        | Simple MLP policy network                              |
| `train_modal.py`      | Training script (local or Modal)                       |
| `test_board_repr.py`  | Unit tests for board representation                    |
| `interface/server.py` | Web UI to visualize model predictions                  |
| `interface/data_server.py` | Web UI to explore training data                   |

## Data Format

**Compact storage format** (136 bytes per sample):
```
boards:     int8[8, 8]    - piece codes (0=empty, 1-6=white PNBRQK, -1..-6=black)
turn:       int8          - 1=white, -1=black
castling:   uint8[4]      - [K, Q, k, q] as 0/1
en_passant: uint8[8, 8]   - binary layer (1 at target square)
from_sq:    uint8         - move origin (0-63)
to_sq:      uint8         - move destination (0-63)
promotion:  uint8         - 0=none, 1=N, 2=B, 3=R, 4=Q
```

**Data structures** (in `board_repr.py`):
- `BoardState` - batched input features with `.to(device)`
- `MoveLabel` - batched labels with `.to(device)` and `.to_uci()`

**Encoder modules** (run on GPU, part of model):
- `CompactToSpatial(state)` → `(batch, 18, 8, 8)` for CNNs
- `CompactToFlat(state)` → `(batch, 837)` for MLPs

## Usage

**Precompute dataset:**
```bash
python precompute.py --num-samples 1M    # -> cache/compact/1M/   (~140 MB)
python precompute.py --num-samples 50M   # -> cache/compact/50M/  (~6.8 GB)
python precompute.py --num-samples full  # -> cache/compact/full/ (~115 GB)
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

Exploring the data with `precompute_conflicts.py` and the data explorer UI shows ~90%+ of positions have multiple analyses, often at similar depths but recommending different moves. This motivates using soft labels instead of picking a single "correct" move.

### Analysis Sessions

Rows with identical `(fen, depth, knodes)` came from the **same analysis session** (one Stockfish run with Multi-PV enabled). Within a session:
- Multiple rows = multiple principal variations (1st best, 2nd best, etc.)
- `cp` scores are directly comparable
- The move with best `cp` is the engine's top choice

Different `(depth, knodes)` = different sessions (different users, times, or Stockfish versions).

### Soft Label Weighting Strategy

To convert multiple analyses into a probability distribution:

1. **Group by FEN**, then by `(depth, knodes)` to identify sessions

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

3. **Within each session**, max-normalize by score:
   ```python
   weight = exp(-(max_score - score) / temperature)  # best move = 1.0
   ```

4. **Across sessions**, take max (not sum) per move:
   ```python
   move_weight[move] = max(move_weight[move], depth * session_weight)
   ```

5. **Normalize** to get final probabilities

**Why max-normalization instead of softmax?**
- Softmax normalizes so sum=1, which underweights the best move in multi-PV sessions
- Max-normalization (max=1) means 3 equally good moves contribute 3x the evidence of 1 move
- Single-PV and multi-PV sessions contribute equally for the same (move, cp, depth)

**Why max across sessions instead of sum?**
- A depth-40 analysis subsumes a depth-20 analysis (same engine, deeper search tree)
- Agreement across sessions doesn't add information - the deeper analysis already "includes" the shallower one
- A depth-40 finding cp=50 should beat depth-38 finding cp=55: the deeper search likely discovered why cp=55 was optimistic
- Shallower analyses only matter if they found a move the deeper analysis missed entirely

**Temperature**: Controls how much cp differences matter within a session. ~50cp (0.5 pawns) is reasonable.


## Architecture

**Training loop pattern:**
```python
for state, label in dataloader:
    state, label = state.to(device), label.to(device)
    logits = model(state)  # model includes encoder
    loss = criterion(logits, label)
```

**Model structure:**
```python
class CNNPolicy(nn.Module):
    def __init__(self):
        self.encoder = CompactToSpatial()  # (batch, 18, 8, 8)
        self.backbone = nn.Sequential(...)
        self.head = nn.Linear(...)

    def forward(self, state: BoardState):
        x = self.encoder(state)
        x = self.backbone(x)
        return self.head(x.flatten(1))
```