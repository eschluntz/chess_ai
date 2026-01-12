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

## Dataset

Source: [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- 844M positions with Stockfish evaluations
- Target: first move from Stockfish's recommended line

| Sample count | Storage size | Precompute time |
|--------------|--------------|-----------------|
| 1M           | 140 MB       | ~30 sec         |
| 50M          | 6.8 GB       | ~25 min         |
| 844M (full)  | 115 GB       | ~8 hours        |

**IMPORTANT NOTES ON DATASET! PAY ATTENTION TO THIS WHEN WRITING RELATED CODE**
- The dataset is NOT uniformly distributed, There's distribution drift over time. Therefor to do accurate science it's very important that we shuffle the test/train split across whatever max data size we're working with.
- I want to be able to do small tests locally, but larger runs will always be on Modal.


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