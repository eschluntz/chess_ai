# Learning a Chess Policy

Learn a policy model to directly predict next moves from chess positions.

## Files

| File                 | Description                                           |
|----------------------|-------------------------------------------------------|
| `train_modal.py`     | Training script - runs locally or on Modal cloud      |
| `data.py`            | Data loading with HuggingFace datasets + Arrow        |
| `features.py`        | Piece-square feature extraction (779 features)        |
| `mlp_model.py`       | Simple 2-layer MLP policy network                     |
| `interface/server.py`| Web UI to visualize model predictions                 |

## Architecture

**Data Pipeline** (`data.py`):
- Uses HuggingFace `datasets` library with Arrow memory-mapping (handles 784M samples without loading into RAM)
- Shuffles dataset, splits into train/eval (10k eval, rest train)
- Feature extraction happens on-the-fly per batch in `collate_fn`
- Target: first move from Stockfish's recommended line

**Features** (`features.py`):
- Piece-square encoding: 768 features (6 piece types × 64 squares × 2 colors)
- Additional: turn, castling rights, material balance (11 features)
- Total: 779 features per position

**Model** (`mlp_model.py`):
- 2-layer MLP: input(779) → hidden(256) → output(1968 moves)
- ~500K parameters at default hidden size

**Training** (`train_modal.py`):
- Time-based training loop with periodic eval and checkpoints
- Supports resume from checkpoint
- Logs to wandb
- Works locally (`python train_modal.py`) or on Modal (`modal run train_modal.py`)

## Usage

**Train locally:**
```bash
python train_modal.py --max-seconds 300 --hidden-size 256
```

**Train on Modal (cloud GPU):**
```bash
modal run train_modal.py
```

**Visualize predictions:**
```bash
python interface/server.py --checkpoint checkpoints/<run_name>.pt
# Open http://127.0.0.1:5000
```

## Data

Dataset: [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
- 784M positions with Stockfish evaluations
- First run downloads and caches to `cache/` directory (~50GB Arrow format)
- Subsequent runs load instantly via memory-mapping

## Resources

- [Lichess standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games) - Full games with ELOs
- [Mastering Chess with a Transformer Model](https://arxiv.org/abs/2409.12272) - Positional embedding variants

---

# Experiments

## 2026-01-01: Dataset Size vs Overfitting

**Goal**: Determine how much training data is needed to avoid overfitting.

**Setup**:
- Model: 2-layer MLP, 256 hidden units, 705k parameters
- Output: 1968 possible UCI moves
- Features: Piece-square encoding (779 features)
- Training: 5 minutes, batch size 256, lr 0.001
- Eval set: 10k held-out positions

### Results

| num_train | Epochs | Train Acc | Eval Acc | Eval Top5 | Eval Loss Trend      |
|-----------|--------|-----------|----------|-----------|----------------------|
| 10k       | 3,904  | 99.8%     | 4.7%     | 15.8%     | 18 → 47 (explodes)   |
| 100k      | 322    | 86.6%     | 7.8%     | 28.8%     | 6 → 18 (explodes)    |
| 1M        | 35     | 24.1%     | 13.4%    | 46.2%     | 3.4 → 3.6 (stable)   |
| 10M       | 3      | 15.3%     | 14.6%    | 50.6%     | 3.4 → 3.1 (improving)|

### Observations

1. **Overfitting threshold at ~1M samples**: Below 1M, eval loss explodes during training. At 1M+, eval metrics stay stable or improve.

2. **Train/eval gap closes with more data**: At 10M samples, train and eval accuracy are nearly identical (15.3% vs 14.6%), indicating the model is learning generalizable patterns rather than memorizing.

### Conclusion
Re-architected the data loader to use the full 784M rows.

