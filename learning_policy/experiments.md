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

| num_train | Epochs | Train Acc | Eval Acc | Eval Top5 | Eval Loss Trend |
|-----------|--------|-----------|----------|-----------|-----------------|
| 10k       | 3,904  | 99.8%     | 4.7%     | 15.8%     | 18 → 47 (explodes) |
| 100k      | 322    | 86.6%     | 7.8%     | 28.8%     | 6 → 18 (explodes) |
| 1M        | 35     | 24.1%     | 13.4%    | 46.2%     | 3.4 → 3.6 (stable) |
| 10M       | 3      | 15.3%     | 14.6%    | 50.6%     | 3.4 → 3.1 (improving) |

### Observations

1. **Overfitting threshold at ~1M samples**: Below 1M, eval loss explodes during training. At 1M+, eval metrics stay stable or improve.

2. **Train/eval gap closes with more data**: At 10M samples, train and eval accuracy are nearly identical (15.3% vs 14.6%), indicating the model is learning generalizable patterns rather than memorizing.