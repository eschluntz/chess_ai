
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

## 2026-01-03: MLP Scaling Experiment and Data Corruption

**Goal**: Find the optimal MLP size for chess move prediction.

**Setup**:
- Sweep over depth (1-3 layers) and width (1024-4096 hidden units)
- 50M training samples, 10k eval
- Training: 1 hour per run, batch size 256, lr 0.001
- All 9 configurations run in parallel on Modal

### Results

| Name              | Eval Acc | Layers | Hidden | Params     |
|-------------------|----------|--------|--------|------------|
| mlp_size_L1_H2048 | 15.59%   | 1      | 2048   | 5,629,872  |
| mlp_size_L1_H4096 | 15.40%   | 1      | 4096   | 11,257,776 |
| mlp_size_L1_H1024 | 14.93%   | 1      | 1024   | 2,815,920  |
| mlp_size_L2_H4096 | 14.59%   | 2      | 4096   | 28,039,088 |
| mlp_size_L2_H2048 | 14.18%   | 2      | 2048   | 9,826,224  |
| mlp_size_L2_H1024 | 14.02%   | 2      | 1024   | 3,865,520  |
| mlp_size_L3_H2048 | 13.11%   | 3      | 2048   | 14,022,576 |
| mlp_size_L3_H4096 | 12.75%   | 3      | 4096   | 44,820,400 |
| mlp_size_L3_H1024 | 12.06%   | 3      | 1024   | 4,915,120  |

![Training loss curves showing periodic spikes](img/train_loss_spikes.png)

### Observations

1. **Deeper models performed worse**: 1-layer MLPs consistently outperformed 2-layer and 3-layer variants, regardless of width. This is counterintuitive - more capacity should help.

2. **Suspicious loss spikes**: All models showed periodic downward spikes in training loss, followed by loss going *higher* than before. These correlated with drops in eval accuracy.

3. **Spikes at identical steps**: When graphing by step instead of wall-clock time, all 9 runs spiked at the exact same training step. This ruled out optimizer dynamics and pointed to corrupted data.

### Root Cause: Corrupted Training Data

The precompute pipeline used memory-mapped numpy files (`np.lib.format.open_memmap`) which pre-allocate with zeros. During an earlier run, one worker stalled mid-write. The resume logic only checked file length, not content - so the partially-written shard (real data followed by thousands of zero rows) was marked as complete.

Result: Contiguous blocks of all-zero feature vectors in the training data. The model could easily "learn" these (predict most common move for zero input), causing:
- Sharp drop in training loss on zero batches
- Overfitting that damaged generalization
- Higher loss on subsequent real data

### Fix

1. Added validation: check that last and middle rows of each shard are non-zero before concatenating
2. Re-ran precomputation from scratch

NOTE: Reran with the fix - spikes were gone, loss curve was the same.

## 2026-01-11: Training Pipeline Optimization

**Goal**: Have a flexible data representation so that I can do experiments on features without needing to rerun pre-compute. 

Iterations:
1. **Compact + 20k output**: Store boards as int8 piece codes (64 bytes/sample), expand to 12 binary planes on GPU. Output all 64×64×5 = 20,480 possible moves.
2. **Compact + vocab**: Same storage, but build vocab of ~1,968 actually-occurring moves. Reduces output layer from 21M to 2M params.
3. **Planes**: Precompute boards as 13 binary uint8 planes (12 pieces + en_passant). GPU work reduces to single `.float()` cast.

**Setup**: L1 H1024, 50M samples.

| Iteration               | Throughput | Speedup | Bottleneck               | Fix                               |
|-------------------------|------------|---------|--------------------------|-----------------------------------|
| Compact + 20k output    | 14,700/sec | 1.0x    | Output layer (21M params)| Build vocab of ~1,968 moves       |
| Compact + vocab         | 29,400/sec | 2.0x    | GPU expansion (fwd 43%)  | Precompute binary planes          |
| Pre-expanded planes     | 83,400/sec | 5.7x    | Backward pass (36%)      | —                                 |

![img](img/throughput_eval.png)

![img](img/throughput_samples.png)

**Storage tradeoff**: Planes format is 42 GB vs 6.8 GB for compact (6x larger), but 2.8x faster than compact+vocab.

**Final state**: Backward pass (actual gradient compute) is now the dominant cost. One epoch over 50M samples takes ~10 minutes.

## 2026-01-17: CNN Kernel Size vs Depth

**Goal**: Compare small-kernel-deep vs large-kernel-shallow CNNs at constant parameter count (~10M).

**Setup**:
- Architecture: ResNet-style CNN with pre-norm, GELU, 2 convs per residual block
- Input: 18 channels (13 piece planes + 5 broadcasted meta channels)
- Output: Linear head to 1,968 moves (~8M params, constant across configs)
- Fixed: hidden_channels=64, batch_size=256, lr=0.001
- Training: 1 hour per run, 50M samples
- Sweep: kernel_size with num_layers adjusted to keep conv params ~2M

| Config | Kernel | Layers | Conv Params | Receptive Field |
|--------|--------|--------|-------------|-----------------|
| K3_L27 | 3      | 27     | ~2.0M       | 55 squares      |
| K5_L10 | 5      | 10     | ~2.0M       | 41 squares      |
| K7_L5  | 7      | 5      | ~2.0M       | 31 squares      |
| K9_L3  | 9      | 3      | ~2.0M       | 25 squares      |
| K11_L2 | 11     | 2      | ~2.0M       | 21 squares      |
| K15_L1 | 15     | 1      | ~1.8M       | 15 squares      |

### Results

| Config   | Kernel | Layers | Params     | Eval Acc |
|----------|--------|--------|------------|----------|
| K3_L27   | 3      | 27     | 10,518,000 | **20.49%** |
| K5_L10   | 5      | 10     | 10,313,072 | 17.92%   |
| K7_L5    | 7      | 5      | 10,217,200 | 17.39%   |
| K9_L3    | 9      | 3      | 10,204,656 | 16.55%   |
| K11_L2   | 11     | 2      | 10,226,032 | 15.87%   |
| K15_L1   | 15     | 1      | 10,190,064 | 15.44%   |

### Conclusions

1. **Small kernels + depth wins**: Despite K15_L1 having best early accuracy (large receptive field learns fast), K3_L27 ultimately won by 5 percentage points. The ranking is almost perfectly monotonic with kernel size.

2. **Depth enables hierarchical features**: Small 3×3 kernels can only see local patterns, but stacking 27 layers lets the network build increasingly abstract representations. This beats "seeing everything at once" with a 15×15 kernel.

**Next steps**: Try even deeper networks with K=3, explore residual connection variants, add torch.compile() for speedup.