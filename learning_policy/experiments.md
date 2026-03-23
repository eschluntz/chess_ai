# CNN State of the Art (2026-03)

### Best result

| Model | H | Params | Steps | Epochs | Eval Acc | Deep Acc | Time |
|-------|---|--------|-------|--------|----------|----------|------|
| scale-20M | 128 | 20.5M | 533k | 1.6 | **50.5%** | **58.0%** | 10h |
| scale-40M | 224 | 41.7M | 199k | 0.6 | 50.4% | 57.9% | 10h |

20M and 40M are tied at 10 hours, but 40M saw only 0.6 epochs vs 1.6 — per-sample, 40M learns faster. Neither is converged; both would improve with more compute.

### Architecture

- **Backbone**: ResNet-style CNN, K=3 kernels, L=14 residual blocks (2 convs each), pre-norm RMSNorm + GELU
- **Input**: 13 piece/EP planes + 4 castling channels (17 total with flip), board normalized to current player's perspective
- **Output**: flatten → `RMSNorm → Linear(H*64, 1968)`

### Training recipe

- **Optimizer**: AdamW, lr=1e-3, wd=0.01, cosine decay
- **Batch**: bs=1024, BF16 autocast, `torch.compile`
- **Data**: full 342M unique positions, soft labels (Boltzmann-weighted cp at max depth, temp=50cp), min_depth=0, flip_board=True
- **Infra**: Modal A10G, background prefetch (queue=2), eval every 120s on 50k held-out

### What we know (solid)

- **K=3 deep > large-kernel shallow** (5pp, monotonic)
- **Flip helps ~2.5pp** — learns color-symmetric patterns once
- **BF16 is free** — 1.6× speedup, no accuracy loss
- **Cosine > constant LR** at iso-step (direction solid, magnitude likely inflated)
- **wd=0.01 > wd=0** (~1pp, borderline)
- **Policy is local** — per-from-square features suffice; spatial_unshared head matches flatten
- **From×to interaction matters** (factored head −4.2pp), **per-square specialization matters** (shared −2.7pp vs unshared)
- **Not data-limited** — train/eval gaps <0.5%, larger models have steeper per-sample log-slopes

### What we don't know (revisit for transformers)

- **Depth**: L=14 was chosen under an old recipe; the 14-vs-27 comparison was 0.16pp (noise). Re-sweep.
- **Batch size**: never tested with LR scaling; bs=1024 might not be optimal.
- **LR warmup**: never tested; larger models (80M+) may need it to not diverge early.
- **Actual scaling ceiling**: log-linear extrapolation suggests 40M→58.7% at 10× compute, but unvalidated.

### What to carry to transformers

- Flip, BF16, prefetch, cosine, wd=0.01 — all validated, port directly
- Use `spatial_unshared` head (per-token C→68 decoder), not flatten — verified equivalent, much smaller
- Start with per-square tokenization (64 tokens) — fixed-length, maps naturally to the spatial head
- Re-sweep depth, batch size, and LR from scratch — CNN values are likely suboptimal for attention

# Transformer Experiments

# TODOs
