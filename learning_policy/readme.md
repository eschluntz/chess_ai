# Learning a Chess Policy
Learn a policy model to directly predict next moves, and in the process, compare different NN architectures and scales.

## Current Status

**Data pipeline** (`data.py`): Loads positions from Lichess/chess-position-evaluations dataset. Caches 10M shuffled positions locally. Reserves first 10K for eval, rest for training. Extracts first move from Stockfish's recommended line as target.

**Features** (`features.py`): Piece-square encoding (768 features: 6 piece types × 64 squares × 2 colors) plus 11 additional features (turn, castling rights, material balance). Total: 779 features.

**Model** (`mlp_model.py`): Simple 2-layer MLP (input → 256 hidden → 1968 move classes). ~500K parameters.

**Training** (`20_train_policy.py`): Time-based training loop (5 min default), wandb logging, periodic checkpoints. Current best: ~40% top-1 accuracy on eval set.

## Web Interface

Visualize the policy model predictions with an interactive chessboard:

```bash
python interface/server.py --checkpoint checkpoints/<run_id>/latest.pt
```

Then open http://127.0.0.1:5000 in your browser.

Features:
- Probability-weighted arrows for top N moves
- Illegal moves shown in red (model predicts but can't be played)
- Sample moves at temp=0 (greedy) or temp=1 (stochastic)
- Drag pieces to play, reset/undo/flip board
- Set arbitrary positions via FEN

## Datasets

`https://huggingface.co/datasets/Lichess/chess-position-evaluations`
- This contains both stockfish scores (cp/mate) and next move (line)
- 785M rows

`https://huggingface.co/datasets/Lichess/standard-chess-games`
 - Contains full games, and player ELOs, with about 6% of the games include Stockfish analysis evaluations.

## Resources

https://arxiv.org/abs/2409.12272
- "Mastering Chess with a Transformer Model"
- Interesting variants of positional embeddings