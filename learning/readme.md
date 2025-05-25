# Supervised Learning Chess

Two goals:
1. Learn a value function to score board positions
2. Learn a policy model to directly predict next moves

Both of these will be "pretraining" incorporated into a future RL Actor Critic Model

## Datasets
I want to train a single model with two heads, predicting both board balue and next move,
so I need a dataset that has both of those in each row. (Ideally also previous move, as I
think that contains some good signal even though the board is techncially markov)

### https://huggingface.co/datasets/Lichess/chess-position-evaluations

```
232,637,106 chess positions evaluated with Stockfish at various depths and node count. Produced by, and for, the Lichess analysis board, running various flavours of Stockfish within user browsers. This version of the dataset is a de-normalized version of the original dataset and contains 548,822,600 rows.

One row of the dataset looks like this:
{
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4",
  "depth": 36,
  "knodes": 206765,
  "cp": 311,
  "mate": None
}
fen: string, the position FEN only contains pieces, active color, castling rights, and en passant square.
line: string, the principal variation, in UCI format.
depth: string, the depth reached by the engine.
knodes: int, the number of kilo-nodes searched by the engine.
cp: int, the position's centipawn evaluation. This is None if mate is certain.
mate: int, the position's mate evaluation. This is None if mate is not certain.
```

Usage:
```
from datasets import load_dataset
ds = load_dataset("Lichess/chess-position-evaluations")
```

### Backup datasets:
https://huggingface.co/datasets/Lichess/standard-chess-games
- Contains full games, but only About 6% of the games include Stockfish analysis evaluations: [%eval 2.35] (235 centipawn advantage), [%eval #-4] (getting mated in 4), always from White's point of view.

https://csslab.cs.toronto.edu/datasets/#maia_kdd

## Board Evaluation

format: `board -> centipawn score`

I should create a eval set to test different algorithms and models on this, and measure loss

Input type should be a `chess.Board object` 

