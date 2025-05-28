# Supervised Learning Chess

Two goals:
1. Learn a value function to score board positions
2. Learn a policy model to directly predict next moves

Both of these will be "pretraining" incorporated into a future RL Actor Critic Model

## Datasets
I want to train a single model with two heads, predicting both board balue and next move,
so I need a dataset that has both of those in each row. (Ideally also previous move, as I
think that contains some good signal even though the board is techncially markov)

`https://huggingface.co/datasets/Lichess/chess-position-evaluations`

This contains both scores (cp/mate) and next move (line)!

```
{
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4",
  ...
  "cp": 311,
  "mate": None
}
```

More details in `1-data-exploration.ipynb`

Backup datasets:
- https://huggingface.co/datasets/Lichess/standard-chess-games
    - Contains full games, but only About 6% of the games include Stockfish analysis evaluations: 
- https://csslab.cs.toronto.edu/datasets/#maia_kdd

### sanity checks on data 
[x] test a very positive and very negative data points, and make sure my baseline agrees with the sign.

## Board Evaluation

See `2-eval-accuracy.py`

eval functions should have the signature: `chess.Board -> centipawn score (max 20k)`

What am I actually optimizing for?
- centipawn loss mean absolute error (MAE): average error
- cp root mean square error (RMSE): same units as cps, but weights extreme differences more.
  - this is probably less important, because distinguishing closely related options is more important
  - BUT I'll probably still use this as my loss function for the NN because it's most standard.
- Spearman correlation: is the ranking of different options correct?
- winner prediction: is advantage positive or negative?


### Results
- `piece_value_eval()` just sums up the value of material
- `piece_position_eval()` heuristic that uses positions of pieces

```
Model                        MAE   RMSE Spearman  Pearson   Win%        N
-------------------------------------------------------------------------
piece_value_eval*           2041   5827    0.375    0.387   47.4    9,590
piece_position_eval         2051   5821    0.339    0.391   57.3    9,590
```


## Value Model Training

### Board Representation
- 8x8 binary grids of piece locations (1 layer for each type of piece for each color) seems to be very standard
- extra bits to represent current turn, castling rights, en passant, etc

optional extra features:
- 8x8 binary grid of all squares under attack for each side?

### Learning a piece value table

### Learning a random forest

### Learning a neural net
- Conv2D layers at start
- residual network

