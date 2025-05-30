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

### Time alignment
There's BIG changes in the distribution over time, so I'm shuffling the first 1M values and drawing everything from there.

## Board Evaluation

See `2-eval-accuracy.py`

eval functions should have the signature: `chess.Board -> centipawn score (max 20k)`

What am I actually optimizing for?
- the ultimate thing I care about is usefulness for a chessbot.
- centipawn loss mean absolute error (MAE): average error
- cp root mean square error (RMSE): same units as cps, but weights extreme differences more.
  - this is probably less important, because distinguishing closely related options is more important
  - BUT I'll probably still use this as my loss function for the NN because it's most standard.
- Spearman correlation: is the ranking of different options correct?
- winner prediction: is advantage positive or negative?
- runtime!


### Results
- `piece_value_eval()` just sums up the value of material
- `piece_position_eval()` heuristic that uses positions of pieces

```
Model                        MAE   RMSE Spearman  Pearson   Win%      k/s        N
------------------------------------------------------------------------------------------
piece_value_eval*           2041   5827    0.375    0.387   47.4      133    9,590
piece_position_eval         2051   5821    0.339    0.391   57.3       94    9,590

(|Score| < 1000cp Subset)
==================================================
Model                        MAE   RMSE Spearman  Pearson   Win%      k/s        N
------------------------------------------------------------------------------------------
piece_value_eval*            119    204    0.272    0.342   43.1      133    8,523
piece_position_eval          132    206    0.235    0.379   54.4       93    8,523
```

## Value Model Training

### Board Representation
- 8x8 binary grids of piece locations (1 layer for each type of piece for each color) seems to be very standard
- extra bits to represent current turn, castling rights, en passant, etc

optional extra features:
- 8x8 binary grid of all squares under attack for each side?

### model 0: Learning a piece value table
[ ] sanity check and understand feature extraction
[ ] is what's printed out including the base value of the piece?
[ ] fix formatting of printed out tables (values too big)



[ ] pick the best value for alpha
[ ] add it to 20-eval-accuracy
[ ] add a function to core/eval.py that uses it

### Learning a random forest

### Learning a neural net
- Conv2D layers at start
- residual network

