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

![evals](learning/assets/eval_comparisons.png)

```
Model                        MAE   RMSE Spearman  Pearson   Win% k runs/s        N
------------------------------------------------------------------------------------------
piece_value_eval*           1494   4955    0.357    0.575   49.0       96    9,537
linear_10e6_a10_no_mates    1494   5017    0.539    0.541   69.5       19    9,537
linear_10e6_a10_yes_mates   1570   3451    0.370    0.722   59.6       19    9,537
piece_position_eval         1506   4952    0.344    0.561   60.1       79    9,537

 (|Score| < 1000cp Subset)
==================================================
Model                        MAE   RMSE Spearman  Pearson   Win% k runs/s        N
------------------------------------------------------------------------------------------
piece_value_eval*            115    198    0.254    0.290   46.2       96    8,784
linear_10e6_a10_no_mates      96    155    0.472    0.516   67.8       19    8,784
linear_10e6_a10_yes_mates    975   1790    0.241    0.308   56.8       19    8,784
piece_position_eval          130    203    0.249    0.300   58.1       79    8,784
```

## Value Model Training

### Board Representation
- 8x8 binary grids of piece locations (1 layer for each type of piece for each color) seems to be very standard
- extra bits to represent current turn, castling rights, en passant, etc

optional extra features:
- 8x8 binary grid of all squares under attack for each side?

### model 0: Learning a piece value table

![graph](learning/assets/linear_piece_square_sweep_no_mates_v2.png)
![graph](learning/assets/linear_piece_square_sweep_with_mates_v2.png)
- more data the better (10e6)
- alpha doesn't matter too much. 
- `learning/saved_models/linear_piece_square_model_1000000_alpha10_no_mates_v2.pkl`
- `learning/saved_models/linear_piece_square_model_1000000_alpha10_with_mates_v2.pkl`

[x] sanity check and understand feature extraction
[x] fix formatting of printed out tables (values too big)
[x] parameter sweep for with and without checkmates
[x] add it to 20-eval-accuracy
[ ] add a function to core/eval.py that uses it

### Learning a random forest

### Learning a neural net
- Conv2D layers at start
- residual network

