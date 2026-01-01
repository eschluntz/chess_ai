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
==================================================
SUMMARY TABLE (All Positions)
==================================================
Model                        MAE   RMSE Spearman  Pearson   Win%      k/s        N
------------------------------------------------------------------------------------------
piece_value_eval*           1506   4941    0.347    0.533   51.7       60    9,577
linear_1000000_alpha10_no   1510   4998    0.488    0.532   67.2       21    9,577
linear_1000000_alpha10_with 1946   3968    0.350    0.619   59.4       21    9,577
piece_position_eval         1517   4938    0.336    0.522   60.2       84    9,577
rf_1000000_estimators500    1517   5059    0.554    0.233   69.6        0    9,577
rf_330000_estimators1000    1502   5053    0.649    0.201   75.5        0    9,577

* = baseline model

==================================================
SUMMARY TABLE (|Score| < 1000cp Subset)
==================================================
Model                        MAE   RMSE Spearman  Pearson   Win%      k/s        N
------------------------------------------------------------------------------------------
piece_value_eval*            132    217    0.257    0.295   49.3       60    8,820
linear_1000000_alpha10_no    118    182    0.415    0.442   65.4       21    8,820
linear_1000000_alpha10_with 1223   2091    0.240    0.282   57.0       21    8,820
piece_position_eval          145    222    0.250    0.304   58.3       84    8,820
rf_1000000_estimators500     108    178    0.524    0.572   68.1        0    8,820
rf_330000_estimators1000      93    159    0.648    0.646   74.3        0    8,820
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

