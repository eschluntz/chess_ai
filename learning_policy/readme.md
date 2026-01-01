# Learning a Chess Policy
Learn a policy model to directly predict next moves, and in the process, compare different NN architectures and scales.

## Datasets

`https://huggingface.co/datasets/Lichess/chess-position-evaluations`
- This contains both stockfish scores (cp/mate) and next move (line)

`https://huggingface.co/datasets/Lichess/standard-chess-games`
 - Contains full games, and player ELOs, with about 6% of the games include Stockfish analysis evaluations.

