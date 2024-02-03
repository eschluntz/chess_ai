# chess_ai
AI algorithms for chess, including classical, ML, and RL.

All algorithms are designed around the [python-chess API](https://python-chess.readthedocs.io/en/latest/), supporting:
```
board.push(move)
board.pop()
board.legal_moves
board.is_game_over()
board.outcome()
board.turn
```

## Performance Testing

### Eval function timing
baseline: 4.79, 4.72
raw game_over eval: 3.92, 3.92
global piece value: 3.93, 3.80, 3.87

# Search trimming! how many calls
default:                          102,579    3.97s
without alpha/beta pruning:     1,592,012   59.0s  
without alpha/beta OR sorting:  1,493,872   56.5s
alpha/beta but NO sorting:         10,291    0.68s

transposition table, no sorting: 1.36s, 15% cache hits
transposition table, sorting: 10.29s, 21% cache hits.

WOW I've been making a big mistake. it's totally not worth sorting 
the candidate moves because a bunch of them aren't going to be explored 
anyway, because of alpha beta pruning.