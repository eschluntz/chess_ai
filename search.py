#!/usr/bin/env python

"""
Contains classical search algorithms for playing chess (or other games)
"""
import numpy as np


def negamax(board, eval_fn, max_depth, alpha=-np.inf, beta=np.inf):
    """Finds the best move using the symmetric, negative form of MiniMax and AlphaBeta pruning.

    board: should follow the python-chess API including
        [...] = board.legal_moves
        board.push_move(move)
        last_move = board.pop()

    eval_fn: a function that transforms a board into a score
        score, over = eval_fn(board)

    max_depth: how many more layers to search.
    alpha:  worst possible score for player1 = -inf
    beta:   worst possible score for player2 = +inf

    returns: (score, move) the expected score down that path.
    """

    TIME_DISCOUNT = 0.99

    # base cases
    score, done = eval_fn(board)
    if done or max_depth == 0:
        return score, None

    # are we maxing or mining?
    direction = 1.0 if board.turn else -1.0

    # loop!
    best_move = None
    best_score = -np.inf * direction

    all_moves = board.legal_moves

    # order these nicely to improve alpha beta pruning
    def score_move_heuristic(move):
        board.push(move)
        score, _ = eval_fn(board)
        board.pop()
        return score
    all_moves.sort(key=score_move_heuristic, reverse=board.turn)

    # we've already sorted, just return now (10% speedup)
    if max_depth == 1:
        move = all_moves[0]
        board.move(move)
        score, _ = eval_fn(board)
        board.pop()
        return score * TIME_DISCOUNT, move

    # search the tree!
    for move in all_moves:
        board.move(move)
        score, _ = negamax(board, eval_fn, max_depth - 1, alpha, beta)
        board.pop()

        if score * direction > best_score * direction:
            best_score = score
            best_move = move

        # update alpha/beta
        if direction > 0:
            alpha = max(alpha, score)  # only if max
        else:
            beta = min(beta, score)  # only if min
        if beta <= alpha:  # we know the parent won't choose us. abandon the search!
            break

    return best_score * TIME_DISCOUNT, best_move
