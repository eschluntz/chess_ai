#!/usr/bin/env python

"""
Contains classical search algorithms for playing chess (or other games)
"""
import chess
import random
import math

from utils import Timer

TIME_DISCOUNT = 0.99


def negamax(board, eval_fn, max_depth, alpha=-999999, beta=999999):
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
    # base cases
    score, done = eval_fn(board)
    if done or max_depth == 0:
        return score, None

    # are we maxing or mining?
    direction = 1 if board.turn else -1

    # loop!
    best_move = None
    best_score = -999999 * direction

    # explore captures first as basic move ordering
    all_caps = set(list(board.generate_legal_captures()))
    all_moves = set(list((board.legal_moves)))
    all_moves -= all_caps
    moves_list = list(all_caps) + list(all_moves)

    # search the tree!
    for move in moves_list:
        board.push(move)
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

    return int(best_score * TIME_DISCOUNT), best_move


class MCTSNode:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)

    def UCB1(self, total_simulations, c_param=1.41):
        if self.visits == 0:
            return float("inf")  # Avoid division by zero
        return self.wins / self.visits + c_param * math.sqrt(math.log(total_simulations) / self.visits)

    def select_child(self):
        # Select a child node with the highest UCB1 score
        return max(self.children, key=lambda child: child.UCB1(self.visits))

    def add_child(self, move):
        new_board = self.board.copy(stack=False)
        new_board.push(move)
        child_node = MCTSNode(new_board, move, self)
        self.children.append(child_node)
        self.untried_moves.remove(move)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


def MCTS(root, iterations=1000):
    for _ in range(iterations):
        node = root
        # Selection
        while not node.untried_moves and node.children:  # Node is fully expanded and non-terminal
            node = node.select_child()

        # Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node = node.add_child(move)

        # Simulation
        simulation_board = node.board.copy(stack=False)
        while not simulation_board.is_game_over():
            move = random.choice(list(simulation_board.legal_moves))
            simulation_board.push(move)
        res_to_score = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
        result = res_to_score[simulation_board.result()]

        # Backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

    # Select the move with the highest win ratio
    best_move = max(
        root.children,
        key=lambda child: child.wins / child.visits if child.visits > 0 else 0,
    ).move
    return best_move


if __name__ == "__main__":
    board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0")
    # with Timer("mcts"):

    #     root = MCTSNode(board)
    #     best_move = MCTS(root, 100)  # Using 100 iterations for quick example, more iterations = better move selection
    # print(f"Best move: {best_move}")

    from eval import piece_value_eval

    with Timer("negamax"):
        for _ in range(5):
            _, best_move = negamax(board, piece_value_eval, 5)
    print(f"Best move: {best_move}")
