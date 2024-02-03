import chess
import pytest

from eval import _game_over_eval, piece_value_eval

########################################################
# Game over conditions should be the same for all evals!
########################################################
eval_functions = [_game_over_eval, piece_value_eval]


@pytest.mark.parametrize("eval_function", eval_functions)
def test_draw(eval_function):
    board = chess.Board()
    # Force a draw by repetition
    for _ in range(5):
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")
    score, done = eval_function(board)
    assert done
    assert score == 0


@pytest.mark.parametrize("eval_function", eval_functions)
def test_white_win(eval_function):
    board = chess.Board()
    # Simulate a checkmate by white
    board.set_fen("7k/5QQ1/8/8/8/8/8/7K b - - 0 1")
    score, done = eval_function(board)
    assert done
    assert score == 100


@pytest.mark.parametrize("eval_function", eval_functions)
def test_black_win(eval_function):
    board = chess.Board()
    # Simulate a checkmate by black
    board.set_fen("7k/8/8/8/8/8/5qq1/7K w - - 0 1")
    score, done = eval_function(board)
    assert done
    assert score == -100


@pytest.mark.parametrize("eval_function", eval_functions)
def test_ongoing_game(eval_function):
    board = chess.Board()
    # Test an ongoing game
    board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    _, done = eval_function(board)
    assert not done


#####################################
# Tests for individual eval functions
#####################################


def test_piece_value_eval_with_only_kings():
    board = chess.Board("7k/8/8/8/8/8/8/7K w - - 0 1")
    score, done = piece_value_eval(board)
    assert done, "insufficient material"
    assert score == 0, "Score should be 0 when only kings are on the board"


def test_piece_value_eval_with_extra_queen_for_white():
    board = chess.Board("7k/8/8/8/8/8/3Q4/7K w - - 0 1")
    score, done = piece_value_eval(board)
    assert not done
    assert score == 9, "Score should be positive when white has more material"


def test_piece_value_eval_with_extra_queen_for_black():
    board = chess.Board("7k/3q4/8/8/8/8/8/7K w - - 0 1")
    score, done = piece_value_eval(board)
    assert not done
    assert score == -9, "Score should be negative when black has more material"


def test_piece_value_eval_with_complex_position():
    board = chess.Board("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1")
    # 5p6P = +1
    # 2n2n = 0
    # 2b1B = -3
    # 2r = -10
    # 1q = -9
    score, done = piece_value_eval(board)
    assert not done
    assert score == -21, "Score should be -21"
