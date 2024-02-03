import chess
import pytest

from eval import _game_over_eval, piece_value_eval, piece_position_eval, CHECKMATE_VALUE

########################################################
# Game over conditions should be the same for all evals!
########################################################
eval_functions = [_game_over_eval, piece_value_eval, piece_position_eval]


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
    assert score == CHECKMATE_VALUE


@pytest.mark.parametrize("eval_function", eval_functions)
def test_black_win(eval_function):
    board = chess.Board()
    # Simulate a checkmate by black
    board.set_fen("7k/8/8/8/8/8/5qq1/7K w - - 0 1")
    score, done = eval_function(board)
    assert done
    assert score == -CHECKMATE_VALUE


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
    assert score == 900, "Score should be positive when white has more material"


def test_piece_value_eval_with_extra_queen_for_black():
    board = chess.Board("7k/3q4/8/8/8/8/8/7K w - - 0 1")
    score, done = piece_value_eval(board)
    assert not done
    assert score == -900, "Score should be negative when black has more material"


def test_piece_value_eval_with_complex_position():
    board = chess.Board("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1")
    # 5p6P = +1
    # 2n2n = 0
    # 2b1B = -3
    # 2r = -10
    # 1q = -9
    score, done = piece_value_eval(board)
    assert not done
    assert score == -2100, "Score should be -21"


def test_piece_position_eval_pawn_advancement():
    board_initial = chess.Board("8/2p5/8/8/8/8/2P5/8 w - - 0 1")
    board_advanced = chess.Board("8/8/2p5/8/8/8/2P5/8 w - - 0 1")
    score_initial, _ = piece_position_eval(board_initial)
    score_advanced, _ = piece_position_eval(board_advanced)
    assert score_advanced > score_initial, "Advancing a pawn should increase the score"


def test_piece_position_eval_knight_center_control():
    board_edge = chess.Board()
    board_center = chess.Board("rnbqkbnr/pppppppp/8/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    score_edge, _ = piece_position_eval(board_edge)
    score_center, _ = piece_position_eval(board_center)
    assert score_center > score_edge, "Placing a knight in the center should increase the score"


def test_piece_position_eval_bishop_center():
    board_short_diagonal = chess.Board()
    board_long_diagonal = chess.Board("rnbqkbnr/pppppppp/8/4B3/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")
    score_short_diagonal, _ = piece_position_eval(board_short_diagonal)
    score_long_diagonal, _ = piece_position_eval(board_long_diagonal)
    assert score_long_diagonal > score_short_diagonal, "Placing a bishop on a long diagonal should increase the score"


def test_piece_position_eval_queen_center_control():
    board_edge = chess.Board("8/8/8/8/8/8/8/4Q3 w - - 0 1")
    board_center = chess.Board("8/8/8/3Q4/8/8/8/8 w - - 0 1")
    score_edge, _ = piece_position_eval(board_edge)
    score_center, _ = piece_position_eval(board_center)
    assert score_center > score_edge, "Placing a queen in the center should increase the score. "


def test_piece_position_eval_king_safety():
    board_exposed = chess.Board("rnbq1bnr/pppppppp/8/8/3k4/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    board_safe = chess.Board()
    score_exposed, _ = piece_position_eval(board_exposed)
    score_safe, _ = piece_position_eval(board_safe)
    assert score_safe > score_exposed, "A safer king position should increase the score"


def test_piece_position_eval_starting_position():
    board_starting = chess.Board()
    score_starting, _ = piece_position_eval(board_starting)
    assert score_starting == 0, "The score should be zero for the starting game board"


test_piece_position_eval_king_safety()
