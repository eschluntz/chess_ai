import chess

from core.search import negamax
from core.eval import piece_value_eval


def test_negamax_finds_checkmates():
    board_moves = [
        ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0", "Nf6+", 3),
        ("1rb4r/pkPp3p/1b1P3n/1Q6/N3Pp2/8/P1P3PP/7K w - - 1 0", "Qd5+", 3),
        ("4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 0", "Qb8+", 3),
        ("6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRN1K b - - 0 1", "Rg1+", 3),
    ]
    for fen, expected_san, depth in board_moves:
        board = chess.Board(fen)
        score, best_move = negamax(board, piece_value_eval, depth)
        assert abs(score) > 90, f"should have found checkmate (taking into account time discounting) but got {score=}"
        assert (
            board.san(best_move) == expected_san
        ), f"{expected_san=} but got {board.san(best_move)} for {depth=} \n{board}"
