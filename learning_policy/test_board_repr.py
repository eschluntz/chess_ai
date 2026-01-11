"""
Unit tests for board_repr.py
"""

import numpy as np
import pytest
import torch

from board_repr import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    fen_to_compact, compact_to_fen, uci_to_compact,
    to_cnn_numpy,
    BoardState, MoveLabel, CompactToSpatial, CompactToFlat,
)


class TestFenToCompact:
    def test_starting_position(self):
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        board, turn, castling, en_passant = fen_to_compact(fen)

        # Check dimensions
        assert board.shape == (8, 8)
        assert board.dtype == np.int8
        assert castling.shape == (4,)
        assert en_passant.shape == (8, 8)

        # White pieces on ranks 1-2
        assert board[0, 0] == ROOK   # a1
        assert board[0, 1] == KNIGHT # b1
        assert board[0, 4] == KING   # e1
        assert board[1, :].tolist() == [PAWN] * 8  # rank 2 all white pawns

        # Black pieces on ranks 7-8
        assert board[7, 0] == -ROOK   # a8
        assert board[7, 4] == -KING   # e8
        assert board[6, :].tolist() == [-PAWN] * 8  # rank 7 all black pawns

        # Empty squares in middle
        assert board[2:6, :].sum() == 0

        # Metadata
        assert turn == 1  # white to move
        assert castling.tolist() == [1, 1, 1, 1]  # all castling rights
        assert en_passant.sum() == 0  # no en passant

    def test_after_e4(self):
        fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'
        board, turn, castling, en_passant = fen_to_compact(fen)

        # Pawn moved from e2 to e4
        assert board[1, 4] == 0     # e2 empty
        assert board[3, 4] == PAWN  # e4 has white pawn

        # Metadata
        assert turn == -1  # black to move
        assert en_passant[2, 4] == 1  # e3 (rank 3, file e)
        assert en_passant.sum() == 1  # only one square

    def test_partial_castling(self):
        fen = 'r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1'
        board, turn, castling, en_passant = fen_to_compact(fen)

        assert castling.tolist() == [1, 0, 0, 1]  # K and q only

    def test_no_castling(self):
        fen = 'r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1'
        _, _, castling, _ = fen_to_compact(fen)
        assert castling.tolist() == [0, 0, 0, 0]

    def test_complex_position(self):
        # Sicilian Dragon position
        fen = 'r1bqk2r/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQkq - 0 1'
        board, turn, castling, en_passant = fen_to_compact(fen)

        # Spot check some pieces (board[rank_idx, file_idx], rank 1 = index 0)
        assert board[6, 6] == -BISHOP  # g7 (rank 7 = index 6, file g = index 6)
        assert board[3, 3] == KNIGHT   # d4 white knight
        assert board[2, 2] == KNIGHT   # c3 white knight
        assert board[5, 5] == -KNIGHT  # f6 black knight
        assert turn == -1

    def test_en_passant_various_files(self):
        # Test en passant on different files
        for file_char, file_idx in [('a', 0), ('d', 3), ('h', 7)]:
            fen = f'8/8/8/8/8/8/8/8 w - {file_char}3 0 1'
            _, _, _, en_passant = fen_to_compact(fen)
            assert en_passant[2, file_idx] == 1  # rank 3
            assert en_passant.sum() == 1

            fen = f'8/8/8/8/8/8/8/8 w - {file_char}6 0 1'
            _, _, _, en_passant = fen_to_compact(fen)
            assert en_passant[5, file_idx] == 1  # rank 6
            assert en_passant.sum() == 1


class TestCompactToFen:
    def test_roundtrip_starting(self):
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
        board, turn, castling, en_passant = fen_to_compact(fen + ' 0 1')
        result = compact_to_fen(board, turn, castling, en_passant)
        assert result == fen

    def test_roundtrip_complex(self):
        fen = 'r1bqk2r/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQkq -'
        board, turn, castling, en_passant = fen_to_compact(fen + ' 0 1')
        result = compact_to_fen(board, turn, castling, en_passant)
        assert result == fen

    def test_roundtrip_en_passant(self):
        fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3'
        board, turn, castling, en_passant = fen_to_compact(fen + ' 0 1')
        result = compact_to_fen(board, turn, castling, en_passant)
        assert result == fen


class TestUciToCompact:
    def test_simple_move(self):
        from_sq, to_sq, promotion = uci_to_compact('e2e4')
        assert from_sq == 1 * 8 + 4  # e2: rank 2 (idx 1), file e (idx 4)
        assert to_sq == 3 * 8 + 4    # e4: rank 4 (idx 3), file e (idx 4)
        assert promotion == 0

    def test_promotion(self):
        from_sq, to_sq, promotion = uci_to_compact('e7e8q')
        assert from_sq == 6 * 8 + 4  # e7
        assert to_sq == 7 * 8 + 4    # e8
        assert promotion == 4        # queen

    def test_knight_promotion(self):
        from_sq, to_sq, promotion = uci_to_compact('a7a8n')
        assert from_sq == 6 * 8 + 0  # a7
        assert to_sq == 7 * 8 + 0    # a8
        assert promotion == 1        # knight

    def test_corner_moves(self):
        # a1 to h8
        from_sq, to_sq, _ = uci_to_compact('a1h8')
        assert from_sq == 0 * 8 + 0  # a1
        assert to_sq == 7 * 8 + 7    # h8


class TestToCnnNumpy:
    def test_shape_single(self):
        board = np.zeros((8, 8), dtype=np.int8)
        cnn = to_cnn_numpy(board)
        assert cnn.shape == (12, 8, 8)
        assert cnn.dtype == np.float32

    def test_shape_batch(self):
        boards = np.zeros((16, 8, 8), dtype=np.int8)
        cnn = to_cnn_numpy(boards)
        assert cnn.shape == (16, 12, 8, 8)

    def test_white_pawn_channel(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[1, 0] = PAWN  # a2
        board[1, 7] = PAWN  # h2

        cnn = to_cnn_numpy(board)

        # Channel 0 = white pawns
        assert cnn[0, 1, 0] == 1.0  # a2
        assert cnn[0, 1, 7] == 1.0  # h2
        assert cnn[0].sum() == 2.0

        # All other channels should be zero
        assert cnn[1:].sum() == 0.0

    def test_black_knight_channel(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[7, 1] = -KNIGHT  # b8
        board[7, 6] = -KNIGHT  # g8

        cnn = to_cnn_numpy(board)

        # Channel 7 = black knights (6 + KNIGHT - 1 = 7)
        assert cnn[7, 7, 1] == 1.0  # b8
        assert cnn[7, 7, 6] == 1.0  # g8
        assert cnn[7].sum() == 2.0

    def test_starting_position_piece_counts(self):
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        board, _, _, _ = fen_to_compact(fen)
        cnn = to_cnn_numpy(board)

        # White pieces: channels 0-5
        assert cnn[0].sum() == 8   # 8 white pawns
        assert cnn[1].sum() == 2   # 2 white knights
        assert cnn[2].sum() == 2   # 2 white bishops
        assert cnn[3].sum() == 2   # 2 white rooks
        assert cnn[4].sum() == 1   # 1 white queen
        assert cnn[5].sum() == 1   # 1 white king

        # Black pieces: channels 6-11
        assert cnn[6].sum() == 8   # 8 black pawns
        assert cnn[7].sum() == 2   # 2 black knights
        assert cnn[8].sum() == 2   # 2 black bishops
        assert cnn[9].sum() == 2   # 2 black rooks
        assert cnn[10].sum() == 1  # 1 black queen
        assert cnn[11].sum() == 1  # 1 black king


class TestBoardState:
    def test_to_device(self):
        board, turn, castling, en_passant = fen_to_compact(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )

        # Just test that .to() works (CPU to CPU)
        state2 = state.to('cpu')
        assert state2.boards.shape == (1, 8, 8)
        assert state2.turn.shape == (1,)
        assert state2.castling.shape == (1, 4)
        assert state2.en_passant.shape == (1, 8, 8)


class TestMoveLabel:
    def test_to_uci(self):
        label = MoveLabel(
            from_sq=torch.tensor([12]),  # e2: rank 2 (1) * 8 + file e (4) = 12
            to_sq=torch.tensor([28]),    # e4: rank 4 (3) * 8 + file e (4) = 28
            promotion=torch.tensor([0]),
        )
        assert label.to_uci(0) == 'e2e4'

    def test_to_uci_promotion(self):
        label = MoveLabel(
            from_sq=torch.tensor([52]),  # e7
            to_sq=torch.tensor([60]),    # e8
            promotion=torch.tensor([4]), # queen
        )
        assert label.to_uci(0) == 'e7e8q'


class TestCompactToSpatial:
    def test_output_shape(self):
        board, turn, castling, en_passant = fen_to_compact(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )

        encoder = CompactToSpatial()
        output = encoder(state)

        assert output.shape == (1, 18, 8, 8)
        assert output.dtype == torch.float32

    def test_piece_channels(self):
        board, turn, castling, en_passant = fen_to_compact(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )

        encoder = CompactToSpatial()
        output = encoder(state)

        # Check piece planes (first 12 channels)
        assert output[0, 0].sum() == 8   # white pawns
        assert output[0, 5].sum() == 1   # white king
        assert output[0, 6].sum() == 8   # black pawns
        assert output[0, 11].sum() == 1  # black king

    def test_turn_channel(self):
        # White to move
        board, turn, castling, en_passant = fen_to_compact(
            '8/8/8/8/8/8/8/8 w - - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )
        encoder = CompactToSpatial()
        output = encoder(state)
        assert output[0, 12, 0, 0] == 1.0  # turn channel

        # Black to move
        _, turn_b, _, _ = fen_to_compact('8/8/8/8/8/8/8/8 b - - 0 1')
        state.turn = torch.tensor([turn_b])
        output = encoder(state)
        assert output[0, 12, 0, 0] == -1.0

    def test_castling_channels(self):
        board, turn, castling, en_passant = fen_to_compact(
            '8/8/8/8/8/8/8/8 w Kq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )
        encoder = CompactToSpatial()
        output = encoder(state)

        # Castling channels 13-16: K, Q, k, q
        assert output[0, 13, 0, 0] == 1.0  # K
        assert output[0, 14, 0, 0] == 0.0  # Q
        assert output[0, 15, 0, 0] == 0.0  # k
        assert output[0, 16, 0, 0] == 1.0  # q

    def test_en_passant_channel(self):
        board, turn, castling, en_passant = fen_to_compact(
            '8/8/8/8/4P3/8/8/8 b - e3 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )
        encoder = CompactToSpatial()
        output = encoder(state)

        # En passant channel 17: should have 1 at e3 (rank 3, file e)
        assert output[0, 17, 2, 4] == 1.0  # e3
        assert output[0, 17].sum() == 1.0  # only one square


class TestCompactToFlat:
    def test_output_shape(self):
        board, turn, castling, en_passant = fen_to_compact(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )

        encoder = CompactToFlat()
        output = encoder(state)

        # 768 (pieces) + 1 (turn) + 4 (castling) + 64 (en passant) = 837
        assert output.shape == (1, 837)
        assert output.dtype == torch.float32

    def test_piece_features(self):
        board, turn, castling, en_passant = fen_to_compact(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )

        encoder = CompactToFlat()
        output = encoder(state)

        # First 768 features are piece planes
        assert output[0, :768].sum() == 32  # 32 pieces

    def test_turn_feature(self):
        board, turn, castling, en_passant = fen_to_compact('8/8/8/8/8/8/8/8 w - - 0 1')
        state = BoardState(
            boards=torch.tensor(board).unsqueeze(0),
            turn=torch.tensor([turn]),
            castling=torch.tensor(castling).unsqueeze(0),
            en_passant=torch.tensor(en_passant).unsqueeze(0),
        )
        encoder = CompactToFlat()
        output = encoder(state)
        assert output[0, 768] == 1.0  # white

        _, turn_b, _, _ = fen_to_compact('8/8/8/8/8/8/8/8 b - - 0 1')
        state.turn = torch.tensor([turn_b])
        output = encoder(state)
        assert output[0, 768] == -1.0  # black


class TestIntegration:
    """Test full pipeline with real positions."""

    def test_various_positions(self):
        positions = [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # starting
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',  # after e4
            'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',  # Italian
            'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',  # after Nc6
            '8/8/8/8/8/8/8/4K2k w - - 0 1',  # endgame kings only
            'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',  # kiwipete
        ]

        spatial_encoder = CompactToSpatial()
        flat_encoder = CompactToFlat()

        for fen in positions:
            board, turn, castling, en_passant = fen_to_compact(fen)

            state = BoardState(
                boards=torch.tensor(board).unsqueeze(0),
                turn=torch.tensor([turn]),
                castling=torch.tensor(castling).unsqueeze(0),
                en_passant=torch.tensor(en_passant).unsqueeze(0),
            )

            spatial = spatial_encoder(state)
            flat = flat_encoder(state)

            # Basic sanity checks
            assert spatial.shape == (1, 18, 8, 8)
            assert flat.shape == (1, 837)

            # Piece count consistency
            piece_count = spatial[0, :12].sum()
            assert 2 <= piece_count <= 32
            assert flat[0, :768].sum() == piece_count

            # Round-trip FEN
            reconstructed = compact_to_fen(board, turn, castling, en_passant)
            fen_prefix = ' '.join(fen.split()[:4])
            assert reconstructed == fen_prefix


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
