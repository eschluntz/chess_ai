"""
Compact (int8 board) representation and transformations - LEGACY.

This format stores boards as int8[8,8] where each value encodes the piece type.
It requires GPU-side expansion to 12 binary planes during training, which adds
~40% overhead to the forward pass.

The planes format (board_repr.py) pre-expands to 12 planes during precompute,
trading disk space for faster training. Use that format for new experiments.

This file is kept for backwards compatibility with existing compact datasets.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Piece codes: positive=white, negative=black
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

PIECE_CHARS = {
    'P': PAWN, 'N': KNIGHT, 'B': BISHOP, 'R': ROOK, 'Q': QUEEN, 'K': KING,
    'p': -PAWN, 'n': -KNIGHT, 'b': -BISHOP, 'r': -ROOK, 'q': -QUEEN, 'k': -KING,
}

PIECE_SYMBOLS = {
    0: '.',
    PAWN: 'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K',
    -PAWN: 'p', -KNIGHT: 'n', -BISHOP: 'b', -ROOK: 'r', -QUEEN: 'q', -KING: 'k',
}

PROMO_CHARS = {0: '', 1: 'n', 2: 'b', 3: 'r', 4: 'q'}
PROMO_CODES = {'n': 1, 'b': 2, 'r': 3, 'q': 4}


@dataclass
class BoardState:
    """Input features for a batch of chess positions (compact int8 format)."""
    boards: torch.Tensor      # (batch, 8, 8) int8
    turn: torch.Tensor        # (batch,) int8
    castling: torch.Tensor    # (batch, 4) uint8
    en_passant: torch.Tensor  # (batch, 8, 8) uint8

    def to(self, device: torch.device) -> 'BoardState':
        return BoardState(
            boards=self.boards.to(device),
            turn=self.turn.to(device),
            castling=self.castling.to(device),
            en_passant=self.en_passant.to(device),
        )


class CompactToSpatial(nn.Module):
    """
    Expand compact board to spatial CNN format.

    Input: BoardState with boards (batch, 8, 8) int8
    Output: (batch, 18, 8, 8) float32
        - Channels 0-11: piece planes (white P,N,B,R,Q,K, black P,N,B,R,Q,K)
        - Channel 12: turn (1=white, -1=black, broadcast to all squares)
        - Channels 13-16: castling rights (broadcast to all squares)
        - Channel 17: en passant
    """

    def forward(self, state: BoardState) -> torch.Tensor:
        batch = state.boards.shape[0]
        device = state.boards.device

        # Piece planes (12 channels)
        pieces = torch.zeros((batch, 12, 8, 8), dtype=torch.float32, device=device)
        for p in range(1, 7):
            pieces[:, p - 1] = (state.boards == p).float()
            pieces[:, p + 5] = (state.boards == -p).float()

        # Turn plane (1 channel, broadcast)
        turn = state.turn.float()[:, None, None, None].expand(-1, 1, 8, 8)

        # Castling planes (4 channels, broadcast)
        castling = state.castling.float()[:, :, None, None].expand(-1, 4, 8, 8)

        # En passant plane (1 channel)
        en_passant = state.en_passant.float()[:, None, :, :]

        return torch.cat([pieces, turn, castling, en_passant], dim=1)


class CompactToFlat(nn.Module):
    """
    Expand compact board to flat MLP format.

    Input: BoardState with boards (batch, 8, 8) int8
    Output: (batch, 837) float32
        - 768: piece planes flattened
        - 1: turn
        - 4: castling
        - 64: en passant flattened
    """

    def forward(self, state: BoardState) -> torch.Tensor:
        batch = state.boards.shape[0]
        device = state.boards.device

        # Piece planes flattened
        pieces = torch.zeros((batch, 12, 8, 8), dtype=torch.float32, device=device)
        for p in range(1, 7):
            pieces[:, p - 1] = (state.boards == p).float()
            pieces[:, p + 5] = (state.boards == -p).float()
        flat_pieces = pieces.reshape(batch, -1)  # (batch, 768)

        # Other features
        turn = state.turn.float()[:, None]  # (batch, 1)
        castling = state.castling.float()  # (batch, 4)
        en_passant = state.en_passant.float().reshape(batch, -1)  # (batch, 64)

        return torch.cat([flat_pieces, turn, castling, en_passant], dim=1)


def fen_to_compact(fen: str) -> tuple[np.ndarray, np.int8, np.ndarray, np.ndarray]:
    """
    Parse FEN string into compact numpy arrays.

    Returns: (board, turn, castling, en_passant)
        - board: int8[8, 8]
        - turn: int8
        - castling: uint8[4]
        - en_passant: uint8[8, 8]
    """
    parts = fen.split()
    piece_placement, active_color, castling_str, en_passant_str = parts[0], parts[1], parts[2], parts[3]

    # Parse piece placement
    board = np.zeros((8, 8), dtype=np.int8)
    rank, file = 7, 0

    for char in piece_placement:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            board[rank, file] = PIECE_CHARS[char]
            file += 1

    # Parse active color
    turn = np.int8(1 if active_color == 'w' else -1)

    # Parse castling rights (as array, not packed bits)
    castling = np.zeros(4, dtype=np.uint8)
    castling[0] = 1 if 'K' in castling_str else 0
    castling[1] = 1 if 'Q' in castling_str else 0
    castling[2] = 1 if 'k' in castling_str else 0
    castling[3] = 1 if 'q' in castling_str else 0

    # Parse en passant as 8x8 layer
    en_passant = np.zeros((8, 8), dtype=np.uint8)
    if en_passant_str != '-':
        file_idx = ord(en_passant_str[0]) - ord('a')
        rank_idx = int(en_passant_str[1]) - 1
        en_passant[rank_idx, file_idx] = 1

    return board, turn, castling, en_passant


def compact_to_fen(board: np.ndarray, turn: np.int8, castling: np.ndarray, en_passant: np.ndarray) -> str:
    """Convert compact representation back to FEN string (without halfmove/fullmove clocks)."""
    # Piece placement
    rows = []
    for rank in range(7, -1, -1):
        row = ''
        empty_count = 0
        for file in range(8):
            piece = board[rank, file]
            if piece == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += PIECE_SYMBOLS[piece]
        if empty_count > 0:
            row += str(empty_count)
        rows.append(row)
    piece_placement = '/'.join(rows)

    # Active color
    active_color = 'w' if turn == 1 else 'b'

    # Castling rights
    castling_str = ''
    if castling[0]:
        castling_str += 'K'
    if castling[1]:
        castling_str += 'Q'
    if castling[2]:
        castling_str += 'k'
    if castling[3]:
        castling_str += 'q'
    if not castling_str:
        castling_str = '-'

    # En passant
    ep_indices = np.argwhere(en_passant)
    if len(ep_indices) == 0:
        en_passant_str = '-'
    else:
        rank_idx, file_idx = ep_indices[0]
        en_passant_str = chr(ord('a') + file_idx) + str(rank_idx + 1)

    return f"{piece_placement} {active_color} {castling_str} {en_passant_str}"


def to_cnn_numpy(boards: np.ndarray) -> np.ndarray:
    """Convert compact board(s) to CNN format (numpy, for testing)."""
    single = boards.ndim == 2
    if single:
        boards = boards[np.newaxis]

    batch_size = boards.shape[0]
    cnn = np.zeros((batch_size, 12, 8, 8), dtype=np.float32)

    for p in range(1, 7):
        cnn[:, p - 1] = (boards == p)
        cnn[:, p + 5] = (boards == -p)

    if single:
        return cnn[0]
    return cnn


def board_to_ascii(board: np.ndarray) -> str:
    """Render board as ASCII art (rank 8 at top)."""
    lines = []
    lines.append('  a b c d e f g h')
    lines.append('  +-+-+-+-+-+-+-+-+')
    for rank in range(7, -1, -1):
        row = f'{rank + 1} |'
        for file in range(8):
            piece = board[rank, file]
            row += PIECE_SYMBOLS[piece] + '|'
        lines.append(row)
    lines.append('  +-+-+-+-+-+-+-+-+')
    return '\n'.join(lines)


def visualize_cnn(cnn: np.ndarray) -> str:
    """Render CNN tensor as ASCII, showing which squares are active per channel."""
    piece_names = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    lines = []

    for ch_idx, name in enumerate(piece_names):
        channel = cnn[ch_idx]
        if channel.sum() == 0:
            continue

        lines.append(f'\n{name}:')
        for rank in range(7, -1, -1):
            row = f'{rank + 1} '
            for file in range(8):
                row += '#' if channel[rank, file] else '.'
            lines.append(row)
        lines.append('  abcdefgh')

    return '\n'.join(lines)
