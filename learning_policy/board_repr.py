"""
Compact board representation and transformations.

Storage format (136 bytes per sample):
    boards:     int8[8, 8]    - piece codes (0=empty, 1-6=white PNBRQK, -1..-6=black)
    turn:       int8          - 1=white, -1=black
    castling:   uint8[4]      - [K, Q, k, q] as 0/1
    en_passant: uint8[8, 8]   - binary layer (1 at target square, 0 elsewhere)
    from_sq:    uint8         - move origin (0-63)
    to_sq:      uint8         - move destination (0-63)
    promotion:  uint8         - 0=none, 1=N, 2=B, 3=R, 4=Q

Indexing: board[rank, file] where rank 0 = rank 1, file 0 = a-file
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


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class BoardState:
    """Input features for a batch of chess positions."""
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


@dataclass
class MoveLabel:
    """Target labels for a batch of chess positions."""
    from_sq: torch.Tensor     # (batch,) uint8
    to_sq: torch.Tensor       # (batch,) uint8
    promotion: torch.Tensor   # (batch,) uint8

    def to(self, device: torch.device) -> 'MoveLabel':
        return MoveLabel(
            from_sq=self.from_sq.to(device),
            to_sq=self.to_sq.to(device),
            promotion=self.promotion.to(device),
        )

    def to_uci(self, idx: int = 0) -> str:
        """Convert a single move to UCI string."""
        files = 'abcdefgh'
        ranks = '12345678'
        f, t, p = self.from_sq[idx].item(), self.to_sq[idx].item(), self.promotion[idx].item()
        uci = files[f % 8] + ranks[f // 8] + files[t % 8] + ranks[t // 8]
        if p:
            uci += PROMO_CHARS[p]
        return uci


# =============================================================================
# Encoder modules (run on GPU)
# =============================================================================

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


# =============================================================================
# FEN parsing (CPU, for data preprocessing)
# =============================================================================

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


def uci_to_compact(uci: str) -> tuple[np.uint8, np.uint8, np.uint8]:
    """
    Parse UCI move string into compact format.

    Returns: (from_sq, to_sq, promotion)
    """
    files = 'abcdefgh'
    from_file = files.index(uci[0])
    from_rank = int(uci[1]) - 1
    to_file = files.index(uci[2])
    to_rank = int(uci[3]) - 1

    from_sq = np.uint8(from_rank * 8 + from_file)
    to_sq = np.uint8(to_rank * 8 + to_file)
    promotion = np.uint8(PROMO_CODES.get(uci[4], 0) if len(uci) > 4 else 0)

    return from_sq, to_sq, promotion


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


# =============================================================================
# Numpy transforms (for testing/visualization, not training)
# =============================================================================

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


# =============================================================================
# Visualization
# =============================================================================

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


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        fen = ' '.join(sys.argv[1:])
    else:
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    print(f'Input FEN: {fen}\n')

    # Parse
    board, turn, castling, en_passant = fen_to_compact(fen)

    # Show compact representation
    print('=== Compact Representation ===')
    print(f'Turn: {"white" if turn == 1 else "black"}')
    print(f'Castling: {castling} (K={castling[0]}, Q={castling[1]}, k={castling[2]}, q={castling[3]})')
    ep_indices = np.argwhere(en_passant)
    ep_str = '-' if len(ep_indices) == 0 else chr(ord('a') + ep_indices[0][1]) + str(ep_indices[0][0] + 1)
    print(f'En passant: {ep_str}')
    print(f'\nBoard array (rank 0=bottom, file 0=left):')
    print(board)
    print(f'\n{board_to_ascii(board)}')

    # Round-trip check
    reconstructed = compact_to_fen(board, turn, castling, en_passant)
    fen_prefix = ' '.join(fen.split()[:4])
    print(f'\nRound-trip FEN: {reconstructed}')
    print(f'Matches input:  {reconstructed == fen_prefix}')

    # Test BoardState and encoders
    print('\n=== BoardState + Encoders ===')
    state = BoardState(
        boards=torch.tensor(board).unsqueeze(0),
        turn=torch.tensor([turn]),
        castling=torch.tensor(castling).unsqueeze(0),
        en_passant=torch.tensor(en_passant).unsqueeze(0),
    )

    spatial_encoder = CompactToSpatial()
    flat_encoder = CompactToFlat()

    spatial = spatial_encoder(state)
    flat = flat_encoder(state)

    print(f'CompactToSpatial output: {spatial.shape}')  # (1, 18, 8, 8)
    print(f'CompactToFlat output:    {flat.shape}')      # (1, 837)
    print(f'Pieces on board: {int(spatial[0, :12].sum())}')
