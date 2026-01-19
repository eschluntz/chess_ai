"""
Board representation using pre-expanded 13-plane format.

Storage format per sample:
    planes: uint8[13, 8, 8] - 12 piece planes + en_passant as 13th
    meta:   int8[5]         - [turn, K, Q, k, q]
    moves:  uint8[3]        - [from_sq, to_sq, promotion]

Plane order: white P,N,B,R,Q,K (0-5), black P,N,B,R,Q,K (6-11), en_passant (12)
Indexing: planes[plane, rank, file] where rank 0 = rank 1, file 0 = a-file
"""

import numpy as np
import torch
import torch.nn as nn

PROMO_CODES = {'n': 1, 'b': 2, 'r': 3, 'q': 4}


class PlanesToFlat(nn.Module):
    """
    Flatten pre-expanded planes to MLP format.

    Input: planes (batch, 13, 8, 8), meta (batch, 5)
    Output: (batch, 837) float32
        - 832: 13 planes flattened (12 pieces + en_passant)
        - 5: meta (turn + 4 castling rights)
    """

    def forward(self, planes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        batch = planes.shape[0]
        flat_planes = planes.reshape(batch, -1).float()  # (batch, 832)
        return torch.cat([flat_planes, meta.float()], dim=1)  # (batch, 837)


def fen_to_planes(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse FEN string into consolidated format.

    Returns: (planes, meta)
        - planes: uint8[13, 8, 8] - 12 piece planes + en_passant
        - meta: int8[5] - [turn, K, Q, k, q]
    """
    parts = fen.split()
    piece_placement, active_color, castling_str, en_passant_str = parts[0], parts[1], parts[2], parts[3]

    # Parse piece placement into 13 planes (12 pieces + en_passant)
    planes = np.zeros((13, 8, 8), dtype=np.uint8)
    rank, file = 7, 0

    plane_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    for char in piece_placement:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            planes[plane_idx[char], rank, file] = 1
            file += 1

    # En passant as 13th plane
    if en_passant_str != '-':
        file_idx = ord(en_passant_str[0]) - ord('a')
        rank_idx = int(en_passant_str[1]) - 1
        planes[12, rank_idx, file_idx] = 1

    # Meta: [turn, K, Q, k, q]
    meta = np.zeros(5, dtype=np.int8)
    meta[0] = 1 if active_color == 'w' else -1
    meta[1] = 1 if 'K' in castling_str else 0
    meta[2] = 1 if 'Q' in castling_str else 0
    meta[3] = 1 if 'k' in castling_str else 0
    meta[4] = 1 if 'q' in castling_str else 0

    return planes, meta


def uci_to_move_tuple(uci: str) -> tuple[np.uint8, np.uint8, np.uint8]:
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
