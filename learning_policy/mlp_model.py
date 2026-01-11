"""
MLP policy network model.
"""

import torch.nn as nn

from board_repr import BoardState, CompactToFlat

# Output: 64 from * 64 to * 5 promo (none, N, B, R, Q) = 20480 possible moves
# In practice, many are illegal, but we predict the full space
NUM_MOVES = 64 * 64 * 5


class PolicyMLP(nn.Module):
    """MLP for move prediction with configurable depth and width.

    Takes BoardState as input, encodes to flat features internally.
    Output is (batch, 64*64*5) logits for (from_sq, to_sq, promotion).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()

        self.encoder = CompactToFlat()  # -> (batch, 837)
        input_size = 837

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, NUM_MOVES))

        self.net = nn.Sequential(*layers)

    def forward(self, state: BoardState):
        x = self.encoder(state)
        return self.net(x)


def move_to_index(from_sq: int, to_sq: int, promotion: int) -> int:
    """Convert move components to output index."""
    return from_sq * 64 * 5 + to_sq * 5 + promotion


def index_to_move(idx: int) -> tuple[int, int, int]:
    """Convert output index to move components."""
    from_sq = idx // (64 * 5)
    remainder = idx % (64 * 5)
    to_sq = remainder // 5
    promotion = remainder % 5
    return from_sq, to_sq, promotion
