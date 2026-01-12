"""
MLP policy network model.
"""

import torch.nn as nn

from board_repr import BoardState, CompactToFlat


class PolicyMLP(nn.Module):
    """MLP for move prediction with configurable depth and width.

    Takes BoardState as input, encodes to flat features internally.
    Output is (batch, num_moves) logits for vocab indices.
    """

    def __init__(
        self,
        num_moves: int,
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
        layers.append(nn.Linear(hidden_size, num_moves))

        self.net = nn.Sequential(*layers)

    def forward(self, state: BoardState):
        x = self.encoder(state)
        return self.net(x)
