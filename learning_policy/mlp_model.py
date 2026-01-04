"""
MLP policy network model.
"""

import torch.nn as nn


class SimplePolicyMLP(nn.Module):
    """MLP for move prediction with configurable depth and width."""

    def __init__(
        self,
        input_size: int,
        num_moves: int,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()

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

    def forward(self, x):
        return self.net(x)
