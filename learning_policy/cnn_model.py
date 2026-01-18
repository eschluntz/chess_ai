"""
CNN policy network model.
"""

import torch
import torch.nn as nn

W, H = 8, 8


class ResCNNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.ln1 = nn.LayerNorm([channels, W, H])
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.gelu1 = nn.GELU()
        self.ln2 = nn.LayerNorm([channels, W, H])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.gelu2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gelu1(self.conv1(self.ln1(x)))
        y = self.gelu2(self.conv2(self.ln2(y)))
        return x + y


class PolicyCNN(nn.Module):
    """CNN for move prediction with configurable depth and width.

    Takes Input:
    - planes uint8 (batch, 13, 8, 8),
    - meta: int8[5] - [turn, K, Q, k, q] (batch, 5)
    Output is (batch, num_moves) logits for vocab indices.
    """

    def combine_planes_and_meta(
        self, planes: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        """concat planes and broadcasted meta information into a single tensor.
        planes (batch, 13, 8, 8). meta (batch, 5). output (batch, 13 + 5, 8, 8)"""

        # unsqueeze w,h dims, then expand (Broadcast) those dims to be the right size
        meta_pl = meta[:, :, None, None].expand(-1, -1, H, W)

        return torch.cat((planes, meta_pl), dim=1).float()  # (batch, 13 + 5, 8, 8)

    def __init__(
        self,
        num_moves: int,
        hidden_channels: int = 32,
        num_layers: int = 1,
        kernel_size: int = 15,
    ):
        super().__init__()

        input_channels = 13 + 5  # 12 pieces + en_passant + 5 broadcasted meta channels
        layers = []

        # input -> hidden channels
        layers.append(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding="same")
        )
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers):
            layers.append(ResCNNBlock(hidden_channels, kernel_size))

        self.layers = nn.Sequential(*layers)
        self.out_ln = nn.LayerNorm(hidden_channels * W * H)
        self.output = nn.Linear(hidden_channels * W * H, num_moves)

    def forward(self, planes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # planes (batch, 13, 8, 8)
        # meta (batch, 5)
        x = self.combine_planes_and_meta(planes, meta)  # (batch, 18, 8, 8)
        x = self.layers(x)  # (batch, hidden_channels, 8, 8)
        x = x.flatten(start_dim=1)  # (batch, hidden_channels * 8 * 8)
        x = self.out_ln(x)
        return self.output(x)  # (batch, num_moves)
