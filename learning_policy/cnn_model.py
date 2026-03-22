"""
CNN policy network model.
"""

import torch
import torch.nn as nn
from einops import pack, repeat

W, H = 8, 8


class ResCNNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.ln1 = nn.RMSNorm([channels, W, H]).to(dtype=torch.bfloat16)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        self.gelu1 = nn.GELU()
        self.ln2 = nn.RMSNorm([channels, W, H]).to(dtype=torch.bfloat16)
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
        meta_pl = repeat(meta, "b n -> b n h w", h=H, w=W)
        combined, _ = pack([planes, meta_pl], "b * h w")
        return combined.float()  # (batch, 13 + 5, 8, 8)

    def combine_planes_and_meta_flipped(
        self, planes: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        """Normalize board to always be from the current player's perspective.

        When it's black's turn (meta[:, 0] == -1):
        - Swap planes 0-5 (white) with 6-11 (black) → "my pieces" vs "their pieces"
        - Flip ranks so current player's pieces are at the bottom
        - Swap castling rights (mine vs theirs)

        planes (batch, 13, 8, 8). meta (batch, 5). output (batch, 13 + 4, 8, 8).
        Turn channel is dropped since it's always "my turn".
        """
        is_black = (meta[:, 0] == -1)  # (batch,)

        if is_black.any():
            # Swap white/black planes and flip ranks for black's perspective
            flipped_planes = torch.cat([planes[:, 6:12], planes[:, 0:6], planes[:, 12:13]], dim=1)
            flipped_planes = flipped_planes.flip(dims=[-2])

            # Swap castling: [turn, K, Q, k, q] → [turn, k, q, K, Q]
            flipped_meta = torch.stack([meta[:, 0], meta[:, 3], meta[:, 4], meta[:, 1], meta[:, 2]], dim=1)

            mask = is_black[:, None, None, None]  # (batch, 1, 1, 1) for planes
            planes = torch.where(mask, flipped_planes, planes)

            mask_meta = is_black[:, None]  # (batch, 1) for meta
            meta = torch.where(mask_meta, flipped_meta, meta)

        # Drop turn channel (index 0), keep castling rights (my_K, my_Q, their_k, their_q)
        castling = meta[:, 1:5]
        meta_pl = repeat(castling, "b n -> b n h w", h=H, w=W)
        combined, _ = pack([planes, meta_pl], "b * h w")
        return combined.float()  # (batch, 13 + 4, 8, 8)

    def __init__(
        self,
        num_moves: int,
        hidden_channels: int = 32,
        num_layers: int = 1,
        kernel_size: int = 15,
        flip_board: bool = False,
    ):
        super().__init__()
        self.flip_board = flip_board

        # 13 piece/ep planes + meta channels (4 if flipped, 5 if not)
        input_channels = 13 + (4 if flip_board else 5)
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
        self.out_ln = nn.RMSNorm(hidden_channels * W * H).to(dtype=torch.bfloat16)
        self.output = nn.Linear(hidden_channels * W * H, num_moves)

    def forward(self, planes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # planes (batch, 13, 8, 8)
        # meta (batch, 5)
        if self.flip_board:
            x = self.combine_planes_and_meta_flipped(planes, meta)  # (batch, 17, 8, 8)
        else:
            x = self.combine_planes_and_meta(planes, meta)  # (batch, 18, 8, 8)
        x = self.layers(x)  # (batch, hidden_channels, 8, 8)
        x = x.flatten(start_dim=1)  # (batch, hidden_channels * 8 * 8)
        x = self.out_ln(x)
        return self.output(x)  # (batch, num_moves)
