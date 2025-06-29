import torch, torch.nn as nn, torch.nn.functional as F

class ChannelLayerNorm(nn.Module):
    in_channels: int

    norm: nn.LayerNorm

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
        
        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        # Move channel dimension to last, apply norm, and move it back.
        # seq: (batch, in_channels, seq_len)
        seq = seq.transpose(1, 2)
        seq = self.norm(seq)
        seq = seq.transpose(1, 2)

        return seq
