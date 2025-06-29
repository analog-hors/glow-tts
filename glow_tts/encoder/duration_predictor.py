import torch, torch.nn as nn, torch.nn.functional as F

from .channel_layer_norm import ChannelLayerNorm

class DurationPredictor(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    p_dropout: float

    conv1: nn.Conv1d
    norm1: ChannelLayerNorm
    conv2: nn.Conv1d
    norm2: ChannelLayerNorm
    proj: nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same")
        self.norm1 = ChannelLayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding="same")
        self.norm2 = ChannelLayerNorm(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, seq_len)`
        """

        # Apply first Conv -> ReLU -> Norm -> Dropout block.
        # seq: (batch, hidden_channels, seq_len)
        seq = self.conv1(seq * mask)
        seq = torch.relu(seq)
        seq = self.norm1(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)

        # Apply second Conv -> ReLU -> Norm -> Dropout block.
        # seq: (batch, hidden_channels, seq_len)
        seq = self.conv2(seq * mask)
        seq = torch.relu(seq)
        seq = self.norm2(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)

        # Project hidden_channels to 1, then squeeze.
        # seq: (batch, seq_len)
        seq = self.proj(seq)
        seq = seq.squeeze(1)

        return seq
