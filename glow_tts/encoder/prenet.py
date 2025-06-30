import torch, torch.nn as nn, torch.nn.functional as F

from .channel_layer_norm import ChannelLayerNorm

class Prenet(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    p_dropout: float

    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    norm1: ChannelLayerNorm
    norm2: ChannelLayerNorm
    norm3: ChannelLayerNorm
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
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding="same")
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding="same")

        self.norm1 = ChannelLayerNorm(hidden_channels)
        self.norm2 = ChannelLayerNorm(hidden_channels)
        self.norm3 = ChannelLayerNorm(hidden_channels)

        self.proj = nn.Conv1d(hidden_channels, in_channels, 1)

        assert self.proj.bias is not None
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        # Apply first Conv -> Norm -> ReLU -> Dropout block.
        # residual: (batch, in_channels, seq_len)
        # seq: (batch, hidden_channels, seq_len)
        residual = seq
        seq = self.conv1(seq * mask)
        seq = self.norm1(seq)
        seq = torch.relu(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)

        # Apply second Conv -> Norm -> ReLU -> Dropout block.
        # seq: (batch, hidden_channels, seq_len)
        seq = self.conv2(seq * mask)
        seq = self.norm2(seq)
        seq = torch.relu(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)

        # Apply third Conv -> Norm -> ReLU -> Dropout block.
        # seq: (batch, hidden_channels, seq_len)
        seq = self.conv3(seq * mask)
        seq = self.norm3(seq)
        seq = torch.relu(seq)
        seq = F.dropout(seq, self.p_dropout, self.training)

        # Project hidden_channels to in_channels and add residual.
        # seq: (batch, in_channels, seq_len)
        seq = self.proj(seq)
        seq = seq + residual

        return seq
