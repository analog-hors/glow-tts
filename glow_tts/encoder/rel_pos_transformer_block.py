import torch, torch.nn as nn, torch.nn.functional as F

from .rel_pos_mha import RelativePositionalMultiHeadAttention
from .channel_layer_norm import ChannelLayerNorm

class RelativePositionalTransformerBlock(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    num_heads: int
    window_size: int
    p_dropout: float

    attn: RelativePositionalMultiHeadAttention
    norm1: ChannelLayerNorm
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    norm2: ChannelLayerNorm

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_heads: int,
        window_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0

        self.attn = RelativePositionalMultiHeadAttention(
            in_channels,
            in_channels,
            num_heads,
            window_size,
            p_dropout,
        )
        self.norm1 = ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size, padding="same")
        self.norm2 = ChannelLayerNorm(in_channels)

    def forward(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        # Apply self-attention.
        # residual: (batch, in_channels, seq_len)
        # output: (batch, in_channels, seq_len)
        residual = seq
        output = self.attn(seq, seq, seq, mask)

        # Apply dropout, masking, and layer norm.
        # output: (batch, in_channels, seq_len)
        output = F.dropout(output, self.p_dropout, self.training)
        output = self.norm1((output + residual) * mask)

        # Process attention context, masking as appropriate.
        # residual: (batch, in_channels, seq_len)
        # output: (batch, in_channels, seq_len)
        residual = output
        output = self.conv1(output * mask)
        output = torch.relu(output)
        output = F.dropout(output, self.p_dropout, self.training)
        output = self.conv2(output * mask)

        # Apply dropout, masking, and layer norm.
        # output: (batch, seq_len, embed_dim)
        output = F.dropout(output, self.p_dropout, self.training)
        output = self.norm2((output + residual) * mask)

        return output
