import torch, torch.nn as nn, torch.nn.functional as F
import math

from .prenet import Prenet
from .rel_pos_transformer_block import RelativePositionalTransformerBlock
from .duration_predictor import DurationPredictor

class TextEncoder(nn.Module):
    num_symbols: int
    out_channels: int
    embed_channels: int
    hidden_channels: int
    hidden_channels_dp: int
    kernel_size: int
    kernel_size_prenet: int
    num_heads: int
    window_size: int
    num_blocks: int
    p_dropout: float

    embed: nn.Embedding
    prenet: Prenet
    blocks: nn.ModuleList
    proj_mean: nn.Conv1d
    duration_pred: DurationPredictor

    def __init__(
        self,
        num_symbols: int,
        out_channels: int,
        embed_channels: int,
        hidden_channels: int,
        hidden_channels_dp: int,
        kernel_size: int,
        kernel_size_prenet: int,
        num_heads: int,
        window_size: int,
        num_blocks: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_dp = hidden_channels_dp
        self.kernel_size = kernel_size
        self.kernel_size_prenet = kernel_size_prenet
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout

        self.embed = nn.Embedding(num_symbols, embed_channels)

        # Unsure if this does anything.
        # Was in the reference implementation.
        nn.init.normal_(self.embed.weight, 0.0, embed_channels ** -0.5)

        self.prenet = Prenet(
            embed_channels,
            embed_channels,
            kernel_size_prenet,
            p_dropout,
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(RelativePositionalTransformerBlock(
                embed_channels,
                hidden_channels,
                kernel_size,
                num_heads,
                window_size,
                p_dropout,
            ))
        
        self.proj_mean = nn.Conv1d(embed_channels, out_channels, 1)
        self.duration_pred = DurationPredictor(
            embed_channels,
            hidden_channels_dp,
            kernel_size,
            p_dropout,
        )

    def forward(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input shapes:
            - seq: `(batch, seq_len)`
            - mask: `(batch, seq_len)`
        
        Returned shapes:
            - mean: `(batch, out_channels, seq_len)`
            - log_stdev: `(batch, out_channels, seq_len)`
            - log_duration: `(batch, seq_len)`
        """

        # Embed seq and transpose.
        # seq: (batch, embed_channels, seq_len)
        seq = self.embed(seq) * math.sqrt(self.embed_channels)
        seq = seq.transpose(1, 2)

        # Reshape mask to match seq.
        # mask: (batch, 1, seq_len)
        mask = mask.unsqueeze(1)

        # Apply prenet.
        # seq: (batch, embed_channels, seq_len)
        seq = self.prenet(seq, mask)

        # Process seq.
        # seq: (batch, embed_channels, seq_len)
        for block in self.blocks:
            seq = block(seq, mask)

        # Project mean and predict duration.
        # Pass detached tensor to duration predictor to
        # prevent it from affecting the MLE training.
        # log_mean: (batch, out_channels, seq_len)
        # duration: (batch, seq_len)
        mean = self.proj_mean(seq)
        log_duration = self.duration_pred(seq.detach(), mask)

        # Predict a constant stdev of one.
        # log_stdev: (batch, out_channels, seq_len)
        log_stdev = torch.zeros_like(mean)

        return mean, log_stdev, log_duration
