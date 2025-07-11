import torch, torch.nn as nn, torch.nn.functional as F

from .act_norm import ActNorm
from .inv_conv_near import InvConvNear
from .coupling_block import CouplingBlock

class FlowDecoder(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    num_wn_layers: int
    num_splits: int
    num_squeeze: int
    num_blocks: int
    p_dropout: float

    flows: nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_wn_layers: int,
        num_splits: int,
        num_squeeze: int,
        num_blocks: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_wn_layers = num_wn_layers
        self.num_splits = num_splits
        self.num_squeeze = num_squeeze
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0
        assert in_channels * num_squeeze % num_splits == 0
        assert num_splits % 2 == 0

        self.flows = nn.ModuleList()
        for i in range(num_blocks):
            self.flows.append(ActNorm(in_channels * num_squeeze))
            self.flows.append(InvConvNear(in_channels * num_squeeze, num_splits))
            self.flows.append(CouplingBlock(
                in_channels * num_squeeze,
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_wn_layers,
                p_dropout,
            ))

    def forward(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
            - logdet_sum: `(batch)`
        """

        batch, in_channels, seq_len = seq.shape
        num_squeeze = self.num_squeeze
        trimmed_seq_len = seq_len - seq_len % num_squeeze

        # Trim seq and mask to be a multiple of num_squeeze.
        # seq: (batch, in_channels, trimmed_seq_len)
        # mask: (batch, 1, trimmed_seq_len)
        seq = seq[:, :, :trimmed_seq_len]
        mask = mask[:, :, :trimmed_seq_len]

        # Squeeze by stacking two timesteps across the channel dimension.
        # The mask selects the last element of each squeeze block.
        # This prevents padding from mixing with non-padding across the new channels.
        # seq: (batch, in_channels * num_squeeze, trimmed_seq_len // num_squeeze)
        # mask: (batch, 1, trimmed_seq_len // num_squeeze)
        seq = seq.reshape((batch, in_channels, trimmed_seq_len // num_squeeze, num_squeeze))
        seq = seq.movedim(3, 1)
        seq = seq.reshape((batch, in_channels * num_squeeze, trimmed_seq_len // num_squeeze))
        mask = mask[:, :, num_squeeze - 1::num_squeeze]

        # Apply flows, and compute Jacobian logdet sum if forwards.
        # seq: (batch, in_channels * num_squeeze, trimmed_seq_len // num_squeeze)
        # logdet_sum: (batch)
        if not reverse:
            logdet_sum = torch.zeros(seq.shape[0], device=seq.device)
            for flow in self.flows:
                seq, logdet = flow(seq, mask)
                logdet_sum += logdet
        else:
            logdet_sum = None
            for flow in reversed(self.flows):
                seq, _logdet = flow(seq, mask, reverse=True)

        # Unsqueeze and pad back to seq_len.
        # seq: (batch, in_channels, seq_len)
        seq = seq.reshape((batch, num_squeeze, in_channels, trimmed_seq_len // num_squeeze))
        seq = seq.movedim(1, 3)
        seq = seq.reshape((batch, in_channels, trimmed_seq_len))
        seq = F.pad(seq, (0, seq_len - trimmed_seq_len), "reflect")

        return seq, logdet_sum

    def prepare_for_inference(self):
        for flow in self.flows:
            flow.prepare_for_inference() # type: ignore
