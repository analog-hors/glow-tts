import torch, torch.nn as nn, torch.nn.functional as F

from .wn_block import WavenetStyleBlock

class CouplingBlock(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    num_layers: int
    p_dropout: float

    wn: WavenetStyleBlock

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        assert self.in_channels % 2 == 0

        self.wn = WavenetStyleBlock(
            in_channels // 2,
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            p_dropout,
        )

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
            - logdet: `(batch)`
        """

        # Split seq into two halves channel-wise.
        # x0: (batch, in_channels / 2, seq_len)
        # x1: (batch, in_channels / 2, seq_len)
        x0 = seq[:, :self.in_channels // 2, :]
        x1 = seq[:, self.in_channels // 2:, :]

        # Compute log_scale and shift as a function of x0.
        # log_scale: (batch, in_channels / 2, seq_len)
        # shift: (batch, in_channels / 2, seq_len)
        trans = self.wn(x0, mask)
        log_scale = trans[:, :self.in_channels // 2, :]
        shift = trans[:, self.in_channels // 2:, :]

        # Compute z1 by applying scale and shift to x1, then masking.
        # Also compute the logdet of the Jacobian if going forwards.
        # z1: (batch, in_channels / 2, seq_len)
        # logdet: (batch)
        if not reverse:
            z1 = (torch.exp(log_scale) * x1 + shift) * mask
            logdet = torch.sum(log_scale * mask, (1, 2))
        else:
            z1 = (x1 - shift) * torch.exp(-log_scale) * mask
            logdet = None

        # Combine x0 with z1 to preserve invertibility, producing z.
        # z: (batch, in_channels, seq_len)
        z = torch.cat((x0, z1), 1)

        return z, logdet

    def prepare_for_inference(self):
        self.wn.remove_parametrizations()
