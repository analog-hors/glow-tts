import torch, torch.nn as nn, torch.nn.functional as F

class ActNorm(nn.Module):
    in_channels: int

    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.scale = nn.Parameter(torch.zeros((1, in_channels, 1)))
        self.shift = nn.Parameter(torch.zeros((1, in_channels, 1)))

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

        # Scale, shift, and mask.
        # Also compute the logdet of the Jacobian if going forwards.
        # seq: (batch, in_channels, seq_len)
        # logdet: (batch)
        if not reverse:
            lengths = torch.sum(mask, (1, 2))
            seq = (torch.exp(self.scale) * seq + self.shift) * mask
            logdet = torch.sum(self.scale) * lengths
        else:
            seq = (seq - self.shift) * torch.exp(-self.scale) * mask
            logdet = None

        return seq, logdet

    def prepare_for_inference(self):
        pass
