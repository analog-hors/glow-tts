import torch, torch.nn as nn, torch.nn.functional as F

class ActNorm(nn.Module):
    in_channels: int

    log_scale: torch.Tensor
    shift: torch.Tensor
    initialized: torch.Tensor

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.log_scale = nn.Parameter(torch.zeros((1, in_channels, 1)))
        self.shift = nn.Parameter(torch.zeros((1, in_channels, 1)))
        self.register_buffer("initialized", torch.tensor(False), persistent=True)

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

        if not self.initialized.item():
            with torch.no_grad():
                # Compute per-channel mean, variance, and log stdev.
                # total: ()
                # mean: (in_channels)
                # mean_sq: (in_channels)
                # variance: (in_channels)
                # log_stdev: (in_channels)
                total = torch.sum(mask)
                mean = torch.sum(seq * mask, (0, 2)) / total
                mean_sq = torch.sum(seq * seq * mask, (0, 2)) / total
                variance = mean_sq - mean * mean
                log_stdev = 0.5 * torch.log(torch.clamp_min(variance, 1e-6))

                # Compute per-channel initializations for shift and log scale.
                # shift_init: (in_channels)
                # log_scale_init: (in_channels)
                shift_init = -mean * torch.exp(-log_stdev)
                log_scale_init = -log_stdev

                # Initialize parameters.
                self.shift.copy_(shift_init.unsqueeze(0).unsqueeze(2))
                self.log_scale.copy_(log_scale_init.unsqueeze(0).unsqueeze(2))
                self.initialized.fill_(True)

        # Scale, shift, and mask.
        # Also compute the logdet of the Jacobian if going forwards.
        # seq: (batch, in_channels, seq_len)
        # logdet: (batch)
        if not reverse:
            lengths = torch.sum(mask, (1, 2))
            seq = (torch.exp(self.log_scale) * seq + self.shift) * mask
            logdet = torch.sum(self.log_scale) * lengths
        else:
            seq = (seq - self.shift) * torch.exp(-self.log_scale) * mask
            logdet = None

        return seq, logdet

    def prepare_for_inference(self):
        pass
