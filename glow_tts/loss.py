import torch
import math

def mle_loss(
    sample: torch.Tensor,
    mean: torch.Tensor,
    log_stdev: torch.Tensor,
    logdet_sum: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute loss for Maximum Likelihood Estimation.

    Input shapes:
        - sample: `(batch, mel_channels, mels_len)`
        - mean: `(batch, mel_channels, mels_len)`
        - log_stdev: `(batch, mel_channels, mels_len)`
        - logdet_sum: `(batch)`
        - mask: `(batch, mels_len)`

    Returned shapes:
        - output: `()`
    """

    # Compute NLL without constant term.
    # loss: ()
    loss = 0.5 * torch.sum(((sample - mean) ** 2) * torch.exp(-2.0 * log_stdev))
    loss = loss + torch.sum(log_stdev)
    loss = loss - torch.sum(logdet_sum)
    
    # Scale down loss by total number of predictions in batch.
    # loss: ()
    loss = loss / torch.sum(torch.ones_like(sample) * mask)
    
    # Apply constant term after scaling.
    # loss: ()
    loss = loss + 0.5 * math.log(2.0 * math.pi)

    return loss
