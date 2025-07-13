import torch, torch.nn as nn, torch.nn.functional as F
import math
from dataclasses import dataclass

from .decoder.flow_decoder import FlowDecoder
from .encoder.text_encoder import TextEncoder
from .hparams import Hyperparameters
from .mas import monotonic_alignment_search

@dataclass
class GlowTTSForwardResult:
    text_mask: torch.Tensor
    mels_mask: torch.Tensor

    latent: torch.Tensor
    mean: torch.Tensor
    log_stdev: torch.Tensor
    logdet_sum: torch.Tensor
    
    log_duration: torch.Tensor
    attn_weights: torch.Tensor

class GlowTTS(nn.Module):
    hparams: Hyperparameters

    encoder: TextEncoder
    decoder: FlowDecoder

    def __init__(self, hparams: Hyperparameters):
        super().__init__()
        self.hparams = hparams

        self.encoder = TextEncoder(
            hparams.num_symbols,
            hparams.mel_channels,
            hparams.enc_embed_channels,
            hparams.enc_hidden_channels,
            hparams.enc_hidden_channels_dp,
            hparams.enc_kernel_size,
            hparams.enc_kernel_size_prenet,
            hparams.enc_num_heads,
            hparams.enc_window_size,
            hparams.enc_num_blocks,
            hparams.dec_p_dropout,
        )
        self.decoder = FlowDecoder(
            hparams.mel_channels,
            hparams.dec_hidden_channels,
            hparams.dec_kernel_size,
            hparams.dec_dilation_rate,
            hparams.dec_num_wn_layers,
            hparams.dec_num_splits,
            hparams.dec_num_squeeze,
            hparams.dec_num_blocks,
            hparams.dec_p_dropout,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mels: torch.Tensor,
        mels_lengths: torch.Tensor,
    ) -> GlowTTSForwardResult:
        """
        Input shapes:
            - text: `(batch, text_len)`
            - text_length: `(batch)`
            - mels: `(batch, mel_channels, mels_len)`
            - mels_length: `(batch)`
        
        Returned shapes:
            - text_mask: `(batch, text_len)`
            - mels_mask: `(batch, text_len)`
            - latent: `(batch, mel_channels, mels_len)`
            - mean: `(batch, mel_channels, mels_len)`
            - log_stdev: `(batch, mel_channels, mels_len)`
            - logdet_sum: `(batch)`
            - log_duration: `(batch, text_len)`
            - attn_weights: `(batch, text_len, mels_len)`
        """

        # Create masks for text and mels.
        # text_mask: (batch, text_len)
        # mels_mask: (batch, 1, mels_len)
        text_mask = _get_mask_from_lengths(text_lengths, text.shape[1])
        mels_mask = _get_mask_from_lengths(mels_lengths, mels.shape[2])
        mels_mask = mels_mask.unsqueeze(1)

        # Encode text into text_mean, text_log_stdev, and duration
        # text_mean: (batch, mel_channels, text_len)
        # text_log_stdev: (batch, mel_channels, text_len)
        # log_duration: (batch, text_len)
        # text_latent_mask: (batch, 1, text_len)
        text_mean, text_log_stdev, log_duration = self.encoder(text, text_mask)

        # Encode mels into latent space and obtain logdet sum.
        # mels_latent: (batch, mel_channels, mels_len)
        # logdet_sum: (batch)
        mels_latent, logdet_sum = self.decoder(mels, mels_mask)

        # Compute scores and attention weights.
        # scores: (batch, text_len, mels_len)
        # attn: (batch, text_len, mels_len)
        with torch.no_grad():
            scores = _compute_log_likelihood_matrix(mels_latent, text_mean, text_log_stdev)
            attn_weights = monotonic_alignment_search(scores, text_lengths, mels_lengths)

        # Upsample text_mean and text_log_stdev to mels_mean and mels_log_stdev.
        # mels_mean: (batch, mel_channels, mels_len)
        # mels_log_stdev: (batch, mel_channels, mels_len)
        mels_mean = torch.matmul(text_mean, attn_weights)
        mels_log_stdev = torch.matmul(text_log_stdev, attn_weights)

        return GlowTTSForwardResult(
            text_mask,
            mels_mask,
            mels_latent,
            mels_mean,
            mels_log_stdev,
            logdet_sum,
            log_duration,
            attn_weights,
        )

    def infer(
        self,
        text: torch.Tensor,
        noise_scale: float = 0.0,
        time_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input shapes:
            - text: `(text_len)`

        Returned shapes:
            - mels: `(mel_channels, mels_len)`
            - duration: `(text_len)`
        """

        return self.infer_for_onnx(
            text,
            torch.tensor(noise_scale),
            torch.tensor(time_scale),
        )

    def infer_for_onnx(
        self,
        text: torch.Tensor,
        noise_scale: torch.Tensor,
        time_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input shapes:
            - text: `(text_len)`
            - noise_scale: `()`
            - time_scale: `()`

        Returned shapes:
            - mels: `(mel_channels, mels_len)`
            - duration: `(text_len)`
        """

        # Introduce dummy batch dimension and create mask.
        # text: (1, text_len)
        # text_mask: (1, text_len)
        text = text.unsqueeze(0)
        text_mask = torch.ones_like(text)

        # Encode text into text_mean and duration
        # text_mean: (1, mel_channels, text_len)
        # text_log_stdev: (1, mel_channels, text_len)
        # log_duration: (1, text_len)
        text_mean, text_log_stdev, log_duration = self.encoder(text, text_mask)

        # Upsample text distribution to mels distribution and create mask.
        # duration: (1, text_len)
        # mels_mean: (1, mel_channels, mels_len)
        # mels_mask: (1, 1, mels_len)
        duration = torch.clamp_min(torch.ceil(torch.exp(log_duration) * time_scale).long(), 1)
        mels_mean = torch.repeat_interleave(text_mean, duration.squeeze(0), 2)
        mels_log_stdev = torch.repeat_interleave(text_log_stdev, duration.squeeze(0), 2)
        mels_mask = torch.ones((1, 1, mels_mean.shape[2]), device=text.device)

        # Randomly sample mels_latent from distribution
        # mels_latent: (1, mel_channels, mels_len)
        mels_latent = mels_mean + torch.exp(mels_log_stdev) * torch.randn_like(mels_mean) * noise_scale

        # Decode mels_latent into mels.
        # mels: (1, mel_channels, mels_len)
        mels, _logdet_sum = self.decoder(mels_latent, mels_mask, reverse=True)

        # Remove dummy batch dimension.
        # mels: (mel_channels, mels_len)
        # duration: (text_len)
        mels = mels.squeeze(0)
        duration = duration.squeeze(0)

        return mels, duration
    
    def prepare_for_inference(self):
        self.decoder.prepare_for_inference()

def _compute_log_likelihood_matrix(
    sample: torch.Tensor,
    mean: torch.Tensor,
    log_stdev: torch.Tensor,
) -> torch.Tensor:
    """
    Input shapes:
        - sample: `(batch, mel_channels, mels_len)`
        - mean: `(batch, mel_channels, text_len)`
        - log_stdev: `(batch, mel_channels, text_len)`
    
    Returned shapes:
        - output: `(batch, text_len, mels_len)`
    """

    # Reshape for broadcasting.
    # sample: (batch, mel_channels, 1, mels_len)
    # mean: (batch, mel_channels, text_len, 1)
    # log_stdev: (batch, mel_channels, text_len, 1)
    sample = sample.unsqueeze(2)
    mean = mean.unsqueeze(3)
    log_stdev = log_stdev.unsqueeze(3)

    # Compute logp.
    # logp: (batch, mel_channels, text_len, mels_len)
    logp = -0.5 * ((sample - mean) ** 2) * torch.exp(-2.0 * log_stdev)
    logp = logp - log_stdev
    logp = logp - 0.5 * math.log(2 * math.pi)

    # Sum over mel_channels.
    # logp: (batch, mel_channels, text_len, mels_len)
    logp = torch.sum(logp, 1)

    return logp

def _get_mask_from_lengths(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Input shapes:
        - lengths: `(batch)`
    
    Returned shapes:
        - output: `(batch, seq_len)`
    """

    # Create an indices tensor and do a broadcasted compare with lengths.
    # indices: (seq_len)
    # mask: (batch, seq_len)
    indices = torch.arange(0, seq_len, device=lengths.device)
    mask = (indices < lengths.unsqueeze(1)).float()

    return mask
