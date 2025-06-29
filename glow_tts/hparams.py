from dataclasses import dataclass

@dataclass(frozen=True)
class Hyperparameters:
    num_symbols: int = 128
    mel_channels: int = 80

    enc_embed_channels: int = 192
    enc_hidden_channels: int = 768
    enc_hidden_channels_dp: int = 256
    enc_kernel_size: int = 3
    enc_kernel_size_prenet: int = 5
    enc_num_heads: int = 2
    enc_window_size: int = 4
    enc_num_blocks: int = 6
    enc_p_dropout: float = 0.1

    dec_hidden_channels: int = 192
    dec_kernel_size: int = 5
    dec_dilation_rate: int = 1
    dec_num_wn_layers: int = 4
    dec_num_splits: int = 4
    dec_num_squeeze: int = 2
    dec_num_blocks: int = 12
    dec_p_dropout: float = 0.05
