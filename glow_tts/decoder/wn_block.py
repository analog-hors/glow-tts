import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

class WavenetStyleBlock(nn.Module):
    in_channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    num_layers: int
    p_dropout: float

    proj_in: nn.Module
    proj_out: nn.Conv1d
    in_layers: nn.ModuleList
    res_skip_layers: nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        assert kernel_size % 2 != 0

        proj_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj_in = weight_norm(proj_in)

        self.proj_out = nn.Conv1d(hidden_channels, out_channels, 1)

        # Every single implementation of this block contains this copy-pasted comment about
        # how initializing the end layer to zeros makes the coupling blocks start with no impact,
        # which stabilizes training. Who am I to ignore this longstanding tradition?
        assert self.proj_out.bias is not None
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(num_layers):
            in_layer = nn.Conv1d(
                hidden_channels,
                hidden_channels * 2,
                kernel_size,
                dilation=dilation_rate ** i,
                padding="same",
            )
            in_layer = weight_norm(
                in_layer,
                name="weight",
            )
            res_skip_layer = torch.nn.Conv1d(
                hidden_channels,
                hidden_channels * 2 if i < num_layers - 1 else hidden_channels,
                1,
            )
            res_skip_layer = weight_norm(
                res_skip_layer,
                name="weight",
            )
            self.in_layers.append(in_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
            - seq: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`
        
        Returned shapes:
            - output: `(batch, out_channels, seq_len)`
        """

        # Project in_channels to hidden_channels.
        # seq: (batch, hidden_channels, seq_len)
        seq = self.proj_in(seq)

        out = torch.zeros_like(seq)
        for i in range(self.num_layers):
            # Apply masking, the dilated convolution layer, and dropout.
            # x: (batch, hidden_channels * 2, seq_len)
            x = self.in_layers[i](seq * mask)
            x = F.dropout(x, self.p_dropout, self.training)

            # Apply gated activation.
            # x: (batch, hidden_channels, seq_len)
            t_act = torch.tanh(x[:, :self.hidden_channels, :])
            s_act = torch.sigmoid(x[:, self.hidden_channels:, :])
            x = t_act * s_act

            if i < self.num_layers - 1:
                # Project a residual half and a skip half, then apply them.
                # x: (batch, hidden_channels * 2, seq_len)
                x = self.res_skip_layers[i](x)
                seq = seq + x[:, :self.hidden_channels, :]
                out = out + x[:, self.hidden_channels:, :]
            else:
                # Last layer, so only compute and apply skip half.
                # x: (batch, hidden_channels, seq_len)
                x = self.res_skip_layers[i](x)
                out = out + x

        # Project hidden_channels to out_channels.
        # out: (batch, out_channels, seq_len)
        out = self.proj_out(out)

        return out

    def remove_parametrizations(self):
        for layer in self.in_layers:
            remove_parametrizations(layer, "weight")
        for layer in self.res_skip_layers:
            remove_parametrizations(layer, "weight")
