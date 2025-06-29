import torch, torch.nn as nn, torch.nn.functional as F

class InvConvNear(nn.Module):
    in_channels: int
    num_splits: int

    matrix: torch.Tensor
    inv_matrix: torch.Tensor | None

    def __init__(self, in_channels: int, num_splits: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_splits = num_splits

        assert in_channels % num_splits == 0
        assert num_splits % 2 == 0

        normal = torch.normal(0.0, 1.0, (num_splits, num_splits))
        matrix = torch.linalg.qr(normal).Q
        if torch.det(matrix) < 0.0:
            matrix[:, 0] *= -1.0

        self.matrix = nn.Parameter(matrix)
        self.inv_matrix = None

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

        batch, in_channels, seq_len = seq.shape
        splits = self.num_splits

        # A complex split-permute-join operation to mix channels with a small matrix.
        # One interpretation is that the channels are first split into two halves.
        # Each half is then interpreted as a matrix, which is transposed.
        # The halves are then joined back together.
        # seq: (batch, splits, in_channels / num_splits, seq_len)
        seq = seq.reshape((batch, 2, in_channels // splits, splits // 2, seq_len))
        seq = seq.transpose(2, 3)
        seq = seq.reshape((batch, splits, in_channels // splits, seq_len))

        # Determine transformation matrix and compute logdet of Jacobian if forwards.
        # matrix: (num_splits, num_splits)
        # logdet: (batch)
        if not reverse:
            lengths = torch.sum(mask, (1, 2))
            matrix = self.matrix
            logdet = torch.logdet(matrix) * (in_channels // splits) * lengths
        elif self.inv_matrix is not None:
            matrix = self.inv_matrix
            logdet = None
        else:
            matrix = self.matrix.inverse()
            logdet = None

        # Apply transform by reshaping matrix to fit.
        # seq: (batch, splits, in_channels / num_splits, seq_len)
        seq = F.conv2d(seq, matrix.unsqueeze(-1).unsqueeze(-1))

        # Undo the initial split-permute-join operation.
        # seq: (batch, in_channels, seq_len)
        seq = seq.reshape((batch, 2, splits // 2, in_channels // splits, seq_len))
        seq = seq.transpose(2, 3)
        seq = seq.reshape((batch, in_channels, seq_len))

        # Apply mask.
        # It seems like the convention is to never assume something was already masked
        # if it's an input and conversely to allow garbage if it's an output,
        # but it was there in the reference implementation.
        # I guess it doesn't hurt to be defensive.
        # seq: (batch, in_channels, seq_len)
        seq = seq * mask

        return seq, logdet

    def prepare_for_inference(self):
        self.inv_matrix = self.matrix.inverse()
