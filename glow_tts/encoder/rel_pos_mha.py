import torch, torch.nn as nn, torch.nn.functional as F
import math

class RelativePositionalMultiHeadAttention(nn.Module):
    in_channels: int
    out_channels: int
    num_heads: int
    window_size: int
    p_dropout: float

    conv_q: nn.Conv1d
    conv_k: nn.Conv1d
    conv_v: nn.Conv1d
    conv_o: nn.Conv1d
    rel_emb_k: nn.Embedding
    rel_emb_v: nn.Embedding

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        window_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.p_dropout = p_dropout

        assert in_channels % self.num_heads == 0

        self.conv_q = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_k = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_v = nn.Conv1d(in_channels, in_channels, 1)
        self.conv_o = nn.Conv1d(in_channels, out_channels, 1)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

        head_channels = in_channels // self.num_heads
        self.rel_emb_k = nn.Embedding(window_size * 2 + 1, head_channels)
        self.rel_emb_v = nn.Embedding(window_size * 2 + 1, head_channels)

        rel_emb_stdev = 1.0 / math.sqrt(head_channels)
        nn.init.normal_(self.rel_emb_k.weight, std=rel_emb_stdev)
        nn.init.normal_(self.rel_emb_v.weight, std=rel_emb_stdev)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input shapes:
            - query: `(batch, in_channels, seq_len)`
            - key: `(batch, in_channels, seq_len)`
            - value: `(batch, in_channels, seq_len)`
            - mask: `(batch, 1, seq_len)`

        Returned shapes:
            - output: `(batch, in_channels, seq_len)`
        """

        batch, in_channels, seq_len = query.shape
        head_channels = in_channels // self.num_heads

        # Project query, key, and value tensors.
        # q: (batch, in_channels, seq_len)
        # k: (batch, in_channels, seq_len)
        # v: (batch, in_channels, seq_len)
        q = self.conv_q(query)
        k = self.conv_k(key)
        v = self.conv_v(value)

        # Split channels across heads and move channel dimension to last.
        # q: (batch, num_heads, seq_len, head_channels)
        # k: (batch, num_heads, seq_len, head_channels)
        # v: (batch, num_heads, seq_len, head_channels)
        q = q.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)
        k = k.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)
        v = v.reshape((batch, self.num_heads, head_channels, seq_len)).transpose(2, 3)

        # Query keys.
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_channels)

        # Query relative embeddings for key, skew to make absolute, and add to scores.
        # rel_emb_k: (seq_len * 2 - 1, head_channels)
        # rel_scores_k: (batch, num_heads, seq_len, seq_len * 2 - 1)
        # scores_k: (batch, num_heads, seq_len, seq_len)
        # scores: (batch, num_heads, seq_len, seq_len)
        rel_emb_k = self._pad_embed_matrix(self.rel_emb_k.weight, seq_len)
        rel_scores_k = torch.matmul(q, rel_emb_k.transpose(0, 1)) / math.sqrt(head_channels)
        scores_k = self._skew_relative_to_absolute(rel_scores_k)
        scores = scores + scores_k

        # Mask out padding in scores.
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = scores.masked_fill(mask.unsqueeze(1) == 0.0, -torch.inf)

        # Compute weights and context.
        # weights: (batch, num_heads, seq_len, seq_len)
        # context: (batch, num_heads, seq_len, head_channels)
        weights = torch.softmax(scores, -1)
        weights = F.dropout(weights, self.p_dropout, self.training)
        context = torch.matmul(weights, v)

        # Skew weights to make relative, apply to relative value embeddings, and add to context.
        # rel_emb_v: (seq_len * 2 - 1, head_channels)
        # rel_weights: (batch, num_heads, seq_len, seq_len * 2 - 1)
        rel_emb_v = self._pad_embed_matrix(self.rel_emb_v.weight, seq_len)
        rel_weights = self._skew_absolute_to_relative(weights)
        context_v = torch.matmul(rel_weights, rel_emb_v)
        context = context + context_v

        # Move channel dimension back and rejoin channels across heads.
        # context: (batch, in_channels, seq_len)
        context = context.transpose(2, 3)
        context = context.reshape((batch, in_channels, seq_len))

        # Project context to have out_channels channels.
        # context: (batch, out_channels, seq_len)
        context = self.conv_o(context)

        return context

    def _pad_embed_matrix(self, embed: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Pad an embedding matrix to make it a relative position matrix.
        Requires that `seq_len >= window_size * 2 + 1`. This is because
        1. It removes the need for slicing down the embedding matrix.
        2. It guarantees at least `window_size` padding, which is required for the skew trick.
        
        Input shapes:
            - embed: `(window_size * 2 + 1, head_channels)`

        Returned shapes:
            - output: `(seq_len * 2 - 1, head_channels)`
        """

        # Ensures that the window fits and there is enough padding.
        # The minimum amount of padding is window_size, so this check is sufficient.
        assert seq_len >= embed.shape[0]

        # Pad embed matrix to fit all distance pairs.
        # Also provides a buffer zone for the shifted values.
        # embed: (seq_len * 2 - 1, head_channels)
        relative_seq_len = seq_len * 2 - 1
        pad = (relative_seq_len - embed.shape[0]) // 2
        embed = F.pad(embed, (0, 0, pad, pad))

        return embed

    def _skew_relative_to_absolute(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Skew a relative tensor to an absolute one.
        Requires at least `window_size` padding to be correct.

        Input shapes:
            - tensor: `(batch, num_heads, seq_len, seq_len * 2 - 1)`

        Returned shapes:
            - output: `(batch, num_heads, seq_len, seq_len)`
        """

        batch, num_heads, seq_len, relative_seq_len = tensor.shape

        # Add extra column for inducing the skew, then flatten.
        # tensor: (batch, num_heads, seq_len * seq_len * 2)
        tensor = F.pad(tensor, (0, 1))
        tensor = tensor.flatten(2)

        # Pad to realign values, then reshape back.
        # tensor: (batch, num_heads, seq_len + 2, seq_len * 2 - 1)
        left_pad = relative_seq_len // 2 + 1
        right_pad = -(left_pad + tensor.shape[2]) % relative_seq_len
        tensor = F.pad(tensor, (left_pad, right_pad))
        tensor = tensor.reshape((batch, num_heads, -1, relative_seq_len))

        # Slice out the relevant block.
        # tensor: (batch, num_heads, seq_len, seq_len)
        tensor = tensor[:, :, 1:seq_len + 1, :seq_len]

        return tensor

    def _skew_absolute_to_relative(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Skew an absolute tensor to a relative one.

        Input shapes:
            - tensor: `(batch, num_heads, seq_len, seq_len)`

        Returned shapes:
            - output: `(batch, num_heads, seq_len, seq_len * 2 - 1)`
        """

        batch, num_heads, seq_len, _seq_len = tensor.shape
        relative_seq_len = seq_len * 2 - 1

        # Pad tensor to fit all distance pairs, then flatten.
        # tensor: (batch, num_heads, seq_len * (seq_len * 2 - 1))
        pad = relative_seq_len - seq_len
        tensor = F.pad(tensor, (pad, 0))
        tensor = tensor.flatten(2)

        # Pad to make full block, then reshape back.
        # tensor: (batch, num_heads, seq_len, seq_len * 2)
        right_pad = -tensor.shape[2] % (relative_seq_len + 1)
        tensor = F.pad(tensor, (0, right_pad))
        tensor = tensor.reshape((batch, num_heads, -1, relative_seq_len + 1))

        # Slice out the relevant block.
        # tensor: (batch, num_heads, seq_len, seq_len * 2 - 1)
        tensor = tensor[:, :, :, :relative_seq_len]

        return tensor
