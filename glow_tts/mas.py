import torch, numpy as np, numba

def monotonic_alignment_search(
    scores: torch.Tensor,
    text_lengths: torch.Tensor,
    mels_lengths: torch.Tensor,
):
    """
    Input shapes:
        - scores: `(batch, text_len, mels_len)`
        - text_lengths: `(batch)`
        - mels_lengths: `(batch)`

    Returned shapes:
        - output: `(batch, text_len, mels_len)`
    """

    # Use numba-accelerated numpy implementation.
    # attn: (batch, text_len, mels_len)
    attn = monotonic_alignment_search_np(
        scores.numpy(force=True),
        text_lengths.numpy(force=True),
        mels_lengths.numpy(force=True),
    )

    return torch.from_numpy(attn).to(scores.device)

@numba.njit(boundscheck=True)
def monotonic_alignment_search_np(
    scores: np.ndarray,
    text_lengths: np.ndarray,
    mels_lengths: np.ndarray,
) -> np.ndarray:
    """
    Input shapes:
        - scores: `(batch, text_len, mels_len)`
        - text_lengths: `(batch)`
        - mels_lengths: `(batch)`

    Returned shapes:
        - output: `(batch, text_len, mels_len)`
    """

    batch, _text_len, _mels_len = scores.shape

    # Compute max_score matrix.
    # max_score[b, i, j] is the maximum score
    # of any path that terminates at [b, i, j].
    # max_score: (batch, text_len, mels_len)
    max_score = np.zeros_like(scores)
    for b in range(batch):
        for i in range(text_lengths[b]):
            for j in range(mels_lengths[b]):
                if i == 0 and j == 0:
                    max_score[b, i, j] = scores[b, i, j]
                elif i == 0:
                    max_score[b, i, j] = scores[b, i, j] + max_score[b, i, j - 1]
                elif j == 0:
                    max_score[b, i, j] = -np.inf
                else:
                    max_score[b, i, j] = scores[b, i, j] + max(
                        max_score[b, i, j - 1],
                        max_score[b, i - 1, j - 1],
                    )

    # Compute attn_weights matrix.
    # attn_weights[b, i, j] is 1.0 iff [b, i, j] belongs
    # to some chosen maximum path terminating at [b, i, j]
    # attn_weights: (batch, text_len, mels_len)
    attn_weights = np.zeros(scores.shape, dtype=np.float32)
    for b in range(batch):
        i = text_lengths[b] - 1
        j = mels_lengths[b] - 1
        attn_weights[b, i, j] = 1.0
        while j > 0:
            if i == 0 or max_score[b, i, j - 1] > max_score[b, i - 1, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
            attn_weights[b, i, j] = 1.0
        
        assert i == 0

    return attn_weights
