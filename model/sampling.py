from __future__ import annotations

from collections import Counter

import torch


def apply_repetition_penalty(
    logits: torch.Tensor, context: list[int], penalty: float
) -> torch.Tensor:
    """CTRL-style: divide positive / multiply negative logits of seen tokens."""
    if penalty == 1.0 or not context:
        return logits
    for t in set(context):
        v = logits[t]
        logits[t] = v / penalty if v > 0 else v * penalty
    return logits


def apply_frequency_presence_penalties(
    logits: torch.Tensor,
    context: list[int],
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """OpenAI-style: subtract freq*count and presence*(seen) from seen tokens."""
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits
    counts = Counter(context)
    for t, c in counts.items():
        logits[t] -= frequency_penalty * c + presence_penalty * (c > 0)
    return logits


def ban_repeat_ngrams(
    logits: torch.Tensor, context: list[int], no_repeat_ngram_size: int
) -> torch.Tensor:
    """Set -inf on any token that would complete an n-gram already in context."""
    n = no_repeat_ngram_size
    if n <= 0 or len(context) < n:
        return logits
    if n == 1:
        for t in set(context):
            logits[t] = float("-inf")
        return logits
    m = n - 1
    suffix = context[-m:]
    for i in range(len(context) - m):
        if context[i : i + m] == suffix:
            logits[context[i + m]] = float("-inf")
    return logits


def top_k_filter(probs: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k >= probs.size(-1):
        return probs
    thresh = torch.topk(probs, k, dim=-1).values[..., -1, None]
    return torch.where(probs < thresh, torch.zeros_like(probs), probs)


def top_p_filter(probs: torch.Tensor, p: float | None) -> torch.Tensor:
    """Keep the minimal nucleus whose cumulative mass reaches p."""
    if p is None or p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    drop = (cum - sorted_probs) >= p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    return torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)


def min_p_filter(probs: torch.Tensor, min_p: float | None) -> torch.Tensor:
    """Keep only tokens with prob >= max_prob * min_p."""
    if min_p is None or min_p <= 0.0:
        return probs
    cutoff = probs.max(dim=-1, keepdim=True).values * min_p
    return torch.where(probs < cutoff, torch.zeros_like(probs), probs)


def sample_token(
    logits: torch.Tensor,
    context: list[int],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    no_repeat_ngram_size: int = 0,
) -> int:
    """Penalize/ban on raw logits, then temperature-scale and filter."""
    logits = logits.clone()
    logits = apply_repetition_penalty(logits, context, repetition_penalty)
    logits = apply_frequency_presence_penalties(
        logits, context, frequency_penalty, presence_penalty
    )
    logits = ban_repeat_ngrams(logits, context, no_repeat_ngram_size)

    if temperature <= 0:
        return int(logits.argmax(dim=-1))

    probs = torch.softmax(logits / temperature, dim=-1)
    probs = min_p_filter(probs, min_p)
    probs = top_k_filter(probs, top_k)
    probs = top_p_filter(probs, top_p)
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        return int(logits.argmax(dim=-1))
    return int(torch.multinomial(probs / total, num_samples=1))
