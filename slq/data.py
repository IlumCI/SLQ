"""Calibration data for the SLQ search.

Every metric SLQ optimizes -- EAR, KL, the Shapley marginals -- is measured on
a calibration set, so this module builds one the way the paper does: contiguous
token windows drawn from a text corpus. The paper uses 512 calibration samples;
that is the default here too.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["build_calibration", "calibration_from_texts", "WIKITEXT"]

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def calibration_from_texts(
    texts: Sequence[str],
    tokenizer,
    n_samples: int = 512,
    seq_len: int = 512,
    batch_size: int = 1,
    seed: int = 0,
    min_chars: int = 1,
) -> list[torch.Tensor]:
    """Tokenize a corpus into fixed-length calibration batches.

    The texts are concatenated and cut into contiguous windows, which is the
    standard protocol for post-training quantization calibration: it preserves
    natural token statistics without padding artifacts.

    Args:
        texts: Raw documents.
        tokenizer: A HuggingFace tokenizer.
        n_samples: Number of ``seq_len`` windows to draw.
        seq_len: Tokens per window.
        batch_size: Windows per returned batch.
        seed: Seed for window placement.
        min_chars: Skip documents shorter than this (WikiText has many blanks).

    Returns:
        A list of ``[batch_size, seq_len]`` integer tensors.
    """
    joined = "\n\n".join(t for t in texts if len(t.strip()) >= min_chars)
    if not joined:
        raise ValueError("no usable text in the supplied corpus")

    ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len + 1:
        raise ValueError(
            f"corpus tokenizes to {ids.numel()} tokens, need at least {seq_len + 1}"
        )

    gen = torch.Generator().manual_seed(seed)
    max_start = ids.numel() - seq_len
    starts = torch.randint(0, max_start, (n_samples,), generator=gen)
    windows = [ids[s : s + seq_len] for s in starts.tolist()]

    return [
        torch.stack(windows[i : i + batch_size])
        for i in range(0, len(windows), batch_size)
        if len(windows[i : i + batch_size]) == batch_size
    ]


def build_calibration(
    tokenizer,
    dataset: tuple[str, str | None] = WIKITEXT,
    split: str = "train",
    n_samples: int = 512,
    seq_len: int = 512,
    batch_size: int = 1,
    seed: int = 0,
    n_docs: int = 20000,
) -> list[torch.Tensor]:
    """Load a corpus from the Hub and build calibration batches.

    Args:
        tokenizer: A HuggingFace tokenizer.
        dataset: ``(repo_id, config)``; defaults to WikiText-2 raw.
        split: Dataset split.
        n_samples: Number of calibration windows (512 in the paper).
        seq_len: Tokens per window.
        batch_size: Windows per batch.
        seed: Seed for window placement.
        n_docs: Cap on documents read, to bound tokenization time.

    Returns:
        A list of ``[batch_size, seq_len]`` integer tensors.

    Raises:
        ImportError: If ``datasets`` is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "build_calibration needs the 'datasets' package; "
            "install slq[hf] or pass your own texts to calibration_from_texts"
        ) from e

    repo, config = dataset
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    texts = [r for r in ds[: min(n_docs, len(ds))]["text"]]
    return calibration_from_texts(
        texts,
        tokenizer,
        n_samples=n_samples,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )
