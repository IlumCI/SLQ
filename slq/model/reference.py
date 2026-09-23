"""A small Llama-style transformer used for tests, demos and CI.

The SLQ pipeline is defined against ``nn.Linear`` layers and traced
normalization sites, so it does not depend on this model in particular. Having
a self-contained one means the whole pipeline -- calibration, GPTQ, smoothing,
Shapley sensitivity, ILP allocation -- runs end to end on CPU in seconds,
without downloading weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["ReferenceConfig", "ReferenceTransformer", "RMSNorm", "make_peaked_reference"]


@dataclass
class ReferenceConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 128
    rms_eps: float = 1e-6


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used by Llama and Qwen."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ReferenceConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.full((t, t), float("-inf"), device=x.device).triu(1)
        att = torch.softmax(att + mask, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ReferenceConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ReferenceConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class ReferenceTransformer(nn.Module):
    """A decoder-only transformer with Llama-style module naming."""

    def __init__(self, cfg: ReferenceConfig | None = None) -> None:
        super().__init__()
        self.config = cfg or ReferenceConfig()
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            # A deliberate per-layer offset makes the weight distribution
            # asymmetric (gamma > 1), which is the regime the gamma-squared law
            # of Lemma 3.2 describes.
            with torch.no_grad():
                m.weight.add_(torch.empty(m.weight.shape[0], 1).uniform_(-0.01, 0.01))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def make_peaked_reference(
    cfg: ReferenceConfig | None = None,
    steps: int = 400,
    batch_size: int = 8,
    seq_len: int = 48,
    lr: float = 3e-3,
    seed: int = 0,
) -> tuple[ReferenceTransformer, Callable[[int, int], torch.Tensor]]:
    """Train a reference model briefly so its output distribution is peaked.

    EAR is bounded above by the top-``K`` probability mass, so it is only
    meaningful on a model whose next-token distribution is concentrated -- the
    property the paper relies on when it calls the top-``K`` restriction "a tight
    approximation". A randomly initialized model is near-uniform, giving a
    top-10 mass of ``K/vocab`` and an EAR that can never approach 0.99 no matter
    how good the quantization is.

    This fits a small induction task (predict the token that followed the last
    occurrence of the current token) for a few hundred steps, which is enough to
    produce a sharply peaked, genuinely structured distribution in seconds on CPU.

    Args:
        cfg: Model configuration.
        steps: Optimizer steps.
        batch_size: Sequences per step.
        seq_len: Tokens per sequence.
        lr: AdamW learning rate.
        seed: Seed for both initialization and data.

    Returns:
        ``(model, sampler)`` where ``sampler(batch_size, seq_len)`` draws fresh
        batches from the same distribution for calibration.
    """
    torch.manual_seed(seed)
    cfg = cfg or ReferenceConfig()
    model = ReferenceTransformer(cfg)

    n_sym = max(8, cfg.vocab_size // 16)
    gen = torch.Generator().manual_seed(seed + 1)
    # A fixed bigram map: with probability `p_det` the next token is a
    # deterministic function of the current one, otherwise it is uniform noise.
    # The learnable part drives the distribution to be peaked; the noise keeps
    # it from collapsing to a delta, so EAR stays a non-trivial measurement.
    follow = torch.randint(0, n_sym, (n_sym,), generator=gen)
    p_det = 0.95

    def sampler(bs: int, sl: int) -> torch.Tensor:
        """Sequences generated from the bigram map, built autoregressively."""
        out = torch.empty(bs, sl, dtype=torch.long)
        out[:, 0] = torch.randint(0, n_sym, (bs,), generator=gen)
        for i in range(1, sl):
            noise = torch.randint(0, n_sym, (bs,), generator=gen)
            keep = torch.rand(bs, generator=gen) < p_det
            out[:, i] = torch.where(keep, follow[out[:, i - 1]], noise)
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x = sampler(batch_size, seq_len)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, sampler
