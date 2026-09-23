"""Partitioning layers into allocation groups (Section 3.3).

Inference engines require some layers to share a bitwidth -- the paper's
example is the fused QKV projection in vLLM -- so bitwidths are assigned to
*groups* of layers rather than to individual layers. The granularity of the
partition is a deployment constraint, not an algorithmic one, so it is exposed
as a policy here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = ["LayerGroup", "GroupingPolicy", "build_groups"]

# Matches "<prefix>.<block index>.<rest>", e.g. "model.layers.12.self_attn.q_proj".
_BLOCK_RE = re.compile(r"^(.*?)\.(\d+)\.(.*)$")

#: Projections that vLLM fuses into a single QKV kernel and which therefore
#: must share a bitwidth when the ``fuse_qkv`` policy flag is set.
QKV_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "query", "key", "value")

#: Projections that vLLM fuses into a single gate/up kernel.
GATE_UP_PROJECTIONS = ("gate_proj", "up_proj")


@dataclass
class LayerGroup:
    """A set of layers constrained to share one bitwidth."""

    name: str
    layers: list[str] = field(default_factory=list)
    numel: int = 0

    def __len__(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class GroupingPolicy:
    """How layers are partitioned into groups.

    Args:
        granularity: ``"layer"`` gives each layer its own group (finest, largest
            ``M``); ``"block"`` groups every layer within a transformer block;
            ``"module_type"`` groups the same projection across all blocks
            (coarsest, smallest ``M``).
        fuse_qkv: Force Q/K/V projections within a block to share a group,
            matching vLLM's fused QKV kernel.
        fuse_gate_up: Force gate/up projections within a block to share a group.
        exclude: Regex patterns for layers that stay in full precision
            (embeddings and the LM head by default).
    """

    granularity: str = "layer"
    fuse_qkv: bool = False
    fuse_gate_up: bool = False
    exclude: tuple[str, ...] = (r"lm_head", r"embed", r"\bwte\b", r"\bwpe\b")

    def __post_init__(self) -> None:
        if self.granularity not in ("layer", "block", "module_type"):
            raise ValueError(
                f"granularity must be 'layer', 'block' or 'module_type', got {self.granularity!r}"
            )

    def is_excluded(self, name: str) -> bool:
        return any(re.search(p, name) for p in self.exclude)


def _split_block(name: str) -> tuple[str | None, str]:
    """Split ``"model.layers.12.self_attn.q_proj"`` into ``("12", "self_attn.q_proj")``."""
    m = _BLOCK_RE.match(name)
    if m is None:
        return None, name
    return m.group(2), m.group(3)


def _fusion_key(leaf: str, policy: GroupingPolicy) -> str:
    """Map a leaf module path onto its fused-kernel key, if any."""
    if policy.fuse_qkv and any(leaf.endswith(p) for p in QKV_PROJECTIONS):
        return leaf.rsplit(".", 1)[0] + ".qkv_proj" if "." in leaf else "qkv_proj"
    if policy.fuse_gate_up and any(leaf.endswith(p) for p in GATE_UP_PROJECTIONS):
        return leaf.rsplit(".", 1)[0] + ".gate_up_proj" if "." in leaf else "gate_up_proj"
    return leaf


def build_groups(
    layers: Iterable[tuple[str, int]],
    policy: GroupingPolicy | None = None,
) -> list[LayerGroup]:
    """Partition ``(layer_name, numel)`` pairs into allocation groups.

    Args:
        layers: Quantizable layers with their parameter counts.
        policy: Partitioning policy; defaults to one group per layer.

    Returns:
        Groups in deterministic order, each with its aggregate parameter count.
        Excluded layers are dropped and stay in full precision.
    """
    policy = policy if policy is not None else GroupingPolicy()
    groups: dict[str, LayerGroup] = {}

    for name, numel in layers:
        if policy.is_excluded(name):
            continue
        block, leaf = _split_block(name)
        leaf = _fusion_key(leaf, policy)

        if policy.granularity == "module_type":
            key = leaf
        elif policy.granularity == "block":
            key = f"block.{block}" if block is not None else name
        elif block is not None:
            key = f"{block}.{leaf}"
        else:
            key = name

        g = groups.setdefault(key, LayerGroup(name=key))
        g.layers.append(name)
        g.numel += numel

    return list(groups.values())


def average_bitwidth(
    groups: Sequence[LayerGroup], assignment: dict[str, float]
) -> float:
    """Parameter-weighted average bitwidth ``b_bar`` (Section 3.2).

        b_bar(b) = sum_l b_l |W_l| / sum_l |W_l|

    Args:
        groups: The allocation groups.
        assignment: Maps group name to its (effective) bitwidth.

    Returns:
        The weighted mean bitwidth over all grouped parameters.
    """
    total = sum(g.numel for g in groups)
    if total == 0:
        return float("nan")
    return sum(assignment[g.name] * g.numel for g in groups) / total
