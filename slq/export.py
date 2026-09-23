"""Export an SLQ allocation to a runnable quantized model.

SLQ decides *which tensor gets which bitwidth*; it does not need to perform the
quantization itself. ``llama-quantize`` accepts per-tensor type overrides:

    llama-quantize --tensor-type-file types.txt in-f32.gguf out.gguf q4_k_m

so an allocation can be handed to it directly. That matters for large models:
the allocation is a few kilobytes of decisions, while holding a 27B model in
memory to quantize it is not possible on the machines this targets. llama.cpp
streams the weights and does the arithmetic; SLQ supplies the policy.

Two name spaces are involved. SLQ works with HuggingFace module paths
(``model.layers.7.self_attn.k_proj``); GGUF uses its own names
(``blk.7.attn_k.weight``). :func:`hf_to_gguf_name` maps between them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "GGML_TYPES",
    "gguf_name_map",
    "gguf_name_to_pattern",
    "bits_to_ggml_type",
    "hf_to_gguf_pattern",
    "write_tensor_type_file",
    "quantize_command",
    "allocation_report",
]

#: ggml types usable with ``--tensor-type``, keyed by the nominal bitwidth SLQ
#: allocates. These are *ggml types*, not ftypes: the ``_M``/``_S``/``_L`` mix
#: suffixes are only valid as the positional base type, not per tensor.
GGML_TYPES: dict[int, str] = {
    2: "q2_K",
    3: "q3_K",
    4: "q4_K",
    5: "q5_K",
    6: "q6_K",
    8: "q8_0",
}

#: Alternative 4-bit type with better quality per bit at the same footprint.
GGML_TYPE_ALIASES = {"iq4_xs": 4.25, "iq4_nl": 4.5}

#: HuggingFace leaf module name -> GGUF tensor stem. A hand-written fallback for
#: standard transformer projections; :func:`gguf_name_map` prefers llama.cpp's
#: own authoritative mapping when gguf-py is importable, which is the only way
#: to get hybrid architectures right (Qwen3.5's linear-attention layers map
#: ``in_proj_a -> ssm_alpha``, ``in_proj_z -> attn_gate`` and so on, none of
#: which is guessable).
_HF_TO_GGUF = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
    "qkv_proj": "attn_qkv",
    "gate_up_proj": "ffn_gate_up",
    # Qwen3.5-style linear-attention blocks.
    "in_proj_qkv": "attn_qkv",
    "in_proj_z": "attn_gate",
    "in_proj_a": "ssm_alpha",
    "in_proj_b": "ssm_beta",
    "out_proj": "ssm_out",
    "conv1d": "ssm_conv1d",
}


def gguf_name_map(arch: str, n_layers: int):
    """Return llama.cpp's own HF-to-GGUF tensor name mapper, if available.

    Hand-maintained name tables go stale and cannot cover architectures that
    did not exist when they were written. llama.cpp ships the authoritative
    mapping in ``gguf-py``; use it whenever it can be imported.

    Args:
        arch: A ``gguf.constants.MODEL_ARCH`` member name, e.g. ``"QWEN35"``.
        n_layers: Block count, which the mapper needs to expand ``{bid}``.

    Returns:
        A callable ``hf_name -> gguf_name | None``, or ``None`` if gguf-py is
        not importable or the architecture is unknown.
    """
    try:
        from gguf.constants import MODEL_ARCH
        from gguf.tensor_mapping import get_tensor_name_map
    except ImportError:
        return None
    member = getattr(MODEL_ARCH, arch, None)
    if member is None:
        return None
    mapper = get_tensor_name_map(member, n_layers)

    def lookup(name: str) -> str | None:
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        # Multimodal checkpoints nest the text tower; the mapper expects the
        # flat form.
        base = base.replace("model.language_model.", "model.")
        return mapper.get_name(base)

    return lookup

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")


def bits_to_ggml_type(bits: int) -> str:
    """Map an allocated bitwidth to the ggml type that realizes it."""
    try:
        return GGML_TYPES[bits]
    except KeyError:
        raise ValueError(
            f"no ggml type for {bits} bits; available: {sorted(GGML_TYPES)}"
        ) from None


def hf_to_gguf_pattern(name: str) -> str | None:
    """Convert an SLQ group or layer name into a GGUF tensor-name regex.

    Args:
        name: An SLQ name, either a full HuggingFace module path
            (``model.layers.7.self_attn.k_proj``) or the ``block.leaf`` form the
            grouping policy produces (``7.self_attn.k_proj``).

    Returns:
        A regex matching the corresponding GGUF tensor, or ``None`` if the name
        does not correspond to a quantizable projection.

    Examples:
        ``model.layers.7.self_attn.k_proj`` -> ``\\.7\\.attn_k\\.weight``
        ``self_attn.k_proj`` (all blocks)   -> ``attn_k\\.weight``
    """
    leaf = name.rsplit(".", 1)[-1]
    stem = _HF_TO_GGUF.get(leaf)
    if stem is None:
        return None

    m = _LAYER_RE.search(name)
    if m is None:
        # No block index: try the "<idx>.<leaf>" shape the grouping policy uses.
        head = name.split(".", 1)[0]
        if head.isdigit():
            return rf"\.{head}\.{stem}\.weight"
        return rf"{stem}\.weight"  # applies to every block
    return rf"\.{m.group(1)}\.{stem}\.weight"


def gguf_name_to_pattern(name: str) -> str:
    """Escape a concrete GGUF tensor name into a regex matching just that tensor.

    ``blk.7.ffn_down`` -> ``blk\\.7\\.ffn_down\\.weight``. The dots are escaped so
    ``blk.1.`` cannot match ``blk.17.``.
    """
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return re.escape(stem) + r"\.weight"


def write_tensor_type_file(
    path: str,
    assignment: Mapping[str, int],
    *,
    type_map: Mapping[int, str] | None = None,
    skip_unmapped: bool = True,
    names_are_gguf: bool = False,
) -> dict[str, str]:
    """Write an allocation in ``--tensor-type-file`` format.

    Args:
        path: Destination file.
        assignment: SLQ group/layer name -> allocated bitwidth.
        type_map: Override the bitwidth-to-ggml-type mapping.
        skip_unmapped: Silently drop names with no GGUF counterpart (norms,
            embeddings). If ``False``, raise on the first one.
        names_are_gguf: Treat the keys as GGUF tensor names already, escaping
            them directly instead of translating from HuggingFace paths. Use
            this when the allocation was built against llama.cpp's own mapper.

    Returns:
        The written ``{pattern: ggml_type}`` mapping.

    Raises:
        ValueError: If ``skip_unmapped`` is false and a name cannot be mapped.
    """
    types = dict(type_map or GGML_TYPES)
    out: dict[str, str] = {}
    for name, bits in sorted(assignment.items()):
        pattern = gguf_name_to_pattern(name) if names_are_gguf else hf_to_gguf_pattern(name)
        if pattern is None:
            if skip_unmapped:
                continue
            raise ValueError(f"cannot map {name!r} to a GGUF tensor name")
        try:
            out[pattern] = types[bits]
        except KeyError:
            raise ValueError(
                f"{name}: no ggml type for {bits} bits; available {sorted(types)}"
            ) from None

    # llama.cpp reads this file with `file >> arg`, i.e. whitespace-separated
    # tokens, and feeds every token to parse_tensor_type. It has no comment
    # syntax, so a header line fails with "malformed tensor type '#'". Write
    # nothing but `pattern=type` entries.
    with open(path, "w", encoding="utf-8") as f:
        for pattern, t in out.items():
            f.write(f"{pattern}={t}\n")
    return out


def quantize_command(
    input_gguf: str,
    output_gguf: str,
    tensor_type_file: str,
    base_type: str = "q4_k_m",
    imatrix: str | None = None,
    threads: int = 8,
    output_tensor_type: str | None = "q6_K",
    token_embedding_type: str | None = "q6_K",
) -> list[str]:
    """Build the ``llama-quantize`` command line for an exported allocation.

    Args:
        input_gguf: The unquantized (F32/F16/BF16) GGUF.
        output_gguf: Destination path.
        tensor_type_file: File written by :func:`write_tensor_type_file`.
        base_type: Fallback ftype for tensors the file does not name.
        imatrix: Optional importance matrix, which llama.cpp uses to improve
            the low-bit types. Strongly recommended below 4 bits.
        threads: Quantization threads.
        output_tensor_type: ggml type for ``output.weight``. The output
            projection is disproportionately sensitive, so it is kept high.
        token_embedding_type: ggml type for the token embedding tensor.

    Returns:
        The command as an argv list. Options must precede the input path,
        which llama.cpp's parser requires.
    """
    cmd = ["llama-quantize"]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    if output_tensor_type:
        cmd += ["--output-tensor-type", output_tensor_type]
    if token_embedding_type:
        cmd += ["--token-embedding-type", token_embedding_type]
    cmd += ["--tensor-type-file", tensor_type_file]
    cmd += [input_gguf, output_gguf, base_type, str(threads)]
    return cmd


def allocation_report(
    assignment: Mapping[str, int], numel: Mapping[str, int]
) -> dict[str, object]:
    """Summarize an allocation: size, mean bits, and spread by projection type."""
    total_params = sum(numel[n] for n in assignment)
    bits_total = sum(assignment[n] * numel[n] for n in assignment)
    by_type: dict[str, list[int]] = {}
    for name, bits in assignment.items():
        leaf = name.rsplit(".", 1)[-1]
        by_type.setdefault(leaf, []).append(bits)
    hist: dict[int, int] = {}
    for b in assignment.values():
        hist[b] = hist.get(b, 0) + 1
    return {
        "params": total_params,
        "mean_bits": bits_total / total_params if total_params else float("nan"),
        "bytes": bits_total / 8,
        "gigabytes": bits_total / 8 / 1e9,
        "histogram": dict(sorted(hist.items())),
        "mean_bits_by_type": {
            k: sum(v) / len(v) for k, v in sorted(by_type.items())
        },
    }
