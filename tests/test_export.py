"""Tests for the llama.cpp export path."""

import pytest

from slq.export import (
    GGML_TYPES,
    ggml_effective_bits,
    allocation_report,
    bits_to_ggml_type,
    hf_to_gguf_pattern,
    quantize_command,
    write_tensor_type_file,
)


def test_maps_hf_projections_to_gguf_stems():
    assert hf_to_gguf_pattern("model.layers.7.self_attn.k_proj") == r"\.7\.attn_k\.weight"
    assert hf_to_gguf_pattern("model.layers.0.mlp.gate_proj") == r"\.0\.ffn_gate\.weight"
    assert hf_to_gguf_pattern("model.layers.31.self_attn.o_proj") == r"\.31\.attn_output\.weight"
    assert hf_to_gguf_pattern("model.layers.2.mlp.down_proj") == r"\.2\.ffn_down\.weight"


def test_maps_grouping_policy_names():
    """The grouping policy emits '<block>.<leaf>' rather than a full path."""
    assert hf_to_gguf_pattern("12.self_attn.v_proj") == r"\.12\.attn_v\.weight"


def test_module_type_names_apply_to_every_block():
    assert hf_to_gguf_pattern("self_attn.q_proj") == r"attn_q\.weight"


def test_unmappable_names_return_none():
    assert hf_to_gguf_pattern("model.norm") is None
    assert hf_to_gguf_pattern("lm_head") is None


def test_bits_map_to_ggml_types():
    assert bits_to_ggml_type(4) == "q4_K"
    assert bits_to_ggml_type(8) == "q8_0"
    with pytest.raises(ValueError, match="no ggml type"):
        bits_to_ggml_type(7)


def test_ggml_types_are_types_not_ftypes():
    """--tensor-type takes ggml types; the _M/_S/_L mix suffixes are invalid there."""
    for name in GGML_TYPES.values():
        assert not name.upper().endswith(("_M", "_S", "_L")) or name.endswith("_K")


def test_tensor_type_file_has_no_comments(tmp_path):
    """llama.cpp tokenizes this file on whitespace and has no comment syntax.

    A '#' header makes parse_tensor_type fail with "malformed tensor type '#'",
    which aborts the whole quantization.
    """
    path = tmp_path / "t.txt"
    write_tensor_type_file(str(path), {"model.layers.0.self_attn.k_proj": 8})
    text = path.read_text()
    assert "#" not in text
    for line in text.splitlines():
        assert line.count("=") == 1
        assert line.split("=")[1] in GGML_TYPES.values()


def test_tensor_type_file_contents(tmp_path):
    path = tmp_path / "t.txt"
    out = write_tensor_type_file(
        str(path),
        {
            "model.layers.0.self_attn.k_proj": 8,
            "model.layers.0.mlp.gate_proj": 4,
            "model.norm": 4,  # unmappable, skipped
        },
    )
    assert out == {r"\.0\.attn_k\.weight": "q8_0", r"\.0\.ffn_gate\.weight": "q4_K"}
    assert len(path.read_text().strip().splitlines()) == 2


def test_unmappable_can_be_strict(tmp_path):
    with pytest.raises(ValueError, match="cannot map"):
        write_tensor_type_file(
            str(tmp_path / "t.txt"), {"model.norm": 4}, skip_unmapped=False
        )


def test_unavailable_bitwidth_raises(tmp_path):
    with pytest.raises(ValueError, match="no ggml type"):
        write_tensor_type_file(
            str(tmp_path / "t.txt"), {"model.layers.0.self_attn.k_proj": 7}
        )


def test_quantize_command_puts_options_before_input():
    """llama.cpp stops option parsing at the first non '--' argument."""
    cmd = quantize_command("in.gguf", "out.gguf", "t.txt", imatrix="im.gguf")
    i_in = cmd.index("in.gguf")
    assert all(not c.startswith("--") for c in cmd[i_in:])
    assert cmd[-2:] == ["q4_k_m", "8"]
    assert cmd[i_in : i_in + 3] == ["in.gguf", "out.gguf", "q4_k_m"]


def test_allocation_report_accounting():
    numel = {"a": 1000, "b": 3000}
    rep = allocation_report({"a": 8, "b": 4}, numel)
    assert rep["params"] == 4000
    assert rep["mean_bits"] == pytest.approx((8 * 1000 + 4 * 3000) / 4000)
    assert rep["bytes"] == pytest.approx((8 * 1000 + 4 * 3000) / 8)
    assert rep["histogram"] == {4: 1, 8: 1}


def test_ggml_bpw_differs_from_slq_int_accounting():
    """A llama.cpp budget must use the k-quant block sizes, not SLQ's grid.

    Budgeting a GGUF with SLQ's own INT-grid accounting (b + 20/128)
    under-counts by ~7%, because a q4_K super-block carries its own scale and
    min structure: 4.5 bpw against 4.156. Two size-matched comparisons were
    invalidated by this before it was caught, both times producing an SLQ file
    4-6% larger than the baseline it was supposed to match.
    """
    from slq.quant.grid import QuantConfig, effective_bits

    f = ggml_effective_bits()
    for bits in (4, 5, 6, 8):
        assert f(bits) > effective_bits(QuantConfig(bits, 128))
    assert f(4) == pytest.approx(4.5)
    assert f(6) == pytest.approx(6.5625)
    assert f(8) == pytest.approx(8.5)


def test_ggml_effective_bits_is_monotonic():
    f = ggml_effective_bits()
    assert f(4) < f(5) < f(6) < f(8)


def test_ggml_effective_bits_rejects_unknown_bitwidth():
    f = ggml_effective_bits()
    with pytest.raises(ValueError, match="no realized bpw"):
        f(7)


def test_every_ggml_type_has_a_known_width():
    from slq.export import GGML_BPW

    for name in GGML_TYPES.values():
        assert name in GGML_BPW
