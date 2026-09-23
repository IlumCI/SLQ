"""SLQ: Statistically-Lossless Quantization of Large Language Models.

Reference implementation of arXiv:2605.02404, integrated with SmoothQuant
(Xiao et al., 2023) activation-difficulty migration for weight-and-activation
configurations.
"""

__version__ = "0.1.0"

from slq.metrics.fidelity import FidelityMeter, FidelityResult, fidelity
from slq.model.grouping import GroupingPolicy, LayerGroup, build_groups
from slq.model.wrapper import QuantizableModel
from slq.quant.act import ActQuantConfig
from slq.quant.grid import QuantConfig, centering_inefficiency, effective_bits, fake_quantize

__all__ = [
    "ActQuantConfig",
    "FidelityMeter",
    "FidelityResult",
    "GroupingPolicy",
    "LayerGroup",
    "QuantConfig",
    "QuantizableModel",
    "__version__",
    "build_groups",
    "centering_inefficiency",
    "effective_bits",
    "fake_quantize",
    "fidelity",
]
