"""SmoothQuant difficulty migration, integrated with SLQ's fidelity search."""

from slq.smooth.alpha import (
    AlphaSearchResult,
    make_scorer,
    search_global_alpha,
    search_per_site_alpha,
    smoothed_copy,
)
from slq.smooth.scales import ActScaleCollector, collect_act_scales
from slq.smooth.smooth import (
    SmoothReport,
    SmoothSite,
    compute_smoothing_scale,
    discover_sites,
    smooth_model,
)

__all__ = [
    "ActScaleCollector",
    "AlphaSearchResult",
    "SmoothReport",
    "SmoothSite",
    "collect_act_scales",
    "compute_smoothing_scale",
    "discover_sites",
    "make_scorer",
    "search_global_alpha",
    "search_per_site_alpha",
    "smooth_model",
    "smoothed_copy",
]
