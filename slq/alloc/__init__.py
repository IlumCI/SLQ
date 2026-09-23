"""Bitwidth allocation: ILP solver, fidelity-targeted search, evolutionary search."""

from slq.alloc.evolution import EvolutionConfig, EvolutionResult, evolutionary_search
from slq.alloc.ilp import AllocationResult, solve_allocation
from slq.alloc.search import (
    SearchResult,
    search_distribution_lossless,
    search_task_lossless,
)

__all__ = [
    "AllocationResult",
    "EvolutionConfig",
    "EvolutionResult",
    "SearchResult",
    "evolutionary_search",
    "search_distribution_lossless",
    "search_task_lossless",
    "solve_allocation",
]
