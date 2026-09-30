"""The memory core. No host, no provider, no transport — just the shape and lifecycle of a fact."""

from .extract import extract, heuristic_extract, merge_paths
from .lifecycle import consolidate, decay
from .models import Category, Importance, Memory, Source, Status
from .rank import context_block, is_passing_state, select, should_persist
from .store import Store

__all__ = [
    "Category", "Importance", "Memory", "Source", "Status", "Store",
    "extract", "heuristic_extract", "merge_paths",
    "consolidate", "decay",
    "context_block", "is_passing_state", "select", "should_persist",
]
