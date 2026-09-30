"""A typed long-term memory layer for agents.

The core owns the shape of a memory and its lifecycle. It knows nothing about
a host, a provider or a transport: those live in `adapters/`, and each one is
thin enough to read in a sitting.
"""

from .extract import extract, heuristic_extract, merge_paths
from .lifecycle import consolidate, decay
from .models import Category, Importance, Memory, Source, Status
from .rank import context_block, select, should_persist
from .store import Store

__all__ = [
    "Category", "Importance", "Memory", "Source", "Status", "Store",
    "extract", "heuristic_extract", "merge_paths",
    "consolidate", "decay",
    "context_block", "select", "should_persist",
]
