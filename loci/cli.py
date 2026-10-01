"""Entry point for the daily maintenance jobs.

Designed to be called by cron, s6, systemd, or any scheduler — one command,
two jobs. Exits 0 on success, 1 on error.
"""

from __future__ import annotations

import os
import sys

from .core.lifecycle import consolidate, decay
from .core.store import Store


def run_maintenance(db_path: str | None = None) -> int:
    path = db_path or os.environ.get("LOCI_DB", "~/.loci/memory.db")
    with Store(path) as store:
        merged = consolidate(store)
        expired = decay(store)
    total = merged + expired
    if total:
        print(f"loci maintenance: {merged} merged, {expired} expired")
    return total


def main() -> None:
    # Accept an optional path argument so cron entries can point at a specific db.
    db = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        run_maintenance(db)
    except Exception as exc:
        print(f"loci maintenance failed: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
