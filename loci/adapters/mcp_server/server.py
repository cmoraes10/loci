"""MCP adapter: the memory core reachable from any MCP client.

What this gives you: explicit tools. The agent calls `remember` and `recall`
on purpose.

What it cannot give you: automatic extraction. MCP is request and response —
nothing calls a server when a turn ends, so there is no moment for the server
to read the exchange on its own. That part needs a host hook, which is what
the Hermes adapter adds over this one.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os

from mcp.server.fastmcp import FastMCP

from loci import Category, Importance, Memory, Source, Store, context_block, should_persist

mcp = FastMCP("loci")

# Lazy init: module-level Store at import time would write ~/.loci/memory.db
# unconditionally — e.g. when a test imports the module. Init on first use.
_store_cache: Store | None = None


def _get_store() -> Store:
    global _store_cache
    if _store_cache is None:
        _store_cache = Store(os.environ.get("LOCI_DB", "~/.loci/memory.db"))
        atexit.register(_store_cache.close)
    return _store_cache


@mcp.tool()
def remember(
    content: str,
    category: str = "preferences",
    importance: str = "medium",
    review_after: str = "",
) -> str:
    """Store one durable fact about the user.

    Use for things that stay true for weeks: a preference, a constraint, a
    decision and its reason, a deadline. Not for what is true only today.

    review_after: optional ISO date (YYYY-MM-DD) — when to revisit this fact.
    """
    from datetime import date as _date

    parsed_date: _date | None = None
    if review_after:
        try:
            parsed_date = _date.fromisoformat(review_after)
        except ValueError:
            return json.dumps({"ok": False, "error": f"invalid review_after date: {review_after!r}"})

    try:
        memory = Memory(
            key=f"{category}_{hashlib.md5(content.encode()).hexdigest()[:12]}",
            title=content[:80],
            content=content,
            category=Category(category),
            importance=Importance(importance),
            source=Source.USER_MESSAGE,
            review_after=parsed_date,
        )
    except ValueError as exc:
        return json.dumps({"ok": False, "error": str(exc)})

    if not should_persist(memory):
        return json.dumps({"ok": True, "stored": False, "reason": "below the write filter"})
    saved = _get_store().upsert(memory)
    return json.dumps({"ok": True, "stored": True, "key": saved.key})


@mcp.tool()
def recall(query: str = "", limit: int = 12) -> str:
    """Retrieve what is known about the user, optionally filtered by a query."""
    store = _get_store()
    # No-query path passes limit * 10 so context_block can rank across a wider
    # candidate set without loading the entire store.
    found = store.search(query, limit) if query else store.active(limit=limit * 10)
    return json.dumps({"ok": True, "context": context_block(found, cap=limit)})


@mcp.tool()
def forget(key: str) -> str:
    """Remove one stored fact by key. Use when the user asks you to forget it."""
    return json.dumps({"ok": True, "removed": _get_store().forget(key)})


if __name__ == "__main__":
    mcp.run()
