"""MCP adapter: the memory core reachable from any MCP client.

What this gives you: explicit tools. The agent calls `remember` and `recall`
on purpose.

What it cannot give you: automatic extraction. MCP is request and response —
nothing calls a server when a turn ends, so there is no moment for the server
to read the exchange on its own. That part needs a host hook, which is what
the Hermes adapter adds over this one.
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from core import Category, Importance, Memory, Source, Store, context_block, should_persist

mcp = FastMCP("loci")
store = Store(os.environ.get("LOCI_DB", "~/.loci/memory.db"))


@mcp.tool()
def remember(content: str, category: str = "preferences", importance: str = "medium") -> str:
    """Store one durable fact about the user.

    Use for things that stay true for weeks: a preference, a constraint, a
    decision and its reason, a deadline. Not for what is true only today.
    """
    try:
        memory = Memory(
            key=f"{category}_{abs(hash(content)) % 10**10}",
            title=content[:80],
            content=content,
            category=Category(category),
            importance=Importance(importance),
            source=Source.USER_MESSAGE,
        )
    except ValueError as exc:
        return json.dumps({"ok": False, "error": str(exc)})

    if not should_persist(memory):
        return json.dumps({"ok": True, "stored": False, "reason": "below the write filter"})
    saved = store.upsert(memory)
    return json.dumps({"ok": True, "stored": True, "key": saved.key})


@mcp.tool()
def recall(query: str = "", limit: int = 12) -> str:
    """Retrieve what is known about the user, optionally filtered by a query."""
    found = store.search(query, limit) if query else store.active()
    return json.dumps({"ok": True, "context": context_block(found, cap=limit)})


@mcp.tool()
def forget(key: str) -> str:
    """Remove one stored fact by key. Use when the user asks you to forget it."""
    return json.dumps({"ok": True, "removed": store.forget(key)})


if __name__ == "__main__":
    mcp.run()
