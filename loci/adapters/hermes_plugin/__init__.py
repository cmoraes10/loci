"""Hermes adapter: the same core, plus the two hooks MCP cannot offer.

`pre_llm_call` injects the context block before the model sees the turn.
`post_llm_call` reads the finished exchange and extracts, in the background,
after the reply is already on its way. Nothing here sits on the path the
person waits on.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading

from loci import (
    Category,
    Importance,
    Memory,
    Source,
    Store,
    context_block,
    extract,
    should_persist,
)

# Lazy init for the same reason as the MCP adapter: importing this module
# must not create a file on disk. Init on first request.
_store_cache: Store | None = None
_model_extractor = None  # ModelExtractor | None — injected via register()


def _get_store() -> Store:
    global _store_cache
    if _store_cache is None:
        _store_cache = Store(os.environ.get("LOCI_DB", "~/.loci/memory.db"))
        atexit.register(_store_cache.close)
    return _store_cache


def _remember(args: dict, **_kwargs) -> str:
    from datetime import date as _date

    parsed_date: _date | None = None
    raw_date = args.get("review_after", "")
    if raw_date:
        try:
            parsed_date = _date.fromisoformat(raw_date)
        except ValueError:
            return json.dumps({"ok": False, "error": f"invalid review_after: {raw_date!r}"})

    category = args.get("category", "preferences")
    try:
        memory = Memory(
            key=f"{category}_{hashlib.md5(args['content'].encode()).hexdigest()[:12]}",
            title=args["content"][:80],
            content=args["content"],
            category=Category(category),
            importance=Importance(args.get("importance", "medium")),
            source=Source.USER_MESSAGE,
            review_after=parsed_date,
        )
    except (KeyError, ValueError) as exc:
        # A typed error back to the model, not an exception. The loop recovers
        # and retries with better arguments instead of dying mid-turn.
        return json.dumps({"ok": False, "error": str(exc)})
    if not should_persist(memory):
        return json.dumps({"ok": True, "stored": False})
    return json.dumps({"ok": True, "key": _get_store().upsert(memory).key})


def _recall(args: dict, **_kwargs) -> str:
    store = _get_store()
    found = store.search(args["query"]) if args.get("query") else store.active()
    return json.dumps({"ok": True, "context": context_block(found)})


def _forget(args: dict, **_kwargs) -> str:
    return json.dumps({"ok": True, "removed": _get_store().forget(args.get("key", ""))})


def _inject_context(payload: dict) -> dict:
    """Eager injection into the system instruction, capped and ranked."""
    block = context_block(_get_store().active())
    if block:
        payload["system"] = f"{payload.get('system', '')}\n\n{block}".strip()
    return payload


def _extract_async(payload: dict) -> dict:
    """Fire and forget. The reply has already gone out."""

    def work() -> None:
        try:
            for memory in extract(payload.get("user", ""), payload.get("assistant", ""), _model_extractor):
                if should_persist(memory):
                    _get_store().upsert(memory)
        except Exception:  # noqa: BLE001 - never let memory break a turn
            pass

    threading.Thread(target=work, daemon=True).start()
    return payload


REMEMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "maxLength": 220},
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "importance": {"type": "string", "enum": [i.value for i in Importance]},
        "review_after": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
            "description": "ISO date (YYYY-MM-DD) — when to revisit this fact. Omit if no deadline.",
        },
    },
    "required": ["content"],
}
RECALL_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}}
FORGET_SCHEMA = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}


def register(ctx, model_extractor=None) -> None:
    global _model_extractor
    _model_extractor = model_extractor
    ctx.register_tool(name="remember", toolset="loci", schema=REMEMBER_SCHEMA, handler=_remember)
    ctx.register_tool(name="recall", toolset="loci", schema=RECALL_SCHEMA, handler=_recall)
    ctx.register_tool(name="forget", toolset="loci", schema=FORGET_SCHEMA, handler=_forget)
    ctx.register_hook("pre_llm_call", _inject_context)
    ctx.register_hook("post_llm_call", _extract_async)
