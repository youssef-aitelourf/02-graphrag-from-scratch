"""Normalize entity keys and merge extraction records per chunk."""

from __future__ import annotations


def normalize_entity_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def entity_node_id(normalized: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in normalized)
    safe = "_".join(s for s in safe.split("_") if s)
    return f"ent:{safe}" if safe else "ent:unknown"
