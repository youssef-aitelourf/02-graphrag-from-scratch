"""Ollama Cloud chat client (structured JSON extraction)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from graphrag_mvp.config import Settings

logger = logging.getLogger(__name__)

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "entity_type", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "relation_type", "description"],
            },
        },
    },
    "required": ["entities", "relations"],
}


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def extract_graph_elements(
    settings: Settings,
    chunk_text: str,
    chunk_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Call Ollama chat with JSON schema; return dict with entities and relations."""
    if not settings.ollama_api_key:
        raise RuntimeError("OLLAMA_API_KEY is required for LLM extraction")

    system = (
        "You are an information extraction system. Extract typed entities and directed "
        "relations from the user text. Use concise descriptions. "
        "Entity names must be canonical (proper nouns, product codenames). "
        f"The text belongs to chunk_id={chunk_id}."
    )
    user = (
        "Text:\n---\n"
        f"{chunk_text}\n---\n"
        "Return ONLY JSON matching the schema: entities (name, entity_type, description), "
        "relations (source, target, relation_type, description). "
        "Relations must connect entity names you list (source/target exact match to entity name)."
    )
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": EXTRACTION_JSON_SCHEMA,
    }
    headers = {"Authorization": f"Bearer {settings.ollama_api_key}"}
    url = f"{settings.ollama_host}/api/chat"
    close = False
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(240.0))
        close = True
    try:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
        content = body.get("message", {}).get("content", "")
        data = _extract_json_object(content)
        if not isinstance(data.get("entities"), list) or not isinstance(
            data.get("relations"), list
        ):
            raise ValueError("LLM JSON missing entities/relations lists")
        return data
    finally:
        if close:
            client.close()
