"""Community (and meta-community) summaries via LLM."""

from __future__ import annotations

import json
from typing import Any

import httpx

from graphrag_mvp.config import Settings


def _post_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    json_schema: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> str:
    if not settings.ollama_api_key:
        raise RuntimeError("OLLAMA_API_KEY is required for summarization")
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
    }
    if json_schema is not None:
        payload["format"] = json_schema
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
        return body.get("message", {}).get("content", "")
    finally:
        if close:
            client.close()


SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "entity_highlights": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["title", "summary", "entity_highlights"],
}


META_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}


def summarize_community(
    settings: Settings,
    community_id: int,
    entity_payload: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    user = json.dumps(
        {
            "community_id": community_id,
            "entities": entity_payload,
        },
        ensure_ascii=False,
    )
    messages = [
        {
            "role": "system",
            "content": "Write a factual community report for retrieval. stay faithful to input.",
        },
        {
            "role": "user",
            "content": "Given this JSON of entities, produce title, summary (3-6 sentences), "
            f"and entity_highlights (up to 8 short phrases). JSON input:\n{user}",
        },
    ]
    raw = _post_chat(settings, messages, json_schema=SUMMARY_SCHEMA, client=client)
    data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    data["community_id"] = community_id
    return data


def summarize_meta(
    settings: Settings,
    community_summaries: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    brief = [
        {"community_id": c.get("community_id"), "title": c.get("title"), "summary": c.get("summary")}
        for c in community_summaries
    ]
    user = json.dumps({"communities": brief}, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": "You compress multiple community reports into one organizational overview.",
        },
        {
            "role": "user",
            "content": "Produce title and summary (max 5 sentences) spanning all communities. JSON:\n"
            + user,
        },
    ]
    raw = _post_chat(settings, messages, json_schema=META_SCHEMA, client=client)
    return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())


def mock_local_summaries(
    artifacts_communities: dict[int, list[str]], graph_nodes: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Offline placeholder when LLM summaries are skipped."""
    reports: list[dict[str, Any]] = []
    for cid, members in sorted(artifacts_communities.items()):
        labels = [
            graph_nodes[m].get("display_name", m)
            for m in members[:20]
            if m in graph_nodes
        ]
        reports.append(
            {
                "community_id": cid,
                "title": f"Community {cid}",
                "summary": "Entities: " + ", ".join(labels[:12]) + ("…" if len(labels) > 12 else ""),
                "entity_highlights": labels[:8],
            }
        )
    meta = None
    if len(reports) > 1:
        meta = {
            "title": "Corpus overview",
            "summary": " ".join(r["summary"] for r in reports[:5]),
        }
    return reports, meta
