"""Heuristic extraction when Ollama is unavailable (offline / CI)."""

from __future__ import annotations

import re
from typing import Any

# Curated phrases improve edges for toy corpus without an LLM.
KNOWN = [
    "Acme Robotics Corporation",
    "Meridian",
    "Nexus-400",
    "Helix-LIDAR mini",
    "Orbit Logistics",
    "VEX Cognition",
    "Kubernetes",
    "Amélie Duarte",
    "Priya Nair",
    "Marco Silva",
    "Kenji Okada",
    "INC-2024-09",
    "Marseille",
    "Rotterdam",
    "Singapore",
    "Toulouse",
    "Helix-LIDAR",
]


def heuristic_extract(chunk_id: str, text: str) -> dict[str, Any]:
    found: set[str] = set()
    low = text

    for phrase in KNOWN:
        if phrase.lower() in low.lower():
            found.add(phrase)

    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b|\b([A-Z]{2,}(?:-[0-9]+[a-z]*)?)\b", text
    ):
        name = (m.group(1) or m.group(2) or "").strip()
        if len(name) >= 3 and not name.isdigit():
            found.add(name)

    entities = [
        {
            "name": name,
            "entity_type": "heuristic",
            "description": f"Mentioned in chunk {chunk_id}: {name}.",
        }
        for name in sorted(found)
    ][:24]

    relations: list[dict[str, str]] = []
    names = [e["name"] for e in entities]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        ins = [n for n in names if n.lower() in sent.lower()]
        for i, a in enumerate(ins):
            for b in ins[i + 1 : i + 3]:
                relations.append(
                    {
                        "source": a,
                        "target": b,
                        "relation_type": "co_occurs",
                        "description": sent[:240],
                    }
                )

    return {"entities": entities, "relations": relations[:40]}
