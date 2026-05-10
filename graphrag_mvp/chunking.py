"""Text chunking with fixed character windows and overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int


def _slug_doc_id(rel_path: str) -> str:
    base = rel_path.replace("/", "_").replace("\\", "_")
    if base.endswith(".txt"):
        base = base[:-4]
    return re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_").lower()


def chunk_text(doc_id: str, full_text: str, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size - 1))
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    pos = 0
    idx = 0
    n = len(full_text)
    while pos < n:
        end = min(pos + size, n)
        piece = full_text[pos:end].strip()
        if piece:
            cid = f"{doc_id}__{idx:03d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    text=piece,
                    char_start=pos,
                    char_end=end,
                )
            )
            idx += 1
        if end >= n:
            break
        pos += step
    return chunks
