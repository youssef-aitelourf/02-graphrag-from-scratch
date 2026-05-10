"""Embedding model + Faiss flat inner-product index (normalized vectors ≈ cosine)."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag_mvp.config import Settings


def load_embedder(settings: Settings) -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def encode_chunks(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
        convert_to_numpy=True,
    )
    if emb.ndim != 2:
        raise RuntimeError("unexpected embedding shape")
    return emb.astype(np.float32)


def build_faiss(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index_bundle(
    out_dir: Path,
    index: faiss.IndexFlatIP,
    vectors: np.ndarray,
    chunk_ids: list[str],
    settings: Settings,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", vectors)
    (out_dir / "chunk_order.json").write_text(
        json.dumps(chunk_ids, indent=2), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps({"embed_model": settings.embed_model}, indent=2), encoding="utf-8"
    )


def load_index_bundle(
    out_dir: Path,
) -> tuple[faiss.IndexFlatIP, np.ndarray, list[str]]:
    index = faiss.read_index(str(out_dir / "faiss.index"))
    vectors = np.load(out_dir / "embeddings.npy")
    chunk_ids = json.loads((out_dir / "chunk_order.json").read_text(encoding="utf-8"))
    return index, vectors, chunk_ids
