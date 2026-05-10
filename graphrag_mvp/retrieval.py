"""Hybrid retrieval: Faiss seeds + graph expansion + cosine rerank."""

from __future__ import annotations

import faiss
import numpy as np
from numpy.typing import NDArray

from graphrag_mvp.config import Settings
from graphrag_mvp.graph_store import expand_entities
import networkx as nx


def faiss_search(
    index: faiss.IndexFlatIP,
    query_vec: NDArray[np.float32],
    k: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, idx = index.search(q, k)
    return scores[0], idx[0]


def vanilla_retrieve(
    index: faiss.IndexFlatIP,
    chunk_ids: list[str],
    query_vec: NDArray[np.float32],
    k: int,
) -> list[tuple[str, float]]:
    scores, idx = faiss_search(index, query_vec, min(k, len(chunk_ids)))
    return [(chunk_ids[int(i)], float(scores[j])) for j, i in enumerate(idx) if i >= 0]


def graph_expand_retrieve(
    settings: Settings,
    index: faiss.IndexFlatIP,
    vectors: NDArray[np.float32],
    chunk_ids: list[str],
    chunk_to_entities: dict[str, list[str]],
    entity_to_chunks: dict[str, list[str]],
    G: nx.MultiDiGraph,
    query_vec: NDArray[np.float32],
    final_k: int,
) -> list[tuple[str, float]]:
    seed_k = min(settings.faiss_seed_k, len(chunk_ids))
    seed_scores, seed_idx = faiss_search(index, query_vec, seed_k)
    seed_chunks = [chunk_ids[int(i)] for i in seed_idx if int(i) >= 0]

    seeds_entities: list[str] = []
    for c in seed_chunks:
        seeds_entities.extend(chunk_to_entities.get(c, []))
    expanded = expand_entities(seeds_entities, G, settings.graph_max_depth)

    cand: set[str] = set(seed_chunks)
    extra = 0
    for e in expanded:
        for ch in entity_to_chunks.get(e, []):
            if ch not in cand:
                cand.add(ch)
                extra += 1
                if extra >= settings.graph_max_extra_chunks:
                    break
        if extra >= settings.graph_max_extra_chunks:
            break

    id_to_row = {cid: i for i, cid in enumerate(chunk_ids)}
    scores: list[tuple[str, float]] = []
    q = query_vec.astype(np.float32)
    for cid in cand:
        row = id_to_row.get(cid)
        if row is None:
            continue
        sim = float(np.dot(q, vectors[row]))
        scores.append((cid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:final_k]
