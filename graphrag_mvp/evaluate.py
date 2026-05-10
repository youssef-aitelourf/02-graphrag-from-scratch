"""Offline recall@k / MRR: vanilla Faiss vs Faiss + graph expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from graphrag_mvp.config import Settings
from graphrag_mvp.embeddings import load_embedder, load_index_bundle
from graphrag_mvp.graph_store import load_graph_json
from graphrag_mvp.retrieval import graph_expand_retrieve, vanilla_retrieve


def load_chunks_jsonl(path: Path) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[row["chunk_id"]] = row["text"]
    return by_id


def gold_set(
    chunks_by_id: dict[str, str],
    substrings: list[str],
    *,
    match: str = "all",
) -> set[str]:
    out: set[str] = set()
    for cid, text in chunks_by_id.items():
        low = text.lower()
        if match == "any":
            ok = any(s.lower() in low for s in substrings)
        else:
            ok = all(s.lower() in low for s in substrings)
        if ok:
            out.add(cid)
    return out


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(ranked[:k])
    return len(gold & top) / len(gold)


def mrr(gold: set[str], ranked: list[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def evaluate_suite(
    artifacts_dir: Path,
    qa_path: Path,
    settings: Settings,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    chunks_by_id = load_chunks_jsonl(artifacts_dir / "chunks.jsonl")
    index, vectors, chunk_ids = load_index_bundle(artifacts_dir)
    G = load_graph_json(artifacts_dir / "graph.json")
    c2e = json.loads(
        (artifacts_dir / "chunk_to_entities.json").read_text(encoding="utf-8")
    )
    e2c = json.loads(
        (artifacts_dir / "entity_to_chunks.json").read_text(encoding="utf-8")
    )

    model = load_embedder(settings)

    questions = json.loads(qa_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {
        "ks": list(ks),
        "settings": {
            "embed_model": settings.embed_model,
            "faiss_seed_k": settings.faiss_seed_k,
            "graph_max_depth": settings.graph_max_depth,
            "graph_max_extra_chunks": settings.graph_max_extra_chunks,
            "leiden_resolution": settings.leiden_resolution,
        },
        "per_question": [],
    }
    meta_path = artifacts_dir / "build_meta.json"
    if meta_path.exists():
        results["build_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    van_recalls = {k: [] for k in ks}
    graph_recalls = {k: [] for k in ks}
    van_mrr: list[float] = []
    graph_mrr: list[float] = []

    max_k = max(ks)

    for item in questions:
        q = item["question"]
        subs = item["gold_substrings"]
        gold = gold_set(
            chunks_by_id,
            subs,
            match=str(item.get("gold_match", "all")),
        )
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(
            np.float32
        )

        van = vanilla_retrieve(index, chunk_ids, qv, max_k)
        ranked_v = [c for c, _ in van]

        graph = graph_expand_retrieve(
            settings,
            index,
            vectors,
            chunk_ids,
            c2e,
            e2c,
            G,
            qv,
            max_k,
        )
        ranked_g = [c for c, _ in graph]

        row = {
            "id": item.get("id"),
            "gold_size": len(gold),
            "vanilla_rank_first_hit": next(
                (i + 1 for i, c in enumerate(ranked_v) if c in gold), None
            ),
            "graph_rank_first_hit": next(
                (i + 1 for i, c in enumerate(ranked_g) if c in gold), None
            ),
        }
        results["per_question"].append(row)

        van_mrr.append(mrr(gold, ranked_v))
        graph_mrr.append(mrr(gold, ranked_g))
        for k in ks:
            van_recalls[k].append(recall_at_k(gold, ranked_v, k))
            graph_recalls[k].append(recall_at_k(gold, ranked_g, k))

    results["vanilla"] = {
        "mrr_mean": float(np.mean(van_mrr)) if van_mrr else 0.0,
        "recall_at_k_mean": {str(k): float(np.mean(van_recalls[k])) for k in ks},
    }
    results["graphrag_hybrid"] = {
        "mrr_mean": float(np.mean(graph_mrr)) if graph_mrr else 0.0,
        "recall_at_k_mean": {str(k): float(np.mean(graph_recalls[k])) for k in ks},
    }

    results["diagnosis"] = _diagnose(results["vanilla"], results["graphrag_hybrid"])
    return results


def _diagnose(van: dict, gr: dict) -> dict[str, str]:
    vk = van["recall_at_k_mean"]
    gk = gr["recall_at_k_mean"]
    k10_v = float(vk.get("10", 0))
    k10_g = float(gk.get("10", 0))
    if k10_g < k10_v - 1e-6:
        return {
            "status": "graphrag_below_vanilla_at_k10",
            "hint": (
                "Increase GRAPH_MAX_DEPTH or GRAPH_MAX_EXTRA_CHUNKS; lower LEIDEN_RESOLUTION "
                "to merge communities if expansion misses bridges; ensure extraction links "
                "co-occurring key entities; check FAISS_SEED_K large enough to seed the walk."
            ),
        }
    if k10_g >= k10_v:
        return {"status": "graphrag_meets_or_beats_vanilla", "hint": ""}
    return {"status": "compare", "hint": ""}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--qa", type=Path, default=Path("eval/qa_labeled.json"))
    p.add_argument("--out", type=Path, default=Path("metrics.json"))
    args = p.parse_args()
    settings = Settings.from_env()
    metrics = evaluate_suite(args.artifacts, args.qa, settings)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["vanilla"], indent=2))
    print(json.dumps(metrics["graphrag_hybrid"], indent=2))
    print(json.dumps(metrics["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
