"""End-to-end indexing: corpus → chunks → extraction → graph → embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import httpx

from graphrag_mvp.chunking import chunk_text
from graphrag_mvp.config import Settings
from graphrag_mvp.embeddings import (
    build_faiss,
    encode_chunks,
    load_embedder,
    save_index_bundle,
)
from graphrag_mvp.graph_store import build_artifacts, save_graph_json
from graphrag_mvp.heuristic_extract import heuristic_extract
from graphrag_mvp.ollama_client import extract_graph_elements
from graphrag_mvp.summaries import mock_local_summaries, summarize_community, summarize_meta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_corpus_chunks(corpus_dir: Path, settings: Settings) -> list:
    from graphrag_mvp.chunking import Chunk

    chunks: list[Chunk] = []
    for p in sorted(corpus_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        doc_id = p.stem.lower()
        chunks.extend(
            chunk_text(doc_id, text, settings.chunk_size, settings.chunk_overlap)
        )
    return chunks


def run_pipeline(
    corpus_dir: Path,
    artifacts_dir: Path,
    settings: Settings,
    *,
    use_heuristic_extraction: bool = False,
    skip_llm_summaries: bool = False,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_corpus_chunks(corpus_dir, settings)
    logger.info("Chunks: %d", len(chunks))

    extractions: list[dict] = []
    shared_client: httpx.Client | None = None
    if not use_heuristic_extraction:
        shared_client = httpx.Client(timeout=httpx.Timeout(240.0))

    try:
        for i, ch in enumerate(chunks):
            if use_heuristic_extraction:
                data = heuristic_extract(ch.chunk_id, ch.text)
            else:
                data = extract_graph_elements(
                    settings,
                    ch.text,
                    ch.chunk_id,
                    client=shared_client,
                )
            row = {
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "text": ch.text,
                "entities": data.get("entities", []),
                "relations": data.get("relations", []),
            }
            extractions.append(row)
            if (i + 1) % 5 == 0:
                logger.info("Extracted %d/%d chunks", i + 1, len(chunks))
    finally:
        if shared_client is not None:
            shared_client.close()

    ext_path = artifacts_dir / "chunks.jsonl"
    with ext_path.open("w", encoding="utf-8") as f:
        for row in extractions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    art = build_artifacts(settings, extractions)
    save_graph_json(art.graph, artifacts_dir / "graph.json")

    e2c_lists = {k: sorted(v) for k, v in art.entity_to_chunks.items()}
    (artifacts_dir / "chunk_to_entities.json").write_text(
        json.dumps(art.chunk_to_entities, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "entity_to_chunks.json").write_text(
        json.dumps(e2c_lists, indent=2), encoding="utf-8"
    )
    part_str = {k: int(v) for k, v in art.partition.items()}
    (artifacts_dir / "partition.json").write_text(
        json.dumps(part_str, indent=2), encoding="utf-8"
    )
    comm_str = {str(k): v for k, v in art.communities.items()}
    (artifacts_dir / "communities.json").write_text(
        json.dumps(comm_str, indent=2), encoding="utf-8"
    )

    G = art.graph
    nodes = dict(G.nodes(data=True))
    community_reports: list[dict] = []

    if skip_llm_summaries or use_heuristic_extraction:
        community_reports, meta = mock_local_summaries(art.communities, nodes)
    else:
        client = httpx.Client(timeout=httpx.Timeout(240.0))
        try:
            for cid, members in sorted(art.communities.items()):
                payload = []
                for m in members:
                    d = nodes.get(m) or {}
                    payload.append(
                        {
                            "id": m,
                            "display_name": d.get("display_name", m),
                            "entity_type": d.get("entity_type", ""),
                            "descriptions": d.get("descriptions", []),
                        }
                    )
                community_reports.append(
                    summarize_community(settings, cid, payload, client=client)
                )
            meta = (
                summarize_meta(settings, community_reports, client=client)
                if len(community_reports) > 1
                else None
            )
        finally:
            client.close()

    out_reports = {
        "communities": community_reports,
        "meta": meta,
    }
    (artifacts_dir / "community_reports.json").write_text(
        json.dumps(out_reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model = load_embedder(settings)
    texts = [row["text"] for row in extractions]
    vectors = encode_chunks(model, texts)
    chunk_ids = [row["chunk_id"] for row in extractions]
    index = build_faiss(vectors)
    save_index_bundle(artifacts_dir, index, vectors, chunk_ids, settings)
    build_meta = {
        "extraction": "heuristic" if use_heuristic_extraction else "ollama",
        "chunk_count": len(chunks),
        "ollama_model": settings.ollama_model,
    }
    (artifacts_dir / "build_meta.json").write_text(
        json.dumps(build_meta, indent=2), encoding="utf-8"
    )
    logger.info("Wrote artifacts under %s", artifacts_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="GraphRAG MVP indexer")
    p.add_argument("--corpus", type=Path, default=Path("corpus"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument(
        "--heuristic",
        action="store_true",
        help="Regex/heuristic extraction (no Ollama). Implies --skip-llm-summaries.",
    )
    p.add_argument(
        "--skip-llm-summaries",
        action="store_true",
        help="Do not call LLM for Leiden community summaries.",
    )
    args = p.parse_args()
    settings = Settings.from_env()
    use_h = args.heuristic or os.environ.get("GRAPHRAG_MOCK_LLM") == "1"
    skip_sum = args.skip_llm_summaries or use_h
    run_pipeline(
        args.corpus,
        args.out,
        settings,
        use_heuristic_extraction=use_h,
        skip_llm_summaries=skip_sum,
    )


if __name__ == "__main__":
    main()
