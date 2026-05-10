"""Build NetworkX graph from extractions + Leiden communities (python-igraph)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import igraph as ig
import networkx as nx

from graphrag_mvp.config import Settings
from graphrag_mvp.entities import entity_node_id, normalize_entity_name


@dataclass
class GraphArtifacts:
    graph: nx.MultiDiGraph
    chunk_to_entities: dict[str, list[str]]
    entity_to_chunks: dict[str, set[str]]
    partition: dict[str, int]
    communities: dict[int, list[str]]


def build_graph_from_extractions(
    extractions: list[dict[str, Any]],
) -> tuple[nx.MultiDiGraph, dict[str, list[str]], dict[str, set[str]]]:
    G = nx.MultiDiGraph()
    chunk_to_entities: dict[str, list[str]] = {}
    entity_to_chunks: dict[str, set[str]] = {}

    def ensure_node(nid: str, display: str, etype: str) -> None:
        if not G.has_node(nid):
            G.add_node(
                nid,
                display_name=display.strip(),
                entity_type=etype.strip() or "unknown",
                descriptions=[],
                chunk_ids=[],
            )

    for row in extractions:
        cid = row["chunk_id"]
        ents = row.get("entities") or []
        rels = row.get("relations") or []
        ordered_unique: list[str] = []

        for ent in ents:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            norm = normalize_entity_name(name)
            nid = entity_node_id(norm)
            desc = (ent.get("description") or "").strip()
            et = (ent.get("entity_type") or "unknown").strip()
            ensure_node(nid, name, et)
            node = G.nodes[nid]
            if desc and desc not in node["descriptions"]:
                node["descriptions"].append(desc)
            if cid not in node["chunk_ids"]:
                node["chunk_ids"].append(cid)
            entity_to_chunks.setdefault(nid, set()).add(cid)
            if nid not in ordered_unique:
                ordered_unique.append(nid)

        chunk_to_entities[cid] = ordered_unique

        for rel in rels:
            s_raw = (rel.get("source") or "").strip()
            t_raw = (rel.get("target") or "").strip()
            if not s_raw or not t_raw:
                continue
            sid = entity_node_id(normalize_entity_name(s_raw))
            tid = entity_node_id(normalize_entity_name(t_raw))
            ensure_node(sid, s_raw, "inferred")
            ensure_node(tid, t_raw, "inferred")
            rtype = (rel.get("relation_type") or "related_to").strip()
            rdesc = (rel.get("description") or "").strip()
            key = f"{cid}::{rtype}::{hash(rdesc) % 1_000_000_000}"
            G.add_edge(
                sid,
                tid,
                key=key,
                relation_type=rtype,
                description=rdesc,
                chunk_id=cid,
            )
            for nid in (sid, tid):
                n = G.nodes[nid]
                if cid not in n["chunk_ids"]:
                    n["chunk_ids"].append(cid)
                entity_to_chunks.setdefault(nid, set()).add(cid)
                if nid not in chunk_to_entities.setdefault(cid, []):
                    chunk_to_entities[cid].append(nid)

    return G, chunk_to_entities, entity_to_chunks


def run_leiden(
    G: nx.MultiDiGraph, resolution: float
) -> tuple[dict[str, int], dict[int, list[str]]]:
    if G.number_of_nodes() == 0:
        return {}, {}
    U = nx.Graph()
    for n in G.nodes():
        U.add_node(n)
    for u, v, _k, _d in G.edges(keys=True, data=True):
        w = 1.0
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)

    nx_nodes = list(U.nodes())
    Gi = ig.Graph.from_networkx(U)
    partition = Gi.community_leiden(weights="weight", resolution=resolution, n_iterations=-1)
    membership = partition.membership
    id_to_comm: dict[str, int] = {}
    comm_to_nodes: dict[int, list[str]] = {}
    for i, c in enumerate(membership):
        node = nx_nodes[i]
        id_to_comm[node] = c
        comm_to_nodes.setdefault(c, []).append(node)
    return id_to_comm, comm_to_nodes


def build_artifacts(settings: Settings, extractions: list[dict[str, Any]]) -> GraphArtifacts:
    G, c2e, e2c = build_graph_from_extractions(extractions)
    part, comms = run_leiden(G, settings.leiden_resolution)
    return GraphArtifacts(
        graph=G,
        chunk_to_entities=c2e,
        entity_to_chunks=e2c,
        partition=part,
        communities=comms,
    )


def save_graph_json(G: nx.MultiDiGraph, path: Path) -> None:
    payload = nx.node_link_data(G)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_graph_json(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    edge_key = "links" if "links" in data else "edges"
    return nx.node_link_graph(data, edges=edge_key, multigraph=True, directed=True)


def expand_entities(
    seed_entities: Iterable[str],
    G: nx.MultiDiGraph,
    max_depth: int,
) -> set[str]:
    frontier = {e for e in seed_entities if G.has_node(e)}
    seen = set(frontier)
    depth = 0
    while frontier and depth < max_depth:
        nxt: set[str] = set()
        for e in frontier:
            nxt.update(G.successors(e))
            nxt.update(G.predecessors(e))
        nxt -= seen
        seen |= nxt
        frontier = nxt
        depth += 1
    return seen
