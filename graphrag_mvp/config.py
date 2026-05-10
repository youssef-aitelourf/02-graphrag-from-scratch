"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_model: str
    ollama_api_key: str | None
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    faiss_seed_k: int
    graph_max_depth: int
    graph_max_extra_chunks: int
    leiden_resolution: float

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            ollama_host=os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "gpt-oss:120b"),
            ollama_api_key=os.environ.get("OLLAMA_API_KEY"),
            embed_model=os.environ.get(
                "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "420")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "90")),
            faiss_seed_k=int(os.environ.get("FAISS_SEED_K", "6")),
            graph_max_depth=int(os.environ.get("GRAPH_MAX_DEPTH", "2")),
            graph_max_extra_chunks=int(os.environ.get("GRAPH_MAX_EXTRA_CHUNKS", "24")),
            leiden_resolution=float(os.environ.get("LEIDEN_RESOLUTION", "1.0")),
        )
