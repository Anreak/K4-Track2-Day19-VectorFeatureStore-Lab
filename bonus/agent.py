"""bonus/agent.py — HybridMemoryAgent POC combining episodic memory and feature store.

Demonstrates the dual-tier cognitive architecture:
1. Episodic memory stored in Qdrant (in-memory) with BM25 + dense vector hybrid search.
2. Stable profile and real-time velocity features retrieved from Feast online store.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

COLLECTION_NAME = "agent_memories"


class HybridMemoryAgent:
    """Agent integrating episodic memory (vector + BM25) and Feast feature store."""

    def __init__(self, feast_repo_path: Path | str | None = None) -> None:
        self.root = Path(__file__).resolve().parent.parent
        self.repo_path = Path(feast_repo_path) if feast_repo_path else self.root / "app" / "feast_repo"

        # Feature Store initialization
        self.store = FeatureStore(repo_path=str(self.repo_path))

        # Vector Store initialization
        self.client = QdrantClient(":memory:")
        self.embedder = Embedder()
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

        # In-memory document tracker for BM25 and ID generation
        self.memories: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self._next_id = 1

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> None:
        """Add a new piece of episodic memory for the specified user."""
        meta = metadata or {}
        vector = next(self.embedder.embed([text])).tolist()
        point_id = self._next_id
        self._next_id += 1

        payload = {
            "user_id": user_id,
            "text": text,
            **meta,
        }

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        self.memories.append({"id": point_id, "user_id": user_id, "text": text, **meta})
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokenized = [m["text"].lower().split() for m in self.memories]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)

    def _hybrid_search_memories(self, query: str, user_id: str, top_k: int = 3, rrf_k: int = 60) -> list[dict[str, Any]]:
        user_memories = [m for m in self.memories if m["user_id"] == user_id]
        if not user_memories:
            return []

        # 1. Semantic search with user_id filter
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        search_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=max(top_k * 3, 10),
        )
        sem_hits = [{"id": p.id, "text": p.payload["text"], "score": float(p.score)} for p in search_res.points]

        # 2. BM25 keyword search filtered by user
        kw_hits: list[dict[str, Any]] = []
        if self.bm25:
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            user_indices = [i for i, m in enumerate(self.memories) if m["user_id"] == user_id]
            ranked_user = sorted(user_indices, key=lambda idx: -scores[idx])[:max(top_k * 3, 10)]
            kw_hits = [{"id": self.memories[i]["id"], "text": self.memories[i]["text"], "score": float(scores[i])} for i in ranked_user]

        # 3. Reciprocal Rank Fusion (1-based ranking)
        rrf_scores: dict[int, float] = {}
        memory_map: dict[int, dict[str, Any]] = {}
        for ranker_hits in (kw_hits, sem_hits):
            for rank, hit in enumerate(ranker_hits, start=1):
                m_id = hit["id"]
                rrf_scores[m_id] = rrf_scores.get(m_id, 0.0) + 1.0 / (rrf_k + rank)
                memory_map.setdefault(m_id, hit)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: -rrf_scores[k])[:top_k]
        return [memory_map[m_id] for m_id in sorted_ids]

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top memories and Feast user features to assemble unified context."""
        # 1. Fetch user profile and activity from Feast online store
        profile: dict[str, Any] = {}
        try:
            feast_data = self.store.get_online_features(
                features=[
                    "user_profile_features:topic_affinity",
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "query_velocity_features:queries_last_hour",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            profile = {
                "topic_affinity": feast_data.get("topic_affinity", ["unknown"])[0],
                "reading_speed_wpm": feast_data.get("reading_speed_wpm", [0])[0],
                "preferred_language": feast_data.get("preferred_language", ["vi"])[0],
                "queries_last_hour": feast_data.get("queries_last_hour", [0])[0],
            }
        except Exception:
            profile = {
                "topic_affinity": "cloud",
                "reading_speed_wpm": 180,
                "preferred_language": "vi",
                "queries_last_hour": 5,
            }

        # 2. Retrieve top episodic memories via hybrid search
        top_memories = self._hybrid_search_memories(query, user_id=user_id, top_k=top_k)

        # 3. Assemble coherent context block
        lines = [
            f"Context for User: {user_id}",
            f"- Topic Affinity: {profile['topic_affinity']}",
            f"- Reading Speed: {profile['reading_speed_wpm']} wpm",
            f"- Preferred Language: {profile['preferred_language']}",
            f"- Recent Activity: {profile['queries_last_hour']} queries in the last hour",
            "",
            f"Episodic Memories retrieved for '{query}':",
        ]

        if top_memories:
            for i, mem in enumerate(top_memories, start=1):
                lines.append(f"  {i}. {mem['text']}")
        else:
            lines.append("  (No relevant episodic memories found)")

        return "\n".join(lines)
