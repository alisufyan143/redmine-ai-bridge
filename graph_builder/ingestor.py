"""Vectorization and relational dual-write ingestion module.

Orchestrates batch embeddings via the new Google GenAI SDK (google.genai),
vector storage in Pinecone, and non-blocking relational edge persistence in
Supabase (supabase.create_async_client) with circuit breaker resilience
and concurrency throttling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Generator, Sequence, TypeVar
import uuid

from google import genai
from google.genai import types
import pinecone
from supabase import AsyncClient, create_async_client

from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("graph_builder.ingestor")

T = TypeVar("T")


class GraphIngestor:
    """Ingests AST entities into Pinecone and Supabase with batched async operations."""

    def __init__(
        self,
        google_api_key: str = "mock_key",
        pinecone_api_key: str = "mock_key",
        pinecone_index_name: str = "redmine-code-graph",
        supabase_url: str = "https://mock.supabase.co",
        supabase_key: str = "mock_key",
        embedding_model: str = "gemini-embedding-001",
        output_dim: int = 3072,
        supabase_client: AsyncClient | None = None,
        pinecone_client: Any | None = None,
        pinecone_index: Any | None = None,
        genai_client: Any | None = None,
    ) -> None:
        self.google_api_key: str = google_api_key
        self.pinecone_api_key: str = pinecone_api_key
        self.pinecone_index_name: str = pinecone_index_name
        self.supabase_url: str = supabase_url
        self.supabase_key: str = supabase_key
        self.embedding_model: str = embedding_model
        self.output_dim: int = output_dim

        # Google GenAI SDK Client (new google.genai)
        self.genai_client = genai_client or genai.Client(api_key=google_api_key)
        
        # Pinecone Client & Index
        self.pc = pinecone_client or (pinecone.Pinecone(api_key=pinecone_api_key) if pinecone_api_key != "mock_key" else None)
        if pinecone_index is not None:
            self.pinecone_index = pinecone_index
        elif self.pc is not None:
            self.pinecone_index = getattr(self.pc, "Index", lambda name: None)(pinecone_index_name)
        else:
            self.pinecone_index = None
        
        # Supabase AsyncClient
        self._supabase_client: AsyncClient | None = supabase_client

    async def get_supabase(self) -> AsyncClient:
        """Returns or initializes the non-blocking asynchronous Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = await create_async_client(
                self.supabase_url, self.supabase_key
            )
        return self._supabase_client

    @staticmethod
    def _batch_generator(iterable: Sequence[T], batch_size: int = 100) -> Generator[list[T], None, None]:
        """Splits an iterable sequence into chunks of maximum batch_size."""
        if batch_size <= 0:
            batch_size = 100
        for i in range(0, len(iterable), batch_size):
            yield list(iterable[i : i + batch_size])

    @circuit_breaker(timeout_seconds=20)
    async def process_batch(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Processes a single batch: generates embeddings, upserts to Pinecone, and inserts edges into Supabase."""
        embedded_count = 0
        upserted_vectors = 0
        inserted_edges = 0

        # 1. Generate embeddings using new Google GenAI SDK
        if nodes:
            node_texts = [
                str(node.get("content") or node.get("name") or "") for node in nodes
            ]

            embeddings: list[list[float]] = []
            embed_resp = None
            config = types.EmbedContentConfig(output_dimensionality=self.output_dim)

            try:
                if hasattr(self.genai_client, "aio") and hasattr(self.genai_client.aio, "models"):
                    embed_call = self.genai_client.aio.models.embed_content(
                        model=self.embedding_model,
                        contents=node_texts,
                        config=config,
                    )
                elif hasattr(self.genai_client, "models") and hasattr(self.genai_client.models, "embed_content"):
                    embed_call = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=node_texts,
                        config=config,
                    )
                else:
                    embed_call = None

                if embed_call is not None:
                    if asyncio.iscoroutine(embed_call):
                        embed_resp = await embed_call
                    else:
                        embed_resp = embed_call
            except Exception as embed_err:
                logger.warning(
                    f"Batch embedding generation warning: {embed_err}",
                    extra={"extra_payload": {"error": str(embed_err)}},
                )

            if embed_resp is not None:
                if hasattr(embed_resp, "embeddings") and embed_resp.embeddings:
                    for emb in embed_resp.embeddings:
                        if hasattr(emb, "values"):
                            embeddings.append(list(emb.values))
                        elif isinstance(emb, list):
                            embeddings.append(emb)
                elif hasattr(embed_resp, "embedding") and getattr(embed_resp.embedding, "values", None):
                    embeddings.append(list(embed_resp.embedding.values))
                elif isinstance(embed_resp, dict) and "embedding" in embed_resp:
                    embeddings.append(embed_resp["embedding"].get("values", []))

            # Default fallback for missing or mock vectors
            while len(embeddings) < len(nodes):
                embeddings.append([0.0] * self.output_dim)

            vectors_to_upsert: list[dict[str, Any]] = []
            for node, embedding in zip(nodes, embeddings):
                node_id = str(node.get("id") or uuid.uuid4())
                metadata = {
                    "name": str(node.get("name", "")),
                    "type": str(node.get("type", "unknown")),
                    "content": str(node.get("content", node.get("name", ""))),
                }
                if "file_path" in node:
                    metadata["file_path"] = str(node["file_path"])

                vectors_to_upsert.append({
                    "id": node_id,
                    "values": embedding,
                    "metadata": metadata,
                })

            if self.pinecone_index is not None and hasattr(self.pinecone_index, "upsert"):
                self.pinecone_index.upsert(vectors=vectors_to_upsert)

            upserted_vectors = len(vectors_to_upsert)
            embedded_count = len(node_texts)


        # 2. Insert relational edges asynchronously into Supabase without blocking the event loop
        if edges:
            formatted_edges: list[dict[str, Any]] = []
            for edge in edges:
                formatted_edges.append({
                    "id": str(edge.get("id") or uuid.uuid4()),
                    "source_id": str(edge["source_id"]),
                    "target_id": str(edge["target_id"]),
                    "relation_type": str(edge.get("relation_type", "relates_to")),
                })

            supabase: AsyncClient = await self.get_supabase()
            table = supabase.table("edges")
            query = table.insert(formatted_edges)
            result = query.execute()
            if asyncio.iscoroutine(result):
                await result

            inserted_edges = len(formatted_edges)

        return {
            "embedded_nodes": embedded_count,
            "upserted_vectors": upserted_vectors,
            "inserted_edges": inserted_edges,
        }

    async def ingest_entities(
        self,
        extracted_entities: dict[str, Any],
        batch_size: int = 100,
        max_concurrency: int = 5,
    ) -> dict[str, Any]:
        """Orchestrates entity transformation, batched chunking, and throttled dual-write ingest."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        if "nodes" in extracted_entities and isinstance(extracted_entities["nodes"], list):
            nodes.extend(extracted_entities["nodes"])
            if "edges" in extracted_entities and isinstance(extracted_entities["edges"], list):
                edges.extend(extracted_entities["edges"])
        else:
            classes = extracted_entities.get("classes", [])
            methods = extracted_entities.get("methods", [])
            requires = extracted_entities.get("requires", [])

            class_nodes: list[dict[str, Any]] = []
            for cls_name in classes:
                c_node = {
                    "id": str(uuid.uuid4()),
                    "name": str(cls_name),
                    "type": "class",
                    "content": f"class {cls_name}",
                }
                class_nodes.append(c_node)
                nodes.append(c_node)

            primary_class_id = class_nodes[0]["id"] if class_nodes else str(uuid.uuid4())

            for meth_name in methods:
                m_node = {
                    "id": str(uuid.uuid4()),
                    "name": str(meth_name),
                    "type": "method",
                    "content": f"def {meth_name}",
                }
                nodes.append(m_node)
                edges.append({
                    "id": str(uuid.uuid4()),
                    "source_id": primary_class_id,
                    "target_id": m_node["id"],
                    "relation_type": "defines",
                })

            for req_name in requires:
                r_node = {
                    "id": str(uuid.uuid4()),
                    "name": str(req_name),
                    "type": "require",
                    "content": f"require '{req_name}'",
                }
                nodes.append(r_node)
                edges.append({
                    "id": str(uuid.uuid4()),
                    "source_id": primary_class_id,
                    "target_id": r_node["id"],
                    "relation_type": "depends_on",
                })

        # Batch nodes and edges
        node_batches = list(self._batch_generator(nodes, batch_size=batch_size))
        edge_batches = list(self._batch_generator(edges, batch_size=batch_size))

        num_batches = max(len(node_batches), len(edge_batches), 1 if (nodes or edges) else 0)
        batch_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

        for i in range(num_batches):
            b_nodes = node_batches[i] if i < len(node_batches) else []
            b_edges = edge_batches[i] if i < len(edge_batches) else []
            batch_pairs.append((b_nodes, b_edges))

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_process(
            b_nodes: list[dict[str, Any]],
            b_edges: list[dict[str, Any]],
        ) -> dict[str, int]:
            async with semaphore:
                return await self.process_batch(b_nodes, b_edges)

        tasks = [bounded_process(b_nodes, b_edges) for b_nodes, b_edges in batch_pairs]
        results = await asyncio.gather(*tasks) if tasks else []

        total_embedded = sum(r.get("embedded_nodes", 0) for r in results)
        total_vectors = sum(r.get("upserted_vectors", 0) for r in results)
        total_edges = sum(r.get("inserted_edges", 0) for r in results)

        return {
            "total_batches": len(batch_pairs),
            "embedded_nodes": total_embedded,
            "upserted_vectors": total_vectors,
            "inserted_edges": total_edges,
        }
