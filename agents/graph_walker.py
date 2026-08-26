"""Graph RAG Walker Agent module.

Performs semantic vector search over Pinecone, traverses relational knowledge edges
via Supabase, builds in-memory NetworkX graphs, and prunes subgraphs using PageRank.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any
from google import genai
from google.genai import types
import networkx as nx
import pinecone
from supabase import AsyncClient, create_async_client

from state import GraphContext, VisualTriageResult
from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("agents.graph_walker")


class GraphWalkerAgent:
    """Agent that performs semantic search, graph traversal, and PageRank pruning for Graph RAG."""

    def __init__(
        self,
        google_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        pinecone_index_name: str = "redmine-code-graph",
        supabase_url: str = "https://mock.supabase.co",
        supabase_key: str = "mock_key",
        embedding_model: str = "gemini-embedding-001",
        output_dim: int = 3072,
        genai_client: Any | None = None,
        pinecone_index: Any | None = None,
        supabase_client: AsyncClient | None = None,
    ) -> None:
        self.embedding_model: str = embedding_model
        self.output_dim: int = output_dim
        self.supabase_url: str = supabase_url
        self.supabase_key: str = supabase_key

        if genai_client is not None:
            self.genai_client: Any = genai_client
        else:
            self.genai_client = genai.Client(api_key=google_api_key).aio

        if pinecone_index is not None:
            self.pinecone_index: Any = pinecone_index
        elif pinecone_api_key:
            pc = pinecone.Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index(pinecone_index_name)
        else:
            self.pinecone_index = None

        self._supabase_client: AsyncClient | None = supabase_client

    async def get_supabase(self) -> AsyncClient:
        """Retrieves or initializes the async Supabase client."""
        if self._supabase_client is None:
            self._supabase_client = await create_async_client(
                self.supabase_url, self.supabase_key
            )
        return self._supabase_client

    @circuit_breaker(timeout_seconds=20)
    async def build_context(self, triage: VisualTriageResult) -> GraphContext:
        """Constructs a pruned GraphContext using semantic search, edge traversal, and PageRank."""
        # a) Formulate query string and generate embedding
        suspected_module = getattr(triage, "Suspected_Module", "")
        error_trace = getattr(triage, "Error_Trace_Visible", "")
        if suspected_module or error_trace:
            query_str = f"{suspected_module} {error_trace}".strip()
        else:
            ui_elements = " ".join(getattr(triage, "ui_elements_detected", []))
            summary = getattr(triage, "triage_summary", "")
            query_str = f"{summary} {ui_elements}".strip()

        if not query_str:
            query_str = "system failure diagnosis"

        logger.info(
            f"Embedding query for Graph RAG walk: '{query_str}'",
            extra={"extra_payload": {"ticket_id": getattr(triage, "ticket_id", "unknown")}},
        )

        vector: list[float] = [0.0] * self.output_dim
        embed_resp = None
        config = types.EmbedContentConfig(output_dimensionality=self.output_dim)

        try:
            if hasattr(self.genai_client, "aio") and hasattr(self.genai_client.aio, "models"):
                embed_call = self.genai_client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=[query_str],
                    config=config,
                )
            elif hasattr(self.genai_client, "models") and hasattr(self.genai_client.models, "embed_content"):
                embed_call = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=[query_str],
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
                f"Query embedding generation warning: {embed_err}",
                extra={"extra_payload": {"error": str(embed_err)}},
            )


        if embed_resp is not None:
            if hasattr(embed_resp, "embeddings") and embed_resp.embeddings:
                emb = embed_resp.embeddings[0]
                vector = list(emb.values if hasattr(emb, "values") else emb)
            elif hasattr(embed_resp, "embedding") and getattr(embed_resp.embedding, "values", None):
                vector = list(embed_resp.embedding.values)
            elif isinstance(embed_resp, dict) and "embedding" in embed_resp:
                vector = list(embed_resp["embedding"].get("values", []))

        # b) Query Pinecone to retrieve top 5 relevant node IDs
        top_node_ids: list[str] = []
        if self.pinecone_index is not None and hasattr(self.pinecone_index, "query"):
            query_res = self.pinecone_index.query(vector=vector, top_k=5, include_metadata=True)
            if asyncio.iscoroutine(query_res):
                query_res = await query_res

            matches = (
                getattr(query_res, "matches", [])
                or (query_res.get("matches", []) if isinstance(query_res, dict) else [])
            )
            for m in matches:
                match_id = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else str(m))
                if match_id:
                    top_node_ids.append(str(match_id))

        # c) Query Supabase for edges connecting top_node_ids
        edges_data: list[dict[str, Any]] = []
        if top_node_ids:
            supabase = await self.get_supabase()
            source_query = supabase.table("edges").select("*").in_("source_id", top_node_ids).execute()
            target_query = supabase.table("edges").select("*").in_("target_id", top_node_ids).execute()

            s_res = await source_query if asyncio.iscoroutine(source_query) else source_query
            t_res = await target_query if asyncio.iscoroutine(target_query) else target_query

            s_data = getattr(s_res, "data", []) or []
            t_data = getattr(t_res, "data", []) or []

            seen_edges: set[str] = set()
            for edge in s_data + t_data:
                if isinstance(edge, dict):
                    edge_key = str(edge.get("id") or f"{edge.get('source_id')}->{edge.get('target_id')}")
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges_data.append(edge)

        # d) Construct undirected NetworkX Graph
        graph = nx.Graph()
        for node_id in top_node_ids:
            graph.add_node(node_id)

        for edge in edges_data:
            src = str(edge.get("source_id", ""))
            dst = str(edge.get("target_id", ""))
            rel = str(edge.get("relation_type", "relates_to"))
            if src and dst:
                graph.add_edge(src, dst, relation_type=rel)

        # e) Token Pruning via PageRank if > 50 nodes
        if graph.number_of_nodes() > 50:
            try:
                pr_scores = nx.pagerank(graph)
                top_50_nodes = [
                    node
                    for node, _ in sorted(
                        pr_scores.items(), key=lambda item: item[1], reverse=True
                    )[:50]
                ]
                graph = graph.subgraph(top_50_nodes).copy()
            except Exception as exc:
                logger.warning(
                    f"PageRank computation failed ({exc}); falling back to degree truncation.",
                    extra={"extra_payload": {"error": str(exc)}},
                )
                sorted_nodes = [
                    n for n, _ in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:50]
                ]
                graph = graph.subgraph(sorted_nodes).copy()

        # f) Build and return GraphContext
        relevant_nodes = list(graph.nodes())
        entity_relationships: dict[str, list[str]] = {}
        for u, v in graph.edges():
            entity_relationships.setdefault(str(u), []).append(str(v))
            entity_relationships.setdefault(str(v), []).append(str(u))

        return GraphContext(
            ticket_id=getattr(triage, "ticket_id", "TICK-000"),
            extracted_at=datetime.now(timezone.utc),
            query=query_str,
            relevant_nodes=relevant_nodes,
            entity_relationships=entity_relationships,
            retrieved_code_snippets=[],
            subgraph_depth=1,
        )
