"""Unit tests for Phase 5: Graph RAG Walker Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.graph_walker import GraphWalkerAgent
from state import GraphContext, VisualTriageResult


@pytest.fixture
def sample_triage_result() -> VisualTriageResult:
    """Fixture providing a standard VisualTriageResult instance."""
    return VisualTriageResult(
        ticket_id="TICK-8001",
        processed_at=datetime.now(timezone.utc),
        screenshot_urls=["https://storage.example.com/checkout_bug.png"],
        ui_elements_detected=["CheckoutButton", "PaymentErrorBanner"],
        anomaly_detected=True,
        triage_summary="Stripe payment gateway timeout exception.",
        visual_confidence_score=0.96,
    )


# ==========================================
# Test 1: Graph Construction & PageRank Pruning
# ==========================================

@pytest.mark.asyncio
async def test_build_context_construction_and_pruning(sample_triage_result: VisualTriageResult) -> None:
    """Test 1: Assert Pinecone 5 seed nodes + Supabase 100 edges (>50 nodes) prunes to exactly 50 nodes."""
    # 1. Mock GenAI Embedding
    mock_genai = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 768
    mock_emb_resp = MagicMock()
    mock_emb_resp.embeddings = [mock_emb]
    mock_genai.aio = MagicMock()
    mock_genai.aio.models = MagicMock()
    mock_genai.aio.models.embed_content = AsyncMock(return_value=mock_emb_resp)

    # 2. Mock Pinecone returning 5 seed node IDs
    mock_pinecone_index = MagicMock()
    seed_nodes = [f"seed_node_{i}" for i in range(5)]
    mock_matches = [{"id": node_id} for node_id in seed_nodes]
    mock_pinecone_index.query.return_value = {"matches": mock_matches}

    # 3. Mock Supabase returning 100 edges linking to 80 unique target nodes (>50 total nodes)
    mock_supabase = MagicMock()
    mock_edges = []
    for i in range(100):
        src = seed_nodes[i % 5]
        dst = f"expanded_node_{i % 80}"
        mock_edges.append({
            "id": f"edge_{i}",
            "source_id": src,
            "target_id": dst,
            "relation_type": "calls",
        })

    mock_source_table = MagicMock()
    mock_source_in = MagicMock()
    mock_source_in.execute = AsyncMock(return_value=MagicMock(data=mock_edges[:50]))
    mock_source_table.select.return_value.in_.return_value = mock_source_in

    mock_target_table = MagicMock()
    mock_target_in = MagicMock()
    mock_target_in.execute = AsyncMock(return_value=MagicMock(data=mock_edges[50:]))
    mock_target_table.select.return_value.in_.return_value = mock_target_in

    # Alternate source_query and target_query executions
    mock_table = MagicMock()
    mock_in_clause = MagicMock()
    mock_in_clause.execute = AsyncMock(side_effect=[
        MagicMock(data=mock_edges[:50]),
        MagicMock(data=mock_edges[50:]),
    ])
    mock_table.select.return_value.in_.return_value = mock_in_clause
    mock_supabase.table.return_value = mock_table

    agent = GraphWalkerAgent(
        genai_client=mock_genai,
        pinecone_index=mock_pinecone_index,
        supabase_client=mock_supabase,
        embedding_model="gemini-embedding-001",
    )

    context = await agent.build_context(sample_triage_result)

    # Assert validated GraphContext return
    assert isinstance(context, GraphContext)
    assert context.ticket_id == "TICK-8001"

    # Assert exactly 50 nodes kept due to PageRank pruning
    assert len(context.relevant_nodes) == 50
    assert len(context.entity_relationships) > 0


# ==========================================
# Test 2: Empty Retrieval Graceful Handling
# ==========================================

@pytest.mark.asyncio
async def test_build_context_empty_retrieval(sample_triage_result: VisualTriageResult) -> None:
    """Test 2: Assert agent gracefully handles empty Pinecone / Supabase responses without crashing."""
    mock_genai = MagicMock()
    mock_genai.aio = MagicMock()
    mock_genai.aio.models = MagicMock()
    mock_genai.aio.models.embed_content = AsyncMock(return_value=MagicMock(embeddings=[]))

    # Empty Pinecone query result
    mock_pinecone_index = MagicMock()
    mock_pinecone_index.query.return_value = {"matches": []}

    mock_supabase = MagicMock()

    agent = GraphWalkerAgent(
        genai_client=mock_genai,
        pinecone_index=mock_pinecone_index,
        supabase_client=mock_supabase,
    )

    context = await agent.build_context(sample_triage_result)

    assert isinstance(context, GraphContext)
    assert context.ticket_id == "TICK-8001"
    assert context.relevant_nodes == []
    assert context.entity_relationships == {}
