"""Unit tests for Phase 2: Vectorization & Relational Dual-Write Ingestor."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from graph_builder.ingestor import GraphIngestor


@pytest.fixture
def mock_ingestor() -> GraphIngestor:
    """Fixture providing a GraphIngestor with mocked clients."""
    mock_genai = MagicMock()
    mock_pinecone = MagicMock()
    mock_supabase = MagicMock()

    ingestor = GraphIngestor(
        google_api_key="mock_google_key",
        pinecone_api_key="mock_pinecone_key",
        pinecone_index_name="mock_index",
        supabase_url="https://mock.supabase.co",
        supabase_key="mock_key",
        genai_client=mock_genai,
        pinecone_client=mock_pinecone,
        supabase_client=mock_supabase,
    )
    return ingestor


# ==========================================
# Test 1: Batching Logic
# ==========================================

@pytest.mark.asyncio
async def test_batch_generator_and_ingest_counts(mock_ingestor: GraphIngestor) -> None:
    """Test 1: Assert 250 entities are split into batches of 100, 100, 50 and process_batch is called 3 times."""
    entities_list = [f"Class_{i}" for i in range(250)]

    # 1. Direct assertion on _batch_generator chunking
    batches = list(GraphIngestor._batch_generator(entities_list, batch_size=100))
    assert len(batches) == 3
    assert len(batches[0]) == 100
    assert len(batches[1]) == 100
    assert len(batches[2]) == 50

    # 2. Assert ingest_entities calls process_batch exactly 3 times
    extracted_entities = {
        "classes": entities_list,
        "methods": [],
        "requires": [],
    }

    with patch.object(
        mock_ingestor,
        "process_batch",
        new_callable=AsyncMock,
        return_value={"embedded_nodes": 100, "upserted_vectors": 100, "inserted_edges": 0},
    ) as mock_process:
        result = await mock_ingestor.ingest_entities(
            extracted_entities,
            batch_size=100,
            max_concurrency=5,
        )

        assert mock_process.call_count == 3
        assert result["total_batches"] == 3


# ==========================================
# Test 2: Database Mocks & Dual-Write Assertions
# ==========================================

@pytest.mark.asyncio
async def test_database_mocks_dual_write() -> None:
    """Test 2: Assert Pinecone upsert and Supabase insert.execute are called with correct schemas."""
    # Mock Google GenAI embed response
    mock_genai = MagicMock()
    mock_emb_1 = MagicMock()
    mock_emb_1.values = [0.1, 0.2, 0.3]
    mock_emb_response = MagicMock()
    mock_emb_response.embeddings = [mock_emb_1]

    mock_genai.aio = MagicMock()
    mock_genai.aio.models = MagicMock()
    mock_genai.aio.models.embed_content = AsyncMock(return_value=mock_emb_response)

    # Mock Pinecone
    mock_pinecone = MagicMock()
    mock_index = MagicMock()
    mock_pinecone.Index.return_value = mock_index

    # Mock Supabase
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_insert.execute = AsyncMock(return_value=MagicMock(data=[{"id": "edge-1"}]))
    mock_table.insert.return_value = mock_insert
    mock_supabase.table.return_value = mock_table

    ingestor = GraphIngestor(
        google_api_key="mock_key",
        pinecone_api_key="mock_key",
        pinecone_index_name="test-index",
        supabase_url="https://test.supabase.co",
        supabase_key="mock_key",
        genai_client=mock_genai,
        pinecone_client=mock_pinecone,
        supabase_client=mock_supabase,
    )

    nodes = [
        {"id": "node-class-1", "name": "PaymentService", "type": "class", "content": "class PaymentService"}
    ]
    edges = [
        {
            "id": "edge-1",
            "source_id": "node-class-1",
            "target_id": "node-method-1",
            "relation_type": "defines",
        }
    ]

    result = await ingestor.process_batch(nodes, edges)

    # Assert embeddings and Pinecone upsert
    mock_genai.aio.models.embed_content.assert_awaited_once()
    mock_index.upsert.assert_called_once()
    upserted_kwargs = mock_index.upsert.call_args[1]
    assert "vectors" in upserted_kwargs
    assert len(upserted_kwargs["vectors"]) == 1
    assert upserted_kwargs["vectors"][0]["id"] == "node-class-1"
    assert upserted_kwargs["vectors"][0]["metadata"]["name"] == "PaymentService"
    assert upserted_kwargs["vectors"][0]["values"] == [0.1, 0.2, 0.3]

    # Assert Supabase insert.execute chain
    mock_supabase.table.assert_called_once_with("edges")
    mock_table.insert.assert_called_once()
    inserted_records = mock_table.insert.call_args[0][0]
    assert len(inserted_records) == 1
    assert inserted_records[0]["source_id"] == "node-class-1"
    assert inserted_records[0]["target_id"] == "node-method-1"
    assert inserted_records[0]["relation_type"] == "defines"
    mock_insert.execute.assert_awaited_once()

    assert result["embedded_nodes"] == 1
    assert result["upserted_vectors"] == 1
    assert result["inserted_edges"] == 1


# ==========================================
# Test 3: Concurrency Throttling with Semaphore
# ==========================================

@pytest.mark.asyncio
async def test_concurrency_throttling(mock_ingestor: GraphIngestor) -> None:
    """Test 3: Assert asyncio.Semaphore limits maximum concurrent batch executions."""
    active_concurrency = 0
    max_observed_concurrency = 0
    lock = asyncio.Lock()

    async def mock_slow_process(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
        nonlocal active_concurrency, max_observed_concurrency
        async with lock:
            active_concurrency += 1
            if active_concurrency > max_observed_concurrency:
                max_observed_concurrency = active_concurrency

        # Simulate network latency
        await asyncio.sleep(0.05)

        async with lock:
            active_concurrency -= 1

        return {"embedded_nodes": len(nodes), "upserted_vectors": len(nodes), "inserted_edges": len(edges)}

    mock_nodes = [{"id": f"node_{i}", "name": f"Node_{i}"} for i in range(100)]
    extracted_data = {"nodes": mock_nodes, "edges": []}

    with patch.object(mock_ingestor, "process_batch", side_effect=mock_slow_process):
        # 10 batches of 10 items, semaphore concurrency limit of 3
        result = await mock_ingestor.ingest_entities(
            extracted_data,
            batch_size=10,
            max_concurrency=3,
        )

        assert result["total_batches"] == 10
        assert max_observed_concurrency <= 3
        assert max_observed_concurrency > 0
