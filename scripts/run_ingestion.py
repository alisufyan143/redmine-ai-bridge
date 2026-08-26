"""Codebase AST extraction and dual-write ingestion script for Redmine AI Bridge."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from supabase import create_async_client

from graph_builder.ast_parser import RubyASTExtractor
from graph_builder.ingestor import GraphIngestor

load_dotenv()


async def main() -> None:
    repo_path = Path("target_repo")
    if not repo_path.exists():
        print("Error: target_repo directory not found. Clone redmine first via: git clone --depth 1 https://github.com/redmine/redmine.git target_repo")
        return

    gemini_key = os.environ.get("GEMINI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "redmine-code-graph")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not all([gemini_key, pinecone_key, supabase_url, supabase_key]):
        print("Error: Missing required environment variables in .env (GEMINI_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY)")
        return

    print("Initializing clients for Graph Ingestion...")
    genai_client = genai.Client(api_key=gemini_key).aio
    pc = Pinecone(api_key=pinecone_key)
    pinecone_index = pc.Index(pinecone_index_name)
    supabase_client = await create_async_client(supabase_url, supabase_key)

    extractor = RubyASTExtractor()
    ingestor = GraphIngestor(
        genai_client=genai_client,
        pinecone_client=pc,
        supabase_client=supabase_client,
        pinecone_index_name=pinecone_index_name,
    )

    ruby_files = list(repo_path.rglob("*.rb"))
    print(f"Found {len(ruby_files)} Ruby files to parse and ingest in '{repo_path}'.")

    total_embedded = 0
    total_edges = 0

    for i, file_path in enumerate(ruby_files, 1):
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
            entities = extractor.extract_entities(code)

            # Attach file path context
            entities["file_path"] = str(file_path.relative_to(repo_path))

            stats = await ingestor.ingest_entities(entities)
            embedded = stats.get("total_embedded_nodes", 0)
            edges = stats.get("total_inserted_edges", 0)
            total_embedded += embedded
            total_edges += edges
            print(f"[{i}/{len(ruby_files)}] Ingested {file_path.name} (+{embedded} nodes, +{edges} edges)")
        except Exception as exc:
            print(f"[{i}/{len(ruby_files)}] Skipping {file_path.name} due to error: {exc}")

    print(f"\nIngestion completed successfully! Total nodes: {total_embedded}, Total edges: {total_edges}")


if __name__ == "__main__":
    asyncio.run(main())
