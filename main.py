"""Main execution entry point for Redmine AI Bridge (System Failure Context Bridge)."""

from __future__ import annotations

import asyncio
import os
import sys
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from supabase import create_async_client

from agents.graph_walker import GraphWalkerAgent
from agents.patch_engineer import PatchEngineerAgent
from agents.security_gate import SecurityGateAgent
from agents.visual_triage import VisualTriageAgent
from interceptor.redmine_client import RedmineClient
from pipeline.orchestrator import AIBridgeOrchestrator
from sandbox.ruby_validator import RubySandboxValidator

load_dotenv()


async def run_pipeline(target_issue_id: int) -> bool:
    """Initializes external dependencies, instantiates the multi-agent graph, and processes a target Redmine issue."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "redmine-code-graph")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    redmine_url = os.environ.get("REDMINE_URL", "https://redmine.example.com")
    redmine_api_key = os.environ.get("REDMINE_API_KEY", "mock_redmine_key")

    if not all([gemini_key, pinecone_key, supabase_url, supabase_key]):
        print("Error: Missing credentials in .env (GEMINI_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY)")
        return False

    print("=" * 60)
    print(f"🚀 Initializing Redmine AI Bridge for Ticket #{target_issue_id}...")
    print("=" * 60)

    # 1. Initialize API and DB clients
    genai_client = genai.Client(api_key=gemini_key).aio
    pc = Pinecone(api_key=pinecone_key)
    pinecone_index = pc.Index(pinecone_index_name)
    supabase_client = await create_async_client(supabase_url, supabase_key)
    redmine_client = RedmineClient(base_url=redmine_url, api_key=redmine_api_key)

    # 2. Instantiate agents & sandbox validator
    visual_triage = VisualTriageAgent(client=genai_client)
    graph_walker = GraphWalkerAgent(
        genai_client=genai_client,
        pinecone_index=pinecone_index,
        supabase_client=supabase_client,
    )
    patch_engineer = PatchEngineerAgent(client=genai_client)
    ruby_validator = RubySandboxValidator()
    security_gate = SecurityGateAgent(client=genai_client)

    # 3. Assemble Orchestrator
    orchestrator = AIBridgeOrchestrator(
        redmine_client=redmine_client,
        visual_triage=visual_triage,
        graph_walker=graph_walker,
        patch_engineer=patch_engineer,
        ruby_validator=ruby_validator,
        security_gate=security_gate,
    )

    # 4. Execute pipeline
    print(f"Running automated diagnosis & remediation on Redmine Ticket #{target_issue_id}...")
    success = await orchestrator.process_ticket(target_issue_id)

    if success:
        print(f"✅ Pipeline finished successfully! Diagnostics and verified patch posted to Redmine issue #{target_issue_id}.")
    else:
        print(f"❌ Pipeline halted or encountered an error processing issue #{target_issue_id}. Check telemetry logs.")

    return success


if __name__ == "__main__":
    issue_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(run_pipeline(issue_id))
