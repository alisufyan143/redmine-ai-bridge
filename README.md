# Redmine AI Bridge (System Failure Context Bridge)

An automated, multi-agent AI pipeline that intercepts Redmine bug tickets, diagnoses UI failures using multimodal reasoning, explores Ruby on Rails codebases via Graph RAG, and produces verified, secure patches in an isolated sandbox.

---

## 🚀 How It Works

```
[ Redmine Ticket & Screenshot ]
             │
             ▼
   1. Redmine Interceptor ──── (Async client with 429 exponential backoff)
             │
             ▼
   2. Visual Triage Agent ──── (Multimodal Gemini 3.5 Flash UI diagnosis)
             │
             ▼
   3. Graph RAG Walker    ──── (Pinecone vectors + Supabase edges + PageRank pruning)
             │
             ▼
   4. Patch Engineer      ──── (Synthesizes minimal unified diff git patches)
             │
             ▼
   5. Docker Sandbox      ──── (Isolated syntax validation in ruby:3.2-alpine)
             │
             ▼
   6. Security Gate Agent ──── (Audits for SQLi, command injection, path traversal)
             │
             ▼
[ Automated Markdown Diagnostic Report Posted to Redmine ]
```

---

## 🧩 Key Architecture

### 1. State Management & Immutability (`state.py`)
- Pydantic v2 ledger enforcing strict schemas, `frozen=True` immutability, and zero extra fields (`extra='forbid'`).
- Hard limits on payload size (10,000 chars) and strict UTC-aware timestamp enforcement to prevent memory and time-drift bugs.

### 2. Telemetry & Circuit Breakers (`utils/telemetry.py`)
- Structured JSON logging for clean observability and aggregation.
- Asynchronous `@circuit_breaker` decorators wrapping external network boundaries (Gemini, Pinecone, Supabase, Redmine) to isolate failures without crashing the event loop.

### 3. Ruby AST Parsing & Graph Ingestion (`graph_builder/`)
- **AST Parser (`ast_parser.py`)**: Uses `tree-sitter` and `tree-sitter-ruby` to extract classes, methods, and `require` dependencies with recursion depth limits and syntax error recovery.
- **Dual-Write Ingestor (`ingestor.py`)**: Batch-embeds AST nodes with Google GenAI (`gemini-embedding-001`), upserts vectors to Pinecone, and persists relational edges to Supabase under concurrency limits (`asyncio.Semaphore`).

### 4. Redmine API Interceptor (`interceptor/`)
- Non-blocking `aiohttp` client for issue metadata and attachment downloads.
- Automatic retry with exponential backoff on HTTP 429 rate limits.
- Strict security guardrails on attachments (5MB max size and `image/png`, `image/jpeg` MIME enforcement).

### 5. Multi-Agent Reasoning Core (`agents/`)
- **Visual Triage Agent (`visual_triage.py`)**: Diagnoses screenshot artifacts and issue text into a structured `VisualTriageResult`.
- **Graph RAG Walker (`graph_walker.py`)**: Queries Pinecone seeds, expands relational graphs from Supabase, and uses `networkx.pagerank` to keep only the 50 most relevant nodes.
- **Patch Engineer (`patch_engineer.py`)**: Generates production-ready unified diffs, conventional commit messages, and target branches.
- **Security Gate (`security_gate.py`)**: Audits proposed diffs against OWASP Top 10 vulnerabilities (ActiveRecord SQL injection, command execution via backticks/`system`, unsafe parameter mass assignment).

### 6. Isolated Sandbox Validator (`sandbox/`)
- Runs `ruby -c` syntax checks inside resource-constrained Docker containers (`ruby:3.2-alpine`, 128MB RAM, 0.5 CPU, `--network none`) with strict execution timeouts.

### 7. End-to-End Orchestrator (`pipeline/`)
- `AIBridgeOrchestrator` wires the entire workflow together and posts comprehensive diagnostic markdown reports directly back to the originating Redmine issue.

---

## 🛠️ Tech Stack

- **Language & Runtime**: Python 3.12, Asyncio, aiohttp
- **AI & Reasoning**: Google GenAI SDK (`gemini-3.5-flash`, `gemini-embedding-001`)
- **Vector & Graph Storage**: Pinecone, Supabase (Async Client), NetworkX
- **Code Parsing & Isolation**: Tree-sitter (Ruby), Docker (`ruby:3.2-alpine`)
- **Validation & State**: Pydantic v2, Pytest, Pytest-Asyncio, aioresponses

---

## ⚙️ Quickstart

### 1. Prerequisites
- Python 3.10+ (or Conda / uv)
- Docker Desktop (running)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/alisufyan143/redmine-ai-bridge.git
cd redmine-ai-bridge

# Install dependencies using uv or pip
uv pip install -r requirements.txt
# Or install directly:
uv pip install pydantic pytest pytest-asyncio tree-sitter tree-sitter-ruby google-genai pinecone supabase aiohttp aiofiles aioresponses networkx
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
# Google Gemini API
GEMINI_API_KEY="your_gemini_api_key"

# Pinecone Vector Database
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX_NAME="redmine-code-graph"

# Supabase Database
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your_supabase_key"

# Redmine Instance (Optional for live ticket polling)
REDMINE_URL="https://redmine.yourdomain.com"
REDMINE_API_KEY="your_redmine_api_key"
```

### 4. Pull the Sandbox Docker Image
```powershell
docker pull ruby:3.2-alpine
```

### 5. Run the Test Suite
```powershell
pytest -v tests/
```
All 41 unit and integration tests covering AST parsing, dual-write batching, client backoff, agents, sandbox execution, and the orchestrator should pass in under 7 seconds.

---

## 🧪 Test Coverage Summary

| Test Module | Coverage Area |
| :--- | :--- |
| `tests/test_phase1.py` | Pydantic immutable state, size limits, UTC timezones, and circuit breaker logging |
| `tests/test_ast_parser.py` | Ruby AST entity extraction, syntax recovery, and recursion depth limits |
| `tests/test_ingestor.py` | Batch chunking (100/100/50), database mocks, and semaphore concurrency limits |
| `tests/test_redmine_client.py` | Async issue fetch, 429 exponential backoff, MIME checks, and 5MB attachment limits |
| `tests/test_visual_triage.py` | Multimodal screenshot diagnosis and structured schema parsing |
| `tests/test_graph_walker.py` | Semantic retrieval, Supabase edge expansion, and PageRank subgraph pruning |
| `tests/test_patch_engineer.py` | Multimodal context compilation and unified diff patch synthesis |
| `tests/test_sandbox.py` | Isolated Docker syntax execution, error capturing, and container kill timeouts |
| `tests/test_security_gate.py` | OWASP security audit validation (SQL injection detection, clean patch approval) |
| `tests/test_orchestrator.py` | Full multi-agent chain execution and graceful failure handling |

---

## 📄 License
MIT
