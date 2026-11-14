# FastAPI RAG Backend

This project is a FastAPI backend for **document ingestion + conversational RAG** with:

- PDF/TXT ingestion
- Azure OpenAI embeddings + chat
- Qdrant vector search
- Redis conversation memory
- PostgreSQL metadata + interview bookings

---

## What it does

1. **Ingest documents** (`.pdf` or `.txt`) via API.
2. Extract and chunk text (`recursive` or `sentence` strategy).
3. Generate embeddings and store vectors in Qdrant.
4. Store chunk metadata in SQL.
5. Answer user questions with a retrieval-based agent.
6. Capture interview bookings from chat into SQL.

---

## Tech stack

- **API:** FastAPI
- **LLM + embeddings:** Azure OpenAI
- **Vector DB:** Qdrant
- **Chat memory:** Redis
- **Relational DB:** PostgreSQL + SQLAlchemy
- **Orchestration:** LangChain + LangGraph

---

## Quick start

### 1) Prerequisites

- Python **3.14+**
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose

### 2) Configure environment

```bash
cp .env.example .env
```

Then fill your Azure OpenAI values in `.env`.

### 3) Start dependencies

```bash
docker compose up -d qdrant redis postgres
```

### 4) Install dependencies

```bash
uv sync
```

### 5) Run the API

```bash
uv run uvicorn main:app --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `APP_NAME` | Yes | — | FastAPI app title |
| `QDRANT_URL` | Yes | — | Qdrant URL |
| `QDRANT_COLLECTION` | No | `documents` | Qdrant collection name |
| `QDRANT_TIMEOUT_SECONDS` | No | `10` | Qdrant client timeout |
| `QDRANT_API_KEY` | No | empty | Qdrant auth key |
| `EMBEDDING_DIMENSION` | No | `1536` | Expected embedding vector size |
| `REDIS_URL` | Yes | — | Redis connection string |
| `DATABASE_URL` | Yes | — | SQLAlchemy DB URL |
| `SESSION_TTL_SECONDS` | Yes | — | Redis chat history TTL |
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Base endpoint for embeddings (and chat fallback) |
| `AZURE_OPENAI_API_KEY` | Yes | — | API key for embeddings (and chat fallback) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | — | Embedding deployment name |
| `AZURE_OPENAI_CHAT_ENDPOINT` | No | falls back to `AZURE_OPENAI_ENDPOINT` | Optional dedicated chat endpoint |
| `AZURE_OPENAI_CHAT_API_KEY` | No | falls back to `AZURE_OPENAI_API_KEY` | Optional dedicated chat key |
| `AZURE_OPENAI_CHAT_API_VERSION` | No | empty | API version for legacy `*.openai.azure.com` endpoints |
| `OPENAI_API_VERSION` | No | empty | Shared fallback API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | No | `gpt-4o` | Chat deployment name |
| `RAG_TOP_K` | No | `5` | Number of retrieved chunks |
| `CHAT_TEMPERATURE` | No | `0.2` | Chat model temperature |

> For `*.services.ai.azure.com` Foundry-style endpoints, keep API version fields empty.

---

## API reference

### `GET /health`

Returns service health:

```json
{"status":"ok"}
```

### `GET /v1/models`

OpenAI-style compatibility route listing configured chat and embedding deployments.

### `POST /api/v1/ingest`

Multipart form:

- `file`: `.pdf` or `.txt`
- `chunking_strategy`: `recursive` or `sentence` (default: `recursive`)

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@/absolute/path/to/file.pdf" \
  -F "chunking_strategy=recursive"
```

Sample response:

```json
{
  "file_name": "file.pdf",
  "chunking_strategy": "recursive",
  "embedding_model": "text-embedding-ada-002",
  "chunks_ingested": 24
}
```

### `POST /api/v1/chat`

Request:

```json
{
  "user_id": "user-1",
  "query": "Summarize the eligibility criteria"
}
```

Response:

```json
{
  "user_id": "user-1",
  "response": "..."
}
```

Use the same `user_id` across requests to preserve multi-turn memory in Redis.

---

## Interview booking flow

The agent has a booking tool that writes to SQL when it has all required fields:

- full name
- email
- date (`YYYY-MM-DD`)
- time (`HH:MM`)

On success, the assistant returns a booking confirmation with a booking reference.

---

## Data model (SQL)

- `chunk_metadata`
  - `file_name`, `chunk_index`, `chunking_strategy`, `embedding_model`, `vector_id`, `text`
- `bookings`
  - `full_name`, `email`, `date`, `time`

Tables are auto-created at startup.

---

## Benchmarking

Run retrieval benchmarks:

```bash
uv run python -m benchmarks.run_benchmark_alt --findings-path benchmarks/findings.md
```

This generates markdown benchmark output in `benchmarks/findings.md`.

---

## Project structure

```text
.
├── app/
│   ├── api/            # FastAPI routes
│   ├── core/           # config + clients
│   ├── db/             # SQLAlchemy models/session
│   ├── repositories/   # DB persistence helpers
│   ├── schemas/        # request/response schemas
│   └── services/       # ingestion, embeddings, vector search, memory, agent
├── benchmarks/         # retrieval benchmark scripts and findings
├── docker-compose.yml  # qdrant, redis, postgres
└── main.py             # FastAPI app entrypoint
```

---

## Common issues

- **400 Unsupported file type** → only `.pdf` and `.txt` are accepted.
- **Qdrant dimension mismatch** → set `EMBEDDING_DIMENSION` to match your embedding deployment.
- **Empty/weak answers** → ingest documents first and verify `RAG_TOP_K`.
- **Booking rejected** → ensure full name, valid email, date format `YYYY-MM-DD`, time format `HH:MM`.
