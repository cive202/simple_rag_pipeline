# palm-mind

Backend for:

- Document ingestion API (`.pdf` / `.txt`)
- Conversational RAG API with Redis chat memory
- Interview booking capture in SQL

## Stack

- FastAPI
- Qdrant (vector DB)
- Redis (chat memory)
- PostgreSQL + SQLAlchemy (metadata + booking)
- Azure OpenAI (embeddings + chat)

## Environment

Create `.env`:

```env
APP_NAME=rag_pipeline
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
QDRANT_TIMEOUT_SECONDS=10
EMBEDDING_DIMENSION=1536
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://app:app@localhost:5432/appdb
SESSION_TTL_SECONDS=86400
AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_ENDPOINT=https://<your-chat-resource>.services.ai.azure.com
AZURE_OPENAI_CHAT_API_KEY=<your-chat-key>
# Leave API version empty for Foundry services.ai.azure.com endpoints.
# Set these only for legacy *.openai.azure.com endpoints.
AZURE_OPENAI_CHAT_API_VERSION=
OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
RAG_TOP_K=5
CHAT_TEMPERATURE=0.2
```

## Run

```bash
docker compose up -d qdrant redis postgres
uv sync
uv run uvicorn main:app --reload
```

## APIs

### 1) Document Ingestion

`POST /api/v1/ingest` (multipart form)

- `file`: `.pdf` or `.txt`
- `chunking_strategy`: `recursive` or `sentence`

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@/path/to/file.pdf" \
  -F "chunking_strategy=recursive"
```

### 2) Conversational RAG

`POST /api/v1/chat`

```json
{
  "user_id": "user-1",
  "query": "Summarize the eligibility criteria"
}
```

The API keeps multi-turn memory in Redis and can capture interview bookings (name, email, date, time) into SQL.
