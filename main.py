from fastapi import FastAPI

from app.api import chat_router, ingestion_router
from app.core.config import settings
from app.db.session import init_db
from app.services.vectorstore import QdrantStore

app = FastAPI(title=settings.app_name)
app.include_router(ingestion_router)
app.include_router(chat_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    QdrantStore().ensure_collection()


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
