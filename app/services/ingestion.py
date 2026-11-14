from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.chunk_metadata import save_chunk_metadata
from app.schemas.ingestion import IngestionResult
from app.services.chunking import CHUNKING_STRATEGIES, chunk_text
from app.services.embeddings import embed_texts
from app.services.extraction import ToMarkdown
from app.services.vectorstore import QdrantStore

_TMP_DIR = Path("/tmp/opencode")


def _read_upload_bytes(file: UploadFile) -> tuple[str, bytes]:
    file_name = file.filename or "uploaded_document"
    content = file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty")
    return file_name, content


def _extract_text(file_name: str, content: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("Document contains no text")
        return text

    if suffix == ".pdf":
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = _TMP_DIR / f"{uuid.uuid4()}.pdf"
        temp_path.write_bytes(content)
        try:
            return ToMarkdown(temp_path).pdf2md()
        finally:
            temp_path.unlink(missing_ok=True)

    raise ValueError("Unsupported file type. Only .pdf and .txt are allowed")


def ingest_document(
    db: Session, file: UploadFile, chunking_strategy: str = "recursive"
) -> IngestionResult:
    if chunking_strategy not in CHUNKING_STRATEGIES:
        allowed = ", ".join(CHUNKING_STRATEGIES)
        raise ValueError(f"Unknown chunking strategy '{chunking_strategy}'. Use: {allowed}")

    file_name, content = _read_upload_bytes(file)
    text = _extract_text(file_name, content)
    chunks = chunk_text(text, strategy=chunking_strategy)
    if not chunks:
        raise ValueError("No chunks were generated from the document")

    vectors = embed_texts(chunks)
    store = QdrantStore()
    store.ensure_collection()

    doc_id = str(uuid.uuid4())
    embedding_model = settings.azure_openai_embedding_deployment
    vector_ids = store.upsert_chunks(
        doc_id=doc_id,
        chunks=chunks,
        vectors=vectors,
        extra_payload={
            "file_name": file_name,
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
        },
    )

    try:
        save_chunk_metadata(
            db,
            file_name=file_name,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_ids=vector_ids,
            chunks=chunks,
        )
    except Exception:
        store.delete_by_doc_id(doc_id)
        raise

    return IngestionResult(
        file_name=file_name,
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        chunks_ingested=len(chunks),
    )
