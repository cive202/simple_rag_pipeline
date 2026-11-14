from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ingestion import IngestionResult
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
def ingest(
    file: UploadFile = File(...),
    chunking_strategy: Literal["recursive", "sentence"] = Form("recursive"),
    db: Session = Depends(get_db),
) -> IngestionResult:
    try:
        return ingest_document(db=db, file=file, chunking_strategy=chunking_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
