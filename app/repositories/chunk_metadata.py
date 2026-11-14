from sqlalchemy.orm import Session

from app.db.models import ChunkMetadata


def save_chunk_metadata(
    db: Session,
    *,
    file_name: str,
    chunking_strategy: str,
    embedding_model: str,
    vector_ids: list[str],
    chunks: list[str],
) -> list[ChunkMetadata]:
    if len(vector_ids) != len(chunks):
        raise ValueError("vector_ids and chunks must have the same length")

    records: list[ChunkMetadata] = []
    for chunk_index, (vector_id, text) in enumerate(zip(vector_ids, chunks)):
        record = ChunkMetadata(
            file_name=file_name,
            chunk_index=chunk_index,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_id=vector_id,
            text=text,
        )
        db.add(record)
        records.append(record)

    db.commit()
    for record in records:
        db.refresh(record)
    return records
