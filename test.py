from app.core.client import qdrant_client
from app.core.config import settings
from app.services.vectorstore import QdrantStore


def main() -> None:
    s = settings
    store = QdrantStore(qdrant_client, s.qdrant_collection, s.embedding_dimension)
    store.ensure_collection()

    vec = [0.1] * s.embedding_dimension
    store.upsert_chunks("doc-test", ["hello world"], [vec], {"filename": "t.txt"})
    print(store.search(vec, top_k=1))
    print(store.search(vec, top_k=1, doc_id="nope"))  # expect []
    store.delete_by_doc_id("doc-test")
    print(store.search(vec, top_k=1))  # expect []


if __name__ == "__main__":
    main()
