"""Benchmark chunking strategies and Qdrant distances with Azure OpenAI embeddings.

Generates retrieval accuracy (hit@k) and latency numbers used in findings.md.
Run with the project venv while Qdrant is up:

    .venv/bin/python -m benchmarks.run_benchmark
"""

from __future__ import annotations

import argparse
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts


DEFAULT_PARQUET_CORPUS = Path("train-00000-of-00001.parquet")
DEFAULT_FINDINGS_PATH = Path("findings.md")

QDRANT = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key.get_secret_value()
    if settings.qdrant_api_key
    else None,
    timeout=settings.qdrant_timeout_seconds,
)
TOP_K = 3
SEARCH_DEPTH = 10  # how deep to look when computing MRR
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

EMBEDDING_LABEL = settings.azure_openai_embedding_deployment
EMBEDDING_DIM = settings.embedding_dimension
CHUNKING_STRATEGIES = ("recursive", "sentence")
DISTANCES = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}

# Default ground-truth set: question -> anchor keyword(s) unique to the
# correct fact. Used when parquet does not provide a question/answer schema.
DEFAULT_QUESTIONS = [
    (
        "What share of my positive predictions were actually right?",
        ["precision measures the fraction of predicted positive"],
    ),
    (
        "How many of the real positive cases did the model catch?",
        ["recall measures the fraction of real positive"],
    ),
    (
        "What single score balances precision and recall?",
        ["f1 score is the harmonic mean"],
    ),
    (
        "Which penalty zeroes out weights to create sparse models?",
        ["l1 regularization", "lasso"],
    ),
    (
        "Which penalty shrinks weights smoothly without forcing zeros?",
        ["l2 regularization", "ridge"],
    ),
    (
        "Why do the earliest layers of a deep net barely update?",
        ["vanishing gradient"],
    ),
    (
        "What happens when gradients blow up and destabilize training?",
        ["exploding gradient"],
    ),
    (
        "Which model slides filters over an image to find shapes?",
        ["convolutional neural network"],
    ),
    (
        "Which network keeps memory across steps for text or audio?",
        ["recurrent neural network maintains"],
    ),
    (
        "What added gates to RNNs to remember over long sequences?",
        ["long short-term memory"],
    ),
    (
        "Which architecture dropped recurrence for attention in 2017?",
        ["transformer architecture replaced recurrence"],
    ),
    (
        "What lets each token weigh the relevance of all others?",
        ["self-attention lets each token"],
    ),
    (
        "What pits a generator against a discriminator?",
        ["generative adversarial"],
    ),
    (
        "Which similarity ignores vector length and looks at angle?",
        ["cosine similarity measures the angle"],
    ),
    (
        "Which distance is the straight-line gap between two points?",
        ["euclidean distance measures the straight-line"],
    ),
    (
        "Which operation grows with vector magnitude?",
        ["dot product multiplies"],
    ),
    (
        "Which learning style invents its own labels, like masked words?",
        ["self-supervised"],
    ),
    (
        "Which method trains many models in parallel and averages them?",
        ["bagging trains many models"],
    ),
    (
        "Which method trains models sequentially to fix prior errors?",
        ["boosting trains models sequentially"],
    ),
    (
        "How do I reduce features by keeping directions of most variance?",
        ["principal component analysis"],
    ),
    (
        "What groups data into k clusters around moving centroids?",
        ["k-means clustering"],
    ),
    (
        "How can I reuse a big-dataset model for a new related task?",
        ["transfer learning"],
    ),
    (
        "What trains an agent using rewards and penalties?",
        ["reinforcement learning trains an agent"],
    ),
    (
        "Which optimizer uses one random example at a time?",
        ["stochastic gradient descent"],
    ),
]

def _candidate_text_column(columns: list[str]) -> str | None:
    preferred = [
        "text",
        "content",
        "document",
        "context",
        "chunk",
        "body",
        "passage",
        "notes",
        "sentence",
        "ground_truth",
        "response",
        "question",
    ]
    for name in preferred:
        if name in columns:
            return name
    return None


def _best_text_column(table: Any) -> str | None:
    """Pick the most content-rich string column by average text length."""
    best_name: str | None = None
    best_avg_len = 0.0

    for name in table.column_names:
        values = table[name].to_pylist()
        texts = [value.strip() for value in values[:200] if isinstance(value, str) and value.strip()]
        if not texts:
            continue

        avg_len = sum(len(text) for text in texts) / len(texts)
        if avg_len > best_avg_len:
            best_avg_len = avg_len
            best_name = name

    return best_name


def _load_corpus_from_parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pyarrow. Install with `uv add pyarrow`."
        ) from exc

    table = pq.read_table(path)
    if table.num_rows == 0:
        raise ValueError(f"Parquet file has no rows: {path}")

    column = _candidate_text_column(list(table.column_names))
    if column is None:
        column = _best_text_column(table)

    if column is None:
        raise ValueError(
            f"No text-like column found in {path}. Columns: {', '.join(table.column_names)}"
        )

    values = table[column].to_pylist()
    lines = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not lines:
        raise ValueError(f"No non-empty text found in '{column}' from {path}")
    return "\n\n".join(lines)


def _load_corpus(parquet_path: Path) -> tuple[str, str]:
    if parquet_path.exists():
        return _load_corpus_from_parquet(parquet_path), str(parquet_path)
    raise FileNotFoundError(f"Parquet corpus not found: {parquet_path}")


def _load_eval_questions_from_parquet(path: Path) -> list[tuple[str, list[str]]]:
    """Build eval questions from parquet rows when possible.

    Expected columns (best effort):
    - question
    - one or more of: ground_truth, response, context
    """

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pyarrow. Install with `uv add pyarrow`."
        ) from exc

    table = pq.read_table(path)
    column_names = set(table.column_names)
    if "question" not in column_names:
        return DEFAULT_QUESTIONS

    answer_columns = [
        name for name in ("ground_truth", "response", "context") if name in column_names
    ]
    if not answer_columns:
        return DEFAULT_QUESTIONS

    rows = table.select(["question", *answer_columns]).to_pylist()
    out: list[tuple[str, list[str]]] = []
    seen_questions: set[str] = set()

    for row in rows:
        raw_question = row.get("question")
        if not isinstance(raw_question, str) or not raw_question.strip():
            continue

        question = raw_question.strip()
        q_key = question.lower()
        if q_key in seen_questions:
            continue

        anchors: list[str] = []
        for col in answer_columns:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                anchors.append(value.strip())

        if not anchors:
            continue

        seen_questions.add(q_key)
        out.append((question, anchors))

    return out or DEFAULT_QUESTIONS


def _embed(texts: list[str]) -> list[list[float]]:
    return embed_texts(texts)


def _recreate(collection: str, dim: int, distance: Distance) -> None:
    if QDRANT.collection_exists(collection):
        QDRANT.delete_collection(collection)
    QDRANT.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=distance),
    )


def _index(
    collection: str,
    chunks: list[str],
) -> None:
    vectors = _embed(chunks)
    QDRANT.upsert(
        collection,
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk})
            for vector, chunk in zip(vectors, chunks)
        ],
    )


def _rank_of_answer(points: list[Any], keywords: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing an anchor keyword."""

    token_pattern = re.compile(r"[a-z0-9]+")

    def tokenize(text: str) -> list[str]:
        return token_pattern.findall(text.lower())

    def anchor_matches(text: str, anchor: str) -> bool:
        normalized_anchor = anchor.lower().strip()
        if not normalized_anchor:
            return False

        if normalized_anchor in text:
            return True

        anchor_tokens = [token for token in tokenize(normalized_anchor) if len(token) >= 3]
        if not anchor_tokens:
            return False

        text_tokens = set(tokenize(text))
        overlap = sum(1 for token in set(anchor_tokens) if token in text_tokens)
        if len(anchor_tokens) <= 2:
            required = 1
        elif len(anchor_tokens) <= 6:
            required = 2
        else:
            required = 3
        return overlap >= required

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        text = str(payload.get("text", "")).lower()
        if any(anchor_matches(text, keyword) for keyword in keywords):
            return rank
    return None


def _evaluate(collection: str, questions: list[tuple[str, list[str]]]) -> dict[str, float]:
    """Return hit@1, hit@3, MRR and average embed/search latency in ms."""
    hit1 = hit3 = 0
    reciprocal_ranks: list[float] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []

    if not questions:
        raise ValueError("No evaluation questions available")

    for question, keywords in questions:
        start = time.perf_counter()
        query_vector = _embed([question])[0]
        embed_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        result = QDRANT.query_points(collection, query=query_vector, limit=SEARCH_DEPTH)
        search_ms.append((time.perf_counter() - start) * 1000)

        rank = _rank_of_answer(result.points, keywords)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= TOP_K:
            hit3 += 1

    n = len(questions)
    return {
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "embed_ms": sum(embed_ms) / n,
        "search_ms": sum(search_ms) / n,
    }


def report_chunking_and_embeddings(
    corpus: str, questions: list[tuple[str, list[str]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in CHUNKING_STRATEGIES:
        chunks = chunk_text(
            corpus,
            strategy=strategy,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        collection = f"bench_openai_{strategy}"
        _recreate(collection, EMBEDDING_DIM, Distance.COSINE)
        _index(collection, chunks)
        metrics = _evaluate(collection, questions)
        QDRANT.delete_collection(collection)
        rows.append(
            {
                "embedding": EMBEDDING_LABEL,
                "chunking": strategy,
                "chunks": len(chunks),
                **metrics,
            }
        )
    return rows


def report_distances(
    corpus: str, questions: list[tuple[str, list[str]]]
) -> list[dict[str, Any]]:
    """Compare Qdrant distance metrics with Azure OpenAI embeddings."""
    chunks = chunk_text(
        corpus,
        strategy="recursive",
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    rows: list[dict[str, Any]] = []
    for label, distance in DISTANCES.items():
        collection = f"bench_dist_{label}"
        _recreate(collection, EMBEDDING_DIM, distance)
        _index(collection, chunks)
        metrics = _evaluate(collection, questions)
        QDRANT.delete_collection(collection)
        rows.append({"distance": label, **metrics})
    return rows


def _format_cell(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(_format_cell(row[h]) for h in headers) + " |" for row in rows
    ]
    return "\n".join([header_row, sep_row, *body])


def _print(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(_format_cell(row[h]) for h in headers))


def _write_findings(
    path: Path,
    corpus_source: str,
    chunking_rows: list[dict[str, Any]],
    distance_rows: list[dict[str, Any]],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    content = [
        "# FINDINGS",
        "",
        f"Generated at: {timestamp}",
        f"Corpus source: `{corpus_source}`",
        f"Embedding deployment: `{EMBEDDING_LABEL}`",
        "",
        "## Chunking x Embedding (distance=cosine)",
        "",
        _to_markdown(chunking_rows),
        "",
        "## Similarity algorithm (Azure OpenAI + recursive)",
        "",
        _to_markdown(distance_rows),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark matrix")
    parser.add_argument(
        "--parquet-corpus",
        default=str(DEFAULT_PARQUET_CORPUS),
        help="Path to parquet corpus file",
    )
    parser.add_argument(
        "--findings-path",
        default=str(DEFAULT_FINDINGS_PATH),
        help="Where to write findings markdown",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet_corpus)
    corpus, source = _load_corpus(parquet_path)
    questions = _load_eval_questions_from_parquet(parquet_path)

    chunking_rows = report_chunking_and_embeddings(corpus, questions)
    distance_rows = report_distances(corpus, questions)

    _print("Chunking x Embedding (distance=cosine)", chunking_rows)
    _print("Similarity algorithm (Azure OpenAI + recursive)", distance_rows)
    _write_findings(Path(args.findings_path), source, chunking_rows, distance_rows)

    print(f"\nSaved results to {args.findings_path}")


if __name__ == "__main__":
    main()
