"""Dual-dataset retrieval benchmark.

Benchmarks Dataset 1 and Dataset 2 in one run, then writes a single
combined finding markdown file.

Default datasets:
  - Dataset 1: benchmarks/corpus.txt
  - Dataset 2: train-00000-of-00001.parquet

Run:

    uv run python -m benchmarks.run_benchmark_alt
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts


DEFAULT_DATASET_1_PATH = Path("benchmarks/corpus.txt")
DEFAULT_DATASET_2_PATH = Path("train-00000-of-00001.parquet")
DEFAULT_FINDINGS_PATH = Path("finding.md")
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_TOP_K = 3
DEFAULT_SEARCH_DEPTH = 10

DISTANCES: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
}


@dataclass(frozen=True)
class EvalCase:
    question: str
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    corpus: str
    eval_cases: list[EvalCase]


@dataclass
class EmbeddingSpec:
    label: str
    dimension: int
    embed: Callable[[list[str], bool], list[list[float]]]


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "What share of my positive predictions were actually right?",
        ("precision measures the fraction of predicted positive",),
    ),
    EvalCase(
        "How many of the real positive cases did the model catch?",
        ("recall measures the fraction of real positive",),
    ),
    EvalCase(
        "What single score balances precision and recall?",
        ("f1 score is the harmonic mean",),
    ),
    EvalCase(
        "Which penalty zeroes out weights to create sparse models?",
        ("l1 regularization", "lasso"),
    ),
    EvalCase(
        "Which penalty shrinks weights smoothly without forcing zeros?",
        ("l2 regularization", "ridge"),
    ),
    EvalCase(
        "Why do the earliest layers of a deep net barely update?",
        ("vanishing gradient",),
    ),
    EvalCase(
        "What happens when gradients blow up and destabilize training?",
        ("exploding gradient",),
    ),
    EvalCase(
        "Which model slides filters over an image to find shapes?",
        ("convolutional neural network",),
    ),
    EvalCase(
        "Which network keeps memory across steps for text or audio?",
        ("recurrent neural network maintains",),
    ),
    EvalCase(
        "What added gates to RNNs to remember over long sequences?",
        ("long short-term memory",),
    ),
    EvalCase(
        "Which architecture dropped recurrence for attention in 2017?",
        ("transformer architecture replaced recurrence",),
    ),
    EvalCase(
        "What lets each token weigh the relevance of all others?",
        ("self-attention lets each token",),
    ),
    EvalCase(
        "What pits a generator against a discriminator?",
        ("generative adversarial",),
    ),
    EvalCase(
        "Which similarity ignores vector length and looks at angle?",
        ("cosine similarity measures the angle",),
    ),
    EvalCase(
        "Which distance is the straight-line gap between two points?",
        ("euclidean distance measures the straight-line",),
    ),
    EvalCase(
        "Which operation grows with vector magnitude?",
        ("dot product multiplies",),
    ),
    EvalCase(
        "Which learning style invents its own labels, like masked words?",
        ("self-supervised",),
    ),
    EvalCase(
        "Which method trains many models in parallel and averages them?",
        ("bagging trains many models",),
    ),
    EvalCase(
        "Which method trains models sequentially to fix prior errors?",
        ("boosting trains models sequentially",),
    ),
    EvalCase(
        "How do I reduce features by keeping directions of most variance?",
        ("principal component analysis",),
    ),
    EvalCase(
        "What groups data into k clusters around moving centroids?",
        ("k-means clustering",),
    ),
    EvalCase(
        "How can I reuse a big-dataset model for a new related task?",
        ("transfer learning",),
    ),
    EvalCase(
        "What trains an agent using rewards and penalties?",
        ("reinforcement learning trains an agent",),
    ),
    EvalCase(
        "Which optimizer uses one random example at a time?",
        ("stochastic gradient descent",),
    ),
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _load_text_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    corpus = path.read_text(encoding="utf-8").strip()
    if not corpus:
        raise ValueError(f"Corpus file is empty: {path}")
    return corpus


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


def _read_parquet_table(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pyarrow. Install with `uv add pyarrow`.") from exc
    return pq.read_table(path)


def _load_corpus_from_parquet(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Parquet corpus not found: {path}")

    table = _read_parquet_table(path)
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


def _load_eval_cases_from_parquet(path: Path) -> list[EvalCase]:
    table = _read_parquet_table(path)
    column_names = set(table.column_names)

    if "question" not in column_names:
        return list(EVAL_CASES)

    answer_columns = [
        name for name in ("ground_truth", "response", "context") if name in column_names
    ]
    if not answer_columns:
        return list(EVAL_CASES)

    rows = table.select(["question", *answer_columns]).to_pylist()
    out: list[EvalCase] = []
    seen_questions: set[str] = set()

    for row in rows:
        raw_question = row.get("question")
        if not isinstance(raw_question, str) or not raw_question.strip():
            continue

        question = raw_question.strip()
        question_key = question.lower()
        if question_key in seen_questions:
            continue

        anchors: list[str] = []
        for col in answer_columns:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                anchors.append(value.strip())

        if not anchors:
            continue

        seen_questions.add(question_key)
        out.append(EvalCase(question=question, anchors=tuple(anchors)))

    return out or list(EVAL_CASES)


def _load_dataset(path: Path, name: str) -> DatasetSpec:
    if path.suffix.lower() == ".parquet":
        corpus = _load_corpus_from_parquet(path)
        eval_cases = _load_eval_cases_from_parquet(path)
    else:
        corpus = _load_text_corpus(path)
        eval_cases = list(EVAL_CASES)
    return DatasetSpec(name=name, source=str(path), corpus=corpus, eval_cases=eval_cases)


def _chunk(corpus: str, strategy: str, chunk_size: int, overlap: int) -> list[str]:
    if strategy in {"recursive", "sentence"}:
        return chunk_text(corpus, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _build_embedding_specs() -> list[EmbeddingSpec]:
    return [
        EmbeddingSpec(
            label=settings.azure_openai_embedding_deployment,
            dimension=settings.embedding_dimension,
            embed=lambda texts, normalize: embed_texts(texts),
        )
    ]


def _recreate_collection(
    client: QdrantClient, collection: str, dimension: int, distance: Distance
) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dimension, distance=distance),
    )


def _index_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[str],
    vectors: list[list[float]],
    batch_size: int = 128,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk})
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection, points=points[start : start + batch_size])


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * 0.95)
    return ordered[index]


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _anchor_matches(text: str, anchor: str) -> bool:
    normalized_anchor = anchor.lower().strip()
    if not normalized_anchor:
        return False
    if normalized_anchor in text:
        return True

    anchor_tokens = [
        token for token in _TOKEN_PATTERN.findall(normalized_anchor) if len(token) >= 3
    ]
    if not anchor_tokens:
        return False

    text_tokens = set(_TOKEN_PATTERN.findall(text))
    overlap = sum(1 for token in set(anchor_tokens) if token in text_tokens)
    if len(anchor_tokens) <= 2:
        required = 1
    elif len(anchor_tokens) <= 6:
        required = 2
    else:
        required = 3
    return overlap >= required


def _rank_of_answer(points: list[Any], anchors: tuple[str, ...]) -> int | None:
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        text = str(payload.get("text", "")).lower()
        if any(_anchor_matches(text, anchor) for anchor in anchors):
            return rank
    return None


def _evaluate(
    client: QdrantClient,
    collection: str,
    embedding: EmbeddingSpec,
    eval_cases: list[EvalCase],
    *,
    top_k: int,
    search_depth: int,
    normalize: bool,
) -> dict[str, float]:
    if not eval_cases:
        raise ValueError("No evaluation cases available")

    hit1 = 0
    hitk = 0
    reciprocal_ranks: list[float] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []

    for case in eval_cases:
        start = time.perf_counter()
        query_vector = embedding.embed([case.question], normalize=normalize)[0]
        embed_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        response = client.query_points(collection, query=query_vector, limit=search_depth)
        search_ms.append((time.perf_counter() - start) * 1000)

        rank = _rank_of_answer(response.points, case.anchors)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= top_k:
            hitk += 1

    n = len(eval_cases)
    return {
        "questions": n,
        "hit@1": hit1 / n,
        f"hit@{top_k}": hitk / n,
        f"mrr@{search_depth}": sum(reciprocal_ranks) / n,
        "embed_ms": statistics.fmean(embed_ms),
        "search_ms": statistics.fmean(search_ms),
        "search_p95_ms": _p95(search_ms),
    }


def _benchmark_chunking_x_embedding(
    client: QdrantClient,
    dataset: DatasetSpec,
    embeddings: list[EmbeddingSpec],
    chunking_strategies: list[str],
    *,
    chunk_size: int,
    overlap: int,
    top_k: int,
    search_depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for embedding in embeddings:
        for strategy in chunking_strategies:
            chunks = _chunk(
                dataset.corpus,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            collection = (
                f"bench_alt_{_slug(dataset.name)}_{_slug(embedding.label)}_{strategy}"
            )
            _recreate_collection(client, collection, embedding.dimension, Distance.COSINE)

            index_start = time.perf_counter()
            try:
                vectors = embedding.embed(chunks, normalize=True)
                _index_chunks(client, collection, chunks, vectors)
                metrics = _evaluate(
                    client,
                    collection,
                    embedding,
                    dataset.eval_cases,
                    top_k=top_k,
                    search_depth=search_depth,
                    normalize=True,
                )
            finally:
                client.delete_collection(collection)

            rows.append(
                {
                    "dataset": dataset.name,
                    "embedding": embedding.label,
                    "chunking": strategy,
                    "chunks": len(chunks),
                    "index_ms": (time.perf_counter() - index_start) * 1000,
                    **metrics,
                }
            )

    return rows


def _benchmark_distances(
    client: QdrantClient,
    dataset: DatasetSpec,
    embedding: EmbeddingSpec,
    *,
    chunk_size: int,
    overlap: int,
    chunking_strategy: str,
    top_k: int,
    search_depth: int,
) -> list[dict[str, Any]]:
    chunks = _chunk(
        dataset.corpus,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    rows: list[dict[str, Any]] = []

    for label, distance in DISTANCES.items():
        collection = f"bench_alt_dist_{_slug(dataset.name)}_{_slug(embedding.label)}_{label}"
        _recreate_collection(client, collection, embedding.dimension, distance)

        index_start = time.perf_counter()
        try:
            vectors = embedding.embed(chunks, normalize=False)
            _index_chunks(client, collection, chunks, vectors)
            metrics = _evaluate(
                client,
                collection,
                embedding,
                dataset.eval_cases,
                top_k=top_k,
                search_depth=search_depth,
                normalize=False,
            )
        finally:
            client.delete_collection(collection)

        rows.append(
            {
                "dataset": dataset.name,
                "embedding": embedding.label,
                "chunking": chunking_strategy,
                "distance": label,
                "chunks": len(chunks),
                "index_ms": (time.perf_counter() - index_start) * 1000,
                **metrics,
            }
        )

    return rows


def _format_cell(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys())
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(_format_cell(row[h]) for h in headers) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _print_section(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    print(_to_markdown(rows))


def _parse_chunking(value: str) -> list[str]:
    allowed = {"recursive", "sentence"}
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise ValueError("No chunking strategies provided")
    invalid = [strategy for strategy in strategies if strategy not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported chunking strategy(ies): {', '.join(invalid)}. "
            "Supported: recursive,sentence"
        )
    return strategies


def _write_findings(
    path: Path,
    sections: list[tuple[DatasetSpec, list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    embedding_label: str,
    distance_chunking: str,
    top_k: int,
    search_depth: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    content: list[str] = [
        "# FINDINGS",
        "",
        f"Generated at: {timestamp}",
        f"Embedding deployment: `{embedding_label}`",
        f"Top-k: {top_k}",
        f"Search depth: {search_depth}",
        "",
    ]

    for dataset, chunk_rows, distance_rows in sections:
        content.extend(
            [
                f"## {dataset.name}",
                "",
                f"Source: `{dataset.source}`",
                f"Evaluation questions: {len(dataset.eval_cases)}",
                "",
                "### Chunking x Embedding (distance=cosine)",
                "",
                _to_markdown(chunk_rows),
                "",
                f"### Distance comparison ({embedding_label} + {distance_chunking})",
                "",
                _to_markdown(distance_rows),
                "",
            ]
        )

    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-dataset retrieval benchmark")
    parser.add_argument("--dataset-1", default=str(DEFAULT_DATASET_1_PATH))
    parser.add_argument("--dataset-2", default=str(DEFAULT_DATASET_2_PATH))
    parser.add_argument("--dataset-1-name", default="dataset_1")
    parser.add_argument("--dataset-2-name", default="dataset_2")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--search-depth", type=int, default=DEFAULT_SEARCH_DEPTH)
    parser.add_argument(
        "--chunking",
        default="recursive,sentence",
        help="Comma-separated list. Supports recursive,sentence",
    )
    parser.add_argument(
        "--distance-chunking",
        default="recursive",
        help="Chunking strategy used in distance benchmark",
    )
    parser.add_argument(
        "--findings-path",
        default=str(DEFAULT_FINDINGS_PATH),
        help="Output markdown path for combined findings",
    )
    args = parser.parse_args()

    chunking_strategies = _parse_chunking(args.chunking)
    datasets = [
        _load_dataset(Path(args.dataset_1), args.dataset_1_name),
        _load_dataset(Path(args.dataset_2), args.dataset_2_name),
    ]
    embeddings = _build_embedding_specs()
    distance_embedding = embeddings[0]

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=settings.qdrant_timeout_seconds,
    )

    sections: list[tuple[DatasetSpec, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for dataset in datasets:
        chunk_rows = _benchmark_chunking_x_embedding(
            client,
            dataset,
            embeddings,
            chunking_strategies,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            top_k=args.top_k,
            search_depth=args.search_depth,
        )
        distance_rows = _benchmark_distances(
            client,
            dataset,
            distance_embedding,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            chunking_strategy=args.distance_chunking,
            top_k=args.top_k,
            search_depth=args.search_depth,
        )
        sections.append((dataset, chunk_rows, distance_rows))

        _print_section(f"{dataset.name}: Chunking x Embedding (distance=cosine)", chunk_rows)
        _print_section(
            f"{dataset.name}: Distance comparison ({distance_embedding.label} + {args.distance_chunking})",
            distance_rows,
        )

    findings_path = Path(args.findings_path)
    _write_findings(
        findings_path,
        sections,
        embedding_label=distance_embedding.label,
        distance_chunking=args.distance_chunking,
        top_k=args.top_k,
        search_depth=args.search_depth,
    )
    print(f"\nSaved combined findings to {findings_path}")


if __name__ == "__main__":
    main()
