# Benchmark Results

Corpus: `benchmarks/corpus.txt`

## Chunking x Embedding (distance=cosine)

| embedding | chunking | chunks | index_ms | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-embedding-ada-002 | recursive | 23 | 14339.8924 | 0.9167 | 1.0000 | 0.9583 | 474.9325 | 6.5666 | 8.4847 |
| text-embedding-ada-002 | sentence | 21 | 14647.9025 | 0.9167 | 1.0000 | 0.9583 | 533.6710 | 5.9396 | 7.0675 |

## Distance comparison (text-embedding-ada-002 + recursive)

| embedding | chunking | distance | chunks | index_ms | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-embedding-ada-002 | recursive | cosine | 23 | 16527.4200 | 0.9167 | 1.0000 | 0.9583 | 428.0244 | 6.8970 | 9.3656 |
| text-embedding-ada-002 | recursive | dot | 23 | 13407.9230 | 0.9167 | 1.0000 | 0.9583 | 409.3514 | 6.4899 | 9.3810 |
| text-embedding-ada-002 | recursive | euclid | 23 | 10286.4957 | 0.9167 | 1.0000 | 0.9583 | 363.8415 | 5.9141 | 7.5631 |
