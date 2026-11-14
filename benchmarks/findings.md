# FINDINGS

Generated at: 2026-11-14T17:03:07.867672+00:00
Embedding deployment: `text-embedding-ada-002`
Top-k: 3
Search depth: 10

## dataset_1

Source: `benchmarks/corpus.txt`
Evaluation questions: 24

### Chunking x Embedding (distance=cosine)

| dataset | embedding | chunking | chunks | index_ms | questions | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | text-embedding-ada-002 | recursive | 23 | 13212.4262 | 24 | 0.9167 | 1.0000 | 0.9583 | 368.7527 | 6.1363 | 8.3432 |
| dataset_1 | text-embedding-ada-002 | sentence | 21 | 10021.2814 | 24 | 0.9167 | 1.0000 | 0.9583 | 336.6480 | 5.9699 | 7.0790 |

### Distance comparison (text-embedding-ada-002 + recursive)

| dataset | embedding | chunking | distance | chunks | index_ms | questions | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_1 | text-embedding-ada-002 | recursive | cosine | 23 | 9992.4590 | 24 | 0.9167 | 1.0000 | 0.9583 | 343.5837 | 6.2824 | 7.1475 |
| dataset_1 | text-embedding-ada-002 | recursive | dot | 23 | 18217.8182 | 24 | 0.9167 | 1.0000 | 0.9583 | 395.1220 | 6.4039 | 7.8963 |
| dataset_1 | text-embedding-ada-002 | recursive | euclid | 23 | 13705.1898 | 24 | 0.9167 | 1.0000 | 0.9583 | 435.6334 | 5.8119 | 6.7721 |

## dataset_2

Source: `train-00000-of-00001.parquet`
Evaluation questions: 60

### Chunking x Embedding (distance=cosine)

| dataset | embedding | chunking | chunks | index_ms | questions | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_2 | text-embedding-ada-002 | recursive | 150 | 32826.0985 | 60 | 1.0000 | 1.0000 | 1.0000 | 383.3649 | 6.0710 | 8.5300 |
| dataset_2 | text-embedding-ada-002 | sentence | 103 | 30901.6647 | 60 | 1.0000 | 1.0000 | 1.0000 | 403.6261 | 6.6410 | 10.5807 |

### Distance comparison (text-embedding-ada-002 + recursive)

| dataset | embedding | chunking | distance | chunks | index_ms | questions | hit@1 | hit@3 | mrr@10 | embed_ms | search_ms | search_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_2 | text-embedding-ada-002 | recursive | cosine | 150 | 48192.2910 | 60 | 1.0000 | 1.0000 | 1.0000 | 630.1095 | 6.7105 | 8.9257 |
| dataset_2 | text-embedding-ada-002 | recursive | dot | 150 | 36702.3955 | 60 | 1.0000 | 1.0000 | 1.0000 | 402.6688 | 6.0650 | 7.8919 |
| dataset_2 | text-embedding-ada-002 | recursive | euclid | 150 | 35275.7632 | 60 | 1.0000 | 1.0000 | 1.0000 | 433.9391 | 6.3300 | 7.8344 |
