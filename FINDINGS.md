# FINDINGS

Generated at: 2026-11-14T16:40:29.705845+00:00
Corpus source: `train-00000-of-00001.parquet`
Embedding deployment: `text-embedding-ada-002`

## Chunking x Embedding (distance=cosine)

| embedding | chunking | chunks | hit@1 | hit@3 | mrr | embed_ms | search_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text-embedding-ada-002 | recursive | 150 | 1.0000 | 1.0000 | 1.0000 | 346.8624 | 5.7961 |
| text-embedding-ada-002 | sentence | 103 | 1.0000 | 1.0000 | 1.0000 | 347.2134 | 5.7796 |

## Similarity algorithm (Azure OpenAI + recursive)

| distance | hit@1 | hit@3 | mrr | embed_ms | search_ms |
| --- | --- | --- | --- | --- | --- |
| cosine | 1.0000 | 1.0000 | 1.0000 | 339.7547 | 4.2428 |
| dot | 1.0000 | 1.0000 | 1.0000 | 374.0716 | 4.2307 |
| euclid | 1.0000 | 1.0000 | 1.0000 | 409.3496 | 6.0507 |
