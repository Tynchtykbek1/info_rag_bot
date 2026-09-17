# Messina Info

Tools for ingesting a Telegram JSON export into a normalized local dataset.

The repository intentionally excludes raw Telegram exports, secrets, databases,
and downloaded model or embedding caches. Tests use only synthetic fixtures.

## Usage

```python
from messina_info import load_telegram_export

export = load_telegram_export("path/to/result.json")
for message in export.messages:
    print(message.date, message.text)
```

Telegram rich-text fragments are flattened into plain text. Service records are
ignored, while malformed message records produce a `TelegramExportError`.

## Multilingual retrieval

Install the project with the retrieval extra to add sentence-transformers and
its runtime dependencies alongside the core NumPy dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,retrieval]"
```

The default embedding model is `intfloat/multilingual-e5-small`. Override it
with `--model` on every command, or set `MESSINA_EMBEDDING_MODEL` when using the
Python API. E5 inputs are encoded with `query: ` and `passage: ` prefixes and
are normalized before cosine search.

Build an index from the read-only message corpus:

```powershell
.\.venv\Scripts\python.exe -m messina_info.cli build `
  --database data/messina.db `
  --index .retrieval-index
```

Run one search:

```powershell
.\.venv\Scripts\python.exe -m messina_info.cli search `
  --index .retrieval-index `
  --query "When is the ERSU scholarship deadline?" `
  -k 5
```

Evaluate the checked-in multilingual cases:

```powershell
.\.venv\Scripts\python.exe -m messina_info.cli evaluate `
  --index .retrieval-index `
  --dataset eval/retrieval_cases.jsonl
```

The generated directory contains normalized vectors in `vectors.npz`, document
metadata, and a manifest with model, dimension, corpus fingerprint, build time,
and format version. Generated indexes are ignored by Git.

Exact NumPy cosine search is intentional: the current corpus is only about 210
language sections, so a linear scan is simple, deterministic, and avoids a
separate vector service. `RetrievalDocument` and the embedding provider protocol
keep storage and encoding concerns separate, allowing the exact backend to be
replaced later by pgvector or Qdrant without changing ingestion or segmentation.

Evaluation reports Hit@1, Hit@3, Hit@5, and MRR@5 for answerable cases. It
reports unanswerable-query scores separately; an optional `--min-score` measures
rejection for an externally selected threshold. A threshold measured on the same
evaluation set is not production-ready, and neither is this baseline retrieval
pipeline without further held-out evaluation and operational hardening.
