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

## Optional quality reranking

Dense retrieval remains the default `fast` mode and does not import or load an
ONNX reranker. Install the separate optional dependency to enable the two-stage
`quality` mode:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,retrieval,reranking]"
```

Quality mode retrieves 15 dense candidates, scores the raw query and raw section
text with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, and then deduplicates by
Telegram message. The default ONNX file is the model's O3-optimized
`onnx/model_O3.onnx`, executed with `CPUExecutionProvider`. Reranker scores are
ordering signals, not probabilities or confidence values.

```powershell
# Default low-latency dense search: no reranker is loaded.
.\.venv\Scripts\python.exe -m messina_info.cli search `
  --index .retrieval-index `
  --query "Quando scade la domanda ERSU?" `
  --mode fast

# Optional quality mode.
.\.venv\Scripts\python.exe -m messina_info.cli search `
  --index .retrieval-index `
  --query "Quando scade la domanda ERSU?" `
  --mode quality `
  --candidate-k 15 `
  --reranker-batch-size 8

# Evaluate the same quality mode.
.\.venv\Scripts\python.exe -m messina_info.cli evaluate `
  --index .retrieval-index `
  --dataset eval/retrieval_cases.jsonl `
  --mode quality
```

On the development CPU, O3 reranking at candidate-k 15 and batch size 8 kept
Hit@1 at 28/30 and Hit@5 at 30/30, with median latency 2.31 seconds and p95
latency 2.64 seconds. This was materially faster than the 4.16-second PyTorch
median but did not meet the experimental 1.5-second target, so quality mode is
optional and is not claimed to be production-ready. Dynamic INT8 reduced the
weights from about 471 MB to 119 MB but was slower than O3 on this CPU.

An equal-weight BM25+dense RRF experiment was rejected because it reduced
overall retrieval quality, especially for Italian cross-lingual queries. It is
not part of the implementation.

# Grounded single-turn answers

Install the optional RAG dependencies with `pip install -e ".[rag]"`. Set
`GEMINI_API_KEY` in the environment and optionally select a model with
`MESSINA_GEMINI_MODEL` (the default is `gemini-2.5-flash`). Secrets and prompts
are not logged.

```powershell
messina-info ask --index data/retrieval-index --language en --mode fast --query "When is the scholarship deadline?"
messina-info ask --index data/retrieval-index --language it --mode quality --query "Quando scade la domanda?"
```

The pipeline retrieves Telegram sections, deduplicates messages, builds a
bounded context whose content is explicitly marked as untrusted, requests a
Pydantic-validated JSON response from Gemini, and validates every citation
against the supplied context. Source links always come from index metadata.
Invalid output, unavailable APIs, missing evidence, or empty retrieval produce
a localized fallback. `fast` uses dense retrieval only and never loads ONNX;
`quality` opts into the existing CPU ONNX reranker. This is single-turn RAG:
there is no conversation memory, query rewriting, calibrated confidence score,
or production retrieval threshold.
