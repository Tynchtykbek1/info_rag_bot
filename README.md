# Messina Info

Messina Info is a multilingual retrieval-augmented assistant for University of
Messina students. It ingests a local Telegram export, segments posts by
language, retrieves relevant RU/EN/IT sections, and can produce grounded Gemini
answers whose citations are checked against the retrieved context.

The repository intentionally excludes raw Telegram exports, secrets, databases,
and downloaded model or embedding caches. Tests use only synthetic fixtures.

## Installation

Python 3.10 or newer is required. Create a virtual environment and install the
features you need:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,retrieval,reranking,rag]"
```

For Gemini answers, copy the safe template and set the key only in the ignored
local file. Existing process environment variables take precedence.

```powershell
Copy-Item .env.example .env
# Edit .env locally; never commit it.
```

Retrieval-only commands do not require a Gemini key.

## Ingestion API

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

On the checked-in 30-case RU/EN/IT evaluation set, dense top-20 retrieval
reached Hit@1 28/30, Hit@3 30/30, Hit@5 30/30, and deduplicated MRR@5 0.9667.
These are development-set retrieval metrics, not production guarantees.

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

## Grounded single-turn answers

Install the optional RAG dependencies with `pip install -e ".[rag]"`. Set
`GEMINI_API_KEY` in the environment and optionally select a model with
`MESSINA_GEMINI_MODEL` (the default is `gemini-2.5-flash`). Secrets and prompts
are not logged.

```powershell
.\.venv\Scripts\python.exe -m messina_info.cli ask `
  --index .retrieval-index `
  --language en `
  --mode fast `
  --query "When is the scholarship deadline?"

.\.venv\Scripts\python.exe -m messina_info.cli ask `
  --index .retrieval-index `
  --language it `
  --mode quality `
  --query "Quando scade la domanda?"
```

The pipeline retrieves Telegram sections, deduplicates messages, builds a
bounded context whose content is explicitly marked as untrusted, requests a
Pydantic-validated JSON response from Gemini, and validates every citation
against the supplied context. Source links always come from index metadata.
Invalid output, unavailable APIs, missing evidence, or empty retrieval produce
a localized fallback. `fast` uses dense retrieval only and never loads ONNX;
`quality` opts into the existing CPU ONNX reranker. Conversation history can be
stored in SQLite and used for deterministic dual retrieval. There is still no
calibrated confidence score or production retrieval threshold.

## Local Telegram bot

Create a bot token with [@BotFather](https://t.me/BotFather), install all local
features, and copy the environment template:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,retrieval,reranking,rag,bot]"
Copy-Item .env.example .env
```

Set `TELEGRAM_BOT_TOKEN` and `GEMINI_API_KEY` only in the ignored `.env` file.
The retrieval index configured by `MESSINA_INDEX_PATH` must already be built.
Start long polling locally with:

```powershell
.\.venv\Scripts\messina-info-bot.exe
```

The first model load may be slow. This MVP supports private chats only; use
`/language ru|en|it` to select a language and `/reset` to clear conversation
history. Answers include Telegram source links but are not guaranteed to be
perfect. Typical end-to-end latency is several seconds or more, and Gemini may
occasionally return transient HTTP 429/503 errors.

Hybrid routing answers common greetings locally and normalizes frequent student
slang. Clear UniME/ERSU questions go straight to grounded RAG; uncertain messages
use one structured Gemini interpretation call before retrieval. Unrelated or
unclear requests receive short replies without source links.

## Conversation routing baseline

The synthetic RU/EN/IT cases in `eval/conversation_routing_cases.jsonl` use an
independent action taxonomy. Local evaluation makes no network calls and reports
uncertain cases as `DEFERRED` rather than guessing an action:

```powershell
python -m messina_info.routing_evaluation --dataset eval/conversation_routing_cases.jsonl --mode local --report .eval-results/local-routing-report.json
```

An explicit live run loads the local `.env` using the existing precedence rules
(process variables win). Use `--dotenv PATH` to select another file. It checks
for `GEMINI_API_KEY` before processing cases and reports only the resolved model
name. The run calls the existing Gemini interpreter only for deferred cases.
It never invokes RAG or Telegram. Use small resumable batches; completed
cases in the JSONL records file are skipped on the next run, while provider
errors can be retried:

```powershell
python -m messina_info.routing_evaluation --dataset eval/conversation_routing_cases.jsonl --mode live --deferred-only --offset 0 --limit 5 --delay-seconds 2 --run-id hybrid-router-4a9169e --records .eval-results/baseline-hybrid-4a9169e-v2-records.jsonl --report .eval-results/baseline-hybrid-4a9169e-v2-report.json
```

`--run-id` identifies the evaluated system version. Every live record also
contains the resolved model and a fingerprint of the complete annotated case.
Resume stops if any of these values differ or if a legacy record lacks them.

Live interpretation adds up to one Gemini request per deferred case and its
latency and API cost; local fast-path cases require none. The live runner disables
provider retries so its reported provider-call count matches actual requests.
Raw records and batch reports in `.eval-results/` are ignored by Git. A stable
final baseline report may be committed separately after all batches are reviewed.

## Tests and repository contents

Run the complete offline suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The public repository contains source code, offline tests with synthetic
fixtures, and the multilingual retrieval evaluation queries. It does not
contain the private/raw Telegram export, normalized message database, generated
retrieval index, downloaded embedding or reranker models, Hugging Face cache,
local `.env`, or API credentials. Those artifacts remain local and are covered
by `.gitignore`.

## Known limitations

This project is not production-ready. Live Gemini calls have shown transient
HTTP 503/504 provider failures. In the latest six-case smoke test, end-to-end
latency was approximately 16–32 seconds per query. Two scholarship questions
were answered and grounded correctly, while the English answerable scholarship
case incorrectly returned `insufficient_evidence` on retry. The system also has
no calibrated confidence score, production retrieval threshold, or service-level
availability guarantees.
