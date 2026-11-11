# real-estate-rag-evals

[![CI](https://github.com/mustafaautomation/real-estate-rag-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaautomation/real-estate-rag-evals/actions/workflows/ci.yml)

Personal open-source demo by Muhammad Mustafa. It is a small RAG assistant over a made-up brokerage, Harbor & Pine Realty, plus an evaluation gate built on RAGAS 0.4.3.

The listings, lease template, building rules, and questions are synthetic. They were written for this repo. There is no customer data and no real agency.

## What it does

A question is checked against two guardrails before retrieval:

- Steering by a protected characteristic (race, color, religion, sex, national origin, disability, familial status) is refused.
- Legal advice on a lease dispute is refused. The assistant may quote the lease template.

If the question uses a word that is not in the corpus, or the retrieved text does not contain the address it asked about, the answer is "I don't know." Dollar amounts in an answer have to appear in the retrieved text.

Otherwise the pipeline chunks the corpus, embeds it, retrieves the nearest chunks, and generates an answer from those chunks.

## Architecture

```mermaid
flowchart LR
  Q[Question] --> G[Guardrails]
  G -->|refuse or unknown| A[Response]
  G -->|in corpus| R[Retriever]
  C[Synthetic corpus] --> CH[Chunker]
  CH --> E[Hashed TF-IDF embeddings]
  E --> F[FAISS or pgvector]
  F --> R
  R --> P[Provider]
  P --> A
  A --> EV[Eval]
  EV --> OFF[Offline metrics and gate]
  EV --> LIVE[RAGAS LLM metrics]
```

Offline retrieval uses a deterministic hashed TF-IDF embedder so tests do not download a model. The same vectors can be stored in FAISS or in Postgres with pgvector. Generation goes through one interface with three implementations: an extractive offline provider, OpenAI, and Anthropic.

The offline provider quotes sentences from the retrieved chunk. It is the provider used by CI. It is not a chat model.

## Run offline

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run harbor-eval --mode offline --check-snapshot
uv run harbor-ask "What is the monthly rent at 18 Lark Street?"
```

`harbor-eval` prints the offline metrics and exits non-zero if any of them is below the threshold in `evals/config.yaml`. `--check-snapshot` also fails if the rerun does not match `data/snapshots/offline_responses.jsonl`.

That snapshot is pipeline output from the extractive provider. `judge_scores` is null. It is not a RAGAS judge run. See `data/snapshots/snapshot_meta.json`.

## Run live RAGAS metrics

Live mode calls OpenAI as the judge for faithfulness, answer relevancy, context precision, and context recall, using the RAGAS 0.4 collections API (`Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`). It runs only on rows whose expected behavior is `answer`. Refusal rows are scored by the offline refusal check instead, because a refusal is not a claim in a listing.

```bash
export OPENAI_API_KEY=...
uv run harbor-eval --mode live
```

If `OPENAI_API_KEY` is missing, the command exits with status 2 and does not write judge scores. This repo does not contain judge scores. None were produced.

There is a `workflow_dispatch` workflow, `.github/workflows/live-eval.yml`. The live step runs only when the `OPENAI_API_KEY` repository secret is set.

## Postgres

```bash
docker compose up -d
export HARBOR_PG_DSN=postgresql://postgres:postgres@localhost:5432/harbor
uv run harbor-ask "What is the late fee in the lease template?" --backend pgvector
```

The compose file starts `pgvector/pgvector:pg16` with the local user and password `postgres`. That password is for a laptop database only. Tests use FAISS and do not need Docker.

## Metrics

Offline metrics are the CI gate. They do not call a model.

| Metric | What it measures | Threshold |
| --- | ---: | ---: |
| hit_rate | Share of answerable questions whose reference document is in the top 4 chunks | 0.95 |
| mrr | Mean reciprocal rank of that document | 0.90 |
| non_llm_context_recall | RAGAS `NonLLMContextRecall`: fraction of reference chunks with string similarity above 0.5 to some retrieved chunk | 0.95 |
| string_presence | RAGAS `StringPresence`: the labelled phrase appears in the response | 0.95 |
| refusal_correctness | Response class matches the label: answer, fair-housing refusal, legal refusal, or unknown | 0.95 |
| grounded_amounts | Every dollar amount in the response also appears in the retrieved text | 1.00 |

For a brokerage, hit rate and context recall measure whether the listing or lease clause was retrieved. String presence and grounded amounts measure whether the reply stuck to that text instead of stating a rent, fee, or date that is not in the file. Refusal correctness measures whether steering questions and dispute questions were refused.

In ragas 0.4.3, `NonLLMContextRecall` is still imported from `ragas.metrics`. The collections package exports `StringPresence` and the four LLM metrics, and it does not export `NonLLMContextRecall` even though the deprecation warning names that path.

Live thresholds, used only when a key is present:

| Metric | Threshold |
| --- | ---: |
| faithfulness | 0.70 |
| answer_relevancy | 0.70 |
| context_precision | 0.70 |
| context_recall | 0.70 |

## This run

Offline eval on the synthetic set (49 questions, 22 documents), extractive provider, hashed TF-IDF, FAISS, k=4. Baseline, the bad retriever, and the no-guardrail variant were run in the same pass.

| Metric | Bad retriever | Baseline | Threshold |
| --- | ---: | ---: | ---: |
| hit_rate | 0.000 | 1.000 | 0.950 |
| mrr | 0.000 | 1.000 | 0.900 |
| non_llm_context_recall | 0.000 | 1.000 | 0.950 |
| string_presence | 0.245 | 1.000 | 0.950 |
| refusal_correctness | 0.245 | 1.000 | 0.950 |
| grounded_amounts | 1.000 | 1.000 | 1.000 |

The bad retriever returns the least similar chunks. Answerable questions then fall through to "I don't know", so it does not invent a dollar amount (`grounded_amounts` stays 1.000) and still fails the gate on retrieval and on answer match. The no-guardrail variant keeps retrieval and drops the refusal: refusal correctness 0.857 and string presence 0.857, both under 0.95.

pytest: 38 passed.

![Offline eval report](docs/eval_report.png)

Full write-up: [docs/eval_report.md](docs/eval_report.md) and [docs/eval_report.html](docs/eval_report.html).

## Broken variants

`harbor-eval --mode offline --variant bad_retriever` and `--variant no_guardrail` are expected to fail the gate. CI runs the baseline variant only. Unit tests run both broken variants and assert that the gate fails.

## Adapting it to another corpus

1. Replace `data/corpus/*.md` with your own documents. Keep one H1 per file. The file stem is the document id.
2. Replace `data/eval/questions.json`. Each row needs an id, question, reference, a phrase that should appear in a correct answer, reference document ids, and `expected_behavior` (`answer`, `refuse_fair_housing`, `refuse_legal`, or `unknown`).
3. Change the guardrail patterns in `src/harbor_rag/guardrails.py` if the refusal rules are different.
4. Run `uv run harbor-eval --mode offline --write-snapshot --write-report docs` and set thresholds in `evals/config.yaml` from that run, above the broken variant and at or below the baseline you accept.
5. For a hosted model, set `OPENAI_API_KEY` and run `--mode live`, or pass `--provider openai` / `--provider anthropic` to `harbor-ask`.

## CI

`.github/workflows/ci.yml` runs lint, format check, mypy, pytest, and the offline eval gate. All four jobs are required. None use `continue-on-error`.
