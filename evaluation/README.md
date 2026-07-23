# Offline evaluation

This package mines high-confidence duplicate candidates, hydrates every issue needed
by an approved dataset, and measures how well the embedding retriever ranks the
canonical issue.

## 1. Import enough historical issues

Import a recent sample so the collector knows which issues to inspect:

```bash
issue-triage import-repo flutter/flutter --limit 1000
```

## 2. Collect candidates

```bash
python -m evaluation.collect_duplicates flutter/flutter --max-comments 10000
```

The collector scans every imported issue body locally, downloads repository comments
in bulk, and extracts explicit phrases such as `duplicate of #123`. Candidates are
written to `evaluation/datasets/candidates.json`.

Duplicate labels increase candidate confidence but are not required. `--max-comments`
limits API work; comments are processed newest-first. A GitHub token is strongly
recommended.

For repositories with a reliable duplicate label, use label-driven collection. This
does not require a repository import first and avoids missing old closing comments:

```bash
python -m evaluation.collect_duplicates flutter/flutter \
  --label "r: duplicate" \
  --target-pairs 300 \
  --sample-seed 42
```

Label mode paginates up to `--max-labeled-issues` matching closed issues, takes a
reproducible random sample across their history, and scans each selected issue's body
and complete comment history. It records the comment author's repository association
so evidence from owners, members, and collaborators can be reviewed first. Confidence
is a review priority only; candidates are never approved automatically.

Re-run with a different seed to expand the candidate set. Existing review decisions
are preserved when the same pair is rediscovered.

## 3. Review the candidates

Open `evaluation/datasets/candidates.json`, follow each evidence URL, and change:

```json
"review_status": "pending"
```

to either:

```json
"review_status": "approved"
```

or:

```json
"review_status": "rejected"
```

Do not approve a pair simply because the regex found it. Confirm that the newer issue
really describes the same underlying problem as the canonical issue.

## 4. Build a fixed evaluation corpus

Build a deterministic corpus containing every approved pair endpoint plus a uniform
sample of historical negatives:

```bash
python -m evaluation.build_corpus flutter/flutter \
  --sample-size 10000 \
  --seed 42
```

The builder deterministically samples issue numbers across the repository's history
and downloads them in GraphQL batches. Pull request numbers are skipped. This avoids
GitHub's deep REST-pagination limit while spreading negatives across the period before
the latest evaluated query. It writes `evaluation/datasets/corpus_manifest.json` with
the dataset hash, sampling parameters, model, and exact issue numbers. Commit that
manifest when publishing results.

This step replaces the local repository store. A GitHub token is required for a
large repository because building the sample may consume many paginated requests.

`hydrate_candidates` remains useful for quickly debugging the pipeline, but a corpus
containing only pair endpoints must not be used for reported retrieval metrics.

## 5. Run evaluation

```bash
python -m evaluation.evaluate flutter/flutter
```

Evaluation skips queries with fewer than 100 historical candidates by default. This
prevents early queries in a sampled corpus from receiving artificially easy ranks.
Change the safeguard explicitly with `--min-candidates`.

The evaluator excludes the query issue itself and issues created after the query. It
reports Recall@1, Recall@5, Recall@10, mean reciprocal rank, and mean, median, and
p95 query latency.

Compare dense retrieval with reciprocal rank fusion (RRF), which combines dense and
TF-IDF rank positions without assuming their raw scores are calibrated:

```bash
python -m evaluation.evaluate flutter/flutter \
  --method dense --method rrf --rrf-k 60 \
  --experiment-name dense-vs-rrf
```

Evaluate a two-stage pipeline that retrieves dense candidates and reranks them with
the local `cross-encoder/ms-marco-MiniLM-L6-v2` model:

```bash
python -m evaluation.evaluate flutter/flutter \
  --method dense --method rerank --rerank-top-n 50 \
  --experiment-name dense-vs-rerank
```

The first reranker run downloads its model. Set `RERANKER_MODEL` in `.env` to test a
different sentence-transformers cross-encoder. A target outside the dense top-N is
correctly counted as not retrieved by the reranking pipeline.

## Dataset fields

| Field | Meaning |
| --- | --- |
| `query_issue` | Duplicate issue used as the search query |
| `duplicate_issue` | Canonical issue expected in the ranking |
| `source` | Whether evidence came from the body or a comment |
| `evidence` | Short text that produced the candidate |
| `confidence` | Collection heuristic, not model confidence |
| `actor_association` | GitHub relationship such as `MEMBER` or `COLLABORATOR` |
| `review_status` | Manual ground-truth decision |
| `query_available` | Whether the query was stored when collected |
| `target_available` | Whether the canonical issue was stored when collected |

## Limitations

- Repositories using only formal duplicate timeline events with no explanatory text
  may be missed. GraphQL timeline collection is a later extension.
- Cross-repository duplicate references are ignored.
- The regex intentionally ignores vague phrases such as `related to #123`.
- Re-running collection refreshes the evidence while preserving existing review
  decisions for pairs that are rediscovered.
- A duplicate label identifies the duplicate query, not necessarily its canonical
  target. Every extracted pair still requires review.
- A sampled corpus measures retrieval against that bounded corpus, not against every
  issue ever created. Record the manifest and corpus size with published metrics.
