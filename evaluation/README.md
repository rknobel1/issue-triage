# Offline evaluation

This package mines high-confidence duplicate candidates and measures how well the
embedding retriever ranks the canonical issue.

## 1. Import enough historical issues

The collector only accepts references whose target is present locally. Import more
than the initial smoke-test dataset:

```bash
issue-triage import-repo fastapi/fastapi --limit 1000
```

## 2. Collect candidates

```bash
python -m evaluation.collect_duplicates fastapi/fastapi
```

The collector filters imported issues for labels containing `duplicate`, downloads
comments for those issues, and extracts explicit phrases such as `duplicate of #123`.
Candidates are written to `evaluation/datasets/candidates.json`.

Only duplicate-labeled issues are inspected, which avoids making one comments request
for every imported issue. A GitHub token is strongly recommended.

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

## 4. Run evaluation

```bash
python -m evaluation.evaluate fastapi/fastapi
```

The evaluator excludes the query issue itself and issues created after the query. It
reports Recall@1, Recall@5, mean reciprocal rank, and mean query latency.

## Dataset fields

| Field | Meaning |
| --- | --- |
| `query_issue` | Duplicate issue used as the search query |
| `duplicate_issue` | Canonical issue expected in the ranking |
| `source` | Whether evidence came from the body or a comment |
| `evidence` | Short text that produced the candidate |
| `confidence` | Collection heuristic, not model confidence |
| `review_status` | Manual ground-truth decision |

## Limitations

- Repositories using formal duplicate timeline events without duplicate labels may be
  missed. GraphQL timeline collection is a later extension.
- Cross-repository duplicate references are ignored.
- The regex intentionally ignores vague phrases such as `related to #123`.
- Re-running collection refreshes the evidence while preserving existing review
  decisions for pairs that are rediscovered.
