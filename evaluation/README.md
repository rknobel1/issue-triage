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

## 4. Hydrate approved pairs

Download any query or canonical issue missing from local storage and embed it:

```bash
python -m evaluation.hydrate_candidates flutter/flutter
```

Hydration fetches only required issue numbers. It is therefore much cheaper than
importing an entire large repository.

## 5. Run evaluation

```bash
python -m evaluation.evaluate flutter/flutter
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
- Hydrating approved pairs only adds the query and target. For credible retrieval
  metrics, build a fixed candidate corpus containing realistic historical negatives.
