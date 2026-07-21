# Issue Triage

An open-source semantic search service that finds likely duplicate GitHub issues.
It downloads issues from a public repository, creates embeddings locally, and ranks
existing issues by cosine similarity.

## Current checkpoint

- Import up to 2,000 issues from a public GitHub repository
- Exclude pull requests returned by GitHub's issues endpoint
- Generate normalized embeddings with `all-MiniLM-L6-v2`
- Persist issue metadata and vectors locally
- Search through a CLI or FastAPI endpoint
- Run retrieval logic without a paid AI API

## Requirements

- Python 3.11+
- Approximately 500 MB of free disk space for Python packages and the embedding model
- A GitHub personal access token is recommended but not required for small imports

## Setup

```bash
git clone https://github.com/rknobel1/issue-triage.git
cd issue-triage
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env  # On Windows: copy .env.example .env
```

Optionally add a fine-grained GitHub token to `.env`. Public read-only repository
access is sufficient. Never commit `.env`.

```dotenv
GITHUB_TOKEN=github_pat_your_token_here
```

## Import your first repository

Start small while verifying the setup:

```bash
issue-triage import-repo fastapi/fastapi --limit 100
```

The first run downloads the embedding model. Later runs use the local cache.

## Search from the command line

```bash
issue-triage search fastapi/fastapi "Validation error response is unclear"
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Example search request:

```bash
curl -X POST http://127.0.0.1:8000/search/duplicates \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "fastapi/fastapi",
    "title": "Validation error response is unclear",
    "body": "The error does not identify the invalid request field.",
    "top_k": 5
  }'
```

## Tests and formatting

```bash
pytest
ruff check .
ruff format --check .
```

## Build an evaluation dataset

After importing a larger issue history, mine duplicate-labeled issues for explicit
references, review the candidates, and calculate retrieval metrics:

```bash
issue-triage import-repo fastapi/fastapi --limit 1000
python -m evaluation.collect_duplicates fastapi/fastapi
python -m evaluation.hydrate_candidates fastapi/fastapi
python -m evaluation.evaluate fastapi/fastapi
```

See `evaluation/README.md` for the review workflow and dataset limitations.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/repositories/import` | Download and embed repository issues |
| `POST` | `/search/duplicates` | Return the most similar imported issues |

Repository import currently runs inside the request. Moving imports to a background
job is an intentional later milestone.

## Next milestones

1. Add TF-IDF as a baseline and report Recall@1, Recall@5, and MRR.
2. Add PostgreSQL and pgvector after the local pipeline is evaluated.
3. Build a small Next.js interface.
4. Add label prediction and human feedback collection.

## License

MIT
