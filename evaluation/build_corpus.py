"""Build a fixed, historically representative duplicate-evaluation corpus."""

import asyncio
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import TypeAdapter

from app.config import settings
from app.embeddings import get_embedder
from app.github import GitHubClient
from app.schemas import Issue
from app.store import RepositoryStore
from evaluation.models import DuplicateCandidate

app = typer.Typer(help="Build a deterministic historical corpus for offline evaluation.")


def load_approved(dataset: Path, repository: str) -> list[DuplicateCandidate]:
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(dataset.read_text())
    return [
        record
        for record in records
        if record.repository == repository and record.review_status == "approved"
    ]


def sampled_issue_numbers(
    max_issue_number: int,
    attempt_limit: int,
    seed: int,
) -> list[int]:
    """Return deterministic issue numbers spread across repository history."""
    count = min(max_issue_number, attempt_limit)
    return random.Random(seed).sample(range(1, max_issue_number + 1), count)


async def sample_historical_issues(
    github: GitHubClient,
    repository: str,
    max_issue_number: int,
    sample_size: int,
    attempt_limit: int,
    seed: int,
    created_before: str,
) -> tuple[list[Issue], int]:
    """Sample issue numbers and fetch historical issues in GraphQL batches."""
    sampled: list[Issue] = []
    attempted = 0
    numbers = sampled_issue_numbers(max_issue_number, attempt_limit, seed)
    for start in range(0, len(numbers), 50):
        batch = numbers[start : start + 50]
        attempted += len(batch)
        issues = await github.get_issues_batch(repository, batch)
        sampled.extend(issue for issue in issues if issue.created_at < created_before)
        if len(sampled) >= sample_size:
            return sampled[:sample_size], attempted
        typer.echo(
            f"Sampled {len(sampled)}/{sample_size} historical issues "
            f"after trying {attempted} numbers"
        )
    return sampled, attempted


def dataset_sha256(dataset: Path) -> str:
    return hashlib.sha256(dataset.read_bytes()).hexdigest()


async def build(
    repository: str,
    dataset: Path,
    manifest: Path,
    sample_size: int,
    scan_limit: int,
    seed: int,
) -> dict:
    pairs = load_approved(dataset, repository)
    if not pairs:
        raise typer.BadParameter("No approved pairs found for that repository")

    github = GitHubClient(settings.github_token)
    required_numbers = {
        number for pair in pairs for number in (pair.query_issue, pair.duplicate_issue)
    }
    try:
        # Query issues are fetched first so the sampling boundary is based on the
        # source-of-truth timestamp rather than potentially stale local data.
        required: dict[int, Issue] = {}
        for position, number in enumerate(sorted(required_numbers), start=1):
            typer.echo(f"[pair {position}/{len(required_numbers)}] Downloading #{number}")
            required[number] = await github.get_issue(repository, number)

        latest_query_at = max(required[pair.query_issue].created_at for pair in pairs)
        max_query_number = max(pair.query_issue for pair in pairs)
        negatives, scanned = await sample_historical_issues(
            github,
            repository=repository,
            max_issue_number=max_query_number,
            sample_size=sample_size,
            attempt_limit=scan_limit,
            seed=seed,
            created_before=latest_query_at,
        )
    finally:
        await github.close()

    corpus_by_number = {issue.number: issue for issue in negatives}
    corpus_by_number.update(required)
    corpus = sorted(corpus_by_number.values(), key=lambda issue: (issue.created_at, issue.number))

    embeddings = get_embedder().encode([issue.text for issue in corpus])
    RepositoryStore(settings.data_dir).save(repository, corpus, embeddings)

    manifest_data = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": repository,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
        "embedding_model": settings.embedding_model,
        "seed": seed,
        "scan_limit": scan_limit,
        "issues_scanned": scanned,
        "negative_sample_size": sample_size,
        "approved_pairs": len(pairs),
        "required_pair_issues": len(required),
        "corpus_size": len(corpus),
        "latest_query_at": latest_query_at,
        "issue_numbers": [issue.number for issue in corpus],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return manifest_data


@app.command()
def run(
    repository: str,
    dataset: Path = Path("evaluation/datasets/candidates.json"),
    manifest: Path = Path("evaluation/datasets/corpus_manifest.json"),
    sample_size: int = typer.Option(
        10_000, min=100, help="Uniformly sampled historical negatives."
    ),
    scan_limit: int = typer.Option(
        25_000, min=100, help="Maximum issue numbers to try while sampling."
    ),
    seed: int = typer.Option(42, help="Historical sampling seed."),
) -> None:
    """Replace local storage with a fixed corpus plus all approved pair endpoints."""
    result = asyncio.run(build(repository, dataset, manifest, sample_size, scan_limit, seed))
    typer.echo(
        f"Built {result['corpus_size']} issue corpus from "
        f"{result['issues_scanned']} historical issues"
    )
    typer.echo(f"Manifest: {manifest}")


if __name__ == "__main__":
    app()
