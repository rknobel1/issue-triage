"""Download and embed issues referenced by an evaluation dataset."""

import asyncio
from pathlib import Path

import httpx
import typer
from pydantic import TypeAdapter

from app.config import settings
from app.embeddings import get_embedder
from app.github import GitHubClient
from app.schemas import Issue
from app.store import RepositoryStore
from evaluation.models import DuplicateCandidate

app = typer.Typer(help="Hydrate issues required by a duplicate evaluation dataset.")


def load_pairs(dataset: Path, repository: str) -> list[DuplicateCandidate]:
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(dataset.read_text())
    return [
        record
        for record in records
        if record.repository == repository and record.review_status == "approved"
    ]


async def download_missing(
    repository: str, issue_numbers: list[int]
) -> tuple[list[Issue], list[int]]:
    github = GitHubClient(settings.github_token)
    downloaded: list[Issue] = []
    failed: list[int] = []
    try:
        for position, number in enumerate(issue_numbers, start=1):
            typer.echo(f"[{position}/{len(issue_numbers)}] Downloading #{number}")
            try:
                downloaded.append(await github.get_issue(repository, number))
            except (httpx.HTTPError, ValueError) as exc:
                typer.echo(f"  Skipped #{number}: {exc}")
                failed.append(number)
    finally:
        await github.close()
    return downloaded, failed


@app.command()
def run(
    repository: str,
    dataset: Path = Path("evaluation/datasets/candidates.json"),
) -> None:
    """Fetch missing issues and merge their embeddings into local storage."""
    pairs = load_pairs(dataset, repository)
    if not pairs:
        raise typer.BadParameter("No approved pairs found for that repository")

    store = RepositoryStore(settings.data_dir)
    try:
        current_issues, _ = store.load(repository)
    except FileNotFoundError:
        current_issues = []
    current_numbers = {issue.number for issue in current_issues}
    required_numbers = {
        number
        for pair in pairs
        for number in (pair.query_issue, pair.duplicate_issue)
    }
    missing = sorted(required_numbers - current_numbers)
    if not missing:
        typer.echo(f"All {len(required_numbers)} required issues are already available")
        return

    issues, failed = asyncio.run(download_missing(repository, missing))
    if issues:
        embeddings = get_embedder().encode([issue.text for issue in issues])
        store.upsert(repository, issues, embeddings)
    typer.echo(f"Hydrated {len(issues)} issues; {len(failed)} failed")
    typer.echo(f"Dataset now has {len(current_numbers) + len(issues)} stored issues")


if __name__ == "__main__":
    app()
