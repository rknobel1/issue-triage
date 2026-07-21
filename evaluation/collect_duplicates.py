"""Mine issue bodies and repository-wide comments for explicit duplicate links."""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import typer
from pydantic import TypeAdapter

from app.config import settings
from app.github import GitHubClient
from app.schemas import Issue
from app.store import RepositoryStore
from evaluation.models import DuplicateCandidate

app = typer.Typer(help="Collect potential duplicate issue pairs from GitHub.")

DUPLICATE_PATTERN = re.compile(
    r"""
    (?:
        duplicates?\s+(?:of\s+)?|
        same\s+as\s+|
        already\s+(?:reported|tracked)\s+(?:in|by)\s+|
        closing\s+(?:this\s+)?(?:as\s+)?(?:a\s+)?duplicates?\s+(?:of\s+)?|
        closing\s+(?:this\s+)?in\s+favou?r\s+of\s+
    )
    (?:https://github\.com/[^/\s]+/[^/\s]+/issues/|\#)
    (?P<issue_number>\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def has_duplicate_label(issue: Issue) -> bool:
    return any("duplicate" in label.casefold() for label in issue.labels)


def extract_references(text: str) -> list[int]:
    return [int(match.group("issue_number")) for match in DUPLICATE_PATTERN.finditer(text)]


def shorten_evidence(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def load_existing(output: Path) -> dict[tuple[str, int, int], DuplicateCandidate]:
    if not output.exists():
        return {}
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(output.read_text())
    return {
        (item.repository, item.query_issue, item.duplicate_issue): item
        for item in records
    }


async def collect(
    repository: str, output: Path, max_comments: int
) -> list[DuplicateCandidate]:
    store = RepositoryStore(settings.data_dir)
    issues, _ = store.load(repository)
    issues_by_number = {issue.number: issue for issue in issues}
    existing = load_existing(output)
    github = GitHubClient(settings.github_token)
    # Preserve reviewed pairs from earlier collection runs and other repositories.
    candidates: dict[tuple[str, int, int], DuplicateCandidate] = dict(existing)

    def add_candidate(
        query_number: int,
        target_number: int,
        source: Literal["issue_body", "issue_comment"],
        evidence: str,
        actor: str | None,
        evidence_url: str | None,
    ) -> None:
        if target_number == query_number:
            return
        key = (repository, query_number, target_number)
        previous = existing.get(key)
        query = issues_by_number.get(query_number)
        candidate = DuplicateCandidate(
            repository=repository,
            query_issue=query_number,
            duplicate_issue=target_number,
            source=source,
            evidence=shorten_evidence(evidence),
            confidence=(
                0.98
                if query is not None and has_duplicate_label(query)
                else 0.95 if source == "issue_comment" else 0.90
            ),
            actor=actor,
            evidence_url=evidence_url,
            discovered_at=datetime.now(UTC).isoformat(),
            review_status=previous.review_status if previous else "pending",
            query_available=query is not None,
            target_available=target_number in issues_by_number,
        )
        current = candidates.get(key)
        if current is None or candidate.confidence > current.confidence:
            candidates[key] = candidate

    # Issue bodies are already local, so every imported issue can be scanned cheaply.
    for issue in issues:
        for target in extract_references(issue.body):
            add_candidate(
                issue.number, target, "issue_body", issue.body, None, issue.html_url
            )

    try:
        scanned = 0
        async for comment in github.iter_repository_comments(
            repository, limit=max_comments
        ):
            scanned += 1
            if scanned % 500 == 0:
                typer.echo(f"Scanned {scanned} repository comments")
            if comment.issue_number not in issues_by_number:
                continue
            for target in extract_references(comment.body):
                add_candidate(
                    comment.issue_number,
                    target,
                    "issue_comment",
                    comment.body,
                    comment.author,
                    comment.html_url,
                )
    finally:
        await github.close()

    result = sorted(
        candidates.values(), key=lambda item: (item.repository, item.query_issue)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.model_dump() for item in result], indent=2), encoding="utf-8"
    )
    return result


@app.command()
def run(
    repository: str,
    output: Path = Path("evaluation/datasets/candidates.json"),
    max_comments: int = typer.Option(10_000, min=1),
) -> None:
    """Collect candidates from an already-imported OWNER/REPOSITORY."""
    try:
        candidates = asyncio.run(collect(repository, output, max_comments))
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc}. Run 'issue-triage import-repo {repository}' first."
        ) from exc
    evaluable = sum(
        item.query_available is True and item.target_available is True
        for item in candidates
    )
    typer.echo(f"Saved {len(candidates)} candidates to {output}")
    typer.echo(f"Currently evaluable pairs: {evaluable}")
    typer.echo("Review each record and change review_status to approved or rejected.")


if __name__ == "__main__":
    app()
