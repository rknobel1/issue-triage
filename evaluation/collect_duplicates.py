import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import typer

from app.config import settings
from app.github import GitHubClient
from app.schemas import Issue
from evaluation.models import DuplicateCandidate

app = typer.Typer(help="Collect potential duplicate issue pairs from GitHub.")

DUPLICATE_PATTERN = re.compile(
    r"""
    (?:
        duplicates?\s+(?:of\s+)?|
        same\s+as\s+|
        already\s+(?:reported|tracked)\s+(?:in|by)\s+|
        closing\s+(?:this\s+)?in\s+favou?r\s+of\s+
    )
    \#(?P<issue_number>\d+)
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


def issue_from_github(record: dict) -> Issue:
    """Convert a GitHub Issues API response into the application's Issue schema."""
    return Issue(
        number=record["number"],
        title=record["title"],
        body=record.get("body") or "",
        state=record["state"],
        labels=[label["name"] for label in record.get("labels", [])],
        created_at=record["created_at"],
        html_url=record["html_url"],
    )


async def iter_duplicate_issues(
    github: GitHubClient,
    repository: str,
    label: str,
):
    """Yield duplicate-labeled issues one page at a time."""
    page = 1
    while True:
        response = await github.client.get(
            f"/repos/{repository}/issues",
            params={
                "state": "closed",
                "labels": label,
                "per_page": 100,
                "page": page,
            },
        )
        response.raise_for_status()
        records = response.json()
        if not records:
            return

        for record in records:
            if "pull_request" not in record:
                yield issue_from_github(record)
        page += 1


async def issue_exists(
    github: GitHubClient,
    repository: str,
    issue_number: int,
) -> bool:
    response = await github.client.get(f"/repos/{repository}/issues/{issue_number}")
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return "pull_request" not in response.json()


async def collect(
    repository: str,
    output: Path,
    limit: int = 100,
    label: str = "r: duplicate",
) -> list[DuplicateCandidate]:
    github = GitHubClient(settings.github_token)
    candidates: dict[tuple[int, int], DuplicateCandidate] = {}

    async def add_candidate(
        query: Issue,
        target_number: int,
        source: str,
        evidence: str,
        actor: str | None,
        evidence_url: str | None,
    ) -> bool:
        if target_number == query.number:
            return False
        key = (query.number, target_number)
        if key in candidates:
            return False
        if not await issue_exists(github, repository, target_number):
            return False
        candidate = DuplicateCandidate(
            repository=repository,
            query_issue=query.number,
            duplicate_issue=target_number,
            source=source,
            evidence=shorten_evidence(evidence),
            confidence=0.95 if source == "issue_comment" else 0.90,
            actor=actor,
            evidence_url=evidence_url,
            discovered_at=datetime.now(UTC).isoformat(),
        )
        candidates[key] = candidate
        return True

    try:
        examined = 0
        async for issue in iter_duplicate_issues(github, repository, label):
            examined += 1
            typer.echo(f"[examined {examined}, paired {len(candidates)}/{limit}] "
                       f"Checking issue #{issue.number}")

            paired = False
            for target in extract_references(issue.body):
                paired = await add_candidate(
                    issue, target, "issue_body", issue.body, None, issue.html_url
                )
                if paired:
                    break

            if paired:
                if len(candidates) >= limit:
                    break
                continue

            async for comment in github.iter_issue_comments(repository, issue.number):
                for target in extract_references(comment.body):
                    paired = await add_candidate(
                        issue,
                        target,
                        "issue_comment",
                        comment.body,
                        comment.author,
                        comment.html_url,
                    )
                    if paired:
                        break
                if paired:
                    break

            if len(candidates) >= limit:
                break
    finally:
        await github.close()

    if output.exists():
        existing_records = {
            (item.query_issue, item.duplicate_issue): item
            for item in (
                DuplicateCandidate.model_validate(record)
                for record in json.loads(output.read_text(encoding="utf-8"))
            )
        }
        for key, candidate in candidates.items():
            existing = existing_records.get(key)
            if existing is not None:
                candidate.review_status = existing.review_status

    result = sorted(candidates.values(), key=lambda item: item.query_issue)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.model_dump() for item in result], indent=2), encoding="utf-8"
    )
    return result


@app.command()
def run(
    repository: str,
    output: Path = Path("evaluation/datasets/candidates.json"),
    limit: int = typer.Option(100, min=1, help="Number of valid pairs to save."),
    label: str = typer.Option(
        "r: duplicate", help="Repository-specific duplicate label."
    ),
) -> None:
    """Stream duplicate-labeled issues until LIMIT valid pairs are collected."""
    candidates = asyncio.run(collect(repository, output, limit, label))
    typer.echo(f"Saved {len(candidates)} candidates to {output}")
    typer.echo("Review each record and change review_status to approved or rejected.")


if __name__ == "__main__":
    app()