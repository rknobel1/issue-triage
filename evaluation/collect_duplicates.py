"""Mine explicit duplicate links from imported issues or duplicate-labeled issues."""

import asyncio
import json
import random
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

TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def has_duplicate_label(issue: Issue) -> bool:
    return any("duplicate" in label.casefold() for label in issue.labels)


def extract_references(text: str) -> list[int]:
    return [int(match.group("issue_number")) for match in DUPLICATE_PATTERN.finditer(text)]


def shorten_evidence(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def candidate_confidence(
    query: Issue | None, source: str, actor_association: str | None
) -> float:
    """Rank evidence for review; confidence never auto-approves a pair."""
    labeled = query is not None and has_duplicate_label(query)
    trusted = actor_association in TRUSTED_ASSOCIATIONS
    if labeled and trusted:
        return 0.99
    if labeled:
        return 0.98
    if trusted:
        return 0.97
    return 0.95 if source == "issue_comment" else 0.90


def load_existing(output: Path) -> dict[tuple[str, int, int], DuplicateCandidate]:
    if not output.exists():
        return {}
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(output.read_text())
    return {(item.repository, item.query_issue, item.duplicate_issue): item for item in records}


async def collect(
    repository: str,
    output: Path,
    max_comments: int,
    label: str | None = None,
    target_pairs: int = 300,
    max_labeled_issues: int = 10_000,
    sample_seed: int = 42,
) -> list[DuplicateCandidate]:
    store = RepositoryStore(settings.data_dir)
    try:
        stored_issues, _ = store.load(repository)
    except FileNotFoundError:
        if label is None:
            raise
        stored_issues = []
    issues_by_number = {issue.number: issue for issue in stored_issues}
    existing = load_existing(output)
    candidates: dict[tuple[str, int, int], DuplicateCandidate] = dict(existing)
    github = GitHubClient(settings.github_token)

    def add_candidate(
        query: Issue,
        target_number: int,
        source: Literal["issue_body", "issue_comment"],
        evidence: str,
        actor: str | None,
        actor_association: str | None,
        evidence_url: str | None,
    ) -> None:
        if target_number == query.number:
            return
        key = (repository, query.number, target_number)
        previous = existing.get(key)
        candidate = DuplicateCandidate(
            repository=repository,
            query_issue=query.number,
            duplicate_issue=target_number,
            source=source,
            evidence=shorten_evidence(evidence),
            confidence=candidate_confidence(query, source, actor_association),
            actor=actor,
            actor_association=actor_association,
            evidence_url=evidence_url,
            discovered_at=datetime.now(UTC).isoformat(),
            review_status=previous.review_status if previous else "pending",
            query_available=query.number in issues_by_number,
            target_available=target_number in issues_by_number,
        )
        current = candidates.get(key)
        if current is None or candidate.confidence > current.confidence:
            candidates[key] = candidate

    async def scan_issue(query: Issue) -> None:
        for target in extract_references(query.body):
            add_candidate(query, target, "issue_body", query.body, None, None, query.html_url)
        async for comment in github.iter_issue_comments(repository, query.number):
            for target in extract_references(comment.body):
                add_candidate(
                    query,
                    target,
                    "issue_comment",
                    comment.body,
                    comment.author,
                    comment.author_association,
                    comment.html_url,
                )

    try:
        if label is not None:
            labeled_issues = [
                issue
                async for issue in github.iter_labeled_issues(
                    repository, label, limit=max_labeled_issues
                )
            ]
            if len(labeled_issues) > target_pairs:
                labeled_issues = random.Random(sample_seed).sample(
                    labeled_issues, target_pairs
                )
                labeled_issues.sort(key=lambda issue: issue.created_at)
            scanned = 0
            for issue in labeled_issues:
                scanned += 1
                await scan_issue(issue)
                if scanned % 25 == 0:
                    typer.echo(f"Scanned {scanned} issues labeled {label!r}")
        else:
            for issue in stored_issues:
                for target in extract_references(issue.body):
                    add_candidate(
                        issue, target, "issue_body", issue.body, None, None, issue.html_url
                    )
            scanned = 0
            async for comment in github.iter_repository_comments(
                repository, limit=max_comments
            ):
                scanned += 1
                if scanned % 500 == 0:
                    typer.echo(f"Scanned {scanned} repository comments")
                query = issues_by_number.get(comment.issue_number)
                if query is None:
                    continue
                for target in extract_references(comment.body):
                    add_candidate(
                        query,
                        target,
                        "issue_comment",
                        comment.body,
                        comment.author,
                        comment.author_association,
                        comment.html_url,
                    )
    finally:
        await github.close()

    result = sorted(candidates.values(), key=lambda item: (item.repository, item.query_issue))
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
    label: str | None = typer.Option(
        None, help="Scan closed issues with this label and each issue's comments."
    ),
    target_pairs: int = typer.Option(
        300, min=1, help="Maximum labeled issues to scan in --label mode."
    ),
    max_labeled_issues: int = typer.Option(
        10_000, min=1, help="Maximum label-matched issues considered for sampling."
    ),
    sample_seed: int = typer.Option(
        42, help="Random seed used for reproducible temporal sampling."
    ),
) -> None:
    """Collect explicit duplicate references for manual review."""
    try:
        candidates = asyncio.run(
            collect(
                repository,
                output,
                max_comments,
                label=label,
                target_pairs=target_pairs,
                max_labeled_issues=max_labeled_issues,
                sample_seed=sample_seed,
            )
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc}. Run 'issue-triage import-repo {repository}' first, or use --label."
        ) from exc
    evaluable = sum(
        item.query_available is True and item.target_available is True for item in candidates
    )
    typer.echo(f"Saved {len(candidates)} candidates to {output}")
    typer.echo(f"Currently evaluable pairs: {evaluable}")
    typer.echo("Review each record and change review_status to approved or rejected.")


if __name__ == "__main__":
    app()
