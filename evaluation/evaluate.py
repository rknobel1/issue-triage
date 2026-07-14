"""Evaluate embedding retrieval with manually approved duplicate pairs."""

import json
import time
from pathlib import Path

import typer
from pydantic import TypeAdapter

from app.config import settings
from app.embeddings import get_embedder
from app.service import TriageService
from app.store import RepositoryStore
from evaluation.models import DuplicateCandidate

app = typer.Typer(help="Evaluate duplicate retrieval against an approved dataset.")


def calculate_metrics(ranks: list[int | None]) -> dict[str, float]:
    if not ranks:
        raise ValueError("At least one evaluated rank is required")
    total = len(ranks)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total,
    }


def load_approved(dataset: Path) -> list[DuplicateCandidate]:
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(dataset.read_text())
    return [record for record in records if record.review_status == "approved"]


@app.command()
def run(
    repository: str,
    dataset: Path = Path("evaluation/datasets/candidates.json"),
) -> None:
    """Report Recall@1, Recall@5, MRR, latency, and individual ranks."""
    approved = [
        item for item in load_approved(dataset) if item.repository == repository
    ]
    if not approved:
        raise typer.BadParameter(
            "No approved pairs found. Set review_status to 'approved' after review."
        )

    store = RepositoryStore(settings.data_dir)
    issues, _ = store.load(repository)
    issues_by_number = {issue.number: issue for issue in issues}
    service = TriageService(store, get_embedder())
    ranks: list[int | None] = []
    latencies_ms: list[float] = []

    for pair in approved:
        query = issues_by_number.get(pair.query_issue)
        target = issues_by_number.get(pair.duplicate_issue)
        if query is None or target is None:
            typer.echo(
                f"Skipping #{pair.query_issue} -> #{pair.duplicate_issue}: issue not imported"
            )
            continue

        started = time.perf_counter()
        results = service.search(
            repository,
            query.title,
            query.body,
            top_k=len(issues),
            exclude_issue_number=query.number,
            created_before=query.created_at,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if result.issue.number == target.number
            ),
            None,
        )
        ranks.append(rank)
        typer.echo(
            f"#{query.number} -> #{target.number}: "
            f"rank {rank if rank is not None else 'not found'}"
        )

    if not ranks:
        raise typer.BadParameter("None of the approved pairs exist in the imported data")
    metrics = calculate_metrics(ranks)
    typer.echo("\nResults")
    typer.echo(f"Pairs evaluated: {len(ranks)}")
    typer.echo(f"Recall@1: {metrics['recall_at_1']:.3f}")
    typer.echo(f"Recall@5: {metrics['recall_at_5']:.3f}")
    typer.echo(f"MRR:      {metrics['mrr']:.3f}")
    typer.echo(f"Mean query latency: {sum(latencies_ms) / len(latencies_ms):.1f} ms")


if __name__ == "__main__":
    app()

