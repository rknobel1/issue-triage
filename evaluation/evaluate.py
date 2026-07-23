"""Compare duplicate-issue retrieval methods and save reproducible experiments."""

import csv
import json
import platform
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TextIO

import typer
import numpy as np
from pydantic import TypeAdapter
from tqdm.auto import tqdm

from app.config import settings
from app.embeddings import get_embedder
from app.store import RepositoryStore
from evaluation.models import DuplicateCandidate
from evaluation.retrievers import get_reranker, rank_issues

app = typer.Typer(help="Evaluate duplicate retrieval against an approved dataset.")


def calculate_metrics(ranks: list[int | None]) -> dict[str, float]:
    if not ranks:
        raise ValueError("At least one evaluated rank is required")
    total = len(ranks)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / total,
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total,
    }


def calculate_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("At least one latency is required")
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_latency_ms": float(np.mean(values)),
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.percentile(values, 95)),
    }


def load_approved(dataset: Path) -> list[DuplicateCandidate]:
    records = TypeAdapter(list[DuplicateCandidate]).validate_json(dataset.read_text())
    return [record for record in records if record.review_status == "approved"]


def format_ranking_lines(
    method: str,
    query_number: int,
    target_number: int,
    target_rank: int | None,
    top_results: list[dict],
) -> list[str]:
    """Format a compact, human-readable ranking for console output."""
    rank_display = "not retrieved" if target_rank is None else str(target_rank)
    lines = [
        f"\n[{method}] query #{query_number} -> target #{target_number} "
        f"(target rank: {rank_display})"
    ]
    for result in top_results:
        marker = " <-- target" if result["issue_number"] == target_number else ""
        lines.append(
            f"  {result['rank']:>3}. #{result['issue_number']} "
            f"score={result['score']:.6f}{marker}"
        )
    return lines


def iter_with_progress(
    pairs: list[DuplicateCandidate],
    enabled: bool,
    stream: TextIO | None = None,
) -> Iterator[DuplicateCandidate]:
    """Yield evaluation pairs through a terminal-friendly tqdm progress bar."""
    yield from tqdm(
        pairs,
        desc="Evaluating duplicate pairs",
        unit="pair",
        dynamic_ncols=stream is None,
        disable=not enabled,
        file=stream,
    )


def _write_experiment(
    output_dir: Path,
    experiment: dict,
    rows: list[dict],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = experiment["experiment_id"]
    json_path = output_dir / f"{experiment_id}.json"
    csv_path = output_dir / f"{experiment_id}.csv"

    json_path.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # UTF-8 with BOM allows Excel on Windows to detect the encoding correctly.
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


@app.command()
def run(
    repository: str,
    dataset: Path = Path("evaluation/datasets/candidates.json"),
    methods: Annotated[
        list[str],
        typer.Option("--method", help="Repeat to compare multiple retrievers."),
    ] = [],
    hybrid_weight: Annotated[
        float, typer.Option(help="Dense contribution to the hybrid score (0..1).")
    ] = 0.5,
    rrf_k: Annotated[
        int, typer.Option(min=1, help="RRF rank constant; larger values flatten ranks.")
    ] = 60,
    rerank_top_n: Annotated[
        int, typer.Option(min=1, help="Dense candidates passed to the cross-encoder.")
    ] = 20,
    top_k_details: Annotated[
        int, typer.Option(min=1, help="Number of ranked candidates saved per query.")
    ] = 10,
    output_dir: Path = Path("evaluation/results"),
    experiment_name: str | None = None,
    min_candidates: Annotated[
        int,
        typer.Option(
            min=1,
            help="Skip queries with fewer historical candidates than this.",
        ),
    ] = 100,
    show_rankings: Annotated[
        bool,
        typer.Option(
            "--show-rankings",
            help="Print each query's target rank and top results.",
        ),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show progress through approved duplicate pairs.",
        ),
    ] = True,
) -> None:
    """Evaluate retrievers and save detailed JSON and CSV results."""
    valid_methods = {"dense", "tfidf", "hybrid", "rrf", "rerank"}
    selected_methods = methods or ["dense", "tfidf", "hybrid"]

    invalid_methods = set(selected_methods) - valid_methods
    if invalid_methods:
        valid_display = ", ".join(sorted(valid_methods))
        invalid_display = ", ".join(sorted(invalid_methods))
        raise typer.BadParameter(
            f"Unknown method(s): {invalid_display}. Valid methods: {valid_display}"
        )
    if not 0.0 <= hybrid_weight <= 1.0:
        raise typer.BadParameter("--hybrid-weight must be between 0 and 1")

    approved = [item for item in load_approved(dataset) if item.repository == repository]
    if not approved:
        raise typer.BadParameter(
            "No approved pairs found. Set review_status to 'approved' after review."
        )

    store = RepositoryStore(settings.data_dir)
    issues, embeddings = store.load(repository)
    issues_by_number = {issue.number: issue for issue in issues}
    embedder = get_embedder()
    reranker = get_reranker() if "rerank" in selected_methods else None
    started_at = datetime.now(UTC)
    safe_repository = repository.replace("/", "-")
    suffix = f"-{experiment_name}" if experiment_name else ""
    experiment_id = f"{started_at:%Y%m%dT%H%M%SZ}-{safe_repository}{suffix}"

    details: list[dict] = []
    csv_rows: list[dict] = []
    ranks_by_method: dict[str, list[int | None]] = {method: [] for method in selected_methods}
    latencies_by_method: dict[str, list[float]] = {method: [] for method in selected_methods}
    skipped: list[dict] = []

    for pair in iter_with_progress(approved, progress):
        query = issues_by_number.get(pair.query_issue)
        target = issues_by_number.get(pair.duplicate_issue)
        if query is None or target is None:
            skipped.append(
                {
                    "query_issue": pair.query_issue,
                    "target_issue": pair.duplicate_issue,
                    "reason": "query or target is not in the imported corpus",
                }
            )
            continue

        candidate_count = sum(
            issue.number != query.number and issue.created_at < query.created_at
            for issue in issues
        )
        if candidate_count < min_candidates:
            skipped.append(
                {
                    "query_issue": query.number,
                    "target_issue": target.number,
                    "reason": "insufficient historical candidates",
                    "candidate_count": candidate_count,
                    "minimum_required": min_candidates,
                }
            )
            continue

        for method in selected_methods:
            started = time.perf_counter()
            ranked = rank_issues(
                method=method,
                issues=issues,
                embeddings=embeddings,
                query=query,
                embedder=embedder,
                hybrid_weight=hybrid_weight,
                rrf_k=rrf_k,
                rerank_top_n=rerank_top_n,
                reranker=reranker,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            rank = next(
                (
                    index
                    for index, result in enumerate(ranked, start=1)
                    if result.issue.number == target.number
                ),
                None,
            )
            ranks_by_method[method].append(rank)
            latencies_by_method[method].append(latency_ms)
            top_results = [
                {
                    "rank": index,
                    "issue_number": result.issue.number,
                    "score": round(result.score, 6),
                    "dense_score": None
                    if result.dense_score is None
                    else round(result.dense_score, 6),
                    "tfidf_score": None
                    if result.tfidf_score is None
                    else round(result.tfidf_score, 6),
                    "reranker_score": None
                    if result.reranker_score is None
                    else round(result.reranker_score, 6),
                }
                for index, result in enumerate(ranked[:top_k_details], start=1)
            ]
            if show_rankings:
                for line in format_ranking_lines(
                    method,
                    query.number,
                    target.number,
                    rank,
                    top_results,
                ):
                    if progress:
                        tqdm.write(line)
                    else:
                        typer.echo(line)
            record = {
                "method": method,
                "query_issue": query.number,
                "target_issue": target.number,
                "target_rank": rank,
                "candidate_count": len(ranked),
                "latency_ms": round(latency_ms, 3),
                "query_title": query.title,
                "target_title": target.title,
                "shared_labels": sorted(set(query.labels) & set(target.labels)),
                "top_results": top_results,
            }
            details.append(record)
            csv_rows.append(
                {
                    "method": method,
                    "query_issue": query.number,
                    "target_issue": target.number,
                    "target_rank": "" if rank is None else rank,
                    "candidate_count": len(ranked),
                    "latency_ms": round(latency_ms, 3),
                    "query_title": query.title,
                    "target_title": target.title,
                    "shared_labels": "|".join(record["shared_labels"]),
                    "top_issue_numbers": "|".join(
                        str(result["issue_number"]) for result in top_results
                    ),
                }
            )

    if not csv_rows:
        raise typer.BadParameter(
            "No approved pairs had enough historical candidates in the corpus"
        )

    summaries = {}
    for method in selected_methods:
        ranks = ranks_by_method[method]
        summaries[method] = {
            "pairs_evaluated": len(ranks),
            **calculate_metrics(ranks),
            **calculate_latency_stats(latencies_by_method[method]),
        }

    experiment = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "started_at": started_at.isoformat(),
        "repository": repository,
        "dataset": str(dataset),
        "corpus_size": len(issues),
        "embedding_model": settings.embedding_model,
        "methods": selected_methods,
        "configuration": {
            "hybrid_weight": hybrid_weight,
            "rrf_k": rrf_k,
            "rerank_top_n": rerank_top_n,
            "reranker_model": settings.reranker_model,
            "top_k_details": top_k_details,
            "min_candidates": min_candidates,
            "show_rankings": show_rankings,
            "progress": progress,
            "temporal_filter": "candidate.created_at < query.created_at",
        },
        "environment": {"python": platform.python_version()},
        "summaries": summaries,
        "skipped": skipped,
        "queries": details,
    }
    json_path, csv_path = _write_experiment(output_dir, experiment, csv_rows)

    typer.echo("\nResults")
    for method, summary in summaries.items():
        typer.echo(
            f"{method:>6}  R@1 {summary['recall_at_1']:.3f}  "
            f"R@5 {summary['recall_at_5']:.3f}  "
            f"R@10 {summary['recall_at_10']:.3f}  "
            f"MRR {summary['mrr']:.3f}"
            f"  latency p50 {summary['median_latency_ms']:.1f}ms"
            f" p95 {summary['p95_latency_ms']:.1f}ms"
        )
    typer.echo(f"\nJSON: {json_path}")
    typer.echo(f"CSV:  {csv_path}")


if __name__ == "__main__":
    app()