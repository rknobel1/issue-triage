from app.schemas import Issue
from evaluation.collect_duplicates import extract_references, has_duplicate_label
from evaluation.evaluate import calculate_latency_stats, calculate_metrics


def test_extracts_explicit_duplicate_references():
    text = "Closing this in favor of #123. This is also a duplicate of #456."
    assert extract_references(text) == [123, 456]


def test_extracts_closing_as_duplicate_and_full_urls():
    text = "Closing as Duplicate of #123; duplicate of https://github.com/acme/demo/issues/456"
    assert extract_references(text) == [123, 456]


def test_ignores_vague_references():
    assert extract_references("This may be related to #123") == []


def test_duplicate_label_is_case_insensitive():
    issue = Issue(
        number=1,
        title="Example",
        labels=["Status: Duplicate"],
        state="closed",
        html_url="https://github.com/acme/demo/issues/1",
        created_at="2026-01-01T00:00:00Z",
    )
    assert has_duplicate_label(issue)


def test_calculates_retrieval_metrics():
    metrics = calculate_metrics([1, 2, None, 5])
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_5"] == 0.75
    assert metrics["mrr"] == (1 + 0.5 + 0 + 0.2) / 4


def test_calculates_latency_statistics():
    stats = calculate_latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["mean_latency_ms"] == 25.0
    assert stats["median_latency_ms"] == 25.0
    assert stats["p95_latency_ms"] == 38.5
