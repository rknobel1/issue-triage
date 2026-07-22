import numpy as np

from app.schemas import Issue
from evaluation.retrievers import rank_issues


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeReranker:
    def predict(self, sentences):
        return np.asarray(
            [1.0 if "exact duplicate" in candidate else 0.0 for _, candidate in sentences],
            dtype=np.float64,
        )


def issue(number: int, title: str, created_at: str) -> Issue:
    return Issue(
        number=number,
        title=title,
        labels=[],
        state="closed",
        html_url=f"https://github.com/acme/demo/issues/{number}",
        created_at=created_at,
    )


def test_tfidf_ranks_exact_terms_and_filters_future_issues():
    issues = [
        issue(1, "NullPointerException in WidgetRenderer", "2025-01-01T00:00:00Z"),
        issue(2, "Dark mode colors", "2025-01-02T00:00:00Z"),
        issue(3, "WidgetRenderer NullPointerException", "2025-01-03T00:00:00Z"),
        issue(4, "WidgetRenderer future report", "2025-01-04T00:00:00Z"),
    ]
    ranked = rank_issues(
        method="tfidf",
        issues=issues,
        embeddings=np.ones((4, 2), dtype=np.float32),
        query=issues[2],
        embedder=FakeEmbedder(),
    )
    assert [result.issue.number for result in ranked] == [1, 2]


def test_hybrid_weight_must_be_between_zero_and_one():
    issues = [
        issue(1, "First", "2025-01-01T00:00:00Z"),
        issue(2, "Second", "2025-01-02T00:00:00Z"),
    ]
    try:
        rank_issues(
            method="hybrid",
            issues=issues,
            embeddings=np.ones((2, 2), dtype=np.float32),
            query=issues[1],
            embedder=FakeEmbedder(),
            hybrid_weight=1.1,
        )
    except ValueError as error:
        assert "hybrid_weight" in str(error)
    else:
        raise AssertionError("Expected invalid hybrid weight to raise ValueError")


def test_rrf_rewards_candidates_ranked_high_by_both_retrievers():
    issues = [
        issue(1, "unrelated", "2025-01-01T00:00:00Z"),
        issue(2, "exact duplicate", "2025-01-02T00:00:00Z"),
        issue(3, "exact duplicate report", "2025-01-03T00:00:00Z"),
    ]
    embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    ranked = rank_issues(
        method="rrf",
        issues=issues,
        embeddings=embeddings,
        query=issues[2],
        embedder=FakeEmbedder(),
    )
    assert ranked[0].issue.number == 2


def test_reranker_only_scores_dense_top_n_and_reorders_them():
    issues = [
        issue(1, "exact duplicate", "2025-01-01T00:00:00Z"),
        issue(2, "other issue", "2025-01-02T00:00:00Z"),
        issue(3, "query", "2025-01-03T00:00:00Z"),
    ]
    embeddings = np.asarray([[0.8, 0.0], [0.9, 0.0], [1.0, 0.0]], dtype=np.float32)
    ranked = rank_issues(
        method="rerank",
        issues=issues,
        embeddings=embeddings,
        query=issues[2],
        embedder=FakeEmbedder(),
        rerank_top_n=2,
        reranker=FakeReranker(),
    )
    assert [result.issue.number for result in ranked] == [1, 2]
    assert ranked[0].reranker_score == 1.0
