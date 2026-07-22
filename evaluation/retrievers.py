"""Retrieval strategies used by the offline evaluation runner."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from app.embeddings import Embedder
from app.config import settings
from app.schemas import Issue

RetrievalMethod = Literal["dense", "tfidf", "hybrid", "rrf", "rerank"]


class Reranker(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]]) -> NDArray[np.float64]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.reranker_model):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def predict(self, sentences: Sequence[tuple[str, str]]) -> NDArray[np.float64]:
        return np.asarray(
            self.model.predict(list(sentences), show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@dataclass(frozen=True)
class RankedIssue:
    issue: Issue
    score: float
    dense_score: float | None = None
    tfidf_score: float | None = None
    reranker_score: float | None = None


def _rank_positions(scores: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return one-based rank positions for scores, with highest score ranked first."""
    order = np.argsort(scores, kind="stable")[::-1]
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def eligible_indices(issues: Sequence[Issue], query: Issue) -> NDArray[np.int64]:
    """Return historical candidates, excluding the query itself."""
    return np.asarray(
        [
            index
            for index, issue in enumerate(issues)
            if issue.number != query.number and issue.created_at < query.created_at
        ],
        dtype=np.int64,
    )


def _tfidf_scores(query_text: str, candidate_texts: list[str]) -> NDArray[np.float64]:
    if not candidate_texts:
        return np.asarray([], dtype=np.float64)
    # Fit only on the historical corpus for this query. Later issues would leak
    # information through document-frequency statistics.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    candidate_matrix = vectorizer.fit_transform(candidate_texts)
    query_vector = vectorizer.transform([query_text])
    return np.asarray((candidate_matrix @ query_vector.T).toarray()).ravel()


def rank_issues(
    *,
    method: RetrievalMethod,
    issues: Sequence[Issue],
    embeddings: NDArray[np.float32],
    query: Issue,
    embedder: Embedder,
    hybrid_weight: float = 0.5,
    rrf_k: int = 60,
    rerank_top_n: int = 50,
    reranker: Reranker | None = None,
) -> list[RankedIssue]:
    """Rank all historical candidates for one query."""
    if not 0.0 <= hybrid_weight <= 1.0:
        raise ValueError("hybrid_weight must be between 0 and 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if rerank_top_n < 1:
        raise ValueError("rerank_top_n must be at least 1")
    if method == "rerank" and reranker is None:
        raise ValueError("reranker is required when method='rerank'")

    indices = eligible_indices(issues, query)
    if not len(indices):
        return []

    candidate_issues = [issues[index] for index in indices]
    dense_scores: NDArray[np.float64] | None = None
    tfidf_scores: NDArray[np.float64] | None = None

    if method in {"dense", "hybrid", "rrf", "rerank"}:
        query_embedding = embedder.encode([query.text])[0]
        dense_scores = np.asarray(embeddings[indices] @ query_embedding, dtype=np.float64)

    if method in {"tfidf", "hybrid", "rrf"}:
        tfidf_scores = _tfidf_scores(query.text, [candidate.text for candidate in candidate_issues])

    if method == "dense":
        assert dense_scores is not None
        combined_scores = dense_scores
    elif method == "tfidf":
        assert tfidf_scores is not None
        combined_scores = tfidf_scores
    elif method == "hybrid":
        assert dense_scores is not None and tfidf_scores is not None
        normalized_dense = np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0)
        combined_scores = hybrid_weight * normalized_dense + (1.0 - hybrid_weight) * tfidf_scores
    elif method == "rrf":
        assert dense_scores is not None and tfidf_scores is not None
        dense_ranks = _rank_positions(dense_scores)
        tfidf_ranks = _rank_positions(tfidf_scores)
        combined_scores = 1.0 / (rrf_k + dense_ranks) + 1.0 / (rrf_k + tfidf_ranks)
    else:
        assert dense_scores is not None and reranker is not None
        dense_order = np.argsort(dense_scores)[::-1][:rerank_top_n]
        candidate_issues = [candidate_issues[position] for position in dense_order]
        dense_scores = dense_scores[dense_order]
        reranker_scores = reranker.predict(
            [(query.text, candidate.text) for candidate in candidate_issues]
        )
        if len(reranker_scores) != len(candidate_issues):
            raise ValueError("reranker returned an unexpected number of scores")
        order = np.argsort(reranker_scores)[::-1]
        return [
            RankedIssue(
                issue=candidate_issues[position],
                score=float(reranker_scores[position]),
                dense_score=float(dense_scores[position]),
                reranker_score=float(reranker_scores[position]),
            )
            for position in order
        ]

    order = np.argsort(combined_scores)[::-1]
    return [
        RankedIssue(
            issue=candidate_issues[position],
            score=float(combined_scores[position]),
            dense_score=None if dense_scores is None else float(dense_scores[position]),
            tfidf_score=None if tfidf_scores is None else float(tfidf_scores[position]),
        )
        for position in order
    ]
