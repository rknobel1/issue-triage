import numpy as np

from app.embeddings import Embedder
from app.github import GitHubClient
from app.schemas import SearchResult
from app.store import RepositoryStore


class TriageService:
    def __init__(self, store: RepositoryStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    async def import_repository(
        self, repository: str, limit: int, github: GitHubClient
    ) -> int:
        issues = [issue async for issue in github.iter_issues(repository, limit)]
        if not issues:
            raise ValueError("No issues were found in that repository")
        embeddings = self.embedder.encode([issue.text for issue in issues])
        self.store.save(repository, issues, embeddings)
        return len(issues)

    def search(
        self,
        repository: str,
        title: str,
        body: str = "",
        top_k: int = 5,
        exclude_issue_number: int | None = None,
        created_before: str | None = None,
    ) -> list[SearchResult]:
        issues, embeddings = self.store.load(repository)
        query = self.embedder.encode([f"{title}\n\n{body}".strip()])[0]
        scores = embeddings @ query
        candidates = [
            index
            for index in np.argsort(scores)[::-1]
            if issues[index].number != exclude_issue_number
            and (created_before is None or issues[index].created_at < created_before)
        ]
        best = candidates[: min(top_k, len(candidates))]
        return [
            SearchResult(issue=issues[index], similarity=round(float(scores[index]), 4))
            for index in best
        ]
