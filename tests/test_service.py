from pathlib import Path

import numpy as np

from app.schemas import Issue
from app.service import TriageService
from app.store import RepositoryStore


class FakeEmbedder:
    vectors = {
        "Login fails\n\nCannot sign in": [1.0, 0.0],
        "Dark mode\n\nPlease add a dark theme": [0.0, 1.0],
        "Authentication is broken": [0.9, 0.1],
    }

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def issue(number: int, title: str, body: str) -> Issue:
    return Issue(
        number=number,
        title=title,
        body=body,
        labels=["bug"],
        state="closed",
        html_url=f"https://github.com/acme/demo/issues/{number}",
        created_at="2026-01-01T00:00:00Z",
    )


def test_search_ranks_most_similar_issue_first(tmp_path: Path):
    store = RepositoryStore(tmp_path)
    issues = [
        issue(1, "Login fails", "Cannot sign in"),
        issue(2, "Dark mode", "Please add a dark theme"),
    ]
    store.save("acme/demo", issues, FakeEmbedder().encode([item.text for item in issues]))
    service = TriageService(store, FakeEmbedder())

    results = service.search("acme/demo", "Authentication is broken", top_k=2)

    assert results[0].issue.number == 1
    assert results[0].similarity > results[1].similarity

