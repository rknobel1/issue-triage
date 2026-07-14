from pathlib import Path

import numpy as np
import pytest

from app.schemas import Issue
from app.store import RepositoryStore


def test_store_round_trip(tmp_path: Path):
    store = RepositoryStore(tmp_path)
    issues = [
        Issue(
            number=1,
            title="Example",
            state="open",
            html_url="https://github.com/acme/demo/issues/1",
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    expected = np.asarray([[1.0, 0.0]], dtype=np.float32)
    store.save("acme/demo", issues, expected)

    loaded_issues, loaded_embeddings = store.load("acme/demo")

    assert loaded_issues == issues
    np.testing.assert_array_equal(loaded_embeddings, expected)


def test_missing_repository(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RepositoryStore(tmp_path).load("missing/repo")

