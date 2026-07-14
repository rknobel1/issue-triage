import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.schemas import Issue


class RepositoryStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(repository: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "__", repository)

    def save(
        self, repository: str, issues: list[Issue], embeddings: NDArray[np.float32]
    ) -> None:
        slug = self._slug(repository)
        metadata = {
            "repository": repository,
            "issues": [issue.model_dump() for issue in issues],
        }
        (self.data_dir / f"{slug}.json").write_text(json.dumps(metadata, indent=2))
        np.save(self.data_dir / f"{slug}.npy", embeddings)

    def load(self, repository: str) -> tuple[list[Issue], NDArray[np.float32]]:
        slug = self._slug(repository)
        metadata_path = self.data_dir / f"{slug}.json"
        embeddings_path = self.data_dir / f"{slug}.npy"
        if not metadata_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(f"Repository '{repository}' has not been imported")
        metadata = json.loads(metadata_path.read_text())
        issues = [Issue.model_validate(record) for record in metadata["issues"]]
        embeddings = np.load(embeddings_path, allow_pickle=False)
        return issues, embeddings

