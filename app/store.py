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

    def upsert(
        self,
        repository: str,
        issues: list[Issue],
        embeddings: NDArray[np.float32],
    ) -> int:
        """Append new issues and replace existing issue metadata and vectors."""
        if len(issues) != len(embeddings):
            raise ValueError("Issue and embedding counts must match")
        try:
            current_issues, current_embeddings = self.load(repository)
        except FileNotFoundError:
            self.save(repository, issues, embeddings)
            return len(issues)

        if len(current_embeddings) and len(embeddings):
            if current_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError("New embeddings use a different vector dimension")

        combined = {
            issue.number: (issue, current_embeddings[index])
            for index, issue in enumerate(current_issues)
        }
        for index, issue in enumerate(issues):
            combined[issue.number] = (issue, embeddings[index])

        ordered = sorted(combined.values(), key=lambda item: item[0].number)
        merged_issues = [item[0] for item in ordered]
        merged_embeddings = np.asarray([item[1] for item in ordered], dtype=np.float32)
        self.save(repository, merged_issues, merged_embeddings)
        return len(issues)
