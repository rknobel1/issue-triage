from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from app.config import settings


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = settings.embedding_model):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()

