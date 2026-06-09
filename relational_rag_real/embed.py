"""Dense text embedding (bge-small on Apple MPS) shared by the retriever and verifier."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # ~33M params, 384-dim, runs on MPS
# bge-v1.5 retrieval requires a query instruction; documents are embedded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _model(name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts, *, model: str = DEFAULT_MODEL, batch_size: int = 256,
          normalize: bool = True, is_query: bool = False) -> np.ndarray:
    """Embed strings into an (n, d) float32 matrix (L2-normalized by default).

    ``is_query=True`` prepends the bge retrieval instruction (use for questions).
    """
    items = list(texts)
    if is_query:
        items = [BGE_QUERY_INSTRUCTION + t for t in items]
    vectors = _model(model).encode(
        items,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embedding_dim(model: str = DEFAULT_MODEL) -> int:
    return int(_model(model).get_sentence_embedding_dimension())
