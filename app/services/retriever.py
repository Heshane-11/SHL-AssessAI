"""
FAISS vector retrieval service.
Loads the pre-built index and performs semantic top-k search.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.catalog import CatalogItem

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.pkl"


class VectorRetriever:
    """
    Singleton-style FAISS retriever.
    Load once, reuse across requests for low latency.
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._metadata: List[CatalogItem] = []
        self._loaded = False

    def _load(self) -> bool:
        """Lazy-load model and index on first use."""
        if self._loaded:
            return True

        index_dir = settings.faiss_index_dir
        index_path = index_dir / _INDEX_FILE
        meta_path = index_dir / _META_FILE

        if not index_path.exists() or not meta_path.exists():
            logger.error(
                "FAISS index not found at %s. Run scripts/build_index.py first.",
                index_dir,
            )
            return False

        try:
            logger.info("Loading embedding model: %s", settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)

            logger.info("Loading FAISS index from %s", index_path)
            self._index = faiss.read_index(str(index_path))

            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)

            self._loaded = True
            logger.info(
                "FAISS retriever ready. Index size: %d vectors.", self._index.ntotal
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def search(self, query: str, top_k: Optional[int] = None) -> List[CatalogItem]:
        """
        Semantic search over the FAISS index.
        Returns top_k CatalogItem objects sorted by similarity.
        """
        if not self._load():
            return []

        k = top_k or settings.top_k_results
        k = min(k, self._index.ntotal)

        try:
            query_vec = self._model.encode([query], convert_to_numpy=True).astype(
                np.float32
            )
            faiss.normalize_L2(query_vec)

            distances, indices = self._index.search(query_vec, k)
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._metadata):
                    results.append(self._metadata[idx])
            return results
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return []

    @property
    def is_ready(self) -> bool:
        return self._loaded or self._load()


# Module-level singleton
retriever = VectorRetriever()
