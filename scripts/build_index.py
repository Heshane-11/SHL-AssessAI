"""
Build FAISS index from catalog.json.

Run: python scripts/build_index.py
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.catalog import CatalogItem

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


def build_document(item: CatalogItem) -> str:
    """
    Concatenate all textual fields of a catalog item into one
    rich string for embedding. More signal = better retrieval.
    """
    skills = " ".join(item.skills_measured) if item.skills_measured else ""
    parts = [
        item.name,
        item.description,
        f"Test type: {item.test_type}",
        f"Duration: {item.duration or ''}",
        f"Skills: {skills}",
        f"Remote testing: {item.remote_testing or ''}",
    ]
    return " | ".join(p for p in parts if p.strip())


def main() -> None:
    catalog_path = settings.catalog_file
    index_dir = settings.faiss_index_dir

    if not catalog_path.exists():
        logger.error(
            "Catalog not found at %s. Run scripts/scrape_catalog.py first.", catalog_path
        )
        return

    # Load catalog
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items: List[CatalogItem] = [CatalogItem(**entry) for entry in raw]
    logger.info("Loaded %d catalog items.", len(items))

    # Build documents
    documents = [build_document(item) for item in items]

    # Encode
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)

    logger.info("Encoding %d documents…", len(documents))
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build FAISS flat index (exact search — fast enough for <10k items)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built. Total vectors: %d, Dimension: %d", index.ntotal, dim)

    # Persist
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(items, f)

    logger.info("Index saved to %s", index_dir)
    logger.info("Done! Run the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
