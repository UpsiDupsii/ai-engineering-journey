import urllib.request
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.embeddings import (
    normalize_vector,
    dot_product,
    cosine_similarity,
    euclidean_distance,
)

class EmbeddingService:
    def __init__(self):
        # Reads the default model directly from app.core.config (settings)
        self.model_name = settings.DEFAULT_EMBEDDING_MODEL
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader: loads the model into memory only when needed."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Converts text into a dense vector."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_image_from_url(self, image_url: str) -> list[float]:
        """Downloads an image and converts it into a dense vector."""
        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            img = Image.open(response).convert('RGB')
        
        embedding = self.model.encode(img)
        return embedding.tolist()

    def compare_raw_vectors(self, vec_a: list[float], vec_b: list[float]) -> dict:
        """Calculates vector metrics using our core/embeddings math logic."""
        a = np.array(vec_a)
        b = np.array(vec_b)

        # Check if vectors are already unit-normalized (|v| == 1.0)
        norm_a_len = np.linalg.norm(a)
        norm_b_len = np.linalg.norm(b)
        is_norm = np.isclose(norm_a_len, 1.0) and np.isclose(norm_b_len, 1.0)

        return {
            "dot_product": dot_product(a, b),
            "cosine_similarity": cosine_similarity(a, b),
            "euclidean_distance": euclidean_distance(a, b),
            "is_normalized": bool(is_norm)
        }

# Global singleton instance
embedding_service = EmbeddingService()
