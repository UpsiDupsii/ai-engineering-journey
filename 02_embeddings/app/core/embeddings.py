import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a 1D vector to unit length (L2 norm).
    ||v||_2 = sqrt(sum(v_i^2))
    """
    
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates standard dot product (sum of element-wise products)."""
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity: (a . b) / (||a|| * ||b||).
    Measures the cosine of the angle between two directional vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates straight-line L2 distance between two points in vector space."""
    return float(np.linalg.norm(a - b))

