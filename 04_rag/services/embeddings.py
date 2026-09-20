# services/embeddings.py
import requests
from typing import List
from core.config import settings

def generate_embedding(text: str) -> List[float]:
    """
    Topic 7: Embedding
    Takes a string of text and calls the local Ollama API to generate a vector embedding.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        # Ollama returns the vector in the 'embedding' key
        if "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError("Unexpected response format from Ollama API: 'embedding' key missing.")
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Ollama at {url}. Is Ollama running? Error: {str(e)}")

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Helper function to generate embeddings for multiple chunks sequentially.
    (Ollama's standard /api/embeddings endpoint processes one prompt at a time).
    """
    return [generate_embedding(text) for text in texts]
