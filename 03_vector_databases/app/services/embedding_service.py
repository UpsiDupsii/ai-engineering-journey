import torch
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, Dict, List
from collections import Counter

# We choose a 768-dim model matching our DENSE_DIM=768 setting
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(f"Loading embedding model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Put model in inference mode

def get_dense_embedding(text: str) -> List[float]:
    """
    Converts text to a 768-dim normalized dense vector using mean pooling.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Token embeddings from last hidden layer
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Mean pooling
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Normalize vector (so cosine similarity is straightforward dot product)
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized[0].tolist()

def get_sparse_embedding(text: str) -> Dict[int, float]:
    """
    Converts text into a dictionary of {token_id: frequency} for sparse keyword matching.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return {}
    
    # Calculate term frequency (TF)
    counts = Counter(token_ids)
    total_tokens = len(token_ids)
    
    # Map token_id to weight
    sparse_vector = {int(token_id): float(count / total_tokens) for token_id, count in counts.items()}
    return sparse_vector

def generate_embeddings(text: str) -> Tuple[List[float], Dict[int, float]]:
    """Helper that returns both dense and sparse representations."""
    return get_dense_embedding(text), get_sparse_embedding(text)
