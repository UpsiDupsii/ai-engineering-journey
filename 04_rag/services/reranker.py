# services/reranker.py
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    # We load a fast, pre-trained model optimized for passage ranking (MS MARCO dataset)
    # The first time this runs, it will download the model weights (a few hundred MBs).
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
except ImportError:
    reranker_model = None
    print("Warning: sentence-transformers is not installed. Run 'pip install sentence-transformers'.")

def rerank_documents(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Topic 12: Reranking
    Uses a Cross-Encoder to deeply analyze the relationship between the query and each chunk,
    returning the chunks sorted by true relevance.
    """
    if not retrieved_chunks or reranker_model is None:
        return retrieved_chunks

    # The CrossEncoder expects data as pairs: [[query, text1], [query, text2], ...]
    query_passage_pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    
    # Predict the relevance scores
    scores = reranker_model.predict(query_passage_pairs)
    
    # Attach the new scores to our chunks
    for idx, chunk in enumerate(retrieved_chunks):
        chunk["cross_encoder_score"] = float(scores[idx])
        
    # Sort descending (for this specific model, higher scores mean higher relevance)
    reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["cross_encoder_score"], reverse=True)
    
    return reranked_chunks