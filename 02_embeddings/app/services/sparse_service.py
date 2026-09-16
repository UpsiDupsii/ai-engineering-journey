import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.embeddings import cosine_similarity
from app.services.embedding_service import embedding_service

class SparseEmbeddingService:
    def generate_sparse_vector(self, text: str) -> dict:
        """Generates a Sparse Vector (TF-IDF) representation.
        Instead of returning thousands of zeros, we return a dict of {word: weight}
        for words that actually appear.
        """
        # Fit vectorizer on the single input string
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])
        
        feature_names = vectorizer.get_feature_names_out()
        dense_array = tfidf_matrix.toarray()[0]
        
        # Filter out 0.0 values so we only hold non-zero keyword weights
        sparse_dict = {
            word: round(float(weight), 4)
            for word, weight in zip(feature_names, dense_array)
            if weight > 0
        }
        
        return {
            "vocabulary_size": len(feature_names),
            "sparse_vector": sparse_dict
        }

    def compare_sparse_vs_dense(self, query: str, document: str) -> dict:
        """Compares exact keyword overlap (Sparse) vs. semantic meaning (Dense)."""
        # 1. Sparse Similarity (TF-IDF)
        vectorizer = TfidfVectorizer()
        # Fit TF-IDF on both query and document together to share the same vocabulary space
        tfidf_matrix = vectorizer.fit_transform([query, document]).toarray()
        
        sparse_query_vec = tfidf_matrix[0]
        sparse_doc_vec = tfidf_matrix[1]
        
        sparse_sim = cosine_similarity(sparse_query_vec, sparse_doc_vec)

        # 2. Dense Similarity (CLIP / Neural Model)
        dense_query_vec = embedding_service.embed_text(query)
        dense_doc_vec = embedding_service.embed_text(document)
        
        dense_sim = cosine_similarity(
            np.array(dense_query_vec), 
            np.array(dense_doc_vec)
        )

        return {
            "query": query,
            "document": document,
            "dense_cosine_similarity": round(float(dense_sim), 4),
            "sparse_cosine_similarity": round(float(sparse_sim), 4)
        }

# Global singleton instance
sparse_service = SparseEmbeddingService()