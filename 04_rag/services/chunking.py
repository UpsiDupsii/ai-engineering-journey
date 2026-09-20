# services/chunking.py
from typing import List
from schemas.document import LoadedDocument
import uuid

def chunk_documents(
    documents: List[LoadedDocument], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[LoadedDocument]:
    """
    Topics 3-6: Parsing, Chunking, Chunk size, Chunk overlap.
    Takes a list of loaded documents and splits them into smaller overlapping chunks.
    """
    chunked_docs: List[LoadedDocument] = []
    
    for doc in documents:
        text = doc.page_content
        text_length = len(text)
        
        # If the text is smaller than the chunk size, just keep it as one chunk
        if text_length <= chunk_size:
            # We copy the metadata to avoid modifying the original by reference
            new_metadata = doc.metadata.copy()
            new_metadata["chunk_index"] = 0
            chunked_docs.append(LoadedDocument(page_content=text, metadata=new_metadata))
            continue
            
        # Sliding window chunking algorithm
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate the end position for the current chunk
            end = start + chunk_size
            
            # Extract the chunk
            chunk_text = text[start:end]
            
            # Create a new document for this chunk
            new_metadata = doc.metadata.copy()
            new_metadata["chunk_index"] = chunk_index
            
            chunked_docs.append(
                LoadedDocument(
                    page_content=chunk_text,
                    metadata=new_metadata
                )
            )
            
            # Move the start forward by (size - overlap)
            # If overlap is 200, we step back 200 characters from the end to start the next chunk
            start += (chunk_size - chunk_overlap)
            chunk_index += 1
            
    return chunked_docs


def parent_child_chunk_documents(
    documents: List[LoadedDocument], 
    parent_chunk_size: int = 1000, 
    child_chunk_size: int = 200,
    overlap: int = 20
) -> List[LoadedDocument]:
    """
    Topic 20: Small-to-Big Chunking
    Splits docs into large parent chunks, then splits those into smaller child chunks.
    Embeds the parent text into the child's metadata.
    """
    child_documents = []
    
    for doc in documents:
        text = doc.page_content
        
        # 1. Create Parent Chunks
        parent_chunks = [text[i:i+parent_chunk_size] for i in range(0, len(text), parent_chunk_size - overlap)]
        
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            # 2. Create Child Chunks from the Parent
            children = [parent_text[i:i+child_chunk_size] for i in range(0, len(parent_text), child_chunk_size - overlap)]
            
            for child_text in children:
                # Merge existing metadata with parent tracking data
                child_meta = doc.metadata.copy() if doc.metadata else {}
                child_meta.update({
                    "parent_id": parent_id,
                    "parent_text": parent_text, # Store the big context right inside the child!
                    "is_child": True
                })
                
                child_documents.append(
                    LoadedDocument(page_content=child_text, metadata=child_meta)
                )
                
    return child_documents

