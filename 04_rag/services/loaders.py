# services/loaders.py
import os
from typing import List
from schemas.document import LoadedDocument

def load_document(file_path: str) -> List[LoadedDocument]:
    """
    Topic 2: Loaders
    Reads a raw file from the disk and extracts its text and metadata.
    Returns a list of LoadedDocument objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    loaded_documents: List[LoadedDocument] = []

    # Handle standard Text files
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            doc = LoadedDocument(
                page_content=text,
                metadata={"source": file_path, "file_type": "txt"}
            )
            loaded_documents.append(doc)
            
    # Handle PDF files
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            
            # In PDFs, it's a standard practice to load each page as a separate document
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                # Only append if the page actually contains text
                if text and text.strip():
                    doc = LoadedDocument(
                        page_content=text,
                        metadata={
                            "source": file_path, 
                            "file_type": "pdf", 
                            "page": page_num + 1
                        }
                    )
                    loaded_documents.append(doc)
        except ImportError:
            raise ImportError("The 'pypdf' library is required to load PDFs. Run: pip install pypdf")
            
    else:
        raise ValueError(f"Unsupported file format: {ext}. Currently supporting .txt and .pdf")
        
    return loaded_documents
