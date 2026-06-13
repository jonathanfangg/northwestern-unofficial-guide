"""Ingestion and chunking pipeline."""

import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(documents_dir: str = DOCUMENTS_DIR) -> list[Document]:
    """Load all 100 review .txt files from the documents directory into LangChain Documents.

    Reads each .txt file's contents, cleans up whitespace, and wraps it in a
    langchain_core.documents.Document with metadata identifying its source filename
    (e.g. {"source": "review1.txt"}).

    Args:
        documents_dir: Path to the directory containing the review .txt files.

    Returns:
        A list of 100 Document objects, one per review file.
    """
    documents = []
    for filename in sorted(os.listdir(documents_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(documents_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"\s+", " ", text).strip()
        documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Applies recursive character splitting to each Document, preserving and
    propagating source metadata to each resulting chunk so chunks can be
    traced back to their originating review file.

    Args:
        documents: List of Documents to split, typically the output of load_documents().
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of Document chunks with source metadata preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
