"""Embedding pipeline."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "northwestern_reviews"
TOP_K = 5


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Embed document chunks and store them in a local ChromaDB collection.

    Uses the all-MiniLM-L6-v2 sentence-transformers model to embed each chunk's
    page_content, then persists the resulting vectors to a local ChromaDB
    collection along with each chunk's source metadata so retrieved results
    can be traced back to their originating review file.

    Args:
        chunks: List of Document chunks to embed, the output of
            chunk_documents() from ingest_and_chunk.py.
        persist_directory: Local directory where ChromaDB stores its data.
        collection_name: Name of the ChromaDB collection to create or update.
        embedding_model: Name of the sentence-transformers model to use for embedding.

    Returns:
        The ChromaDB vector store instance containing the embedded chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model}")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    return vectorstore


def load_vectorstore(
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Load an existing local ChromaDB vector store collection.

    Connects to a previously persisted ChromaDB collection using the same
    embedding model that was used to build it, without re-embedding any chunks.

    Args:
        persist_directory: Local directory where ChromaDB stores its data.
        collection_name: Name of the ChromaDB collection to load.
        embedding_model: Name of the sentence-transformers model used for embedding.

    Returns:
        The ChromaDB vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def retrieve(query: str, vectorstore, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a query using cosine similarity.

    Embeds the query with the same embedding model used to build the vector
    store, then performs a similarity search to return the most relevant
    chunks along with their source metadata for attribution.

    Args:
        query: The user's natural language question.
        vectorstore: The ChromaDB vector store to search, typically the output
            of build_vectorstore() or load_vectorstore().
        k: Number of top chunks to retrieve.

    Returns:
        A list of the k most relevant Document chunks, each with source metadata.
    """
    return vectorstore.similarity_search(query, k=k)
