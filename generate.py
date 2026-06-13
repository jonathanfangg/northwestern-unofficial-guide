"""Generation pipeline using Groq's llama-3.3-70b-versatile."""

from langchain_core.documents import Document

GROQ_MODEL = "llama-3.3-70b-versatile"
NO_RESULTS_RESPONSE = "I don't have enough information to answer that."


def build_prompt(query: str, chunks: list[Document]) -> str:
    """Build a grounded prompt from the user query and retrieved chunks.

    Combines the retrieved review chunks into a context block (each labeled
    with its source filename) and instructs the model to answer only using
    that context, to represent the full range of opinions when reviews
    disagree rather than defaulting to the majority view, and to cite source
    filenames for claims made in its answer.

    Args:
        query: The user's natural language question.
        chunks: List of retrieved Document chunks to use as context, typically
            the output of retrieve() from embed.py.

    Returns:
        A formatted prompt string ready to send to the Groq chat completion API.
    """


def generate_answer(query: str, chunks: list[Document], model: str = GROQ_MODEL) -> str:
    """Generate a grounded answer to the query using Groq's llama-3.3-70b-versatile.

    If no chunks were retrieved, returns NO_RESULTS_RESPONSE without calling
    the LLM. Otherwise, builds a grounded prompt via build_prompt() and sends
    it to the Groq chat completion API, returning the model's response text.

    Args:
        query: The user's natural language question.
        chunks: List of retrieved Document chunks to use as context, typically
            the output of retrieve() from embed.py.
        model: Name of the Groq model to use for generation.

    Returns:
        The generated answer text, including source attribution.
    """


if __name__ == "__main__":
    from ingest_and_chunk import load_documents, chunk_documents
    from embed import build_vectorstore, retrieve

    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)

    query = "What do students say about professors at Northwestern?"
    retrieved = retrieve(query, vectorstore)
    answer = generate_answer(query, retrieved)
    print(answer)
