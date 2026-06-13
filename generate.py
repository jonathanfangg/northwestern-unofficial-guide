"""Generation pipeline using Groq's llama-3.3-70b-versatile."""

import os

from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
NO_RESULTS_RESPONSE = "I do not have the information to answer that."

SYSTEM_PROMPT = (
    "You are an assistant answering questions about Northwestern "
    "University using student reviews. Answer the question using only the "
    "context provided and do not use any outside knowledge. If the reviews "
    "express differing or conflicting opinions, summarize the full range of "
    "views rather than picking one side. Cite the source filename(s) for "
    "each claim you make, e.g. (review12.txt). If the context does not "
    f"contain enough information to answer, respond exactly with: "
    f'"{NO_RESULTS_RESPONSE}"'
)


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
    context = "\n\n".join(
        f"[{chunk.metadata['source']}]\n{chunk.page_content}" for chunk in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above."
    )


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
    if not chunks:
        return NO_RESULTS_RESPONSE

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)},
        ],
    )
    return response.choices[0].message.content


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
