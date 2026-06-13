"""Gradio app."""

import os

import gradio as gr

from embed import build_vectorstore, load_vectorstore, retrieve
from generate import generate_answer
from ingest_and_chunk import chunk_documents, load_documents

PERSIST_DIRECTORY = "chroma_db"


def get_vectorstore():
    """Load the persisted vector store, building it first if it doesn't exist yet.

    Checks for an existing ChromaDB collection on disk. If found, loads it
    directly via load_vectorstore(). Otherwise, runs the full ingestion and
    chunking pipeline and builds a new vector store via build_vectorstore().

    Returns:
        The ChromaDB vector store instance.
    """
    if os.path.isdir(PERSIST_DIRECTORY):
        return load_vectorstore()
    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


def answer_question(query: str) -> str:
    """Answer a user's question about Northwestern using the RAG pipeline.

    Retrieves the top-k most relevant review chunks for the query, generates
    a grounded answer with source attribution via the Groq LLM, and formats
    the answer alongside a list of cited source files for display.

    Args:
        query: The user's natural language question.

    Returns:
        A formatted string containing the generated answer and its sources.
    """
    chunks = retrieve(query, vectorstore)
    answer = generate_answer(query, chunks)
    sources = sorted({chunk.metadata["source"] for chunk in chunks})
    sources_text = "\n".join(f"- {source}" for source in sources)
    return f"{answer}\n\n**Sources:**\n{sources_text}"


vectorstore = get_vectorstore()

NORTHWESTERN_PURPLE = "#4E2A84"

CSS = f"""
.gradio-container {{
    background-color: {NORTHWESTERN_PURPLE} !important;
}}
* {{
    color: #FFFFFF !important;
}}
input, textarea {{
    background-color: #FFFFFF !important;
    color: #000000 !important;
}}
#submit-btn {{
    display: none !important;
}}
"""

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question about Northwestern", lines=2),
    outputs=gr.Markdown(label="Answer"),
    title="The Unofficial Guide to Northwestern",
    description="Ask questions about academics, social life, dining, housing, and more, based on real student reviews. Press Enter to submit.",
    submit_btn=gr.Button("Submit", elem_id="submit-btn"),
    clear_btn=None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(css=CSS)
