# The Unofficial Guide — Project 1

## Domain

Student reviews of Northwestern University sourced from Niche.com, covering academics, social life, dining, housing, campus culture, weather, and cost. Reviews span the past 4 years and represent unfiltered student opinions. This knowledge is valuable and hard to find through official channels because Northwestern's own materials are promotional. These honest assessments of what daily life is actually like only exist in student-generated content like these reviews.

## Document Sources

| #       | Source                    | Type | URL or file path                                                |
| ------- | ------------------------- | ---- | --------------------------------------------------------------- |
| 1 - 100 | Niche.com Student Reviews | .txt | https://www.niche.com/colleges/northwestern-university/reviews/ |

## Chunking Strategy

**Chunk size:** 500 characters

**Overlap:** 50 characters

**Why these choices fit your documents:** Niche reviews vary in length from ~200 characters (short opinions) to ~1000 characters (detailed multi-topic reviews). Recursive character splitting is used over fixed-size chunking because it respects natural sentence and paragraph boundaries, splitting only when necessary and preferring cleaner breaks over random cuts mid-sentence. A 500-character cap keeps each chunk focused on one coherent thought while accommodating the average review length without splitting it at all. A 50-character overlap ensures that context is not lost at chunk boundaries for the minority of longer reviews that get split into two chunks. Semantic chunking was considered but rejected because it is computationally expensive and unnecessary for short reviews. Strings are stripped of whitespaces and newlines for preprocessing.

**Final chunk count:** 130

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** all-MiniLM-L6-v2 via sentence-transformers. It runs locally with no API key or rate limits.

**Production tradeoff reflection:** For a production deployment, the main tradeoffs to consider would be accuracy vs. cost vs. latency. all-MiniLM-L6-v2 is fast and free but was trained on general text. It would be more effective to use a domain-specific model fine-tuned on student review data which would likely produce better semantic matches. OpenAI's text-embedding-3-large would offer higher accuracy at the cost of per-call API fees and latency. If the system needed to serve international students, a multilingual model like paraphrase-multilingual-MiniLM-L12-v2 would be worth considering despite its larger size. At scale, local models are lower accuracy but cheaper.

## Grounded Generation

**System prompt grounding instruction:**

```
You are an assistant answering questions about Northwestern University using
student reviews. Answer the question using only the context provided and do
not use any outside knowledge. If the reviews express differing or conflicting
opinions, summarize the full range of views rather than picking one side. Cite
the source filename(s) for each claim you make, e.g. (review12.txt). If the
context does not contain enough information to answer, respond exactly with:
"I do not have the information to answer that."
```

**How source attribution is surfaced in the response:**

The system prompt instructs the model to cite the source filename in parentheses (e.g., `(review12.txt)`) next to each claim in its answer. `app.py` also collects the set of source filenames from the retrieved chunks and appends a "Sources" list below the generated answer, so attribution doesn't only rely on the model following instructions correctly.


## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| #   | Question                                                               | Expected answer                                                                                  | System response (summarized)                                                                                                | Retrieval quality | Response accuracy |
| --- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------- |
| 1   | What do people say about Northwestern during winter?                   | It gets really cold and depressing.                                                              | Winters are cold, can be challenging, and there are limited dorm options.                                                   | Relevant          | Accurate          |
| 2   | What do students say about the quarter system?                         | It's fast paced but opportunity for more classes.                                                | Mixed opinions: some think it allows them to explore their interests and others find it hard to adjust to and overwhelming. | Relevant          | Accurate          |
| 3   | What is the difference in social scene between North and South campus? | North campus is louder and has more parties and South campus is quieter.                         | North area has more of a party scene, while south area is quieter.                                                          | Relevant          | Accurate          |
| 4   | What do students say about the cost of attending Northwestern?         | It is very expensive. Some felt it was worth it and other's didn't.                              | It is expensive and financial aid staff is not helpful.                                                                     | Relevant          | Accurate          |
| 5   | What do students say about professors at Northwestern?                 | Professors are generally caring and willing to help students academically and with career goals. | They are knowledgeable and try to help the students.                                                                        | Relevant          | Accurate          |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

## Failure Case Analysis

**Question that failed:** How do I like the lakefill?

**What the system returned:** A response saying that I have a positive opinion of the lakefill but then notes that answers are based on reviews and not the user.

**Root cause (tied to a specific pipeline stage):** Generation

**What you would change to fix it:** Add a system prompt to clearly state that the user did not write any of the reviews and should not be assumed to be the author when using the pronoun "I". The initial response should be that the files are only from reviewers and to clarify what the user meant.


## Spec Reflection

**One way the spec helped you during implementation:** The spec helped me split up each part of implementation and only focus on that part until it was done. It allowed me to isolate the ingesting and chunking from the retrieving from the generating.

**One way your implementation diverged from the spec, and why:** The Chunking Strategy section of planning.md does not mention text cleanup. `load_documents()` in `ingest_and_chunk.py` strips and collapses whitespace before chunking. This was added because the review files had inconsistent line breaks and extra whitespace from being copy-pasted from Niche.com, which would have produced chunks with awkward mid-sentence breaks.


## AI Usage

**Instance 1**

- _What I gave the AI:_ The Chunking Strategy section of planning.md and a request to implement `load_documents()` and `chunk_documents()` in `ingest_and_chunk.py`.
- _What it produced:_ A `load_documents()` function that read each `.txt` file into a LangChain `Document` and a `chunk_documents()` function using `RecursiveCharacterTextSplitter` with the specified 500/50 chunk size and overlap.
- _What I changed or overrode:_ After printing sample chunks, I noticed messy whitespace and line breaks from the raw review files. I had the AI add a whitespace-normalization step (`re.sub(r"\s+", " ", text).strip()`) to `load_documents()` that wasn't in the original spec.

**Instance 2**

- _What I gave the AI:_ The Retrieval Approach section of planning.md and a request to implement `build_vectorstore()`, `load_vectorstore()`, and `retrieve()` in `embed.py`.
- _What it produced:_ An implementation using `Chroma.from_documents()` with all-MiniLM-L6-v2 embeddings, using a local `chroma_db` directory, and a `retrieve()` function returning the top-k similar chunks.
- _What I changed or overrode:_ `Chroma.from_documents()` appended to the existing persisted collection every time the script ran, so re-running it doubled the stored chunks and retrieval returned duplicate results for the same review. I had the AI change `build_vectorstore()` to call `vectorstore.reset_collection()` before `add_documents()` so the collection is rebuilt each time.
