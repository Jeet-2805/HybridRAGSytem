# Hybrid Search RAG System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over a PDF by combining keyword search (BM25) with dense vector search, then reranking and self-checking its own retrieval before generating an answer with Groq's LLaMA 3.3.

## How it works

1. **Ingest** — the source PDF is loaded and split into ~300-character chunks (`PyMuPDFLoader` + `RecursiveCharacterTextSplitter`), embedded with `all-MiniLM-L6-v2`, and stored in a persistent ChromaDB collection.
2. **Hybrid retrieval** — every query is scored two ways: BM25 keyword matching and cosine similarity on the embeddings. Both scores are normalized to a 0–1 range before being blended (`alpha` controls the mix), since raw BM25 and cosine-similarity scores aren't on comparable scales.
3. **Multi-query expansion** — the LLM rewrites the question into a few alternate phrasings before retrieving, to catch chunks a single phrasing might miss.
4. **Reranking** — retrieved chunks are deduplicated and reranked with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) to push the most relevant ones to the top.
5. **Self-check & retry** — an LLM judge checks whether the retrieved context actually answers the question; if not, the system retries with a wider search before giving up.
6. **Generation** — the final context goes to Groq's `llama-3.3-70b-versatile`, with the system prompt constrained to answer only from retrieved context.

## Example

Asked *"What is File Handling in Java, with an example?"*, the system retrieves the relevant chunks from the source PDF and returns a grounded answer that correctly explains `FileInputStream`/`FileOutputStream` with a working code snippet — rather than falling back on the model's general knowledge.

## Setup

```bash
pip install -r requirements.txt
```

Open `hybridsearch.ipynb` and run the cells top to bottom. The first run builds the ChromaDB index (`hybrid_search/`, gitignored) from your source PDF — point the ingestion cell at any PDF of your own to reuse this on different content.

## Stack

Python · LangChain · ChromaDB · Sentence-Transformers · BM25 (`rank_bm25`) · Groq (LLaMA 3.3)
