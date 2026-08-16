# Hybrid Search + Reranking + Generation — End-to-End RAG Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** system that combines **hybrid search (sparse + dense retrieval)**, **neural reranking**, and **LLM-based generation** to answer questions grounded in your own documents.

Built with **LangChain**, **ChromaDB**, **BM25**, and **Cohere's** embedding, rerank, and chat models.

---

## Why this project

Most basic RAG tutorials stop at "embed chunks → similarity search → stuff into a prompt." That's fragile:

- **Pure dense retrieval** misses exact keyword/terminology matches (e.g., acronyms, proper nouns, numbers).
- **Pure keyword search (BM25)** misses semantically related phrasing.
- **Raw retrieval scores** from embeddings aren't true relevance judgments — they're just vector distances.

This project addresses all three problems by chaining three retrieval-quality techniques together **before** the LLM ever sees the context:

```
Hybrid Search  →  fixes recall (keyword + semantic coverage)
Reranking      →  fixes precision (true relevance scoring)
Generation     →  produces the final grounded answer
```

---

## Architecture / Flow

```
                         ┌─────────────────────────┐
                         │        PDF Corpus        │
                         └────────────┬─────────────┘
                                      │  pdfminer.six (text extraction)
                                      ▼
                         ┌─────────────────────────┐
                         │  RecursiveCharacterText   │
                         │  Splitter (2048 / 512)    │
                         └────────────┬─────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                             ▼
     ┌────────────────────┐                       ┌───────────────────────┐
     │   BM25Retriever      │                       │  Cohere Embeddings     │
     │   (sparse/keyword)    │                       │  (embed-english-v3.0)  │
     └──────────┬───────────┘                       └───────────┬───────────┘
                │                                                 ▼
                │                                     ┌───────────────────────┐
                │                                     │   ChromaDB (dense)     │
                │                                     └───────────┬───────────┘
                └───────────────────┬─────────────────────────────┘
                                    ▼
                     ┌───────────────────────────────┐
                     │   EnsembleRetriever             │
                     │   Hybrid fusion (weights 0.3/0.7)│
                     └────────────────┬───────────────┘
                                      ▼
                     ┌───────────────────────────────┐
                     │   CohereRerank (rerank-v4.0-pro)│
                     │   via ContextualCompression      │
                     │   Retriever                       │
                     └────────────────┬───────────────┘
                                      ▼
                     ┌───────────────────────────────┐
                     │   Top-K Reranked Context         │
                     └────────────────┬───────────────┘
                                      ▼
                     ┌───────────────────────────────┐
                     │   Prompt Template                │
                     │   {context} + {question}         │
                     └────────────────┬───────────────┘
                                      ▼
                     ┌───────────────────────────────┐
                     │   ChatCohere (command-a-plus)    │
                     └────────────────┬───────────────┘
                                      ▼
                     ┌───────────────────────────────┐
                     │   StrOutputParser → Final Answer │
                     └───────────────────────────────┘
```

**Indexing (offline, run once per corpus):**
`PDF → text extraction → chunking → embeddings → ChromaDB` (dense) and `PDF → text extraction → chunking → BM25 index` (sparse), built in parallel from the same source documents.

**Querying (runtime, per user question):**
`Question → EnsembleRetriever (hybrid fusion of BM25 + Chroma) → CohereRerank (relevance rescoring) → top reranked chunks → prompt template → ChatCohere → parsed answer.`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain (`RunnableParallel`, `ChatPromptTemplate`, LCEL chains) |
| Sparse retrieval | `BM25Retriever` (`rank_bm25`) |
| Dense retrieval | `ChromaDB` + Cohere `embed-english-v3.0` |
| Hybrid fusion | `EnsembleRetriever` (weighted reciprocal rank fusion) |
| Reranking | Cohere `rerank-v4.0-pro` via `ContextualCompressionRetriever` |
| Generation | Cohere `ChatCohere` (`command-a-plus-05-2026`) |
| PDF parsing | `pdfminer.six` |
| Data inspection | `pandas` |

---

## Project Structure

```
hybrid-rag-rerank-cohere/
├── RAG_Hybrid_Search_Rerank_EndToEnd.ipynb   # Main end-to-end pipeline (Colab-ready)
├── docs/                                       # (optional) sample PDFs used for the demo
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/hybrid-rag-rerank-cohere.git
cd hybrid-rag-rerank-cohere
```

### 2. Install dependencies
```bash
pip install langchain-cohere langchain langchain-classic pdfminer.six chromadb rank_bm25 langchain-community langchain-text-splitters
```

### 3. Set your Cohere API key
Get a free key from [dashboard.cohere.com](https://dashboard.cohere.com/api-keys), then:

```bash
export COHERE_API_KEY="your-api-key-here"
```

> If running in **Google Colab**, add `COHERE_KEY` under *Secrets* (🔑 icon in the left sidebar) instead — the notebook reads it via `google.colab.userdata`.

### 4. Add your documents
Drop your PDF(s) into the working directory (or `/content/` if using Colab) and update the file paths in the ingestion cell.

### 5. Run the notebook
Open `RAG_Hybrid_Search_Rerank_EndToEnd.ipynb` and run all cells top to bottom. Each section is self-contained and prints/validates its own output.

---

## Example Usage

```python
response = chain.invoke("What is Self-Attention in Transformers?")
print(response)
```

```python
# Batch multiple questions at once
responses = chain.batch([
    "What is YOLO?",
    "How is Transformer different from YOLO?"
])
```

```python
# Stream the answer token-by-token
for chunk in chain.stream("What are the 3 vectors in the Transformer architecture?"):
    print(chunk, end="", flush=True)
```

---

## Key Design Decisions

- **Hybrid retrieval over single-method search** — BM25 catches exact terminology/keyword matches that embeddings can blur together; Chroma's dense vectors catch paraphrased/semantically similar queries. `EnsembleRetriever` fuses both result sets with tunable weights (`0.3` BM25 / `0.7` dense) rather than picking one strategy.
- **Reranking sits on top of the hybrid retriever, not a single retriever** — the reranker's `base_retriever` is the `EnsembleRetriever` itself, so every candidate reaching the LLM has already survived both a keyword *and* a semantic filter before being relevance-scored by a dedicated cross-encoder-style model (`rerank-v4.0-pro`). This measurably improves precision over similarity-score-only ranking.
- **Composable LCEL chain** — retrieval, prompting, generation, and output parsing are wired with LangChain's `RunnableParallel` and pipe (`|`) syntax, making each stage independently swappable (e.g., drop in a different LLM or vector store without touching the rest of the pipeline).
- **Validation at every stage** — retrieval and reranking outputs are inspected via `pandas` DataFrames before ever reaching the LLM, which is critical for debugging silent retrieval-quality issues (a common failure mode in production RAG systems).
- **Three invocation patterns demonstrated** — `.invoke()`, `.batch()`, and `.stream()` show awareness of latency/throughput tradeoffs for real-world deployment (single request vs. batch jobs vs. user-facing streaming UX).

---

## Possible Extensions

- Swap `ChatCohere` for another LLM provider (OpenAI, Anthropic, local models) — LCEL makes this a one-line change.
- Add conversational memory for multi-turn follow-up questions.
- Add an evaluation harness (e.g., RAGAS) to quantitatively measure retrieval precision/recall before and after reranking.
- Deploy as a FastAPI service or Streamlit app for interactive querying.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
