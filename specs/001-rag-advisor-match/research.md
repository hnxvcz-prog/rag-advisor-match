# Research: Rag Advisor Match

## Document Parsing
- **Decision**: `python-docx` for text extraction, paired with `langchain` + OpenAI Structured Outputs for metadata extraction.
- **Rationale**: The spec requires converting Word to unstructured text AND structured JSON. Since Word files lack native semantic tags, we extract raw text with `python-docx` and pass it to an LLM strictly bound by `pydantic` schemas to reliably extract `expertise`, `clients`, `communication_style`.
- **Alternatives considered**: Regular expressions (rejected due to fragility against variable Word formats).

## Vector Storage
- **Decision**: `faiss-cpu` (Facebook AI Similarity Search).
- **Rationale**: Extremely lightweight, runs completely in-memory, no external server required. Perfect for a 10-document MVP without the overhead of Docker or separate services.
- **Alternatives considered**: ChromaDB, Pinecone (overkill for 10 docs).

## Frontend Framework
- **Decision**: `Streamlit`.
- **Rationale**: Fastest way to build a UI for Python data/AI applications without writing custom React/HTML/CSS. Highly suitable for LLM demos.
- **Alternatives considered**: FastAPI + React (too high overhead for an MVP demo), Gradio (less flexible layout than Streamlit).

## LLM Gateway
- **Decision**: `langchain-openai`.
- **Rationale**: Provides out-of-the-box support for RAG flows, embeddings generation, and structured outputs generation (Tool Calling / Structured Output APIs).

## Future Optimization: Re-Ranking (Phase 2/3)
- **Proposed Enhancement**: `Cross-Encoder Reranker` (e.g., Cohere Rerank, BGE-Reranker).
- **Rationale**: Currently, Phase 1 relies on FAISS vector similarities (Dense Embeddings with Bi-Encoder logic), which merges the scores via mathematical average (50/50). A Cross-Encoder performs late-interaction processing by feeding both the `User Query` and the `Advisor Document` into the transformer simultaneously. This provides a much deeper, word-by-word semantic evaluation.
- **Implementation Strategy**: 
  1. Retrieve a broader initial pool (e.g., Top 10-20) from the FAISS indexing phase.
  2. Implement mathematical hard filters (branch, asset scale).
  3. Send the filtered pool to the API (like `Cohere Rerank`) or run a local lightweight model (`BAAI/bge-reranker-v2-m3`).
  4. Yield the final Top 3 based on the fine-grained Cross-Encoder scoring.
- **Expected Outcome**: Drastic improvement in retrieval precision, reducing cases where advisors are mismatched based purely on overlapping but contextually disconnected vocabulary.
