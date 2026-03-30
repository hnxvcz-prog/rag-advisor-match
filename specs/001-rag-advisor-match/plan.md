# Implementation Plan: Rag Advisor Match MVP

**Branch**: `001-rag-advisor-match` | **Date**: 2026-03-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-rag-advisor-match/spec.md`

## Summary

This MVP validates the end-to-end pipeline of natural language advisor matching. The technical approach involves using Python for document processing (`python-docx`), Pydantic + LLM structured outputs to enforce JSON extraction of criteria from unstructured Word docs. For indexing, we will use in-memory `FAISS` with `langchain-openai` embeddings. A lightweight frontend will be built using `Streamlit` to collect user queries and render LLM-generated explanations alongside source evidence.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: `langchain`, `langchain-openai`, `faiss-cpu`, `python-docx`, `pydantic`, `streamlit`
**Storage**: In-memory FAISS and Local JSON files. No external databases required.
**Testing**: `pytest` for unit testing the scoring and matching logic.
**Target Platform**: Local executable (Web interface via Streamlit)
**Project Type**: RAG Web Application MVP
**Performance Goals**: < 15s latency for end-to-end query processing
**Constraints**: 10 Word docs limit; assumes OpenAI API access.
**Scale/Scope**: Proof of concept; single user session at a time.

## Constitution Check

*GATE: Passed. No specific constitution file rules violate this lightweight POC.*

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-advisor-match/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── quickstart.md        # Phase 1 output
```

### Source Code (repository root)

```text
src/
├── core/
│   ├── document_parser.py    # Reads Word docs and extracts JSON/Text
│   ├── indexer.py            # Manages FAISS embeddings
│   ├── query_parser.py       # Converts NL to standard requirements
│   ├── matcher.py            # Scoring logic (Semantic + Weighted fields)
│   └── generator.py          # LLM reasoning with citations
├── models/
│   └── schemas.py            # Pydantic data models
└── ui/
    └── app.py                # Streamlit frontend

data/
├── raw/                      # 10 Word documents
└── processed/                # JSON metadata and parsed text
```

**Structure Decision**: A modular Python project with a Streamlit UI entry point. Separating `core/` logic from `ui/` ensures backend components can be tested independently.
