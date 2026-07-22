# Agentic RAG Research Assistant — Implementation Plan

## Goal Definition

### What is being built?
A full-stack **Agentic RAG (Retrieval-Augmented Generation) Research Assistant** — a web application where users upload documents (PDF, DOCX, TXT) and ask natural-language questions, receiving source-grounded, cited answers produced by a multi-agent LangGraph pipeline. The system rewrites ambiguous queries, retrieves relevant chunks from a vector store, grades their relevance, checks for hallucinations, and generates answers with inline citations back to the source document and page number.

### What does "done" look like?
1. A FastAPI backend that accepts document uploads (`/upload`), answers questions (`/ask`), and exposes a health check (`/health`).
2. A LangGraph multi-agent pipeline with 4 core agents: Query Rewriter → Retriever → Relevance Grader → Answer Generator, plus an optional Hallucination Checker.
3. A ChromaDB-backed vector store with chunked, embedded document data and rich metadata (doc_id, page_number, chunk_id).
4. A Streamlit frontend with file upload, chat interface, expandable citation cards, agent step indicators, and error handling.
5. RAGAS evaluation comparing naive RAG vs. agentic RAG on 15–20 test Q&A pairs.
6. Docker packaging (`Dockerfile` + `docker-compose.yml`) for reproducible deployment.
7. Deployment-ready for HuggingFace Spaces / Render / Railway.

### What is explicitly out of scope?
- User authentication / persistent multi-session history beyond in-memory
- Streaming token-by-token responses
- Hybrid search (keyword + semantic)
- Multi-modal support (tables/images in PDFs)
- Feedback loop (thumbs up/down)
- Agent self-correction loop
- React frontend (we'll use Streamlit for speed; React is listed as a future enhancement)
- Production-scale Pinecone deployment (we'll use local ChromaDB)

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem alignment with LangChain, FastAPI, ML libraries |
| **Backend Framework** | FastAPI + Uvicorn | Auto-generated Swagger docs, async support, Pydantic validation |
| **Agent Orchestration** | LangChain + LangGraph | Explicit, debuggable state-machine-based agent pipeline |
| **Vector Store** | ChromaDB (local) | Zero cost, no external dependency, sufficient for portfolio scale |
| **Embeddings** | `text-embedding-3-small` (OpenAI) | Strong quality-to-cost ratio; fallback to `all-MiniLM-L6-v2` (HuggingFace) if zero-cost required |
| **LLM** | GPT-4o-mini (OpenAI) | Cost-effective, fast, sufficient for grading + generation tasks |
| **Document Parsing** | `pdfplumber` (PDF), `python-docx` (DOCX), built-in (TXT) | Lightweight, reliable for clean documents |
| **Text Splitting** | `RecursiveCharacterTextSplitter` (LangChain) | 500-token chunks, 50-token overlap, paragraph→sentence→word priority |
| **Frontend** | Streamlit | Fastest path to a demoable UI; no frontend build tooling needed |
| **Evaluation** | RAGAS | Direct LangChain integration, standard RAG metrics |
| **Observability** | LangSmith | Free tier, full agent step tracing |
| **Containerization** | Docker + docker-compose | Reproducible deployment, portfolio credibility |
| **Deployment Target** | HuggingFace Spaces (Docker SDK) | Free, community-recognized, simple |
| **Environment Management** | `venv` + `pip` | Standard Python workflow; `requirements.txt` for deps |
| **Config Management** | `python-dotenv` + `.env` | Standard secret handling; never committed to Git |

---

## Session Modularization

---

### Session 1: Project Scaffolding & Environment Setup

**Objective:** Create the full project directory structure, virtual environment, install all dependencies, configure environment variables, and verify everything runs.

**Scope:**
- Files created: full directory tree per Section 7.1 of the master doc
- `requirements.txt` with all dependencies
- `.env` template (`.env.example` committed, `.env` gitignored)
- `.gitignore`
- `config.py` — env var loading via `python-dotenv`
- `main.py` — bare FastAPI app with `/health` endpoint only
- `README.md` stub

**Output:** Running `uvicorn main:app --reload` starts the server, `GET /health` returns `{"status": "ok"}`, and all dependencies import without error.

**Connects To:** Session 2 depends on the directory structure, working environment, and `config.py` for API keys.

**Failure Surface:**
- Dependency version conflicts (especially `langchain` vs `langgraph` vs `langchain-openai` — these evolve rapidly)
- Python version mismatch (need 3.11+)
- API key misconfiguration (OpenAI key invalid or missing)

---

### Session 2: Document Ingestion Pipeline — Text Extraction & Chunking

**Objective:** Build the complete ingestion pipeline: accept a file, extract text, chunk it, generate embeddings, and store in ChromaDB with metadata.

**Scope:**
- `core/ingestion.py` — text extraction (PDF via `pdfplumber`, DOCX via `python-docx`, TXT via built-in read)
- Chunking with `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap)
- Page-number tracking during extraction (critical for citations)
- `core/vector_store.py` — ChromaDB client wrapper (create collection, add documents, query)
- Metadata schema: `{chunk_id, doc_id, doc_name, page_number, chunk_index, char_count, chunk_text}`

**Output:** A test script that loads a sample PDF, ingests it, and queries ChromaDB with a hardcoded question, printing top-k matching chunks with their metadata (including page numbers).

**Connects To:** Session 3 uses `vector_store.py` for retrieval and the metadata schema for citation generation. Session 4 wraps ingestion in the `/upload` endpoint.

**Failure Surface:**
- `pdfplumber` fails on complex/scanned PDFs — we'll handle gracefully with error messages, not crash
- Page number tracking breaks if PDF has no page structure (single-page exports) — fallback to `page_number: 0`
- ChromaDB collection name collisions across sessions — use session_id-scoped collections or filter by doc_id metadata

---

### Session 3: Basic RAG Chain (No Agents)

**Objective:** Build a simple retrieve-then-generate chain — the "naive RAG" baseline that we'll later upgrade with agents and also compare against in RAGAS evaluation.

**Scope:**
- `core/agents/retriever.py` — wraps ChromaDB similarity search, returns top-k chunks
- `core/agents/answer_generator.py` — takes chunks + question, calls LLM with a grounded prompt template ("Answer only using the provided context. If the answer isn't present, say so. Cite the page number.")
- A simple chain function that wires retriever → generator (no LangGraph yet)
- `models/schemas.py` — Pydantic models: `AskRequest`, `AskResponse`, `Citation`

**Output:** A callable function `run_basic_rag(question, doc_ids) -> AskResponse` that retrieves chunks, generates an answer, and returns citations. Verified with 5+ test questions against a real document.

**Connects To:** Session 4 exposes this as the `/ask` endpoint. Session 6 replaces this chain with the full LangGraph pipeline. Session 9 uses this as the "naive RAG" baseline for RAGAS comparison.

**Failure Surface:**
- LLM generates answers not grounded in context (prompt engineering issue) — iterate on the system prompt
- Citations don't map correctly to source chunks — ensure chunk metadata is passed through cleanly
- Top-k too low misses relevant info, too high adds noise — start with k=6, tune later

---

### Session 4: FastAPI Endpoints — `/upload` and `/ask`

**Objective:** Expose the ingestion pipeline and RAG chain as proper REST endpoints with validation, error handling, and structured responses.

**Scope:**
- `routers/upload.py` — `POST /upload` accepting multipart file + session_id, validating file type/size, calling ingestion pipeline, returning `{doc_id, filename, chunks_created, status}`
- `routers/ask.py` — `POST /ask` accepting JSON body (`AskRequest`), calling RAG chain, returning `AskResponse`
- `routers/health.py` — already done in Session 1, just wire into router
- `main.py` — register all routers
- File validation: type check (PDF/DOCX/TXT), size limit (20MB), error responses (422)
- `core/memory.py` — session-based chat history store (in-memory dict)

**Output:** Full API testable via Swagger UI at `/docs`. Upload a document, ask a question, get a cited answer — all via HTTP.

**Connects To:** Session 5 connects the Streamlit frontend to these endpoints. Session 6 swaps the RAG chain behind `/ask` with the LangGraph pipeline (endpoint interface stays the same).

**Failure Surface:**
- Large file uploads timing out — set appropriate timeout, async processing if needed
- Session management edge cases (expired sessions, concurrent uploads) — keep simple with in-memory dict, document limitations
- Multipart form parsing issues — test with real PDF files, not just curl

---

### Session 5: Minimal Streamlit Frontend (MVP)

**Objective:** Build a working Streamlit UI that connects to the FastAPI backend — file upload, chat interface, basic citation display.

**Scope:**
- `frontend/app.py` — main Streamlit entrypoint
- `frontend/components/upload_widget.py` — file upload with progress indication
- `frontend/components/chat_window.py` — chat message rendering (user + assistant bubbles)
- `frontend/components/citation_card.py` — basic citation display (doc name, page number, chunk text)
- `frontend/components/sidebar.py` — document list, session controls
- `frontend/utils/api_client.py` — wrapper functions for `/upload` and `/ask` endpoints

**Output:** A running Streamlit app where a user can: upload a document → see "indexed" confirmation → type a question → receive an answer with basic citations. End-to-end demo-ready.

**Connects To:** Session 8 polishes this UI significantly. Sessions 6–7 upgrade the backend behind the same API, so the frontend benefits without changes.

**Failure Surface:**
- Streamlit session state management quirks (reruns on every interaction) — use `st.session_state` carefully
- Backend connection issues (CORS, port conflicts) — configure backend to allow Streamlit's origin
- File upload size limits in Streamlit vs. FastAPI — align both to 20MB

---

### Session 6: LangGraph Agentic Pipeline — Query Rewriter + Relevance Grader

**Objective:** Replace the naive RAG chain with a full LangGraph state machine featuring Query Rewriter and Relevance Grader agents.

**Scope:**
- `core/agents/graph.py` — LangGraph `StateGraph` definition with `AgentState` TypedDict
- `core/agents/query_rewriter.py` — takes raw question + chat history, outputs standalone disambiguated query
- `core/agents/relevance_grader.py` — takes retrieved chunks + query, scores each as relevant/irrelevant, filters out irrelevant ones
- Wire the full pipeline: `rewrite → retrieve → grade → generate`
- Integrate chat history from `core/memory.py` into the query rewriter
- Configure LangSmith tracing to log each agent step
- Update `routers/ask.py` to call the LangGraph pipeline instead of the basic chain
- Add `agent_trace` field to `AskResponse` (list of agent names that fired)

**Output:** Multi-turn conversations work correctly (follow-up questions resolved). Irrelevant chunks are filtered before generation. LangSmith trace shows all 4 agent steps. The `/ask` endpoint returns `agent_trace` in the response.

**Connects To:** Session 7 adds the Hallucination Checker as a 5th node in this graph. Session 9 evaluates this pipeline vs. the naive baseline from Session 3.

**Failure Surface:**
- LangGraph API changes (evolves fast) — pin version in `requirements.txt`, verify against current docs
- Query rewriter over-rewrites (changes the question's meaning) — careful prompt engineering, test with edge cases
- Relevance grader too aggressive (filters out relevant chunks) or too lenient — tune the grading prompt and threshold
- Chat history context window overflow — limit history to last N turns

---

### Session 7: Hallucination Checker Agent

**Objective:** Add a 5th agent node that cross-checks the generated answer against the source chunks, flagging unsupported claims.

**Scope:**
- `core/agents/hallucination_checker.py` — takes generated answer + relevant chunks, verifies each claim is supported, returns a confidence assessment
- Update `graph.py` — add `hallucination_check` node after `generate`, with conditional routing (pass/flag)
- Update `AgentState` — add `hallucination_check_result` field
- Update `AskResponse` — add optional `hallucination_warning` field
- If flagged: return the answer with a warning ("⚠️ Some claims may not be fully supported by the source documents")

**Output:** Deliberately test with a question that should trigger the hallucination checker (e.g., ask about something partially in the document). The checker correctly flags unsupported claims.

**Connects To:** Session 8 displays hallucination warnings in the UI. Session 9 evaluates the impact on faithfulness scores.

**Failure Surface:**
- Hallucination checker is itself an LLM call — it can make mistakes (meta-hallucination)
- Adds latency to every request — measure and document the tradeoff
- False positives (correct answers flagged as hallucinated) annoy users — tune the prompt for precision over recall

---

### Session 8: Frontend Polish & Error Handling

**Objective:** Make the Streamlit UI demo-ready with polished UX, expandable citations, agent step indicators, error handling, and visual styling.

**Scope:**
- Expandable citation cards (collapsible, showing exact source text + page number)
- Agent step indicator during "thinking" state ("Rewriting query… Retrieving… Grading… Generating…")
- Hallucination warning display (from Session 7)
- Error handling UX: unsupported file type toast, "no relevant info found" message, API timeout handling
- Multi-document support in sidebar (upload multiple docs, optionally scope questions to specific docs)
- Visual styling pass: consistent colors, spacing, typography, clean layout
- Document filtering dropdown (scope questions to specific uploaded documents)

**Output:** A visually polished, demo-ready application. All error flows from Section 6.3/6.4 of the master doc are handled gracefully. The UI impresses on first look.

**Connects To:** Session 10 deploys this polished app. Session 9 uses it for demo recordings.

**Failure Surface:**
- Streamlit styling limitations (less control than React) — work within constraints, use `st.markdown` with custom CSS where needed
- Agent step indicator timing (hard to show real-time steps in Streamlit without websockets) — use status containers or progress bars
- State management complexity with multiple documents — keep session state schema clean

---

### Session 9: RAGAS Evaluation & Benchmarking

**Objective:** Quantitatively prove that the agentic pipeline outperforms naive RAG using RAGAS metrics.

**Scope:**
- `eval/test_set.json` — 15–20 question/ground-truth-answer pairs written against a test document
- `eval/run_evaluation.py` — script that runs both naive RAG and agentic RAG against the test set, collects RAGAS metrics
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Comparison table: naive vs. agentic, targeting the benchmarks from Section 11 of the master doc
- Generate a chart/visualization of the comparison
- Update `README.md` with the evaluation results

**Output:** A reproducible evaluation script, a comparison table showing agentic RAG outperforms naive RAG on all metrics, and the results documented in the README.

**Connects To:** Session 10 includes these results in deployment documentation and portfolio materials.

**Failure Surface:**
- RAGAS requires specific input format (question, answer, contexts, ground_truth) — follow their schema exactly
- Scores may not hit targets on first run — iterate on chunking strategy, prompts, top-k, grading threshold
- Ground truth answers must be carefully written to be fair evaluations — avoid answers that require info not in the document

---

### Session 10: Dockerization, Deployment & Documentation

**Objective:** Package the full application in Docker, deploy to a public URL, and finalize all documentation.

**Scope:**
- `Dockerfile` — multi-stage build for backend
- `docker-compose.yml` — backend + frontend services
- Deploy to HuggingFace Spaces (Docker SDK) or Render/Railway
- Finalize `README.md`: project overview, architecture diagram, setup instructions, eval results, demo link, "What I Tried and What Failed" section
- Record 2-minute demo video (Loom or similar)
- Clean up repo: remove dead code, verify `.gitignore`, clean commit history

**Output:** A publicly accessible deployed application with a single shareable URL. A professional README with architecture diagram, evaluation results, and demo video link.

**Connects To:** This is the final deliverable.

**Failure Surface:**
- Docker build failures due to system dependencies (ChromaDB C extensions, etc.) — test on clean environment
- HuggingFace Spaces resource limits (RAM, disk) — ChromaDB index size must fit within free tier
- Environment variable configuration in deployment platform — verify secrets are set correctly
- Cold start latency on free tier — document this in README

---

## Open Questions

**Domain framing:** The master doc recommends picking a domain (Legal, Academic, Corporate). Which domain should we frame this project around? This affects the test documents, Q&A evaluation pairs, UI copy/branding, and README narrative.

**Recommendation:** Academic (Research paper assistant for literature review) — natural fit, easy to source test documents, relatable to a wide audience.

**LLM Provider:** The master doc lists OpenAI (GPT-4o-mini) as primary, with Gemini and Ollama as alternatives. Which should we use?
- **OpenAI** — best ecosystem integration with LangChain, but requires API key with billing
- **Gemini** (via Google AI Studio) — generous free tier, good quality
- **Ollama** (local) — fully free but needs local compute

**Recommendation:** Start with Gemini (free tier) for zero-cost development; add OpenAI as an alternative config.

**Embedding Model:** Same question for embeddings:
- `text-embedding-3-small` (OpenAI) — paid per token
- `all-MiniLM-L6-v2` (HuggingFace) — free, runs locally

**Recommendation:** Use `all-MiniLM-L6-v2` for zero-cost local development.

---

## Progress Checklist

- [x] **Session 1: Project Scaffolding & Environment Setup**
  - [x] Full directory tree created per Section 7.1
  - [x] `requirements.txt` with all pinned dependencies
  - [x] `.env.example` template committed, `.env` in `.gitignore`
  - [x] `config.py` loads env vars correctly
  - [x] `main.py` starts with `uvicorn`, `/health` returns `{"status": "ok"}`
  - [x] All dependencies import without error
  - [x] Git repo initialized with `.gitignore` and README stub

- [x] **Session 2: Document Ingestion Pipeline**
  - [x] PDF text extraction with page numbers works (`pdfplumber`)
  - [x] DOCX text extraction works (`python-docx`)
  - [x] TXT text extraction works
  - [x] `RecursiveCharacterTextSplitter` chunks at 500 tokens / 50 overlap
  - [x] ChromaDB wrapper: create collection, add docs, query
  - [x] Metadata schema includes `doc_id`, `page_number`, `chunk_id`, `chunk_index`
  - [x] Test script prints top-k chunks with metadata for a hardcoded query

- [x] **Session 3: Basic RAG Chain**
  - [x] Retriever function returns top-k chunks from ChromaDB
  - [x] Answer generator uses grounded prompt template with citation instruction
  - [x] `run_basic_rag()` returns `AskResponse` with answer + citations
  - [x] Pydantic schemas (`AskRequest`, `AskResponse`, `Citation`) defined
  - [x] Integration flow verified with test script

- [x] **Session 4: FastAPI Endpoints**
  - [x] `POST /upload` accepts PDF/DOCX/TXT, validates type/size, returns chunk count
  - [x] `POST /ask` accepts question, returns grounded answer with citations
  - [x] 422 error for unsupported file types / oversized files
  - [x] Session-based chat history store (in-memory)
  - [x] Swagger UI at `/docs` fully functional for manual testing

- [x] **Session 5: Minimal Streamlit Frontend (MVP)**
  - [x] File upload widget with progress indication
  - [x] Chat interface with user/assistant message bubbles
  - [x] Basic citation display (doc name, page, text)
  - [x] Sidebar with uploaded document list
  - [x] End-to-end flow works: upload → ask → answer with citations

- [x] **Session 6: LangGraph Agentic Pipeline**
  - [x] `AgentState` TypedDict defined
  - [x] LangGraph `StateGraph` with 4 nodes: rewrite → retrieve → grade → generate
  - [x] Query Rewriter resolves follow-up questions correctly (multi-turn test)
  - [x] Relevance Grader filters irrelevant chunks (log/print proof)
  - [x] LangSmith tracing captures full agent trace
  - [x] `/ask` returns `agent_trace` in response
  - [x] Multi-document sessions supported

- [/] **Session 7: Hallucination Checker Agent**
  - [ ] Hallucination checker node added to LangGraph
  - [ ] Cross-checks answer claims against source chunks
  - [ ] Flags unsupported claims with a warning
  - [ ] At least 1 deliberately tested unsupported claim detected correctly
  - [ ] `AskResponse` includes optional `hallucination_warning` field

- [ ] **Session 8: Frontend Polish & Error Handling**
  - [ ] Expandable citation cards with page number + exact source text
  - [ ] Agent step indicator visible during "thinking" state
  - [ ] Hallucination warning displayed when triggered
  - [ ] Error states (bad file, no answer found, API timeout) handled gracefully
  - [ ] Multi-document upload + document filtering dropdown
  - [ ] Visual styling pass complete (consistent, non-default appearance)

- [ ] **Session 9: RAGAS Evaluation & Benchmarking**
  - [ ] 15–20 test Q&A pairs written and saved in `eval/test_set.json`
  - [ ] Evaluation script runs both naive and agentic RAG
  - [ ] RAGAS scores generated: Faithfulness, Answer Relevancy, Context Precision, Context Recall
  - [ ] Comparison table: naive vs. agentic (agentic outperforms)
  - [ ] Results documented in README with chart/table

- [ ] **Session 10: Dockerization, Deployment & Documentation**
  - [ ] `Dockerfile` builds successfully
  - [ ] `docker-compose.yml` runs full app (backend + frontend)
  - [ ] App deployed to public URL (HuggingFace Spaces / Render / Railway)
  - [ ] README finalized: architecture diagram, setup steps, eval results, demo link
  - [ ] 2-minute demo video recorded and linked
  - [ ] Repo cleaned: no `.env` committed, no dead code, clear history
