# Agentic RAG Research Assistant
## End-to-End Project Documentation

**Project Type:** Generative AI / Applied LLM Engineering
**Duration:** 6 Weeks
**Author:** _[Your Name]_
**Mentor Review Status:** _Pending_
**Version:** 1.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives & Learning Outcomes](#2-objectives--learning-outcomes)
3. [Prerequisites](#3-prerequisites)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack — Detailed Breakdown](#5-tech-stack--detailed-breakdown)
6. [User Flows](#6-user-flows)
7. [Backend Documentation](#7-backend-documentation)
8. [Frontend Documentation](#8-frontend-documentation)
9. [Data Layer & Vector Store Schema](#9-data-layer--vector-store-schema)
10. [End-to-End Functionality Walkthrough](#10-end-to-end-functionality-walkthrough)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Security & Reliability Considerations](#12-security--reliability-considerations)
13. [Week-Wise Build Plan](#13-week-wise-build-plan)
14. [Deployment Guide](#14-deployment-guide)
15. [Portfolio Presentation Strategy](#15-portfolio-presentation-strategy)
16. [Future Enhancements (Beyond MVP)](#16-future-enhancements-beyond-mvp)
17. [Appendix](#17-appendix)

---

## 1. Project Overview

The **Agentic RAG Research Assistant** is a web application that allows a user to upload one or more documents (PDF, DOCX, TXT) and ask natural-language questions about their contents. Unlike a basic "chat with PDF" tool, this system uses a **multi-agent pipeline** — built with LangGraph — to rewrite ambiguous queries, retrieve relevant chunks from a vector database, grade the relevance of what was retrieved, and only then generate a final answer that includes citations back to the source document and page number.

**Problem it solves:** Manually searching through long documents (research papers, legal contracts, internal policy documents) is slow and error-prone. This assistant gives accurate, source-grounded answers in seconds, with verifiable citations — addressing the core weakness of plain LLM chat (hallucination) through retrieval grounding and an explicit relevance-grading step.

**Recommended domain framing** (pick one to give the project a clear narrative — do not build a generic "upload any PDF" tool):

| Domain | Example Use Case | Why It Works |
|---|---|---|
| Legal | Contract clause Q&A ("What's the termination notice period?") | High business value, easy to demo |
| Academic | Research paper assistant for literature review | Natural fit for student reviewers/mentors |
| Corporate | Internal policy / HR handbook bot | Relatable to any interviewer |

---

## 2. Objectives & Learning Outcomes

By the end of this project, you will be able to demonstrate:

- Building a **production-style RAG pipeline** (chunking, embeddings, retrieval, generation)
- Designing and orchestrating a **multi-agent system** using LangGraph
- Working with **vector databases** (ChromaDB / Pinecone) and embedding models
- Building a **REST API backend** with FastAPI, including request validation and async processing
- Building a usable **frontend** (Streamlit or React) for a non-technical end user
- Running **quantitative RAG evaluation** (faithfulness, relevancy, precision) using RAGAS
- **Deploying** a full-stack AI application to the cloud, for free
- Producing **professional technical documentation** of your own system (this very document is a model of that)

---

## 3. Prerequisites

### 3.1 Knowledge Prerequisites
- Python (functions, classes, async/await basics, virtual environments)
- Basic REST API concepts (GET/POST, JSON, status codes)
- Git & GitHub fundamentals (clone, commit, push, branches)
- Basic command-line comfort (cd, pip install, running scripts)
- *Not required but helpful:* prior exposure to ML/NLP concepts

### 3.2 Accounts to Create (all free tier)
- [ ] GitHub account
- [ ] OpenAI API account (or Google AI Studio for Gemini — has a generous free tier)
- [ ] HuggingFace account (for free embedding models + deployment)
- [ ] LangSmith account (free tier, for tracing/debugging agents)
- [ ] Pinecone account *(only if you choose cloud vector DB over local ChromaDB)*

### 3.3 Local Environment Setup
```bash
# Python 3.11+ required
python --version

# Create project folder and virtual environment
mkdir agentic-rag-assistant && cd agentic-rag-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Core dependencies
pip install fastapi uvicorn langchain langgraph langchain-openai \
            chromadb pypdf python-docx pydantic python-multipart \
            streamlit ragas python-dotenv
```

### 3.4 Environment Variables
Create a `.env` file (never commit this to GitHub):
```
OPENAI_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag-assistant
```

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
                         ┌─────────────────────────────────────┐
                         │           FRONTEND (UI)              │
                         │     Streamlit / React + Tailwind     │
                         │  - File upload                       │
                         │  - Chat interface                    │
                         │  - Source citation display            │
                         └───────────────┬───────────────────────┘
                                         │ HTTP (REST / JSON)
                         ┌───────────────▼───────────────────────┐
                         │          BACKEND API (FastAPI)         │
                         │  /upload   /ask   /history  /health    │
                         └───────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
        ┌───────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
        │  INGESTION MODULE  │ │ AGENT ORCHESTRATOR│ │   SESSION / MEMORY  │
        │  - Text extraction │ │   (LangGraph)      │ │   (per-user chat    │
        │  - Chunking         │ │  query rewriter →  │ │   history store)    │
        │  - Embedding        │ │  retriever →        │ └─────────────────────┘
        └─────────┬───────────┘ │  relevance grader → │
                   │             │  answer generator   │
                   ▼             └─────────┬───────────┘
        ┌───────────────────┐              │
        │   VECTOR STORE     │◄─────────────┘
        │  ChromaDB/Pinecone  │
        └─────────┬───────────┘
                   │
                   ▼
        ┌───────────────────┐
        │     LLM API         │
        │ OpenAI / Gemini /   │
        │ Ollama (local)       │
        └───────────────────┘
```

### 4.2 Agent Pipeline (LangGraph State Machine)

| Step | Agent | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | **Query Rewriter** | Raw user question + chat history | Standalone, disambiguated query | Resolves pronouns/context from follow-up questions |
| 2 | **Retrieval Agent** | Rewritten query | Top-k chunks from vector store | Semantic search against embeddings |
| 3 | **Relevance Grader** | Retrieved chunks + query | Filtered, relevant-only chunks | Removes noisy/irrelevant chunks before generation (reduces hallucination) |
| 4 | **Answer Generator** | Relevant chunks + query | Final answer with inline citations | Produces grounded, source-cited response |
| *(optional, Week 4)* | **Hallucination Checker** | Answer + source chunks | Verified/flagged answer | Cross-checks answer claims against source text |

### 4.3 Ingestion Pipeline (runs once per uploaded document)

```
Upload (PDF/DOCX/TXT)
   → Text Extraction (pdfplumber / python-docx)
   → Chunking (500 tokens, 50-token overlap)
   → Embedding (text-embedding-3-small or all-MiniLM-L6-v2)
   → Store in Vector DB with metadata {doc_name, page_number, chunk_id}
```

---

## 5. Tech Stack — Detailed Breakdown

| Layer | Primary Choice | Alternative | Notes |
|---|---|---|---|
| **Frontend** | Streamlit | React + Tailwind | Streamlit is faster to build; React is more impressive if you have frontend skill already |
| **Backend** | FastAPI (Python 3.11+) | Flask | FastAPI gives free auto-generated API docs (Swagger UI) |
| **Orchestration** | LangChain + LangGraph | LlamaIndex | LangGraph gives explicit, debuggable agent state machines |
| **Vector Store** | ChromaDB (local) | Pinecone (cloud) | Start local/free; migrate to Pinecone if you want a "production scale" story |
| **Embeddings** | `text-embedding-3-small` (OpenAI) | `all-MiniLM-L6-v2` (HuggingFace, free/local) | Use the free local model if you want zero API cost |
| **LLM** | GPT-4o-mini | Gemini 1.5 Flash / Ollama (Llama 3) | Ollama = fully free but needs local GPU/CPU capacity |
| **Document Parsing** | `pdfplumber`, `python-docx` | `unstructured` library | `unstructured` handles messier/scanned documents better |
| **Evaluation** | RAGAS | TruLens | RAGAS integrates cleanly with LangChain objects |
| **Observability** | LangSmith | — | Free tier is enough; gives you a trace of every agent step |
| **Deployment** | HuggingFace Spaces | Render / Railway | HF Spaces is free, Gen-AI-community-recognized, and simple |
| **Containerization** | Docker | — | Optional but strongly recommended for portfolio credibility |

---

## 6. User Flows

### 6.1 Primary Flow — Ask a Question

1. User lands on the app and sees an upload zone + empty chat window.
2. User uploads a document (PDF/DOCX/TXT).
3. System shows an ingestion progress indicator ("Extracting text… Chunking… Embedding… Done ✅").
4. User types a question in the chat box and hits send.
5. UI shows a "thinking" state (optionally streaming intermediate agent steps: "Rewriting query…", "Searching documents…", "Grading relevance…", "Generating answer…").
6. Answer appears in the chat, with **expandable citation cards** below it showing the exact source chunk and page number.
7. User can ask a follow-up question — chat history is retained for context.

### 6.2 Secondary Flow — Multi-Document Session

1. User uploads a second document mid-session.
2. System re-indexes and tags the new document separately in metadata.
3. User can optionally scope a question to "only Document B" via a dropdown filter.
4. Citations clearly indicate which document the answer was drawn from.

### 6.3 Error Flow — Unsupported / Corrupted File

1. User uploads an unsupported file type or a corrupted PDF.
2. Backend returns a structured error (`422 Unprocessable Entity`) with a human-readable message.
3. Frontend displays a friendly error toast: *"We couldn't read this file. Please upload a PDF, DOCX, or TXT under 20MB."*

### 6.4 Error Flow — No Relevant Information Found

1. User asks a question unrelated to the uploaded document.
2. Relevance Grader agent filters out all retrieved chunks as irrelevant.
3. System responds honestly: *"I couldn't find relevant information in the uploaded document(s) to answer this."* — **never** fabricates an answer.

---

## 7. Backend Documentation

### 7.1 Project Structure
```
backend/
├── main.py                  # FastAPI app entrypoint
├── routers/
│   ├── upload.py             # /upload endpoint
│   ├── ask.py                 # /ask endpoint
│   └── health.py               # /health endpoint
├── core/
│   ├── ingestion.py            # text extraction + chunking + embedding
│   ├── vector_store.py          # ChromaDB client wrapper
│   ├── agents/
│   │   ├── graph.py              # LangGraph state machine definition
│   │   ├── query_rewriter.py
│   │   ├── retriever.py
│   │   ├── relevance_grader.py
│   │   └── answer_generator.py
│   └── memory.py                # session/chat history store
├── models/
│   └── schemas.py               # Pydantic request/response models
├── config.py                   # env var loading
├── requirements.txt
└── Dockerfile
```

### 7.2 API Endpoints

#### `POST /upload`
Uploads and ingests a document.

**Request:** `multipart/form-data`
| Field | Type | Required |
|---|---|---|
| `file` | File (PDF/DOCX/TXT, max 20MB) | Yes |
| `session_id` | string | Yes |

**Response — 200 OK**
```json
{
  "doc_id": "doc_8f3a1c",
  "filename": "employee_handbook.pdf",
  "chunks_created": 142,
  "status": "indexed"
}
```

**Response — 422 Unprocessable Entity**
```json
{ "error": "unsupported_file_type", "message": "Please upload a PDF, DOCX, or TXT file." }
```

#### `POST /ask`
Submits a question and receives a grounded, cited answer.

**Request Body**
```json
{
  "session_id": "sess_4e21",
  "question": "What is the notice period for resignation?",
  "doc_ids": ["doc_8f3a1c"]
}
```

**Response — 200 OK**
```json
{
  "answer": "The notice period for resignation is 30 days, as outlined in Section 4.2.",
  "citations": [
    { "doc_id": "doc_8f3a1c", "page": 6, "chunk_text": "Employees must provide a minimum of 30 days written notice..." }
  ],
  "agent_trace": ["query_rewriter", "retriever", "relevance_grader", "answer_generator"]
}
```

#### `GET /health`
Simple liveness check for deployment monitoring. Returns `{"status": "ok"}`.

### 7.3 Pydantic Schemas (excerpt)
```python
class AskRequest(BaseModel):
    session_id: str
    question: str
    doc_ids: list[str] | None = None

class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_text: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    agent_trace: list[str]
```

### 7.4 LangGraph State Definition (conceptual)
```python
class AgentState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    answer: str
    chat_history: list[dict]

graph = StateGraph(AgentState)
graph.add_node("rewrite", query_rewriter)
graph.add_node("retrieve", retriever)
graph.add_node("grade", relevance_grader)
graph.add_node("generate", answer_generator)

graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_edge("grade", "generate")
graph.set_entry_point("rewrite")
graph.set_finish_point("generate")
```

---

## 8. Frontend Documentation

### 8.1 Page/Component Structure (Streamlit version)
```
frontend/
├── app.py                    # Main Streamlit entrypoint
├── components/
│   ├── upload_widget.py        # File upload + progress
│   ├── chat_window.py           # Chat message rendering
│   ├── citation_card.py          # Expandable source citation UI
│   └── sidebar.py                # Document list / session controls
└── utils/
    └── api_client.py            # Wrapper functions calling the FastAPI backend
```

### 8.2 Key UI States
| State | What the User Sees |
|---|---|
| Empty | Upload zone + "Upload a document to get started" |
| Ingesting | Progress spinner with step labels |
| Ready | Chat input enabled, document name shown in sidebar |
| Thinking | Animated "agent steps" indicator (great for demo videos) |
| Answered | Chat bubble + collapsible citation cards below it |
| Error | Red toast/banner with a clear, non-technical message |

### 8.3 React Alternative (if chosen)
- `UploadZone.jsx` — drag-and-drop file upload with progress bar
- `ChatThread.jsx` — message list with auto-scroll
- `CitationCard.jsx` — expandable card showing source snippet + page number
- `Sidebar.jsx` — list of uploaded documents with delete/filter controls
- State management: React `useState`/`useReducer` is sufficient at this scale — no need for Redux

---

## 9. Data Layer & Vector Store Schema

### 9.1 Chunk Metadata Schema (stored alongside each embedding)
```json
{
  "chunk_id": "doc_8f3a1c_chunk_017",
  "doc_id": "doc_8f3a1c",
  "doc_name": "employee_handbook.pdf",
  "page_number": 6,
  "chunk_text": "...",
  "chunk_index": 17,
  "char_count": 487
}
```

### 9.2 Session/Memory Schema (in-memory dict or Redis for production)
```json
{
  "session_id": "sess_4e21",
  "doc_ids": ["doc_8f3a1c"],
  "chat_history": [
    { "role": "user", "content": "What is the notice period?" },
    { "role": "assistant", "content": "30 days, per Section 4.2." }
  ]
}
```

### 9.3 Chunking Strategy
- **Chunk size:** 500 tokens
- **Overlap:** 50 tokens (preserves context across chunk boundaries)
- **Splitter:** `RecursiveCharacterTextSplitter` (LangChain) — splits on paragraph → sentence → word boundaries in that priority order

---

## 10. End-to-End Functionality Walkthrough

1. **Document ingestion** — file uploaded → parsed → chunked → embedded → stored with metadata.
2. **Question submitted** — frontend sends question + session_id to `/ask`.
3. **Query rewriting** — agent resolves the question against chat history into a standalone query (handles "what about the second one?"-type follow-ups).
4. **Retrieval** — top-k (e.g. 6) chunks retrieved via cosine similarity search.
5. **Relevance grading** — an LLM call scores each retrieved chunk as relevant/irrelevant; irrelevant ones are dropped.
6. **Answer generation** — remaining chunks + question are passed to the LLM with a strict prompt: *"Answer only using the provided context. If the answer isn't present, say so."*
7. **Citation mapping** — each claim in the answer is linked back to its source chunk_id/page.
8. **Response rendering** — frontend displays the answer with expandable citation cards.
9. **Memory update** — the exchange is appended to session chat history for future context.
10. **Tracing** — the full agent trace is logged to LangSmith for debugging and for your evaluation report.

---

## 11. Evaluation Strategy

Use **RAGAS** to quantitatively score your system on a held-out set of 15–20 question/answer pairs you write yourself against your test document(s):

| Metric | What It Measures | Target |
|---|---|---|
| **Faithfulness** | Does the answer only contain claims supported by retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer actually address the question asked? | > 0.85 |
| **Context Precision** | Are the retrieved chunks actually relevant (low noise)? | > 0.75 |
| **Context Recall** | Did retrieval find all the necessary information? | > 0.75 |

**Comparison table to include in your README:**

| Configuration | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| Naive RAG (no agent, no grading) | _e.g. 0.71_ | _e.g. 0.79_ | _e.g. 0.62_ |
| Agentic RAG (with grading agent) | _e.g. 0.89_ | _e.g. 0.91_ | _e.g. 0.83_ |

This before/after comparison is one of the **strongest pieces of evidence** you can put on your CV — it proves your agentic design choices had a measurable impact, not just added complexity.

---

## 12. Security & Reliability Considerations

*(Thinking beyond the MVP — mentioning these in your documentation signals senior-level thinking, even if you only implement a subset.)*

- **File validation:** restrict file types and size (max 20MB) server-side, not just in the UI
- **Prompt injection awareness:** documents may contain text designed to hijack the LLM (e.g. "Ignore previous instructions…") — mention this as a known risk and isolate retrieved content clearly within the prompt template
- **Rate limiting:** add basic per-session rate limiting on `/ask` to prevent API cost abuse if deployed publicly
- **API key handling:** never expose keys client-side; all LLM calls happen server-side only
- **Data retention:** clarify in your README whether uploaded documents are stored persistently or deleted after the session — important for a "legal/HR document" framing
- **Graceful degradation:** if the LLM API call fails or times out, return a clear error rather than a hung request

---

## 13. Week-Wise Build Plan

> Each week ends with a **visible, demoable milestone**. Treat each "Demo" as a short recording you can show your mentor — this builds a running portfolio of progress, not just a final reveal.

---

### 🟦 Week 1 — Environment Setup + Document Ingestion Pipeline

**Goal:** Get raw documents into a searchable vector store.

**Tasks:**
- Set up Python environment, repo, and folder structure
- Implement file upload handling (PDF/DOCX/TXT) in FastAPI
- Implement text extraction (`pdfplumber`, `python-docx`)
- Implement chunking with `RecursiveCharacterTextSplitter`
- Generate embeddings and store in local ChromaDB
- Write a simple test script that queries ChromaDB directly and prints top-k matches

**Demo at end of week:** Run a script that uploads a PDF and prints the top 3 most relevant chunks for a hardcoded test question — proving the retrieval foundation works.

**✅ Checklist**
- [ ] Repo created with structure from Section 7.1
- [ ] `.env` configured, API keys working
- [ ] `/upload` endpoint accepts PDF/DOCX/TXT and returns chunk count
- [ ] ChromaDB collection populated with embeddings + metadata
- [ ] Test script retrieves relevant chunks for a sample query
- [ ] Code pushed to GitHub with a working README stub

---

### 🟩 Week 2 — Basic RAG Loop + Minimal Frontend (MVP)

**Goal:** A user can ask a question and get an LLM-generated answer grounded in the document — no agents yet, just a straight RAG chain.

**Tasks:**
- Build `/ask` endpoint: retrieve top-k chunks → pass to LLM with a grounded prompt → return answer
- Build minimal Streamlit UI: file upload + chat input + answer display
- Add basic citation display (just show which chunk was used, unformatted is fine for now)
- Connect frontend to backend end-to-end

**Demo at end of week:** Live screen recording — upload a document in the UI, ask a real question, get a correct, source-grounded answer displayed in the chat.

**✅ Checklist**
- [ ] `/ask` endpoint returns a grounded answer (not hallucinated)
- [ ] Streamlit app runs locally and connects to backend
- [ ] Citations (even basic) are shown alongside the answer
- [ ] At least 5 test questions answered correctly against a real document
- [ ] README updated with setup instructions + screenshot

---

### 🟨 Week 3 — Agentic Layer: Query Rewriting + Relevance Grading

**Goal:** Upgrade the straight RAG chain into a true multi-agent pipeline using LangGraph.

**Tasks:**
- Define `AgentState` and build the LangGraph state machine (Section 7.4)
- Implement Query Rewriter agent (handles follow-up/ambiguous questions using chat history)
- Implement Relevance Grader agent (filters irrelevant retrieved chunks before generation)
- Add session-based chat history (in-memory dict is fine for now)
- Set up LangSmith tracing to visualize each agent step

**Demo at end of week:** Ask a multi-turn conversation (e.g. "What's the notice period?" → "What about for senior employees?") and show the query rewriter correctly resolving the follow-up — plus show the LangSmith trace of all 4 agent steps firing in sequence.

**✅ Checklist**
- [ ] LangGraph state machine runs end-to-end with all nodes connected
- [ ] Follow-up questions correctly resolved using chat history
- [ ] Irrelevant chunks are visibly filtered out before generation (log/print proof)
- [ ] LangSmith trace captured and screenshotted for documentation
- [ ] Multi-document upload supported (at least 2 docs in one session)

---

### 🟧 Week 4 — Hallucination Checking + UI Polish

**Goal:** Increase answer trustworthiness and make the UI demo-ready.

**Tasks:**
- Add Hallucination Checker agent (cross-checks generated answer against retrieved context, flags unsupported claims)
- Improve citation UI — expandable cards showing exact source text + page number
- Add an "agent steps" visual indicator in the UI ("Rewriting query… Retrieving… Grading… Generating…")
- Handle error flows from Section 6.3/6.4 gracefully in the UI
- Style pass on the frontend (clean layout, consistent spacing, readable typography)

**Demo at end of week:** Full polished walkthrough — upload, multi-turn chat, expandable citations, and a deliberately irrelevant question that correctly triggers the "no relevant info found" response instead of a hallucinated answer.

**✅ Checklist**
- [ ] Hallucination checker flags at least one deliberately tested unsupported claim correctly
- [ ] Citation cards are expandable and show page number + exact source text
- [ ] Error states (bad file, no answer found) display cleanly, not as raw errors
- [ ] UI has a consistent, non-default visual style
- [ ] Agent step indicator visible during "thinking" state

---

### 🟪 Week 5 — Evaluation with RAGAS + Dockerization

**Goal:** Prove your system's quality quantitatively and package it for deployment.

**Tasks:**
- Write 15–20 question/answer test pairs against your chosen test document(s)
- Run RAGAS evaluation on both a "naive RAG" baseline and your full agentic pipeline
- Document the before/after comparison table (Section 11)
- Write a `Dockerfile` and `docker-compose.yml` for backend + frontend
- Test full app running from Docker locally

**Demo at end of week:** Show the RAGAS evaluation report (scores table or chart) comparing naive vs. agentic RAG — this is your strongest engineering-credibility artifact. Also demo the app running fully from `docker-compose up`.

**✅ Checklist**
- [ ] 15–20 test Q&A pairs written and saved in repo (`/eval/test_set.json`)
- [ ] RAGAS scores generated for both naive and agentic configurations
- [ ] Comparison table/chart created for README
- [ ] Dockerfile builds successfully
- [ ] Full app runs via `docker-compose up` with no manual steps

---

### 🟥 Week 6 — Deployment + Documentation + Portfolio Packaging

**Goal:** Ship it publicly and package it for recruiters/mentors to review.

**Tasks:**
- Deploy backend + frontend to HuggingFace Spaces (or Render/Railway)
- Write the final `README.md`: overview, architecture diagram, setup steps, eval results, demo link
- Record a 2-minute demo video (Loom) showing the full user flow
- Write a short technical blog post: "How I Built an Agentic RAG Assistant"
- Publish a LinkedIn post announcing the project with the demo clip
- Final mentor review and presentation

**Demo at end of week:** A single shareable link to the **live, publicly deployed app** — this is the final deliverable.

**✅ Checklist**
- [ ] App deployed and accessible via public URL
- [ ] README finalized with architecture diagram + eval results table
- [ ] 2-minute demo video recorded and linked in README
- [ ] Blog post published (Medium/Dev.to)
- [ ] LinkedIn post published
- [ ] GitHub repo cleaned: no committed `.env`, no dead code, clear commit history
- [ ] Final walkthrough presented to mentor

---

## 14. Deployment Guide

### 14.1 HuggingFace Spaces (Recommended — Free)
1. Create a new Space → select **Docker** as the Space SDK
2. Push your repo (with `Dockerfile`) to the Space's git remote
3. Add your API keys under **Settings → Repository Secrets** (never hardcode them)
4. Space auto-builds and deploys — you get a public URL like `https://huggingface.co/spaces/yourname/agentic-rag`

### 14.2 Alternative: Render / Railway
- Connect your GitHub repo directly
- Set environment variables in the dashboard
- Both offer free tiers sufficient for a portfolio-scale demo (note: free tiers may sleep after inactivity — mention this in your README so reviewers aren't confused by a slow first load)

---

## 15. Portfolio Presentation Strategy

| Asset | Purpose |
|---|---|
| **Live demo link** | Primary proof — one click, zero setup, for any reviewer |
| **Clean GitHub repo** | README with architecture diagram, setup steps, eval results |
| **2-minute demo video** | For reviewers who won't click through a live app (recruiters skimming) |
| **Blog post** | Signals communication skill, not just coding skill |
| **RAGAS metrics table** | Quantitative proof of engineering rigor |
| **LinkedIn post** | Visibility — Gen AI content gets strong recruiter engagement |

**Strongest possible addition:** include a "What I Tried and What Failed" section in your README — e.g. an early chunking strategy that performed poorly, a prompt that caused hallucinations before you added the grading agent. This signals genuine experimentation, which reviewers consistently rate above polished-but-shallow projects.

---

## 16. Future Enhancements (Beyond MVP)

These are **not required** for the 6-week build but are worth listing in your README under "Future Work" — it shows you understand the system's limits and have a roadmap, which is itself a strong signal to reviewers:

- **Streaming responses** (token-by-token, like ChatGPT) for better perceived latency
- **Hybrid search** (keyword + semantic) for queries with exact terms/numbers that pure embedding search can miss
- **Multi-modal support** — extract and reason over tables/images within PDFs
- **User authentication** and persistent multi-session history (currently in-memory only)
- **Feedback loop** — thumbs up/down on answers, logged for future fine-tuning of the grading prompt
- **Agent self-correction loop** — if the hallucination checker flags an answer, route back to retrieval with a refined query instead of just returning a warning

---

## 17. Appendix

### 17.1 Suggested `.gitignore`
```
venv/
.env
__pycache__/
*.pyc
chroma_db/
.DS_Store
```

### 17.2 Useful Commands Reference
```bash
# Run backend locally
uvicorn main:app --reload --port 8000

# Run frontend locally
streamlit run app.py

# Run RAGAS evaluation
python eval/run_evaluation.py

# Build and run via Docker
docker-compose up --build
```

### 17.3 Recommended Reading
- LangGraph official documentation (agent state machines)
- RAGAS documentation (evaluation metrics methodology)
- "Lost in the Middle" paper — informs why chunk ordering matters in long contexts

---

*This document is a living plan — update the checklists weekly and treat deviations as documented decisions, not failures. A README that explains **why** you changed direction is more valuable than one that pretends the original plan was followed exactly.*
