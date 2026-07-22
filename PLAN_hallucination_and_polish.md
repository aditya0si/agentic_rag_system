# Session 7 & 8: Hallucination Checker Integration & Frontend Polish

## Goal Definition

### What is being built?
Integration of the `Hallucination Checker` agent into the LangGraph state machine, propagation of the groundedness warning through the FastAPI backend to the Streamlit UI, and polishing of the Streamlit frontend to display agent execution steps, hallucination warnings, multi-document selection filters, and expandable citation cards with clean styling.

### What does "done" look like?
1. The LangGraph state machine in `backend/core/agents/graph.py` contains `hallucination_check` as a node following the `generate` node.
2. The `run_agentic_rag` helper includes the hallucination node execution, updating the state's `hallucination_warning` field (either `None` or a groundedness warning message).
3. The `/ask` endpoint returns the `hallucination_warning` field.
4. An integration test in `backend/test_hallucination.py` validates that when hallucination is detected (via a mocked or real LLM check), the warning is set and returned, and `hallucination_checker` is in the agent trace.
5. The Streamlit UI displays a prominent warning block (using styled alerts) under any response that has a hallucination warning.
6. The Streamlit UI displays progress step indicators (e.g. status component or spinner updates) during the query lifecycle: "Rewriting query..." → "Retrieving document context..." → "Grading chunk relevance..." → "Checking answer groundedness..." → "Generating response...".
7. Citations are displayed in beautiful, expandable cards showing document name, page number, and the source excerpt text.
8. Sidebar supports multi-document selection for query scoping.

### What is explicitly out of scope?
- Automatic self-correction or regeneration loops within the graph (i.e. retry generation upon hallucination detection).
- Persistent multi-session history in a SQL DB.

---

## Tech Stack
The implementation uses the existing stack:
- **LangGraph** (StateGraph nodes & routing)
- **FastAPI** (Uvicorn backend server)
- **Streamlit** (Python frontend framework)
- **ChromaDB** (local vector database)
- **Pydantic** (schemas and validation)

---

## Session Modularization

### Session 7: Hallucination Checker Agent Integration
- **Objective**: Wire the hallucination checker into the LangGraph pipeline, update backend states, and test integration.
- **Scope**:
  - `backend/core/agents/graph.py` — Add hallucination node & route generating node to hallucination check node, then to END.
  - `backend/test_hallucination.py` — Write test cases verifying the check and trace.
- **Output**: The `/ask` endpoint returns answers with hallucination warnings and trace. Passing automated pipeline tests.
- **Connects To**: Session 8 (UI displays warnings).
- **Failure Surface**: Extra API latency. Mitigated by fallback parsing.

### Session 8: Frontend Polish, Multi-Doc Scoping & UX Error Handling
- **Objective**: Integrate step status messages, show hallucination warnings, polish expandable citations, and support multiple document scoping.
- **Scope**:
  - `frontend/app.py` — Update chat display, warnings, and steps.
  - `frontend/components/citation_card.py` — Polish visual cards.
  - `frontend/components/sidebar.py` — Multi-doc check & scope selection.
- **Output**: Beautiful, responsive user interface with premium visuals, warnings, and citations.
- **Connects To**: Session 9 (RAGAS Evaluation).
- **Failure Surface**: Streamlit state resetting bugs. Mitigated by robust `st.session_state` management.

---

## Progress Checklist

- [ ] Session 7: Hallucination Checker Agent Integration
  - [ ] Update `AgentState` TypedDict to include `hallucination_warning` in `backend/core/agents/graph.py`
  - [ ] Add `hallucination_node` to graph in `backend/core/agents/graph.py`
  - [ ] Add `hallucination_check` node to the `StateGraph` and update edges: `generate` -> `hallucination_check` -> `END`
  - [ ] Update `run_agentic_rag` to initialize `hallucination_warning`
  - [ ] Create `backend/test_hallucination.py` to test the integrated hallucination node with both real and mocked checks
  - [ ] Verify that `backend/test_hallucination.py` passes successfully
- [ ] Session 8: Frontend Polish, Multi-Doc Scoping & UX Error Handling
  - [ ] Update frontend chat flow to show thinking steps dynamically in Streamlit
  - [ ] Display the `hallucination_warning` as a warning block below the assistant message
  - [ ] Update `frontend/components/citation_card.py` to render citations as premium, clean expandable cards
  - [ ] Implement multi-document upload and check session document listings in the sidebar
  - [ ] Verify frontend connects and handles errors gracefully without crashing
