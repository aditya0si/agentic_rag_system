"""
Agentic RAG Research Assistant — Streamlit Frontend.
Integrates chat interface, document management sidebar, citation card display,
conversation export, and real-time agent pipeline visualization.
"""

import uuid
import json
import streamlit as st
from datetime import datetime
from utils.api_client import ask_question_api, check_backend_health
from components.sidebar import render_sidebar
from components.citation_card import render_citations

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Page Configuration
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Agentic RAG Research Assistant — AI-powered document Q&A with hallucination detection."
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Premium CSS — Modern Glassmorphism + Dark Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Typography ── */
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', sans-serif;
    color: #1E293B;
}
h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; }
h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }

/* ── Product dashboard elements ── */
.product-kicker { color: #4F46E5; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
.welcome-panel { background: radial-gradient(circle at top right, #E0E7FF 0%, transparent 38%), #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; padding: 28px; margin: 16px 0 24px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06); }
.metric-card { background: rgba(255,255,255,0.76); border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px; }
.metric-value { color: #0F172A; font-size: 1.15rem; font-weight: 750; }
.metric-label { color: #64748B; font-size: 0.75rem; margin-top: 2px; }

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #F0F4FF 0%, #F8FAFC 50%, #EEF2FF 100%);
}

/* ── Dark Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] button {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #C7D2FE !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(99, 102, 241, 0.3) !important;
    border-color: #818CF8 !important;
}

/* ── Chat Messages ── */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 16px;
    padding: 20px 24px;
    animation: fadeSlideIn 0.35s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User message bubble */
div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="stChatMessageIconUser"]) {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border-left: 4px solid #3B82F6;
    margin-left: 40px;
}

/* Assistant message bubble */
div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="stChatMessageIconAssistant"]) {
    background: #FFFFFF;
    border-left: 4px solid #10B981;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
    margin-right: 40px;
}

/* ── Status / Progress ── */
div[data-testid="stStatus"] {
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Expanders (Citations) ── */
div[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #F8FAFC;
    transition: box-shadow 0.2s;
}
div[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
}

/* ── Warning / Info boxes ── */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 2px dashed rgba(99,102,241,0.3);
    background: rgba(99,102,241,0.04);
    padding: 8px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── Code blocks ── */
code, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []

if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = None

if "query_in_flight" not in st.session_state:
    st.session_state.query_in_flight = False

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

render_sidebar(st.session_state.session_id)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main Header
# ═══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<div class='product-kicker'>Your research workspace</div>", unsafe_allow_html=True)
    st.title("🔍 Ask better questions of your documents")
    st.markdown(
        "<span style='color:#64748B; font-size:0.95rem;'>"
        "Upload a source, ask in plain English, and verify every answer with citations."
        "</span>",
        unsafe_allow_html=True
    )

with col2:
    # Backend health indicator
    if st.session_state.backend_healthy is None:
        with st.spinner(""):
            st.session_state.backend_healthy = check_backend_health()
    if st.session_state.backend_healthy:
        st.markdown(
            "<div style='background:#ECFDF5; border:1px solid #A7F3D0; border-radius:20px; "
            "padding:6px 14px; text-align:center; font-size:0.82rem; color:#065F46; margin-top:8px;'>"
            "🟢 Backend Online</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#FEF2F2; border:1px solid #FECACA; border-radius:20px; "
            "padding:6px 14px; text-align:center; font-size:0.82rem; color:#991B1B; margin-top:8px;'>"
            "🔴 Backend Offline</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# ── First-use onboarding ────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-panel">
          <div class="product-kicker">Grounded AI, not guesswork</div>
          <h2 style="margin:8px 0 8px;">Turn dense files into confident answers.</h2>
          <p style="color:#64748B; margin:0; max-width:680px;">Start by adding PDFs, Word files, notes, or Markdown in the left panel. Every answer is based on the documents you choose and includes its source passages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(3)
    for column, value, label in zip(
        metric_cols,
        ("Focused scope", "Cited answers", "Multi-agent review"),
        ("Choose exactly which documents to search", "Inspect the original passage and page", "Retrieval, relevance checks, and grounding"),
    ):
        with column:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    if st.session_state.uploaded_docs:
        st.markdown("##### Try one of these prompts")
        starter_cols = st.columns(3)
        starters = [
            "Give me a concise summary of these documents.",
            "What are the most important findings and why do they matter?",
            "Compare the key arguments across the selected documents.",
        ]
        for column, prompt in zip(starter_cols, starters):
            with column:
                if st.button(prompt, key=f"starter_{starters.index(prompt)}", use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Chat History Rendering
# ═══════════════════════════════════════════════════════════════════════════════

for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        # Hallucination warning
        if role == "assistant" and msg.get("hallucination_warning"):
            st.warning(f"⚠️ {msg['hallucination_warning']}")

        # Citations
        if role == "assistant" and msg.get("citations"):
            render_citations(msg["citations"])

        # Agent trace (collapsible)
        if role == "assistant" and msg.get("agent_trace"):
            with st.expander("🔎 View Agent Pipeline Trace", expanded=False):
                trace = msg["agent_trace"]
                icons = {
                    "query_rewriter": ("🔍", "Query Rewriter — Resolved conversational context"),
                    "retriever": ("📂", "Retriever — Fetched chunks from ChromaDB"),
                    "relevance_grader": ("⚖️", "Relevance Grader — Filtered irrelevant chunks"),
                    "answer_generator": ("✨", "Answer Generator — Produced grounded response"),
                    "hallucination_checker": ("🛡️", "Hallucination Checker — Verified groundedness"),
                }
                for step in trace:
                    icon, desc = icons.get(step, ("➡️", step))
                    st.markdown(f"{icon} **{desc}**")

        # Timestamp
        if msg.get("timestamp"):
            st.caption(f"_{msg['timestamp']}_")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Chat Input & Processing
# ═══════════════════════════════════════════════════════════════════════════════

# Disable input while a query is in flight
disabled = st.session_state.query_in_flight
question = st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=disabled
)
question = question or st.session_state.pop("pending_question", None)

if question:
    # Guard: no documents
    if not st.session_state.uploaded_docs:
        st.warning("⚠️ Please upload at least one document in the sidebar to get started.")
    # Guard: backend down
    elif not st.session_state.backend_healthy:
        st.error("🔴 Cannot reach the backend. Please ensure the API server is running.")
    else:
        st.session_state.query_in_flight = True

        # Append user message
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": timestamp
        })
        with st.chat_message("user"):
            st.markdown(question)
            st.caption(f"_{timestamp}_")

        # Query backend
        with st.chat_message("assistant"):
            status_container = st.status(
                "🧠 **Agentic RAG pipeline running...**",
                expanded=True
            )

            with status_container:
                st.write("🔍 Rewriting query to resolve conversational context...")
                selected_docs = st.session_state.get("selected_doc_ids", None)
                response = ask_question_api(
                    question,
                    st.session_state.session_id,
                    selected_docs
                )

                if "error" in response:
                    status_container.update(
                        label="❌ Pipeline failed",
                        state="error",
                        expanded=True
                    )
                    st.error(f"**Backend Error:** {response['message']}")
                    # Save error as assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ _Request failed: {response['message']}_",
                        "citations": [],
                        "hallucination_warning": None,
                        "agent_trace": [],
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                else:
                    # Show trace steps
                    trace = response.get("agent_trace", [])
                    step_icons = {
                        "query_rewriter": "🔍 Query Rewriter — Resolving conversational context...",
                        "retriever": "📂 Retriever — Fetching document chunks from ChromaDB...",
                        "relevance_grader": "⚖️ Relevance Grader — Filtering irrelevant chunks...",
                        "answer_generator": "✨ Answer Generator — Producing grounded response...",
                        "hallucination_checker": "🛡️ Hallucination Checker — Verifying answer groundedness...",
                    }
                    for node in trace:
                        if node in step_icons:
                            st.write(step_icons[node])

                    status_container.update(
                        label="✅ Analysis complete",
                        state="complete",
                        expanded=False
                    )

            if "error" not in response:
                answer = response["answer"]
                citations = response.get("citations", [])
                warning = response.get("hallucination_warning")
                trace = response.get("agent_trace", [])

                st.markdown(answer)

                if warning:
                    st.warning(f"⚠️ {warning}")

                if citations:
                    render_citations(citations)

                if trace:
                    with st.expander("🔎 Agent Pipeline Trace", expanded=False):
                        icons = {
                            "query_rewriter": ("🔍", "Query Rewriter"),
                            "retriever": ("📂", "Retriever"),
                            "relevance_grader": ("⚖️", "Relevance Grader"),
                            "answer_generator": ("✨", "Answer Generator"),
                            "hallucination_checker": ("🛡️", "Hallucination Checker"),
                        }
                        for step in trace:
                            icon, name = icons.get(step, ("➡️", step))
                            st.markdown(f"{icon} **{name}**")

                timestamp = datetime.now().strftime("%H:%M")
                st.caption(f"_{timestamp}_")

                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "hallucination_warning": warning,
                    "agent_trace": trace,
                    "timestamp": timestamp
                })

        st.session_state.query_in_flight = False
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Footer — Export & Quick Actions
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.messages:
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 1, 4])

    with col_a:
        # Export conversation as JSON
        export_data = json.dumps(st.session_state.messages, indent=2, default=str)
        st.download_button(
            label="📥 Export Chat (JSON)",
            data=export_data,
            file_name=f"rag_conversation_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_b:
        # Export as readable Markdown
        md_lines = [f"# Agentic RAG Conversation — {st.session_state.session_id}\n"]
        for msg in st.session_state.messages:
            role_emoji = "🧑" if msg["role"] == "user" else "🤖"
            md_lines.append(f"### {role_emoji} {msg['role'].title()}")
            md_lines.append(msg["content"])
            if msg.get("citations"):
                md_lines.append("\n**Sources:**")
                for c in msg["citations"]:
                    md_lines.append(f"- {c.get('doc_name','?')} p.{c.get('page','?')}")
            md_lines.append("")
        st.download_button(
            label="📝 Export Chat (MD)",
            data="\n".join(md_lines),
            file_name=f"rag_conversation_{st.session_state.session_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_c:
        st.markdown(
            "<div style='text-align:right; color:#94A3B8; font-size:0.78rem; padding-top:8px;'>"
            f"Session: <code>{st.session_state.session_id}</code> &nbsp;|&nbsp; "
            f"{len(st.session_state.messages)} messages &nbsp;|&nbsp; "
            f"{len(st.session_state.uploaded_docs)} docs indexed"
            "</div>",
            unsafe_allow_html=True
        )
