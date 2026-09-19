"""
Sidebar component — handles file uploads, displays indexed documents with
management controls, query scoping, and session management.
"""

import streamlit as st
from utils.api_client import upload_document_api


def render_sidebar(session_id: str):
    """
    Renders the sidebar with document upload controls, indexed document list
    with per-document actions, query scope selector, and session management.
    """
    with st.sidebar:
        # ── Branding Header ──
        st.markdown(
            """
            <div style="text-align:center; padding:8px 0 16px 0;">
                <div style="font-size:2.4rem; margin-bottom:4px;">🔍</div>
                <div style="font-size:1.1rem; font-weight:700; color:#E2E8F0;">
                    Agentic RAG
                </div>
                <div style="font-size:0.72rem; color:#94A3B8; margin-top:2px;">
                    Research Assistant
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════
        # 1. File Uploader
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("### 📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Drag & drop PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Maximum 20MB per file. Supported: PDF, DOCX, TXT, Markdown.",
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_uploaded = False
            for uploaded_file in uploaded_files:
                uploaded_names = [d["filename"] for d in st.session_state.uploaded_docs]
                if uploaded_file.name not in uploaded_names:
                    with st.spinner(f"📥 Ingesting **{uploaded_file.name}**..."):
                        file_bytes = uploaded_file.read()
                        res = upload_document_api(uploaded_file.name, file_bytes, session_id)

                        if "error" in res:
                            st.error(
                                f"❌ Failed to index **{uploaded_file.name}**: {res['message']}"
                            )
                        else:
                            st.session_state.uploaded_docs.append(
                                {
                                    "doc_id": res["doc_id"],
                                    "filename": res["filename"],
                                    "chunks": res["chunks_created"],
                                }
                            )
                            st.toast(
                                f"✅ Indexed {res['filename']} ({res['chunks_created']} chunks)",
                                icon="✅",
                            )
                            new_uploaded = True
            if new_uploaded:
                st.rerun()

        # ═══════════════════════════════════════════════════════════════════════
        # 2. Indexed Documents List
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("---")
        st.markdown("### 📚 Indexed Documents")

        if st.session_state.uploaded_docs:
            # Summary stats
            total_chunks = sum(d["chunks"] for d in st.session_state.uploaded_docs)
            st.caption(
                f"{len(st.session_state.uploaded_docs)} document(s) · {total_chunks} total chunks"
            )

            for i, doc in enumerate(st.session_state.uploaded_docs):
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"<div style='padding:6px 0;'>"
                        f"<span style='font-weight:600;'>📄 {doc['filename']}</span><br>"
                        f"<span style='font-size:0.75rem; color:#94A3B8;'>"
                        f"ID: <code>{doc['doc_id']}</code> · {doc['chunks']} chunks"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{doc['doc_id']}", help=f"Remove {doc['filename']}"):
                        st.session_state.uploaded_docs.pop(i)
                        # Also remove from scope if selected
                        if doc["doc_id"] in st.session_state.get("selected_doc_ids", []):
                            st.session_state.selected_doc_ids.remove(doc["doc_id"])
                        st.toast(f"Removed {doc['filename']}", icon="🗑️")
                        st.rerun()
        else:
            st.info("📭 No documents indexed yet. Upload files above to begin.")

        # ═══════════════════════════════════════════════════════════════════════
        # 3. Query Scope Selector
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("---")
        st.markdown("### 🎯 Query Scope")

        if st.session_state.uploaded_docs:
            doc_options = {doc["filename"]: doc["doc_id"] for doc in st.session_state.uploaded_docs}

            # Quick-select buttons
            st.caption("Quick select:")
            qcol1, qcol2 = st.columns(2)
            with qcol1:
                if st.button("📋 Select All", use_container_width=True, key="scope_all"):
                    st.session_state.selected_doc_ids = list(doc_options.values())
                    st.rerun()
            with qcol2:
                if st.button("✖️ Clear All", use_container_width=True, key="scope_none"):
                    st.session_state.selected_doc_ids = []
                    st.rerun()

            selected_filenames = st.multiselect(
                "Documents to search:",
                options=list(doc_options.keys()),
                default=[
                    name
                    for name, did in doc_options.items()
                    if did in st.session_state.get("selected_doc_ids", [])
                ]
                or list(doc_options.keys()),
                help="Only selected documents will be searched for answers.",
                label_visibility="collapsed",
            )
            st.session_state.selected_doc_ids = [doc_options[name] for name in selected_filenames]

            if selected_filenames:
                st.caption(f"🔎 Searching across {len(selected_filenames)} document(s)")
            else:
                st.warning("⚠️ No documents selected — queries will return no results.")
        else:
            st.session_state.selected_doc_ids = []
            st.info("Upload files to configure query scope.")

        # ═══════════════════════════════════════════════════════════════════════
        # 4. Session Actions
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("---")
        st.markdown("### ⚙️ Session")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.toast("Chat history cleared", icon="🧹")
                st.rerun()

        with col_b:
            if st.button("🔄 Reset All", use_container_width=True, key="reset_all"):
                st.session_state.messages = []
                st.session_state.uploaded_docs = []
                st.session_state.selected_doc_ids = []
                st.toast("Session fully reset", icon="🔄")
                st.rerun()

        # Session info footer
        st.markdown("---")
        st.caption(
            f"🆔 Session: `{session_id}`\n\n"
            f"💬 {len(st.session_state.messages)} messages\n\n"
            f"📄 {len(st.session_state.uploaded_docs)} documents"
        )
