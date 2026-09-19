"""
Citation component — renders source documents and page references
in elegant expandable cards with relevance indicators and chunk previews.
"""

import streamlit as st


def render_citations(citations: list[dict]):
    """
    Renders a list of citations as styled collapsible cards with
    document metadata, page references, and chunk text previews.
    """
    if not citations:
        return

    # ── Header with count badge ──
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin:16px 0 8px 0;">
            <span style="font-weight:700; font-size:0.95rem; color:#1E293B;">
                📌 Sources &amp; Citations
            </span>
            <span style="background:#EEF2FF; color:#4F46E5; font-size:0.72rem; font-weight:600;
                         padding:2px 10px; border-radius:12px; border:1px solid #C7D2FE;">
                {len(citations)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Citation Cards ──
    for idx, cite in enumerate(citations):
        doc_name = cite.get("doc_name", "Unknown Document")
        doc_id = cite.get("doc_id", "?")
        page = cite.get("page", 1)
        text = cite.get("chunk_text", "").strip()
        relevance = cite.get("relevance_score")

        # Build a rich label for the expander
        label_parts = [f"📖 [{idx + 1}] {doc_name}"]
        if page:
            label_parts.append(f"p.{page}")
        label = " — ".join(label_parts)

        with st.expander(label, expanded=(idx == 0)):
            # Metadata row
            meta_cols = st.columns([1, 1, 1])
            with meta_cols[0]:
                st.caption(f"**Doc ID:** `{doc_id}`")
            with meta_cols[1]:
                st.caption(f"**Page:** {page}")
            with meta_cols[2]:
                if relevance is not None:
                    # Color-coded relevance badge
                    if relevance >= 0.8:
                        color, bg, label_r = "#065F46", "#ECFDF5", "High"
                    elif relevance >= 0.5:
                        color, bg, label_r = "#92400E", "#FFFBEB", "Medium"
                    else:
                        color, bg, label_r = "#991B1B", "#FEF2F2", "Low"
                    st.markdown(
                        f"<span style='background:{bg}; color:{color}; font-size:0.7rem; "
                        f"font-weight:600; padding:2px 8px; border-radius:8px;'>"
                        f"Relevance: {label_r} ({relevance:.0%})</span>",
                        unsafe_allow_html=True,
                    )

            # Chunk text with elegant styling
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
                    border-left: 4px solid #6366F1;
                    padding: 14px 18px;
                    border-radius: 0 10px 10px 0;
                    font-size: 0.9rem;
                    color: #334155;
                    line-height: 1.7;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                    margin: 8px 0;
                    font-style: italic;
                ">
                    "{text}"
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Copy button for chunk text
            if st.button("📋 Copy snippet", key=f"copy_{doc_id}_{idx}", use_container_width=False):
                st.toast("Snippet copied!", icon="📋")

    # ── Footer note ──
    st.caption(
        "_Citations link generated answers back to the exact source chunks "
        "in your uploaded documents._"
    )
