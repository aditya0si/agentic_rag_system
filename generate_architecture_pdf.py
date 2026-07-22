"""
Professional PDF Generator for Agentic RAG Research Assistant Architecture Report.
Customized for HCL Internship Submission: rag_architecture_HCL_internship.pdf
Author: Aditya Singh | Repository: https://github.com/aditya0si/agentic_rag_system
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Group, Polygon

class NumberedCanvas(canvas.Canvas):
    """
    Custom canvas that performs two-pass page numbering ('Page X of Y')
    and draws consistent executive headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 25, "HCL Internship Technical Report — Agentic RAG Research Assistant")
            self.drawRightString(8.5 * inch - 36, 11 * inch - 25, "Author: Aditya Singh | GitHub: aditya0si/agentic_rag_system")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(36, 36, 8.5 * inch - 36, 36)
        
        self.drawString(36, 22, "CONFIDENTIAL — HCL INTERNSHIP TECHNICAL PORTFOLIO")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 22, page_str)
        self.restoreState()


def create_langgraph_flowchart_drawing():
    """
    Creates a clean vector flowchart drawing of the 5-Agent LangGraph Pipeline.
    """
    d = Drawing(540, 95)
    
    nodes = [
        ("User Query", "#1E293B", 10),
        ("Query Rewriter", "#2563EB", 90),
        ("Retriever + Reranker", "#7C3AED", 180),
        ("Relevance Grader", "#D97706", 280),
        ("Answer Generator", "#2563EB", 375),
        ("Hallucination Check", "#059669", 465),
    ]
    
    for label, bg_hex, x in nodes:
        d.add(Rect(x, 25, 65, 45, rx=5, ry=5, fillColor=colors.HexColor(bg_hex), strokeColor=colors.HexColor("#0F172A"), strokeWidth=1))
        words = label.split(" ")
        if len(words) == 1:
            d.add(String(x + 32, 44, words[0], fontName="Helvetica-Bold", fontSize=8, textAnchor="middle", fillColor=colors.white))
        elif len(words) == 2:
            d.add(String(x + 32, 50, words[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.white))
            d.add(String(x + 32, 36, words[1], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.white))
        else:
            d.add(String(x + 32, 54, words[0], fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.white))
            d.add(String(x + 32, 43, words[1], fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.white))
            d.add(String(x + 32, 32, words[2], fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.white))
            
        if x < 465:
            arrow_start_x = x + 65
            arrow_end_x = x + 88
            d.add(Line(arrow_start_x, 47, arrow_end_x, 47, strokeColor=colors.HexColor("#64748B"), strokeWidth=1.5))
            d.add(Polygon([arrow_end_x, 47, arrow_end_x - 4, 50, arrow_end_x - 4, 44], fillColor=colors.HexColor("#64748B"), strokeColor=colors.HexColor("#64748B")))
            
    d.add(String(270, 6, "Figure 1: State transitions in the 5-Agent LangGraph State Machine", fontName="Helvetica-Oblique", fontSize=8.5, textAnchor="middle", fillColor=colors.HexColor("#475569")))
    return d


def create_ingestion_flowchart_drawing():
    """
    Creates a clean vector flowchart drawing of the Document Ingestion & Chunking Pipeline.
    """
    d = Drawing(540, 70)
    steps = [
        ("PDF / DOCX Upload", "#0F172A", 15),
        ("Text & Page Extract", "#334155", 130),
        ("500-Token Chunking", "#2563EB", 245),
        ("all-MiniLM-L6 Embed", "#7C3AED", 360),
        ("ChromaDB Vector Store", "#059669", 455)
    ]
    for label, bg_hex, x in steps:
        d.add(Rect(x, 20, 75, 38, rx=4, ry=4, fillColor=colors.HexColor(bg_hex), strokeColor=colors.HexColor("#0F172A"), strokeWidth=1))
        words = label.split(" ")
        if len(words) == 2:
            d.add(String(x + 37.5, 42, words[0], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.white))
            d.add(String(x + 37.5, 30, words[1], fontName="Helvetica-Bold", fontSize=7.5, textAnchor="middle", fillColor=colors.white))
        elif len(words) == 3:
            d.add(String(x + 37.5, 45, words[0], fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.white))
            d.add(String(x + 37.5, 34, f"{words[1]} {words[2]}", fontName="Helvetica-Bold", fontSize=7, textAnchor="middle", fillColor=colors.white))
        else:
            d.add(String(x + 37.5, 36, label, fontName="Helvetica-Bold", fontSize=8, textAnchor="middle", fillColor=colors.white))

        if x < 455:
            d.add(Line(x + 75, 39, x + 98, 39, strokeColor=colors.HexColor("#64748B"), strokeWidth=1.5))
            d.add(Polygon([x + 98, 39, x + 94, 42, x + 94, 36], fillColor=colors.HexColor("#64748B"), strokeColor=colors.HexColor("#64748B")))
            
    d.add(String(270, 4, "Figure 2: Document Ingestion, Token Chunking, and Embedding Storage Pipeline", fontName="Helvetica-Oblique", fontSize=8.5, textAnchor="middle", fillColor=colors.HexColor("#475569")))
    return d


def generate_pdf(output_filename="rag_architecture_HCL_internship.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    repo_url = "https://github.com/aditya0si/agentic_rag_system"
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=5
    )
    
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    code_path_style = ParagraphStyle(
        "CodePathStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1D4ED8")
    )

    story = []

    # =========================================================================
    # PAGE 1: HEADER BANNER, EXECUTIVE SUMMARY & LANGGRAPH DIAGRAM
    # =========================================================================
    
    header_data = [
        [
            Paragraph("<b>AGENTIC RAG RESEARCH ASSISTANT</b>", title_style),
            Paragraph(f"<b>HCL INTERNSHIP PORTFOLIO</b><br/><a href='{repo_url}'><u>github.com/aditya0si/agentic_rag_system</u></a>", meta_style)
        ],
        [
            Paragraph("Production-Grade Multi-Agent RAG System with LangGraph, Cross-Encoder Reranking & Hallucination Guardrails", subtitle_style),
            Paragraph("<b>Author:</b> Aditya Singh<br/><b>Date:</b> July 2026", meta_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[350, 190])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=8, spaceBefore=4))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary:</b> The <i>Agentic RAG Research Assistant</i> is an enterprise-grade document intelligence system "
        "engineered by <b>Aditya Singh</b> to eliminate hallucinations and low-precision retrieval common in traditional RAG implementations. "
        "Powered by a 5-agent <b>LangGraph state machine</b>, the system orchestrates query rewriting, vector retrieval, cross-encoder re-ranking, "
        "LLM relevance grading, grounded answer generation, and strict post-generation hallucination verification. "
        "Every generated claim is directly tied to page-level document citations. Repository: "
        f"<a href='{repo_url}'><u>github.com/aditya0si/agentic_rag_system</u></a>"
    )
    summary_table = Table([[Paragraph(summary_text, body_style)]], colWidths=[540])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # Key Differentiators Table
    story.append(Paragraph("Key System Differentiators & Performance Highlights", h1_style))
    diff_data = [
        [Paragraph("Architectural Feature", table_header_style), Paragraph("Naive RAG Baseline", table_header_style), Paragraph("Agentic RAG Implementation", table_header_style), Paragraph("Impact / Benefit", table_header_style)],
        [Paragraph("State Orchestration", table_cell_bold), Paragraph("Linear single-pass chain", table_cell_style), Paragraph("5-Node LangGraph State Machine", table_cell_style), Paragraph("Dynamic routing, state trace & node retries", table_cell_style)],
        [Paragraph("Retrieval Precision", table_cell_bold), Paragraph("Top-K vector cosine distance", table_cell_style), Paragraph("Bi-Encoder + Cross-Encoder Re-ranker", table_cell_style), Paragraph("Re-ranks ms-marco models for semantic precision", table_cell_style)],
        [Paragraph("Context Filtering", table_cell_bold), Paragraph("Raw chunks passed directly", table_cell_style), Paragraph("LLM Binary Relevance Grader", table_cell_style), Paragraph("Filters out irrelevant noise prior to synthesis", table_cell_style)],
        [Paragraph("Hallucination Defense", table_cell_bold), Paragraph("None (relies on system prompt)", table_cell_style), Paragraph("Independent Hallucination Checker", table_cell_style), Paragraph("Cross-verifies generated claims vs source context", table_cell_style)],
        [Paragraph("Observability", table_cell_bold), Paragraph("Basic console print statements", table_cell_style), Paragraph("structlog + Prometheus + SSE Stream", table_cell_style), Paragraph("Per-node latency tracking & request correlation", table_cell_style)],
    ]
    diff_table = Table(diff_data, colWidths=[110, 125, 150, 155])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(diff_table)
    story.append(Spacer(1, 8))

    # Architecture Diagram 1: LangGraph Flowchart
    story.append(Paragraph("System Architecture — Multi-Agent LangGraph Pipeline", h1_style))
    story.append(create_langgraph_flowchart_drawing())
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: REAL UI SCREENSHOTS & COMPONENT WALKTHROUGH
    # =========================================================================
    story.append(Paragraph("User Interface & System Telemetry Walkthrough", h1_style))
    story.append(Paragraph(
        "The application features a modern Streamlit frontend (<a href='https://github.com/aditya0si/agentic_rag_system/tree/main/frontend/app.py'><u>frontend/app.py</u></a>) "
        "communicating asynchronously with the FastAPI backend (<a href='https://github.com/aditya0si/agentic_rag_system/tree/main/backend/main.py'><u>backend/main.py</u></a>). "
        "Users receive real-time node progress updates via Server-Sent Events (SSE) and grounded responses with page-level citations.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Embed Chat UI Screenshot
    if os.path.exists("ui_screenshot_chat.png"):
        story.append(Paragraph("<b>UI Screenshot 1: Interactive Chat Interface & Grounded Citations</b>", h2_style))
        story.append(Image("ui_screenshot_chat.png", width=540, height=303))
        story.append(Spacer(1, 8))

    # Embed Metrics UI Screenshot
    if os.path.exists("ui_screenshot_metrics.png"):
        story.append(Paragraph("<b>UI Screenshot 2: Real-time Telemetry & RAGAS Benchmark Dashboard</b>", h2_style))
        story.append(Image("ui_screenshot_metrics.png", width=540, height=225))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: DETAILED AGENT MODULES & INGESTION PIPELINE
    # =========================================================================
    story.append(Paragraph("Document Ingestion & Storage Architecture", h1_style))
    story.append(Paragraph(
        "Documents uploaded via <code>/upload</code> pass through an automated ingestion pipeline "
        "(<a href='https://github.com/aditya0si/agentic_rag_system/tree/main/backend/core/ingestion.py'><u>backend/core/ingestion.py</u></a>). "
        "Text is extracted while retaining page numbers, split into 500-token chunks with 50-token overlap, embedded with <code>all-MiniLM-L6-v2</code>, "
        "and persisted in ChromaDB (<a href='https://github.com/aditya0si/agentic_rag_system/tree/main/backend/core/vector_store.py'><u>backend/core/vector_store.py</u></a>).",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(create_ingestion_flowchart_drawing())
    story.append(Spacer(1, 10))

    story.append(Paragraph("Deep-Dive: The 5 LangGraph Agent Modules", h1_style))

    agents_info = [
        ("1. Query Rewriter Agent", "backend/core/agents/query_rewriter.py", 
         "Analyzes multi-turn conversational history to reformulate implicit follow-up questions into standalone, context-rich search queries."),
        
        ("2. Retriever & Cross-Encoder Reranker", "backend/core/agents/retriever.py & reranker.py", 
         "Over-fetches top candidate chunks from ChromaDB, then re-ranks them using ms-marco-MiniLM-L-6-v2 cross-encoder to elevate high-precision semantic matches."),
        
        ("3. Relevance Grader Agent", "backend/core/agents/relevance_grader.py", 
         "Performs binary relevance checks using LLM structured output parsing on each retrieved chunk. Irrelevant noise is filtered out before synthesis."),
        
        ("4. Answer Generator Agent", "backend/core/agents/answer_generator.py", 
         "Synthesizes grounded answers strictly using filtered chunks. Embeds inline bracketed citations ([1], [2]) linking claims to exact page numbers."),
        
        ("5. Hallucination Checker Agent", "backend/core/agents/hallucination_checker.py", 
         "Performs an independent audit verifying generated statements against original source facts. Attaches explicit warnings if unsupported claims are found.")
    ]

    for title, relpath, desc in agents_info:
        file_url = f"{repo_url}/tree/main/{relpath.split(' ')[0]}"
        card_content = [
            Paragraph(f"<b>{title}</b> — <a href='{file_url}'><u>{relpath}</u></a>", h2_style),
            Paragraph(desc, body_style)
        ]
        card_table = Table([[card_content]], colWidths=[540])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 5))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: TECH STACK TABLE, SECURITY, EVALUATION & CODE REFERENCES
    # =========================================================================
    story.append(Paragraph("Technology Stack & Engineering Rationale", h1_style))
    
    tech_data = [
        [Paragraph("Layer", table_header_style), Paragraph("Technology", table_header_style), Paragraph("Rationale / Key Features", table_header_style)],
        [Paragraph("Backend Framework", table_cell_bold), Paragraph("FastAPI + Uvicorn", table_cell_style), Paragraph("Asynchronous execution, auto OpenAPI docs, Pydantic data validation", table_cell_style)],
        [Paragraph("Agent Orchestration", table_cell_bold), Paragraph("LangGraph 0.2+", table_cell_style), Paragraph("Explicit cyclic state machine, state persistence, debuggable node traces", table_cell_style)],
        [Paragraph("Vector Database", table_cell_bold), Paragraph("ChromaDB (Local)", table_cell_style), Paragraph("Zero-cost, embedded persistence, metadata filtering by document ID", table_cell_style)],
        [Paragraph("Embeddings", table_cell_bold), Paragraph("all-MiniLM-L6-v2", table_cell_style), Paragraph("Free local inference, 384-dim semantic embeddings with low latency", table_cell_style)],
        [Paragraph("Re-ranking Engine", table_cell_bold), Paragraph("ms-marco-MiniLM-L-6-v2", table_cell_style), Paragraph("Cross-encoder model improving retrieval precision over bi-encoders", table_cell_style)],
        [Paragraph("LLM Provider", table_cell_bold), Paragraph("Google Gemini 1.5 / OpenAI", table_cell_style), Paragraph("High token window, fast response times, cost-effective structured output", table_cell_style)],
        [Paragraph("Frontend UI", table_cell_bold), Paragraph("Streamlit", table_cell_style), Paragraph("Rapid reactive web interface with native chat components & SSE streaming", table_cell_style)],
        [Paragraph("Observability", table_cell_bold), Paragraph("structlog + Prometheus", table_cell_style), Paragraph("JSON formatted logs, X-Request-ID request tracing, custom metrics", table_cell_style)],
        [Paragraph("Security", table_cell_bold), Paragraph("Token-Bucket Rate Limiter", table_cell_style), Paragraph("Defensive rate limiting, prompt injection detection, input sanitization", table_cell_style)],
        [Paragraph("Evaluation", table_cell_bold), Paragraph("RAGAS Framework", table_cell_style), Paragraph("Standardized evaluation for faithfulness, relevancy, precision, recall", table_cell_style)],
    ]
    tech_table = Table(tech_data, colWidths=[120, 150, 270])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 8))

    # Defense-in-Depth Security & Observability Box
    sec_title = "<b>Security & Enterprise Readiness Guardrails:</b>"
    sec_text = (
        "• <b>Token-Bucket Rate Limiting:</b> Enforced via <code>backend/middleware/rate_limiter.py</code> per IP.<br/>"
        "• <b>Prompt Injection Guardrails:</b> Input text checked in <code>backend/core/security.py</code> for injection vectors.<br/>"
        "• <b>Distributed Request Tracing:</b> Every HTTP call injects an <code>X-Request-ID</code> correlation header.<br/>"
        "• <b>In-Memory Caching:</b> TTL-based caching layer (<code>backend/core/cache.py</code>) caches embeddings (24h) and LLM queries (1h)."
    )
    sec_table = Table([[Paragraph(sec_title, h2_style)], [Paragraph(sec_text, body_style)]], colWidths=[540])
    sec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(sec_table)
    story.append(Spacer(1, 8))

    # Codebase File Index Reference Table with GitHub Links
    story.append(Paragraph("Codebase File Index & Repository Reference", h1_style))
    
    file_index_data = [
        [Paragraph("Module Role", table_header_style), Paragraph("Repository File Path", table_header_style), Paragraph("Key Symbols / Classes", table_header_style)],
        [Paragraph("FastAPI App Entry", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/main.py'><u>backend/main.py</u></a>", code_path_style), Paragraph("app, lifespan, routers", table_cell_style)],
        [Paragraph("LangGraph Pipeline", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/core/agents/graph.py'><u>backend/core/agents/graph.py</u></a>", code_path_style), Paragraph("AgentState, build_graph()", table_cell_style)],
        [Paragraph("Query Rewriter", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/core/agents/query_rewriter.py'><u>backend/core/agents/query_rewriter.py</u></a>", code_path_style), Paragraph("rewrite_query()", table_cell_style)],
        [Paragraph("Vector Store Manager", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/core/vector_store.py'><u>backend/core/vector_store.py</u></a>", code_path_style), Paragraph("VectorStoreManager", table_cell_style)],
        [Paragraph("Cross-Encoder Reranker", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/core/agents/reranker.py'><u>backend/core/agents/reranker.py</u></a>", code_path_style), Paragraph("rerank_chunks()", table_cell_style)],
        [Paragraph("Hallucination Checker", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/backend/core/agents/hallucination_checker.py'><u>backend/core/agents/hallucination_checker.py</u></a>", code_path_style), Paragraph("check_hallucination()", table_cell_style)],
        [Paragraph("Streamlit UI Application", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/frontend/app.py'><u>frontend/app.py</u></a>", code_path_style), Paragraph("main(), render_chat()", table_cell_style)],
        [Paragraph("RAGAS Evaluation Framework", table_cell_bold), Paragraph(f"<a href='{repo_url}/tree/main/eval/run_evaluation.py'><u>eval/run_evaluation.py</u></a>", code_path_style), Paragraph("evaluate_ragas()", table_cell_style)],
    ]
    file_table = Table(file_index_data, colWidths=[130, 210, 200])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(file_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled PDF: {output_filename}")

if __name__ == "__main__":
    generate_pdf("rag_architecture_HCL_internship.pdf")
