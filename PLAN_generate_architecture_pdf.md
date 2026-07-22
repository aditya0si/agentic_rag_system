# PLAN_generate_architecture_pdf.md

## SECTION A — GOAL DEFINITION

1. **What is being built or changed?**
   Creating a Python PDF generation script (`generate_architecture_pdf.py`) to build a high-impact, professional PDF document named `rag_architecture_HCL_internship.pdf` for an HCL internship portfolio presentation.

2. **What does "done" look like?**
   A polished, multi-page PDF document named [rag_architecture_HCL_internship.pdf](file:///c:/Users/oliad/Desktop/Agentic%20Research/rag_architecture_HCL_internship.pdf) in the project root containing:
   - HCL Internship project header and executive architecture overview.
   - Flowchart diagrams for the 5-Agent LangGraph State Machine & Ingestion Pipeline.
   - High-fidelity UI screenshots showing the document chat interface, inline citations, agent execution trace badges, and observability metrics dashboard.
   - In-depth technical breakdown of Frontend (Streamlit), Backend (FastAPI, middleware, security), Vector Store (ChromaDB, cross-encoder re-ranking), and RAGAS evaluation benchmark framework.
   - Full technology stack comparison table.

3. **What is explicitly out of scope for this task?**
   - Modifying backend service logic or production application code.

---

## SECTION B — TECH STACK

- **Language**: Python 3.11+
- **PDF Engine**: `reportlab` (ReportLab Platypus, Flowables, Paragraphs, Tables, Drawing/Shapes)
- **Asset Generation**: High-resolution UI screenshots (`ui_screenshot_chat.png`, `ui_screenshot_metrics.png`) generated via Pillow and embedded into ReportLab flowables.

---

## SECTION C — SESSION MODULARIZATION

### Session 1: Asset Generation & Script Architecture Setup
- **OBJECTIVE**: Generate UI screenshot assets (`ui_screenshot_chat.png`, `ui_screenshot_metrics.png`), build architectural flowcharts, and initialize `generate_architecture_pdf.py`.
- **SCOPE**: Image assets in project root and `generate_architecture_pdf.py`.
- **OUTPUT**: UI screenshots generated and saved as PNG files; `generate_architecture_pdf.py` created with custom layout, color schemes, cover page, and image drawing helpers.
- **CONNECTS TO**: Session 2 (Compiles the PDF document).
- **FAILURE SURFACE**: ReportLab image sizing/scaling issues or missing dependencies.

### Session 2: PDF Compilation, Rendering & Verification
- **OBJECTIVE**: Run `generate_architecture_pdf.py` to create `rag_architecture_HCL_internship.pdf`, verify layout across all pages, and deliver the final document.
- **SCOPE**: `rag_architecture_HCL_internship.pdf`.
- **OUTPUT**: Verified, perfectly formatted `rag_architecture_HCL_internship.pdf` file.
- **CONNECTS TO**: Completion of user request.
- **FAILURE SURFACE**: Page overflow or text clipping.

---

## SECTION D — PROGRESS CHECKLIST

- [x] Session 1: Asset Generation & Script Architecture Setup
  - [x] Update PLAN file for HCL Internship PDF requirements
  - [x] Generate `ui_screenshot_chat.png` (Streamlit UI with document upload, chat, citations, and agent traces)
  - [x] Generate `ui_screenshot_metrics.png` (Analytics dashboard with node latencies, RAGAS scores, and relevance metrics)
  - [x] Write `generate_architecture_pdf.py` with custom reportlab styles, tables, diagrams, and image flowables
- [x] Session 2: PDF Compilation, Rendering & Verification
  - [x] Execute `generate_architecture_pdf.py` to produce `rag_architecture_HCL_internship.pdf`
  - [x] Verify PDF generation, page layout, image rendering, and text alignment
  - [x] Deliver final output link and summary to user
