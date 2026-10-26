"""Generate a comprehensive architecture & operational PDF report for the Financial Relevance Intelligence Engine."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Financial Relevance Intelligence Engine (FRIE) — Technical Architecture & Operation")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential — Internal Architecture & Engineering Guide")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        
        self.restoreState()

def create_system_pdf(filename="Financial_Relevance_Intelligence_Engine_Architecture.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1E3A8A")   # Dark Navy
    secondary_color = colors.HexColor("#0284C7") # Teal/Cyan
    text_dark = colors.HexColor("#0F172A")       # Slate 900
    text_muted = colors.HexColor("#475569")      # Slate 600

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=6,
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("Financial Relevance Intelligence Engine", title_style))
    story.append(Paragraph("End-to-End Technical Architecture, Multi-Stage Classification Pipeline & System Blueprint", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=14))

    # Executive Summary Box
    exec_summary_text = (
        "<b>System Purpose:</b> In modern automated financial and contract analysis workflows, feeding every raw document or unvetted text directly into expensive Large Language Models (LLMs) incurs massive compute costs, token waste, and high operational latency. The <b>Financial Relevance Intelligence Engine (FRIE)</b> serves as a production-grade filter and extraction preparatory engine. It ingests heterogenous file formats (PDF, DOCX, XLSX, XLS, CSV, TXT, PNG, JPG), determines document and chunk-level financial relevance, pinpoints concrete obligation and transaction passages, and generates structured handoff payloads for downstream financial extraction."
    )
    summary_table = Table(
        [[Paragraph(exec_summary_text, body_style)]],
        colWidths=[504],
    )
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BORDER', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Section 1: Architecture Overview
    story.append(Paragraph("1. High-Level Architectural Pipeline", h1_style))
    story.append(Paragraph(
        "FRIE processes incoming data through a sequence of six decoupled, idempotent processing stages designed to ensure maximum throughput, robust security, and auditable zero-data-loss execution:",
        body_style
    ))

    stages_data = [
        ["Stage", "Name", "Core Technology & Strategy"],
        ["Stage A", "Ingestion & Security", "SHA-256 deduplication, magic bytes validation, zip-bomb ratio checks, filename sanitization, persistent local storage."],
        ["Stage B", "Format-Aware Extraction", "PyMuPDF (native PDF & layout), RapidOCR/PaddleOCR (scanned pages & images), openpyxl/xlrd (tabular coordinate preservation), python-docx (structural hierarchy)."],
        ["Stage C", "Structure-Aware Chunking", "Sentence boundary chunking with heading breadcrumb preservation, tabular row-slice grouping with header/footnote binding, stable chunk hashing."],
        ["Stage D", "Reference Dataset & FAISS", "Curated 14-category financial ontology + hard negatives. Sentence-Transformers (all-MiniLM-L6-v2) embedding with SQLite caching and FAISS L2/IP vector index."],
        ["Stage E", "Hybrid Classification", "Dual-engine scoring: Inexpensive lexical & obligation pattern heuristics blended with contrastive multi-example vector similarity."],
        ["Stage F", "Document-Level Aggregation", "Ensemble scoring: FINANCIAL_RELEVANT, MIXED_CONTENT, NEEDS_REVIEW, or NOT_FINANCIAL. Downstream handoff JSON schema generation."]
    ]
    t_stages = Table(stages_data, colWidths=[60, 140, 304])
    t_stages.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_stages)
    story.append(Spacer(1, 14))

    # Section 2: Format-Aware Parsing & OCR Strategy
    story.append(Paragraph("2. Format-Aware Parsing & OCR Strategy", h1_style))
    story.append(Paragraph(
        "A foundational principle of FRIE is <b>intelligent format awareness</b>: never treating structured files as flat text, and never incurring unnecessary OCR overhead on searchable documents.",
        body_style
    ))
    story.append(Paragraph("• <b>PDF Processing:</b> Native digital text is extracted via PyMuPDF dict inspection, which retains font metrics, spans, and bounding coordinates. When a page has fewer than 40 native characters, it is recognized as scanned or low-quality and routed to the OCR engine with image deskewing and adaptive contrast enhancements.", bullet_style))
    story.append(Paragraph("• <b>Spreadsheets (XLSX, XLS, CSV):</b> Instead of flattening rows into arbitrary text, sheets are split into table regions. The system preserves sheet names, cell coordinates (e.g., Sheet1!B4:F12), headers, formulas (recorded as untrusted text, never evaluated to prevent spreadsheet formula injection), and number formats.", bullet_style))
    story.append(Paragraph("• <b>Images (PNG, JPG):</b> Processed through RapidOCR (ONNX Runtime engine) or Tesseract with multi-line coordinate grouping and per-paragraph confidence tracking.", bullet_style))
    story.append(Paragraph("• <b>Word (DOCX) & Text:</b> Preserves paragraph indices, heading hierarchies (H1/H2 breadcrumbs), and embedded table cell structures.", bullet_style))

    story.append(PageBreak())

    # Section 3: Reference Dataset & Vector Indexing
    story.append(Paragraph("3. Financial Reference Dataset & 14-Category Ontology", h1_style))
    story.append(Paragraph(
        "FRIE relies on a curated, versioned reference dataset (`financial_reference_v1.jsonl`) spanning 14 key financial domains alongside carefully constructed hard-negative non-financial entries:",
        body_style
    ))

    cat_table_data = [
        ["#", "Category Slug", "Domain Description"],
        ["1", "income_revenue", "Gross/net revenue, subscription earnings, royalties, deferred revenue"],
        ["2", "expenses_operating_costs", "OpEx, COGS, SG&A expenses, payroll, operational expenditures"],
        ["3", "accounts_payable_receivable", "Vendor ledgers, trade receivables, aging schedules, trade payables"],
        ["4", "invoices_purchase_orders", "Purchase order milestones, commercial invoices, line totals, Net 30/60 terms"],
        ["5", "cash_flow_bank_transactions", "Operating cash flow, bank statements, wire transfers (SWIFT/ACH), opening balances"],
        ["6", "loans_interest_repayment", "Term facilities, promissory notes, principal installments, SOFR/LIBOR margins"],
        ["7", "rent_leases_recurring_payments", "Base rent, security deposits, lease escalations, recurring retainers"],
        ["8", "contracts_payment_obligations", "Firm covenants: 'shall pay', milestones, consideration, fees for service"],
        ["9", "insurance", "Premiums, deductibles, coverage limits, indemnities, claim payouts"],
        ["10", "taxes_duties", "Corporate income tax, VAT/GST withholding, statutory duties, customs"],
        ["11", "penalties_late_fees", "Default interest, liquidated damages, overdue fees, 1.5% late penalties"],
        ["12", "assets_liabilities_investments", "Balance sheet assets, shareholders' equity, goodwill, book/fair value"],
        ["13", "maturity_renewals_deadlines", "Debenture maturity dates, auto-renewal covenants, payment deadlines"],
        ["14", "refunds_credits_settlements", "Credit memos, warranty settlement payouts, customer chargebacks"],
        ["15", "non_financial (Hard Negatives)", "Scientific charge/bond/deposit, $var code, NDAs, boilerplate legal terms"]
    ]
    t_cats = Table(cat_table_data, colWidths=[24, 180, 300])
    t_cats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_cats)
    story.append(Spacer(1, 14))

    # Section 4: Hybrid Classification Engine
    story.append(Paragraph("4. The Hybrid Classification & Scoring Engine", h1_style))
    story.append(Paragraph(
        "Rather than relying strictly on keyword searches or raw semantic embeddings (both of which suffer high false-positive rates when isolated), FRIE implements a <b>calibrated hybrid engine</b>:",
        body_style
    ))

    story.append(Paragraph("<b>A. Inexpensive Signal Analysis (Lexical & Grammatical):</b>", h2_style))
    story.append(Paragraph("Evaluates regex and token patterns without embedding overhead: currency quantities ($4.2M, EUR 12,850.00, Rs. 45,00,000), payment phrases ('shall pay', 'equal monthly installments', 'net 30 days'), financial parties ('Borrower', 'Lessor', 'Tenant'), rate indicators ('SOFR + 2.75%', '8.5% p.a.'), and table headers. Crucially, it applies negative suppression for code syntax (e.g. `$HOME`, `${var}`, regex `^$`) and non-financial polysemy (e.g. 'electric charge', 'chemical bond', 'mineral deposit').", body_style))

    story.append(Paragraph("<b>B. Semantic Embedding & Category Similarity:</b>", h2_style))
    story.append(Paragraph("Generates L2-normalized 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. Computes cosine similarity across the FAISS vector index. The score is calculated as a <b>contrastive multi-example aggregation</b> (mean of top-2 matches in a category minus the top hard-negative matches), preventing single matching words from polluting relevance.", body_style))

    story.append(Paragraph("<b>C. Calibrated Three-Way Decision:</b>", h2_style))
    story.append(Paragraph(
        "Each chunk receives an authoritative decision based on empirically validated thresholds: "
        "<b>RELEVANT (Score ≥ 0.60)</b>, <b>UNCERTAIN (0.35 &lt; Score &lt; 0.60)</b>, or <b>IRRELEVANT (Score ≤ 0.35)</b>. "
        "Uncertain chunks can optionally invoke an isolated secondary LLM tie-breaker with strict JSON output schemas.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 5: Document Aggregation & Downstream Handoff
    story.append(Paragraph("5. Document-Level Aggregation & Downstream Handoff", h1_style))
    story.append(Paragraph(
        "Chunk-level verdicts are assembled into an overall document state:",
        body_style
    ))
    story.append(Paragraph("• <b>FINANCIAL_RELEVANT:</b> Dominantly financial documents (e.g. quarterly balance sheets, loan agreements).", bullet_style))
    story.append(Paragraph("• <b>MIXED_CONTENT:</b> Long commercial contracts or company announcements that contain both non-financial sections (e.g., NDA clauses, governing law) and concrete financial commitments. <i>Only relevant regions are prepared for downstream LLM extraction</i>, preventing massive token waste.", bullet_style))
    story.append(Paragraph("• <b>NEEDS_REVIEW:</b> Flagged when all chunks appear non-financial but OCR confidence is substandard or pages have sparse text, safeguarding against false negatives.", bullet_style))
    story.append(Paragraph("• <b>NOT_FINANCIAL:</b> Cleanly filtered documents preserved in the audit log without LLM invocation.", bullet_style))

    story.append(PageBreak())

    # Section 6: UI & Operational Features
    story.append(Paragraph("6. Operational Interface & User Experience", h1_style))
    story.append(Paragraph(
        "The system provides a responsive, production-ready React (TypeScript) dashboard running on Vite:",
        body_style
    ))

    ui_features = [
        ["Module", "Route", "Capabilities & Features"],
        ["Dashboard", "/", "Telemetry cards, document status breakdown, average latency, embedding cache hit-rates, chunk decision distribution charts, and live evaluation metrics."],
        ["Upload", "/upload", "Multi-file drag-and-drop & native file picker, client-side format & size validation (50MB cap), real-time progress bars, and instant SHA-256 duplicate detection."],
        ["Document Results", "/documents/:id", "Document status timeline, overall relevance badge, heuristic confidence breakdown, extraction quality alerts, and raw original file downloads."],
        ["Chunk Explorer", "/documents/:id/chunks", "Interactive passage inspector with signal highlighting, source bounding coordinates, table structure renders, and human-in-the-loop reviewer correction tools."],
        ["Reference Dataset", "/reference", "Vector database browser, category distribution metrics, dynamic entry editor, and one-click vector index rebuilding."],
        ["Evaluation", "/evaluation", "Precision-Recall curves, latency benchmarks, OCR failure metrics, and per-format performance statistics."]
    ]
    t_ui = Table(ui_features, colWidths=[90, 110, 304])
    t_ui.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_ui)
    story.append(Spacer(1, 14))

    # Section 7: Security & Production Deployment
    story.append(Paragraph("7. Enterprise Security & Container Deployment", h1_style))
    story.append(Paragraph("• <b>Prompt Injection Isolation:</b> Extracted text from uploaded documents is strictly treated as untrusted data. It is never interpolated into executable scripts or unescaped HTML.", bullet_style))
    story.append(Paragraph("• <b>Local Offline Execution:</b> Both the embedding models (Sentence-Transformers) and the vector index (FAISS) run entirely locally on the server CPU. No document content is uploaded to third-party APIs.", bullet_style))
    story.append(Paragraph("• <b>Macro & Archive Hardening:</b> Rejects macro-enabled Office extensions (`.docm`, `.xlsm`) and evaluates zip-bomb uncompressed size ratios before parsing.", bullet_style))
    story.append(Paragraph("• <b>Dockerized Architecture:</b> A multi-container `docker-compose.yml` configures the FastAPI backend on port 8000 and the production Nginx web server on port 5173.", bullet_style))
    story.append(Spacer(1, 14))

    # Summary Sign-off Box
    signoff_data = [
        ["System Status", "Live & Hosted Locally"],
        ["Backend API", "http://localhost:8000 (FastAPI / OpenAPI)"],
        ["Frontend UI", "http://localhost:5173 (React / Vite)"],
        ["Active Model", "sentence-transformers/all-MiniLM-L6-v2 | FAISS CPU Index"],
        ["OCR Engine", "RapidOCR (ONNX Runtime)"]
    ]
    t_signoff = Table(signoff_data, colWidths=[130, 374])
    t_signoff.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#94A3B8")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_signoff)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    create_system_pdf()
