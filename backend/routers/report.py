import uuid
import io
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from models.db_models import User
from routers.auth import get_current_user
from fastapi.responses import StreamingResponse

from models.schemas import ReportRequest, MonitorBatchItem, MonitorResponse
from routers.analyze import get_analysis
from core.bias_engine import BiasEngine

router = APIRouter()
be = BiasEngine()


@router.post("/report/generate")
async def generate_report(
    request: ReportRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate a full PDF compliance report."""
    analysis = get_analysis(request.analysis_id)

    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()

        NAVY = HexColor("#0C447C")
        TEAL = HexColor("#1D9E75")
        RED = HexColor("#E24B4A")
        AMBER = HexColor("#EF9F27")
        GRAY = HexColor("#F8F8F6")
        DARK_GRAY = HexColor("#4A4A4A")

        title_style = ParagraphStyle("title", parent=styles["Title"],
            textColor=white, fontSize=24, alignment=TA_CENTER, spaceAfter=6)
        h1_style = ParagraphStyle("h1", parent=styles["Heading1"],
            textColor=NAVY, fontSize=16, spaceBefore=16, spaceAfter=8)
        h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
            textColor=NAVY, fontSize=13, spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle("body", parent=styles["Normal"],
            textColor=DARK_GRAY, fontSize=10, spaceAfter=6, leading=14)
        small_style = ParagraphStyle("small", parent=styles["Normal"],
            textColor=DARK_GRAY, fontSize=9, leading=12)

        elements = []
        bias_score = analysis["bias_score"]
        risk_level = analysis["risk_level"]
        domain = analysis["domain"]
        metrics = analysis["metrics"]
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %Human:%M UTC")

        # Cover Page
        elements.append(Spacer(1, 0.3*inch))
        cover_data = [[Paragraph(
            f"<font color='white'><b>FairSight AI Bias Compliance Report</b></font>",
            ParagraphStyle("ct", fontSize=22, textColor=white, alignment=TA_CENTER, leading=28)
        )]]
        cover_table = Table(cover_data, colWidths=[6.5*inch])
        cover_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), NAVY),
            ("TOPPADDING", (0,0), (-1,-1), 24),
            ("BOTTOMPADDING", (0,0), (-1,-1), 24),
            ("LEFTPADDING", (0,0), (-1,-1), 16),
            ("RIGHTPADDING", (0,0), (-1,-1), 16),
        ]))
        elements.append(cover_table)
        elements.append(Spacer(1, 0.2*inch))

        info_data = [
            ["Organization:", request.organization_name],
            ["Analyst:", request.analyst_name],
            ["Domain:", domain.capitalize()],
            ["Generated:", datetime.utcnow().strftime("%B %d, %Y")],
            ["Tool Version:", "FairSight v1.0"],
            ["Risk Level:", risk_level],
            ["Overall Score:", f"{bias_score:.3f} / 1.000"],
        ]
        info_table = Table(info_data, colWidths=[1.8*inch, 4.7*inch])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("TEXTCOLOR", (0,0), (0,-1), NAVY),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [white, GRAY]),
        ]))
        elements.append(info_table)
        elements.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=12))
        elements.append(PageBreak())

        # Executive Summary
        elements.append(Paragraph("Executive Summary", h1_style))
        risk_color = RED if risk_level in ["CRITICAL","HIGH"] else (AMBER if risk_level == "MEDIUM" else TEAL)
        elements.append(Paragraph(
            f"This report presents the results of an AI fairness audit conducted on {request.organization_name}'s "
            f"{domain} decision-making system. The analysis identified a <b>composite fairness score of "
            f"{bias_score:.3f}</b> (scale 0–1, where 1.0 is perfectly fair), representing a "
            f"<b>{risk_level} risk level</b>.",
            body_style
        ))

        passing = sum(1 for m in metrics if m.get("status") == "PASS")
        failing = sum(1 for m in metrics if m.get("status") == "FAIL")
        elements.append(Paragraph(
            f"Of {len(metrics)} fairness metrics evaluated, <b>{passing} passed</b> and <b>{failing} failed</b> "
            f"compliance thresholds. The legal threshold per the EEOC 4/5ths rule is 0.80 for disparate impact.",
            body_style
        ))

        # Metrics Table
        elements.append(Spacer(1, 0.15*inch))
        elements.append(Paragraph("Fairness Metrics", h1_style))
        metric_table_data = [["Metric", "Score", "Threshold", "Status"]]
        for m in metrics:
            status = m["status"]
            score_str = f"{m['score']:.4f}"
            metric_table_data.append([m["name"], score_str, str(m["threshold"]), status])

        mt = Table(metric_table_data, colWidths=[2.4*inch, 1.1*inch, 1.1*inch, 1.9*inch])
        mt_styles = [
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("TEXTCOLOR", (0,0), (-1,0), white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, GRAY]),
            ("GRID", (0,0), (-1,-1), 0.25, DARK_GRAY),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]
        for i, m in enumerate(metrics, start=1):
            color = TEAL if m["status"] == "PASS" else (AMBER if m["status"] == "BORDERLINE" else RED)
            mt_styles.append(("TEXTCOLOR", (3, i), (3, i), color))
            mt_styles.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        mt.setStyle(TableStyle(mt_styles))
        elements.append(mt)

        # Legal Compliance
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Legal Compliance References", h1_style))

        legal_items = [
            ("EU AI Act — Article 10", "Requires technical measures to address bias in training data for high-risk AI systems. Applicable to all systems making decisions affecting individuals."),
            ("US EEOC 4/5ths Rule", f"Disparate impact ratio must be ≥0.80. Current score: {next((m['score'] for m in metrics if 'Disparate' in m['name']), 'N/A'):.3f}."),
        ]
        if domain == "lending":
            legal_items += [
                ("US Fair Housing Act", "Prohibits discrimination in housing-related lending based on race, color, national origin, religion, sex, familial status, or disability."),
                ("Equal Credit Opportunity Act (ECOA)", "Prohibits credit discrimination. Lenders must apply consistent standards across all applicants."),
            ]
        if domain == "hiring":
            legal_items.append(("Title VII (Civil Rights Act)", "Prohibits employment discrimination based on race, color, religion, sex, or national origin."))

        for law, desc in legal_items:
            elements.append(Paragraph(f"<b>{law}</b>", body_style))
            elements.append(Paragraph(desc, small_style))
            elements.append(Spacer(1, 0.05*inch))

        # Audit Trail
        elements.append(PageBreak())
        elements.append(Paragraph("Audit Trail", h1_style))
        audit_data = [
            ["Analysis ID:", analysis["analysis_id"]],
            ["Dataset ID:", analysis["dataset_id"]],
            ["Analyst:", request.analyst_name],
            ["Organization:", request.organization_name],
            ["Timestamp:", datetime.utcnow().isoformat() + "Z"],
            ["Protected Attributes:", ", ".join(analysis["protected_attrs"])],
            ["Outcome Column:", analysis["outcome"]],
            ["Domain:", domain],
            ["Tool Version:", "FairSight v1.0"],
        ]
        audit_table = Table(audit_data, colWidths=[1.8*inch, 4.7*inch])
        audit_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [white, GRAY]),
            ("GRID", (0,0), (-1,-1), 0.25, DARK_GRAY),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        elements.append(audit_table)

        # Signature Section
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("Sign-off", h1_style))
        sig_data = [
            ["Compliance Officer:", "___________________________", "Date:", "______________"],
            ["Technical Lead:", "___________________________", "Date:", "______________"],
        ]
        sig_table = Table(sig_data, colWidths=[1.5*inch, 2.2*inch, 0.7*inch, 2.1*inch])
        sig_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ]))
        elements.append(sig_table)

        doc.build(elements)
        buf.seek(0)

        filename = f"fairsight_report_{request.organization_name.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.post("/monitor/batch", response_model=MonitorResponse)
async def analyze_batches(
    batches: list,
    current_user: User = Depends(get_current_user)
):
    """Analyze multiple dataset batches for drift monitoring."""
    if not batches:
        raise HTTPException(status_code=400, detail="No batches provided.")

    # Process each batch
    from routers.upload import _dataset_store
    be = BiasEngine()

    time_series = []
    alerts = []
    baseline_score = None

    for batch in batches:
        dataset_id = batch.get("dataset_id")
        label = batch.get("label", "Batch")
        timestamp = batch.get("timestamp", datetime.utcnow().isoformat())

        if dataset_id not in _dataset_store:
            continue

        # Simplified batch analysis
        score = 0.7 + (hash(dataset_id) % 30) / 100  # Deterministic mock for demo
        time_series.append({"label": label, "timestamp": timestamp, "score": round(score, 3)})

        if baseline_score is None:
            baseline_score = score

        if score < 0.65:
            alerts.append({
                "label": label,
                "timestamp": timestamp,
                "score": round(score, 3),
                "message": f"{label} — Disparate impact dropped to {score:.2f}. ALERT triggered.",
                "severity": "HIGH",
            })

    drift_score = 0.0
    if baseline_score and time_series:
        latest = time_series[-1]["score"]
        drift_score = round(abs(latest - baseline_score) / baseline_score, 4)

    recommendations = [
        "Review model training data for the latest quarter",
        "Check for shifts in applicant pool composition",
        "Re-run full bias audit with updated dataset",
        "Consider reweighing if drift exceeds 15%",
    ]

    return MonitorResponse(
        time_series=time_series,
        drift_score=drift_score,
        alerts=alerts,
        recommendations=recommendations,
    )
