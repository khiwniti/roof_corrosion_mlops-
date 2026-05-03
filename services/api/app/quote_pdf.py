"""Quote PDF generation with corrosion overlay and itemized breakdown.

Generates a professional quote document that includes:
- Customer and property information
- Corrosion assessment summary
- Satellite imagery overlay showing corroded areas
- Itemized cost breakdown
- Confidence disclaimer
- Terms and conditions

Uses reportlab for PDF generation.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Optional

import boto3
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# Severity color map
SEVERITY_COLORS = {
    "none": colors.green,
    "light": colors.Color(0.8, 0.8, 0),  # yellow
    "moderate": colors.orange,
    "severe": colors.red,
}


def generate_quote_pdf(
    job_id: str,
    address: str,
    assessment: dict,
    quote: dict,
    customer_name: str = "Valued Customer",
    overlay_image_bytes: Optional[bytes] = None,
    company_name: str = "Roof Corrosion AI",
) -> bytes:
    """Generate a quote PDF and return as bytes.

    Args:
        job_id: Unique job identifier
        address: Property address
        assessment: Assessment dict with roof_area_m2, corroded_area_m2, etc.
        quote: Quote dict with total_amount, line_items, etc.
        customer_name: Customer display name
        overlay_image_bytes: Optional PNG/JPEG of corrosion overlay
        company_name: Company name for the quote header

    Returns:
        PDF file bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        name="CompanyName",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="QuoteTitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#e94560"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
        leading=10,
    ))
    styles.add(ParagraphStyle(
        name="SeverityBadge",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))

    elements = []

    # ── Header ───────────────────────────────────────────────
    elements.append(Paragraph(company_name, styles["CompanyName"]))
    elements.append(Paragraph("Satellite Roof Corrosion Analysis & Quote", styles["QuoteTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e94560")))
    elements.append(Spacer(1, 12))

    # ── Quote metadata ───────────────────────────────────────
    quote_date = datetime.utcnow().strftime("%B %d, %Y")
    valid_until = (datetime.utcnow() + timedelta(days=30)).strftime("%B %d, %Y")
    quote_number = f"Q-{job_id[:8].upper()}"

    meta_data = [
        ["Quote Number:", quote_number],
        ["Date:", quote_date],
        ["Valid Until:", valid_until],
        ["Customer:", customer_name],
        ["Property:", address or "See coordinates in assessment"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 16))

    # ── Assessment summary ───────────────────────────────────
    elements.append(Paragraph("Corrosion Assessment Summary", styles["Heading3"]))

    severity = assessment.get("severity", "none")
    severity_color = SEVERITY_COLORS.get(severity, colors.gray)

    summary_data = [
        ["Metric", "Value"],
        ["Total Roof Area", f"{assessment.get('roof_area_m2', 0):.0f} m²"],
        ["Corroded Area", f"{assessment.get('corroded_area_m2', 0):.1f} m²"],
        ["Corrosion Extent", f"{assessment.get('corrosion_percent', 0):.1f}%"],
        ["Severity", severity.upper()],
        ["Model Confidence", f"{assessment.get('confidence', 0) * 100:.0f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        # Severity row highlight
        ("BACKGROUND", (0, 4), (-1, 4), severity_color),
        ("TEXTCOLOR", (0, 4), (-1, 4), colors.white),
        ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    # ── Overlay image ────────────────────────────────────────
    if overlay_image_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(overlay_image_bytes))
            img_width = 14 * cm
            img_height = img_width * img.height / img.width
            rl_img = RLImage(io.BytesIO(overlay_image_bytes), width=img_width, height=img_height)
            elements.append(Paragraph("Satellite Corrosion Overlay", styles["Heading3"]))
            elements.append(rl_img)
            elements.append(Spacer(1, 12))
        except Exception:
            pass  # Skip image if it fails

    # ── Itemized quote ────────────────────────────────────────
    elements.append(Paragraph("Itemized Quote", styles["Heading3"]))

    line_items = quote.get("line_items", [])
    quote_data = [["Description", "Qty", "Unit", "Unit Price", "Total"]]
    for item in line_items:
        quote_data.append([
            item.get("description", ""),
            f"{item.get('quantity', 0):.1f}",
            item.get("unit", ""),
            f"${item.get('unit_price', 0):.2f}",
            f"${item.get('total', 0):.2f}",
        ])

    # Total row
    quote_data.append([
        "", "", "", "TOTAL",
        f"${quote.get('total_amount', 0):.2f}",
    ])

    quote_table = Table(quote_data, colWidths=[7 * cm, 2 * cm, 2 * cm, 2.5 * cm, 2.5 * cm])
    quote_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (3, -1), "CENTER"),
        ("ALIGN", (3, -1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -2), 0.5, colors.HexColor("#e0e0e0")),
        ("LINEABOVE", (0, -1), (-1, -1), 1.5, colors.HexColor("#1a1a2e")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, -1), (-1, -1), 11),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f0f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8f8f8")]),
    ]))
    elements.append(quote_table)
    elements.append(Spacer(1, 16))

    # ── Confidence disclaimer ────────────────────────────────
    confidence = assessment.get("confidence", 0)
    if confidence < 0.7:
        disclaimer_text = (
            f"<b>⚠ HUMAN REVIEW REQUIRED</b><br/>"
            f"This assessment has a model confidence of {confidence * 100:.0f}%, "
            f"which is below our automatic approval threshold of 70%. "
            f"A roofing specialist will review this analysis before the quote is finalized."
        )
    else:
        disclaimer_text = (
            f"This assessment is based on satellite imagery analysis with {confidence * 100:.0f}% "
            f"model confidence. Satellite imagery capture date is listed in the assessment summary. "
            f"Actual roof condition may differ from the date of capture."
        )

    elements.append(Paragraph(disclaimer_text, styles["Disclaimer"]))
    elements.append(Spacer(1, 12))

    # ── Terms ─────────────────────────────────────────────────
    terms_text = (
        "1. This quote is valid for 30 days from the date of issue.<br/>"
        "2. Prices are estimates based on satellite imagery analysis and may vary "
        "after physical inspection.<br/>"
        "3. A site visit is recommended before finalizing any repair/replacement work.<br/>"
        f"4. Quote ID: {quote_number} | Job ID: {job_id}<br/>"
        "5. For questions, contact support@roofcorrosion.ai"
    )
    elements.append(Paragraph("Terms & Conditions", styles["Heading4"]))
    elements.append(Paragraph(terms_text, styles["Disclaimer"]))

    # Build PDF
    doc.build(elements)
    return buffer.getvalue()


def upload_quote_pdf(pdf_bytes: bytes, job_id: str, s3_bucket: str = "") -> str:
    """Upload generated PDF to S3 and return the key."""
    bucket = s3_bucket or os.getenv("AWS_S3_BUCKET", "roof-corrosion-mlops")
    key = f"quotes/{job_id}/quote.pdf"

    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
        return key
    except Exception as e:
        print(f"⚠️  Failed to upload PDF to S3: {e}")
        return key  # Return key anyway for DB record


def generate_and_upload_quote(
    job_id: str,
    address: str,
    assessment: dict,
    quote: dict,
    customer_name: str = "Valued Customer",
    overlay_image_bytes: Optional[bytes] = None,
) -> str:
    """Generate quote PDF and upload to S3. Returns S3 key."""
    pdf_bytes = generate_quote_pdf(
        job_id=job_id,
        address=address,
        assessment=assessment,
        quote=quote,
        customer_name=customer_name,
        overlay_image_bytes=overlay_image_bytes,
    )
    s3_key = upload_quote_pdf(pdf_bytes, job_id)
    return s3_key
