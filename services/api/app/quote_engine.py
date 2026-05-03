"""Quote engine: corrosion assessment → itemized quote.

Converts corrosion mask areas (m²) and severity into a dollar quote
using the price book. Includes confidence gating for human review.
"""

from dataclasses import dataclass
from typing import Optional

from app.db import PriceBookEntry, get_supabase


@dataclass
class QuoteResult:
    """Output of the quote engine."""

    total_amount: float
    line_items: list[dict]
    currency: str = "USD"
    requires_human_review: bool = False
    review_reason: Optional[str] = None


# Confidence thresholds for human review gating
CONFIDENCE_REVIEW_THRESHOLD = 0.7  # below this → force human review
SEVERITY_REVIEW_THRESHOLD = "severe"  # severe always gets review
AREA_REVIEW_THRESHOLD_M2 = 500  # large roofs get review


def compute_quote(
    roof_area_m2: float,
    corroded_area_m2: float,
    corrosion_percent: float,
    severity: str,
    confidence: float,
    material: str = "corrugated_metal",
    region: str = "default",
) -> QuoteResult:
    """Compute an itemized quote from a corrosion assessment.

    Logic:
    - If corrosion < 20%: offer repair + coating
    - If corrosion 20–60%: offer repair + coating + partial replacement
    - If corrosion > 60%: full replacement recommended
    - Always include inspection fee
    - Confidence gating: if confidence < 0.7, flag for human review
    """
    line_items = []
    requires_review = False
    review_reason = None

    # ── Confidence gate ─────────────────────────────────────
    if confidence < CONFIDENCE_REVIEW_THRESHOLD:
        requires_review = True
        review_reason = f"Low model confidence ({confidence:.0%})"

    if severity == SEVERITY_REVIEW_THRESHOLD:
        requires_review = True
        review_reason = (review_reason or "") + " Severe corrosion classification."

    if roof_area_m2 > AREA_REVIEW_THRESHOLD_M2:
        requires_review = True
        review_reason = (review_reason or "") + f" Large roof area ({roof_area_m2:.0f} m²)."

    # ── Fetch prices from price book ────────────────────────
    try:
        supabase = get_supabase()
        prices = supabase.table("price_book").select("*").eq("material", material).eq("region", region).execute()
        price_map = {p["service_type"]: p["price_per_m2"] for p in prices.data}
    except Exception:
        # Fallback prices if DB not available
        price_map = {
            "replacement": 45.00,
            "repair": 25.00,
            "coating": 15.00,
        }

    # ── Inspection fee (flat) ───────────────────────────────
    line_items.append({
        "description": "Satellite roof inspection & corrosion analysis",
        "quantity": 1,
        "unit": "each",
        "unit_price": 150.00,
        "total": 150.00,
    })

    # ── Service recommendations based on severity ───────────
    if corrosion_percent < 20:
        # Light corrosion: repair + protective coating
        repair_area = corroded_area_m2
        coating_area = roof_area_m2  # coat entire roof

        line_items.append({
            "description": f"Corrosion repair ({severity})",
            "quantity": round(repair_area, 1),
            "unit": "m²",
            "unit_price": price_map.get("repair", 25.00),
            "total": round(repair_area * price_map.get("repair", 25.00), 2),
        })
        line_items.append({
            "description": "Protective anti-corrosion coating (full roof)",
            "quantity": round(coating_area, 1),
            "unit": "m²",
            "unit_price": price_map.get("coating", 15.00),
            "total": round(coating_area * price_map.get("coating", 15.00), 2),
        })

    elif corrosion_percent < 60:
        # Moderate: partial replacement + coating
        replace_area = corroded_area_m2
        coating_area = roof_area_m2 - corroded_area_m2

        line_items.append({
            "description": f"Corroded section replacement ({severity})",
            "quantity": round(replace_area, 1),
            "unit": "m²",
            "unit_price": price_map.get("replacement", 45.00),
            "total": round(replace_area * price_map.get("replacement", 45.00), 2),
        })
        line_items.append({
            "description": "Protective coating (remaining roof)",
            "quantity": round(max(coating_area, 0), 1),
            "unit": "m²",
            "unit_price": price_map.get("coating", 15.00),
            "total": round(max(coating_area, 0) * price_map.get("coating", 15.00), 2),
        })

    else:
        # Severe: full replacement recommended
        line_items.append({
            "description": f"Full roof replacement recommended ({severity}, {corrosion_percent:.0f}% affected)",
            "quantity": round(roof_area_m2, 1),
            "unit": "m²",
            "unit_price": price_map.get("replacement", 45.00),
            "total": round(roof_area_m2 * price_map.get("replacement", 45.00), 2),
        })

    # ── Waste factor (10% for replacement, 5% for repair) ───
    for item in line_items[1:]:  # skip inspection fee
        if "replacement" in item["description"].lower():
            waste_qty = round(item["quantity"] * 0.10, 1)
            line_items.append({
                "description": "Waste allowance (10%)",
                "quantity": waste_qty,
                "unit": "m²",
                "unit_price": item["unit_price"],
                "total": round(waste_qty * item["unit_price"], 2),
            })
            break  # only one waste line

    total = sum(item["total"] for item in line_items)

    return QuoteResult(
        total_amount=round(total, 2),
        line_items=line_items,
        requires_human_review=requires_review,
        review_reason=review_reason.strip() if review_reason else None,
    )
