import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

export const maxDuration = 30;

const quotationSchema = z.object({
  jobId: z.string(),
  roofAreaM2: z.number().positive(),
  corrodedAreaM2: z.number().min(0),
  corrosionPercent: z.number().min(0).max(100),
  severity: z.enum(["none", "light", "moderate", "severe"]),
  confidence: z.number().min(0).max(1),
  material: z.string().default("corrugated_metal"),
  region: z.string().default("TH"),
});

/**
 * Generate a quotation PDF for a roof analysis.
 * In production, this would use @react-pdf/renderer.
 * For Phase 0, returns a JSON quotation with line items.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const parsed = quotationSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request body", issues: parsed.error.issues },
        { status: 400 }
      );
    }

    const { roofAreaM2, corrodedAreaM2, corrosionPercent, severity, confidence, material, region } = parsed.data;

    // Simple THB pricing logic (stub)
    const isThai = region === "TH";
    const currency = isThai ? "THB" : "USD";
    const pricePerM2 = isThai ? 850 : 45; // replacement price per m²
    const coatingPricePerM2 = isThai ? 280 : 15;

    const lineItems = [];

    // Always include coating for light/moderate corrosion
    if (severity === "light" || severity === "moderate") {
      lineItems.push({
        description: "Roof coating (anti-corrosion)",
        quantity: Math.ceil(roofAreaM2),
        unit: "m²",
        unitPrice: coatingPricePerM2,
        total: Math.ceil(roofAreaM2) * coatingPricePerM2,
      });
    }

    // Replacement for severe corrosion
    if (severity === "severe" || corrosionPercent > 30) {
      lineItems.push({
        description: "Corroded section replacement",
        quantity: Math.ceil(corrodedAreaM2),
        unit: "m²",
        unitPrice: pricePerM2,
        total: Math.ceil(corrodedAreaM2) * pricePerM2,
      });
    }

    // Labour
    const labourRate = isThai ? 200 : 12; // per m²
    lineItems.push({
      description: "Labour (rip + re-mug)",
      quantity: Math.ceil(roofAreaM2),
      unit: "m²",
      unitPrice: labourRate,
      total: Math.ceil(roofAreaM2) * labourRate,
    });

    const total = lineItems.reduce((sum, item) => sum + item.total, 0);
    const requiresHumanReview = confidence < 0.7 || total > (isThai ? 100000 : 3000);

    return NextResponse.json({
      currency,
      total_amount: total,
      line_items: lineItems,
      requires_human_review: requiresHumanReview,
      review_reason: requiresHumanReview
        ? confidence < 0.7
          ? "Low model confidence"
          : "Quote exceeds mandatory sign-off threshold"
        : null,
      valid_until: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days
    });
  } catch (err) {
    console.error("POST /api/quotation error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
