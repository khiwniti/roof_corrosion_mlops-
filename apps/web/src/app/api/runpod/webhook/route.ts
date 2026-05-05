import { NextRequest, NextResponse } from "next/server";
import { supabaseServer } from "@/lib/supabase";

export const maxDuration = 30;

// In production, verify X-RunPod-Signature HMAC here.
// For Phase 0 stub we accept all calls.

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { jobId, status, mask_url, model_version, ingestion_meta } = body;

    if (!jobId) {
      return NextResponse.json({ error: "Missing jobId" }, { status: 400 });
    }

    const supabase = supabaseServer();
    const updatePayload: any = {
      status: status ?? "completed",
      overlay_image_s3_key: mask_url ?? null,
      roof_model_version: model_version ?? null,
      completed_at: status === "completed" ? new Date().toISOString() : undefined,
    };
    if (ingestion_meta) {
      updatePayload.metadata = ingestion_meta;
    }
    // Tier-0 / Tier-1 result fields stored in metadata
    if (body.area_m2 !== undefined || body.building_count !== undefined || body.coarse_breakdown) {
      updatePayload.metadata = {
        ...(updatePayload.metadata || {}),
        area_m2: body.area_m2,
        building_count: body.building_count,
        class_percentages: body.class_percentages,
        coarse_breakdown: body.coarse_breakdown,
        confidence: body.confidence,
      };
    }
    // Tier-1 extra fields
    if (body.cost_estimate_eur !== undefined || body.quote_band) {
      updatePayload.metadata = {
        ...(updatePayload.metadata || {}),
        cost_estimate_eur: body.cost_estimate_eur,
        cost_estimate_thb: body.cost_estimate_thb,
        quote_band: body.quote_band,
        requires_human_review: body.requires_human_review,
      };
    }

    const { error } = await supabase
      .from("jobs")
      .update(updatePayload)
      .eq("id", jobId);

    if (error) {
      console.error("Webhook Supabase update error:", error);
      return NextResponse.json({ error: "Failed to update job" }, { status: 500 });
    }

    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("POST /api/runpod/webhook error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
