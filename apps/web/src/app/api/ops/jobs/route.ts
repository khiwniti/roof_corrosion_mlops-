import { NextRequest, NextResponse } from "next/server";
import { supabaseServer } from "@/lib/supabase";

export const maxDuration = 30;

/**
 * Ops dashboard: fetch recent jobs with optional status filter.
 * Uses service role key to bypass RLS.
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status");
    const limit = Math.min(parseInt(searchParams.get("limit") ?? "50", 10), 100);

    const supabase = supabaseServer();
    let query = supabase
      .from("jobs")
      .select("id, status, source, aoi_geojson, submitted_at, completed_at, roof_model_version, overlay_image_s3_key, metadata")
      .order("submitted_at", { ascending: false })
      .limit(limit);

    if (status) {
      query = query.eq("status", status);
    }

    const { data, error } = await query;

    if (error) {
      console.error("Supabase error:", error);
      return NextResponse.json({ error: "Failed to fetch jobs" }, { status: 500 });
    }

    return NextResponse.json({ jobs: data ?? [] });
  } catch (err) {
    console.error("GET /api/ops/jobs error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
