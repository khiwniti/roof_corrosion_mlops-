import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { supabaseServer } from "@/lib/supabase";

export const maxDuration = 60;

const createJobSchema = z.object({
  polygon: z.object({
    type: z.literal("Polygon"),
    coordinates: z.array(z.array(z.array(z.number()))),
  }),
  tier: z.number().int().min(0).max(3).default(0),
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const parsed = createJobSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request body", issues: parsed.error.issues },
        { status: 400 }
      );
    }

    const { polygon, tier } = parsed.data;

    // Insert job into Supabase (UUID auto-generated)
    const supabase = supabaseServer();
    const { data: inserted, error: insertError } = await supabase
      .from("jobs")
      .insert({
        status: "queued",
        aoi_geojson: polygon,
        source: tier === 0 ? "maxar" : tier === 1 ? "nearmap" : "drone",
        metadata: { tier },
      })
      .select("id")
      .single();

    if (insertError || !inserted) {
      console.error("Supabase insert error:", insertError);
      return NextResponse.json({ error: "Failed to create job" }, { status: 500 });
    }

    const jobId = inserted.id;

    // Call RunPod stub (asynchronous, non-blocking)
    const runpodApiKey = process.env.RUNPOD_API_KEY;
    const runpodEndpointId = process.env.RUNPOD_ENDPOINT_ID;
    const appUrl = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

    if (runpodApiKey && runpodEndpointId) {
      fetch(`https://api.runpod.ai/v2/${runpodEndpointId}/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${runpodApiKey}`,
        },
        body: JSON.stringify({
          input: { polygon, tier, jobId },
          webhook: `${appUrl}/api/runpod/webhook`,
        }),
      }).catch((err) => {
        console.error("RunPod call error (non-blocking):", err);
      });
    } else {
      // Dev fallback: simulate webhook after 3s with a stub mask URL
      setTimeout(() => {
        fetch(`${appUrl}/api/runpod/webhook`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            jobId,
            status: "completed",
            mask_url: `${appUrl}/api/stub-mask?jobId=${jobId}`,
            area_m2: 0,
            class_areas: {},
            model_version: "stub-v0",
            confidence_stats: {},
          }),
        }).catch(() => {});
      }, 3000);
    }

    return NextResponse.json({ jobId, status: "queued" }, { status: 201 });
  } catch (err) {
    console.error("POST /api/jobs error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
