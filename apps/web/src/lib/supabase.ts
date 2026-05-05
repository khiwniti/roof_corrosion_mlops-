import { createClient, SupabaseClient } from "@supabase/supabase-js";

let _browserClient: SupabaseClient | null = null;

// Browser client (for client components / hooks) — lazy init to survive SSR/build
export const getSupabaseBrowser = (): SupabaseClient => {
  if (_browserClient) return _browserClient;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    throw new Error("NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY must be set");
  }
  _browserClient = createClient(url, key, {
    realtime: { params: { eventsPerSecond: 10 } },
  });
  return _browserClient;
};

// Re-export for compat with existing imports that expect supabaseBrowser
export const supabaseBrowser = new Proxy({} as SupabaseClient, {
  get(_target, prop) {
    const client = getSupabaseBrowser();
    // @ts-expect-error dynamic prop access
    return client[prop];
  },
});

// Server client (for Route Handlers / Server Components)
export const supabaseServer = (): SupabaseClient => {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set");
  }
  return createClient(url, key, {
    auth: { autoRefreshToken: false, persistSession: false },
  });
};

// Types for the jobs table (aligned with existing schema)
export interface JobRow {
  id: string;
  status: "queued" | "processing" | "completed" | "failed" | "requires_review";
  overlay_image_s3_key?: string | null;
  roof_model_version?: string | null;
  aoi_geojson?: GeoJSON.Polygon | null;
  metadata?: any;
  created_at: string;
  updated_at: string;
}
