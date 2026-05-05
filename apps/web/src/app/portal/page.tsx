"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MaplibreTerradrawControl } from "@watergis/maplibre-gl-terradraw";
import "@watergis/maplibre-gl-terradraw/dist/maplibre-gl-terradraw.css";
import { area } from "@turf/area";
import { polygon as turfPolygon } from "@turf/helpers";
import { supabaseBrowser } from "@/lib/supabase";
import type { JobRow } from "@/lib/supabase";

const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_KEY ?? "";

export default function PortalPage() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const drawControlRef = useRef<MaplibreTerradrawControl | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobRow | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [polygonArea, setPolygonArea] = useState<number | null>(null);
  const [drawnPolygon, setDrawnPolygon] = useState<GeoJSON.Polygon | null>(null);
  const [selectedTier, setSelectedTier] = useState<number>(0);
  const [quote, setQuote] = useState<any | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(false);

  // Initialize MapLibre + terra-draw
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAPTILER_KEY
        ? `https://api.maptiler.com/maps/streets/style.json?key=${MAPTILER_KEY}`
        : "https://demotiles.maplibre.org/style.json",
      center: [100.5018, 13.7563], // Bangkok
      zoom: 12,
    });

    const draw = new MaplibreTerradrawControl({
      modes: ["polygon", "select", "delete-selection"],
      open: true,
    });

    map.addControl(draw, "top-left");
    mapRef.current = map;
    drawControlRef.current = draw;

    // Listen for terra-draw changes via the control's internal terra-draw instance
    const td = (draw as any).getTerraDrawInstance?.();
    if (td) {
      td.on("change", () => {
        const snapshot = td.getSnapshot();
        const poly = snapshot.find((f: any) => f.geometry.type === "Polygon");
        if (poly) {
          const geoJsonPolygon: GeoJSON.Polygon = {
            type: "Polygon",
            coordinates: poly.geometry.coordinates,
          };
          setDrawnPolygon(geoJsonPolygon);
          const a = area(turfPolygon(poly.geometry.coordinates));
          setPolygonArea(a);
        } else {
          setDrawnPolygon(null);
          setPolygonArea(null);
        }
      });
    }

    return () => {
      map.remove();
    };
  }, []);

  // Supabase Realtime subscription
  useEffect(() => {
    if (!jobId) return;

    const channel = supabaseBrowser
      .channel(`job:${jobId}`)
      .on(
        "postgres_changes",
        {
          event: "UPDATE",
          schema: "public",
          table: "jobs",
          filter: `id=eq.${jobId}`,
        },
        (payload) => {
          setJob(payload.new as JobRow);
        }
      )
      .subscribe();

    return () => {
      supabaseBrowser.removeChannel(channel);
    };
  }, [jobId]);

  // Mask overlay on MapLibre when job completes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !drawnPolygon || job?.status !== "completed" || !job?.overlay_image_s3_key) return;

    // Compute bounding box from polygon coordinates
    const coords = drawnPolygon.coordinates[0];
    const lons = coords.map((c) => c[0]);
    const lats = coords.map((c) => c[1]);
    const bounds: [number, number, number, number] = [
      Math.min(...lons),
      Math.min(...lats),
      Math.max(...lons),
      Math.max(...lats),
    ];

    // Add image source + raster layer
    if (!map.getSource("mask-overlay")) {
      map.addSource("mask-overlay", {
        type: "image",
        url: job.overlay_image_s3_key,
        coordinates: [
          [bounds[0], bounds[3]], // top-left
          [bounds[2], bounds[3]], // top-right
          [bounds[2], bounds[1]], // bottom-right
          [bounds[0], bounds[1]], // bottom-left
        ],
      });
      map.addLayer({
        id: "mask-overlay-layer",
        type: "raster",
        source: "mask-overlay",
        paint: { "raster-opacity": 0.6 },
      });
    }

    return () => {
      if (map.getLayer("mask-overlay-layer")) map.removeLayer("mask-overlay-layer");
      if (map.getSource("mask-overlay")) map.removeSource("mask-overlay");
    };
  }, [job, drawnPolygon]);

  const submitJob = useCallback(async () => {
    if (!drawnPolygon) {
      setError("Please draw a polygon on the map first.");
      return;
    }
    const a = polygonArea ?? 0;
    if (a < 10 || a > 10000) {
      setError(`Polygon area ${a.toFixed(0)} m² is out of range (10–10,000 m²).`);
      return;
    }

    setLoading(true);
    setError(null);
    setJob(null);

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon: drawnPolygon, tier: selectedTier }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setJobId(data.jobId);
      setJob({ id: data.jobId, status: "queued", aoi_geojson: drawnPolygon, created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as JobRow);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [drawnPolygon, polygonArea]);

  const generateQuote = useCallback(async () => {
    if (!jobId || !polygonArea) return;
    setQuoteLoading(true);
    try {
      const meta = (job?.metadata as any) || {};
      const areaM2 = meta.area_m2 ?? polygonArea;
      const confidence = meta.confidence ?? 0.5;
      const materialPct = meta.coarse_breakdown?.metal_percent ?? 50;
      const material = materialPct > 50 ? "corrugated_metal" : "clay_tile";
      const corrosionProb = meta.corrosion_prob ?? 0;
      const severity = meta.severity ?? "none";
      const corrodedAreaM2 = areaM2 * corrosionProb;
      const res = await fetch("/api/quotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId,
          roofAreaM2: areaM2,
          corrodedAreaM2,
          corrosionPercent: Math.round(corrosionProb * 100),
          severity,
          confidence,
          material,
          region: "TH",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Quote failed");
      setQuote(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Quote failed");
    } finally {
      setQuoteLoading(false);
    }
  }, [jobId, polygonArea, job]);

  const clearDrawing = useCallback(() => {
    const draw = drawControlRef.current;
    if (draw) {
      const td = (draw as any).getTerraDrawInstance?.();
      if (td) {
        const snapshot = td.getSnapshot();
        snapshot.forEach((f: any) => td.removeFeature(f.id));
      }
    }
    setDrawnPolygon(null);
    setPolygonArea(null);
    setJobId(null);
    setJob(null);
    setQuote(null);
    setError(null);
  }, []);

  const statusColor = (s: string) => {
    switch (s) {
      case "queued": return "bg-slate-100 text-slate-700";
      case "processing": return "bg-blue-100 text-blue-700";
      case "completed": return "bg-green-100 text-green-700";
      case "failed": return "bg-red-100 text-red-700";
      case "requires_review": return "bg-orange-100 text-orange-700";
      default: return "bg-slate-100 text-slate-700";
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-3 bg-white border-b">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold text-slate-900">Roof Corrosion AI</span>
        </Link>
        <div className="flex gap-4 items-center">
          <span className="text-sm text-slate-500">Customer Portal</span>
          <button className="px-4 py-2 text-sm bg-slate-100 rounded-md hover:bg-slate-200">
            Sign Out
          </button>
        </div>
      </nav>

      <div className="flex flex-col lg:flex-row h-[calc(100vh-60px)]">
        {/* Map */}
        <div className="flex-1 relative">
          <div ref={mapContainerRef} className="absolute inset-0" />
          {/* Area badge */}
          {polygonArea !== null && (
            <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur rounded-lg px-3 py-2 shadow text-sm font-medium">
              Area: {polygonArea.toFixed(0)} m²
              {polygonArea >= 10 && polygonArea <= 10000 ? (
                <span className="ml-2 text-green-600">✓ Valid</span>
              ) : (
                <span className="ml-2 text-red-600">✗ Must be 10–10,000 m²</span>
              )}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-full lg:w-96 bg-white border-l p-6 overflow-y-auto">
          <h1 className="text-2xl font-bold text-slate-900 mb-2">Roof Analysis</h1>
          <p className="text-slate-500 mb-6 text-sm">
            {selectedTier === 0
              ? "Free Tier-0 preliminary analysis using Sentinel-2 imagery (±30% range)."
              : "Tier-1 binding quote using Pléiades 50cm VHR imagery (±15% range). Imagery cost billed separately."}
          </p>

          {/* Tier selector */}
          <div className="flex gap-1 mb-4 bg-slate-100 rounded-lg p-1">
            {[0, 1].map((tier) => (
              <button
                key={tier}
                onClick={() => setSelectedTier(tier)}
                className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition ${
                  selectedTier === tier
                    ? tier === 0
                      ? "bg-white text-orange-600 shadow-sm"
                      : "bg-white text-blue-600 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                Tier {tier}
              </button>
            ))}
          </div>

          <div className="flex gap-2 mb-4">
            <button
              onClick={submitJob}
              disabled={loading || !drawnPolygon || (polygonArea ?? 0) < 10 || (polygonArea ?? 0) > 10000}
              className={`flex-1 px-4 py-3 rounded-lg font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed ${
                selectedTier === 1
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-orange-500 text-white hover:bg-orange-600"
              }`}
            >
              {loading ? "Submitting..." : selectedTier === 1 ? "Get Binding Quote" : "Analyze Roof"}
            </button>
            <button
              onClick={clearDrawing}
              className="px-4 py-3 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 font-medium"
            >
              Clear
            </button>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm mb-4">
              {error}
            </div>
          )}

          {/* Job status */}
          {jobId && job && (
            <div className="bg-slate-50 rounded-lg border p-4 mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-500 font-mono">{jobId.slice(0, 12)}...</span>
                <span className={`text-xs px-2 py-1 rounded-full font-medium ${statusColor(job.status)}`}>
                  {job.status}
                </span>
              </div>
              {job.status === "queued" && (
                <p className="text-sm text-slate-500">Waiting in queue...</p>
              )}
              {job.status === "processing" && (
                <p className="text-sm text-slate-500">Fetching satellite imagery and running analysis...</p>
              )}
              {job.status === "completed" && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-green-700 font-medium">Analysis complete</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                      (job.metadata as any)?.tier === 1
                        ? "bg-blue-100 text-blue-700"
                        : "bg-slate-100 text-slate-600"
                    }`}>
                      Tier {(job.metadata as any)?.tier ?? 0}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Buildings</p>
                      <p className="font-semibold text-slate-900">{(job.metadata as any)?.building_count ?? "--"}</p>
                    </div>
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Roof Area</p>
                      <p className="font-semibold text-slate-900">
                        {((job.metadata as any)?.area_m2 ?? polygonArea ?? 0).toFixed(0)} m²
                      </p>
                    </div>
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Corrosion</p>
                      <p className="font-semibold text-slate-900">
                        {(job.metadata as any)?.corrosion_prob !== undefined
                          ? `${((job.metadata as any).corrosion_prob * 100).toFixed(0)}%`
                          : "--"}
                      </p>
                    </div>
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Severity</p>
                      <p className={`font-semibold ${
                        (job.metadata as any)?.severity === "severe"
                          ? "text-red-600"
                          : (job.metadata as any)?.severity === "moderate"
                            ? "text-orange-600"
                            : (job.metadata as any)?.severity === "light"
                              ? "text-yellow-600"
                              : "text-green-600"
                      }`}>
                        {(job.metadata as any)?.severity ?? "--"}
                      </p>
                    </div>
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Metal</p>
                      <p className="font-semibold text-slate-700">
                        {(job.metadata as any)?.coarse_breakdown?.metal_percent ?? "--"}%
                      </p>
                    </div>
                    <div className="bg-white rounded p-2 border">
                      <p className="text-xs text-slate-500">Tile</p>
                      <p className="font-semibold text-orange-600">
                        {(job.metadata as any)?.coarse_breakdown?.tile_percent ?? "--"}%
                      </p>
                    </div>
                    <div className="bg-white rounded p-2 border col-span-2">
                      <p className="text-xs text-slate-500">Confidence</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-green-500 h-full rounded-full"
                            style={{ width: `${((job.metadata as any)?.confidence ?? 0.5) * 100}%` }}
                          />
                        </div>
                        <span className="font-semibold text-slate-900 text-xs">
                          {(((job.metadata as any)?.confidence ?? 0.5) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  {!quote && (
                    <button
                      onClick={generateQuote}
                      disabled={quoteLoading}
                      className="w-full px-4 py-2 bg-orange-500 text-white rounded-lg font-medium hover:bg-orange-600 disabled:opacity-50 transition text-sm"
                    >
                      {quoteLoading ? "Generating..." : "Generate Quotation"}
                    </button>
                  )}
                  {quote && (
                    <div className="bg-white rounded-lg border p-3 text-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-slate-900">Quotation</span>
                        {quote.requires_human_review && (
                          <span className="text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded-full">Pending Review</span>
                        )}
                      </div>
                      <table className="w-full text-xs mb-2">
                        <tbody>
                          {quote.line_items.map((item: any, i: number) => (
                            <tr key={i} className="border-b">
                              <td className="py-1 text-slate-700">{item.description}</td>
                              <td className="py-1 text-right text-slate-500">{item.quantity} {item.unit}</td>
                              <td className="py-1 text-right font-medium">{quote.currency} {item.total.toLocaleString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div className="flex justify-between items-center pt-1 border-t">
                        <span className="font-bold text-slate-900">Total</span>
                        <span className="text-lg font-bold text-orange-500">{quote.currency} {quote.total_amount.toLocaleString()}</span>
                      </div>
                    </div>
                  )}
                  {job.metadata && (
                    <details className="text-xs text-slate-500">
                      <summary className="cursor-pointer font-medium text-slate-600 hover:text-slate-800">
                        Ingestion details
                      </summary>
                      <div className="mt-1 pl-2 border-l-2 border-slate-200 space-y-0.5">
                        <p>Method: {(job.metadata as any).ingestion_method ?? "unknown"}</p>
                        {(job.metadata as any).coverage !== undefined && (
                          <p>Coverage: {((job.metadata as any).coverage * 100).toFixed(0)}%</p>
                        )}
                        {(job.metadata as any).time_steps !== undefined && (
                          <p>Time steps: {(job.metadata as any).time_steps}</p>
                        )}
                        {(job.metadata as any).fallback_level !== undefined && (
                          <p>Fallback: level {(job.metadata as any).fallback_level}</p>
                        )}
                        {(job.metadata as any).deferred_precision && (
                          <p className="text-orange-600">Deferred precision (90d median)</p>
                        )}
                        {(job.metadata as any).error && (
                          <p className="text-red-500">Error: {(job.metadata as any).error}</p>
                        )}
                      </div>
                    </details>
                  )}
                  {(job.metadata as any)?.tier === 1 && (
                    <div className="bg-blue-50 rounded p-2 text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-600">Imagery cost (est.)</span>
                        <span className="font-medium text-slate-900">
                          {(job.metadata as any)?.cost_estimate_thb
                            ? `THB ${(job.metadata as any).cost_estimate_thb.toLocaleString()}`
                            : "Pending"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-600">Quote band</span>
                        <span className="font-medium text-blue-700">{(job.metadata as any)?.quote_band ?? "±15%"}</span>
                      </div>
                      {(job.metadata as any)?.requires_human_review && (
                        <p className="text-orange-600 font-medium">
                          Flagged: exceeds THB 100k threshold — estimator review required
                        </p>
                      )}
                    </div>
                  )}
                  <p className="text-xs text-slate-400">
                    {(job.metadata as any)?.tier === 1
                      ? "Tier-1 binding estimate. Imagery cost billed separately."
                      : "Tier-0 preliminary estimate (±30% range). Binding quotes require Tier-1 or Tier-3."}
                  </p>
                  {job.overlay_image_s3_key && (
                    <a
                      href={job.overlay_image_s3_key}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block text-sm text-orange-600 hover:underline font-medium"
                    >
                      View mask GeoTIFF →
                    </a>
                  )}
                </div>
              )}
              {job.status === "failed" && (
                <p className="text-sm text-red-600">Analysis failed. Please try again or contact support.</p>
              )}
              {job.status === "requires_review" && (
                <p className="text-sm text-orange-600">Result flagged for human review before quoting.</p>
              )}
            </div>
          )}

          {/* Instructions */}
          {!jobId && (
            <div className="text-sm text-slate-500 space-y-2">
              <p className="font-medium text-slate-700">How to use:</p>
              <ol className="list-decimal pl-4 space-y-1">
                <li>Zoom to your roof on the map.</li>
                <li>Draw a polygon covering the roof area.</li>
                <li>Click <strong>{selectedTier === 1 ? "Get Binding Quote" : "Analyze Roof"}</strong> to submit.</li>
              </ol>
              <p className="pt-2 text-xs text-slate-400">
                {selectedTier === 1
                  ? "Tier-1 binding quote with Pléiades 50cm imagery. Imagery cost estimated separately before purchase."
                  : "Free preliminary estimate (Tier-0). For a binding quote, switch to Tier-1 or Tier-3."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
