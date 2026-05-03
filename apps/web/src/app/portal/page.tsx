"use client";

import { useState, useCallback } from "react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}`
  : "/api";

interface JobResult {
  job_id: string;
  status: string;
  assessment?: {
    roof_area_m2: number;
    corroded_area_m2: number;
    corrosion_percent: number;
    severity: string;
    confidence: number;
    overlay_image_s3_key?: string;
  };
  quote?: {
    total_amount: number;
    currency: string;
    line_items: Array<{
      description: string;
      quantity: number;
      unit: string;
      unit_price: number;
      total: number;
    }>;
    requires_human_review: boolean;
  };
}

export default function PortalPage() {
  const [address, setAddress] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitJob = useCallback(async () => {
    if (!address.trim()) return;
    setLoading(true);
    setError(null);
    setJobResult(null);
    setJobId(null);

    try {
      // Use the synchronous endpoint — runs the FM pipeline inline
      const res = await fetch(`${API_BASE}/quote/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ address }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setJobId(data.job_id);
      setJobResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [address]);

  const pollJob = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API_BASE}/quote/${jobId}`);
      const data = await res.json();
      setJobResult(data);
    } catch {
      setError("Failed to fetch job status");
    }
  }, [jobId]);

  // Currency-aware price formatter. THB → ฿X,XXX.XX, USD → $X,XXX.XX, etc.
  const formatPrice = (amount: number, currency: string) => {
    try {
      return new Intl.NumberFormat(currency === "THB" ? "th-TH" : "en-US", {
        style: "currency",
        currency,
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      }).format(amount);
    } catch {
      return `${currency} ${amount.toLocaleString()}`;
    }
  };

  const severityColor = (s: string) => {
    switch (s) {
      case "none": return "text-green-400";
      case "light": return "text-yellow-400";
      case "moderate": return "text-orange-400";
      case "severe": return "text-red-400";
      default: return "text-slate-400";
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-4 bg-white border-b">
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

      <div className="max-w-4xl mx-auto px-8 py-12">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Roof Analysis</h1>
        <p className="text-slate-500 mb-8">
          Enter your building address to get a satellite-derived corrosion analysis and quote.
        </p>

        {/* Address input */}
        <div className="flex gap-3 mb-8">
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="e.g. 88 Bangna-Trad Rd, Samut Prakan, Thailand"
            className="flex-1 px-4 py-3 border rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-orange-400"
            onKeyDown={(e) => e.key === "Enter" && submitJob()}
          />
          <button
            onClick={submitJob}
            disabled={loading || !address.trim()}
            className="px-6 py-3 bg-orange-500 text-white rounded-lg font-semibold hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? "Submitting..." : "Analyze"}
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 mb-6">
            {error}
          </div>
        )}

        {/* Job status */}
        {jobId && (
          <div className="bg-white rounded-lg border p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-slate-900">Job Status</h2>
              <button
                onClick={pollJob}
                className="px-3 py-1 text-sm bg-slate-100 rounded hover:bg-slate-200"
              >
                Refresh
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-slate-500">Job ID:</span>
                <span className="ml-2 font-mono text-slate-900">{jobId.slice(0, 8)}...</span>
              </div>
              <div>
                <span className="text-slate-500">Status:</span>
                <span className="ml-2 font-semibold text-slate-900">
                  {jobResult?.status || "queued"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Assessment results */}
        {jobResult?.assessment && (
          <div className="bg-white rounded-lg border p-6 mb-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Corrosion Assessment</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-slate-500">Roof Area</p>
                <p className="text-2xl font-bold text-slate-900">
                  {jobResult.assessment.roof_area_m2.toFixed(0)} m²
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Corroded Area</p>
                <p className="text-2xl font-bold text-red-500">
                  {jobResult.assessment.corroded_area_m2.toFixed(1)} m²
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Corrosion %</p>
                <p className="text-2xl font-bold text-orange-500">
                  {jobResult.assessment.corrosion_percent.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Severity</p>
                <p className={`text-2xl font-bold ${severityColor(jobResult.assessment.severity)}`}>
                  {jobResult.assessment.severity.charAt(0).toUpperCase() + jobResult.assessment.severity.slice(1)}
                </p>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-slate-500">
                Model confidence: {(jobResult.assessment.confidence * 100).toFixed(0)}%
                {jobResult.assessment.confidence < 0.7 && (
                  <span className="ml-2 text-orange-600 font-medium">
                    (Low confidence — pending human review)
                  </span>
                )}
              </p>
            </div>
          </div>
        )}

        {/* Quote */}
        {jobResult?.quote && (
          <div className="bg-white rounded-lg border p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-slate-900">Quote</h2>
              {jobResult.quote.requires_human_review && (
                <span className="px-3 py-1 bg-orange-100 text-orange-700 text-sm rounded-full font-medium">
                  Pending Review
                </span>
              )}
            </div>
            <table className="w-full text-sm mb-4">
              <thead>
                <tr className="border-b text-slate-500">
                  <th className="text-left py-2">Description</th>
                  <th className="text-right py-2">Qty</th>
                  <th className="text-right py-2">Unit</th>
                  <th className="text-right py-2">Unit Price</th>
                  <th className="text-right py-2">Total</th>
                </tr>
              </thead>
              <tbody>
                {jobResult.quote.line_items.map((item, i) => (
                  <tr key={i} className="border-b">
                    <td className="py-2 text-slate-900">{item.description}</td>
                    <td className="py-2 text-right">{item.quantity}</td>
                    <td className="py-2 text-right">{item.unit}</td>
                    <td className="py-2 text-right">
                      {formatPrice(item.unit_price, jobResult.quote!.currency)}
                    </td>
                    <td className="py-2 text-right font-medium">
                      {formatPrice(item.total, jobResult.quote!.currency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="flex justify-between items-center pt-4 border-t">
              <span className="text-lg font-bold text-slate-900">Total</span>
              <span className="text-2xl font-bold text-orange-500">
                {formatPrice(jobResult.quote.total_amount, jobResult.quote.currency)}
              </span>
            </div>
          </div>
        )}

        {/* Feedback */}
        {jobResult?.status === "completed" && (
          <div className="bg-white rounded-lg border p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Was this assessment accurate?</h2>
            <div className="flex gap-3">
              <button className="px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 font-medium">
                Yes, correct
              </button>
              <button className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 font-medium">
                No, something is wrong
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
