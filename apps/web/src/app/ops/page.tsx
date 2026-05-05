"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface Job {
  id: string;
  status: string;
  source: string;
  submitted_at: string;
  completed_at: string | null;
  roof_model_version: string | null;
  overlay_image_s3_key: string | null;
}

export default function OpsPage() {
  const [activeTab, setActiveTab] = useState<"jobs" | "models" | "drift" | "relabel">("jobs");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);

  useEffect(() => {
    fetch("/api/ops/jobs?limit=20")
      .then((res) => res.json())
      .then((data) => {
        setJobs(data.jobs ?? []);
        setJobsLoading(false);
      })
      .catch(() => setJobsLoading(false));
  }, []);

  const tabs = [
    { id: "jobs", label: "Job Queue" },
    { id: "models", label: "Models" },
    { id: "drift", label: "Drift Monitor" },
    { id: "relabel", label: "Relabel Queue" },
  ] as const;

  const tierFromSource = (source: string) => {
    switch (source) {
      case "maxar": return "Tier 0 (S2)";
      case "nearmap": return "Tier 1 (VHR)";
      case "drone": return "Tier 3 (Drone)";
      default: return source;
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "queued": return "bg-slate-100 text-slate-700";
      case "processing": return "bg-blue-100 text-blue-700";
      case "completed": return "bg-green-100 text-green-700";
      case "failed": return "bg-red-100 text-red-700";
      case "requires_review": return "bg-orange-100 text-orange-700";
      default: return "bg-slate-100 text-slate-700";
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-4 border-b border-slate-700">
        <Link href="/" className="text-xl font-bold">Roof Corrosion AI</Link>
        <div className="flex gap-4 items-center">
          <span className="text-sm text-slate-400">Ops Dashboard</span>
          <span className="px-2 py-1 bg-green-900 text-green-400 text-xs rounded">Internal</span>
        </div>
      </nav>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-48 min-h-[calc(100vh-64px)] border-r border-slate-700 p-4">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full text-left px-3 py-2 rounded text-sm transition ${
                  activeTab === tab.id
                    ? "bg-slate-700 text-white"
                    : "text-slate-400 hover:text-white hover:bg-slate-800"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-8">
          {activeTab === "jobs" && (
            <div>
              <h1 className="text-2xl font-bold mb-6">Job Queue</h1>
              <div className="grid grid-cols-5 gap-3 mb-6">
                {["queued", "processing", "completed", "failed", "requires_review"].map((s) => (
                  <div key={s} className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                    <p className="text-xs text-slate-400 capitalize">{s.replace("_", " ")}</p>
                    <p className="text-2xl font-bold">{jobs.filter((j) => j.status === s).length}</p>
                  </div>
                ))}
              </div>
              <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400 text-left">
                      <th className="py-2 px-4">ID</th>
                      <th className="py-2 px-4">Status</th>
                      <th className="py-2 px-4">Tier</th>
                      <th className="py-2 px-4">Area</th>
                      <th className="py-2 px-4">Corrosion</th>
                      <th className="py-2 px-4">Severity</th>
                      <th className="py-2 px-4">Submitted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobsLoading ? (
                      <tr><td colSpan={7} className="py-8 text-center text-slate-500">Loading...</td></tr>
                    ) : jobs.length === 0 ? (
                      <tr><td colSpan={7} className="py-8 text-center text-slate-500">No jobs yet</td></tr>
                    ) : (
                      jobs.map((job) => {
                        const meta = (job as any).metadata || {};
                        const tier = meta.tier ?? 0;
                        const area = meta.area_m2 ?? "—";
                        const corrosion = meta.corrosion_prob !== undefined ? `${(meta.corrosion_prob * 100).toFixed(0)}%` : "—";
                        const severity = meta.severity ?? "—";
                        const severityColor = severity === "severe" ? "text-red-400" : severity === "moderate" ? "text-orange-400" : severity === "light" ? "text-yellow-400" : "text-green-400";
                        return (
                          <tr key={job.id} className="border-b border-slate-700/50">
                            <td className="py-2 px-4 font-mono text-xs">{job.id.slice(0, 8)}...</td>
                            <td className="py-2 px-4">
                              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(job.status)}`}>
                                {job.status}
                              </span>
                            </td>
                            <td className="py-2 px-4">
                              <span className={`text-xs px-2 py-0.5 rounded font-medium ${tier === 1 ? "bg-blue-900 text-blue-300" : "bg-slate-700 text-slate-300"}`}>
                                Tier {tier}
                              </span>
                            </td>
                            <td className="py-2 px-4 text-slate-300 text-xs">{typeof area === "number" ? `${area.toFixed(0)} m²` : area}</td>
                            <td className="py-2 px-4 text-slate-300 text-xs">{corrosion}</td>
                            <td className={`py-2 px-4 text-xs font-medium ${severityColor}`}>{severity}</td>
                            <td className="py-2 px-4 text-slate-400 text-xs">{new Date(job.submitted_at).toLocaleString()}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === "models" && (
            <div>
              <h1 className="text-2xl font-bold mb-6">Model Registry</h1>
              <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400">
                      <th className="text-left py-2">Model</th>
                      <th className="text-left py-2">Version</th>
                      <th className="text-left py-2">Stage</th>
                      <th className="text-right py-2">Material IoU</th>
                      <th className="text-right py-2">Corrosion IoU</th>
                      <th className="text-right py-2">Severity IoU</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-slate-700">
                      <td className="py-2">Clay v1.5 + ViT-Adapter + Mask2Former (3-head)</td>
                      <td className="py-2 font-mono">v0.1.0-dev</td>
                      <td className="py-2"><span className="px-2 py-0.5 bg-yellow-900 text-yellow-400 text-xs rounded">dev</span></td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                    </tr>
                    <tr className="border-b border-slate-700">
                      <td className="py-2">Tier-0 Baseline (smp U-Net + EfficientNet-B7)</td>
                      <td className="py-2 font-mono">v0.0.1-stub</td>
                      <td className="py-2"><span className="px-2 py-0.5 bg-slate-700 text-slate-300 text-xs rounded">stub</span></td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                    </tr>
                  </tbody>
                </table>
                <p className="text-slate-500 text-xs mt-4">
                  Metrics populate after Phase 1a open-data baseline training.
                  Clay model gates: material IoU ≥ 0.70, corrosion IoU ≥ 0.45, severity IoU ≥ 0.50.
                </p>
              </div>
            </div>
          )}

          {activeTab === "drift" && (
            <div>
              <h1 className="text-2xl font-bold mb-6">Drift Monitor</h1>
              <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
                <p className="text-slate-400 text-sm mb-4">
                  Evidently drift reports will appear here. Monitors input distribution
                  (tile GSD, region, cloud cover) and prediction distribution (severity mix,
                  area estimates) for distribution shift.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-700 rounded p-4">
                    <p className="text-sm text-slate-300">Input Drift</p>
                    <p className="text-xs text-slate-500">No data yet</p>
                  </div>
                  <div className="bg-slate-700 rounded p-4">
                    <p className="text-sm text-slate-300">Prediction Drift</p>
                    <p className="text-xs text-slate-500">No data yet</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "relabel" && (
            <div>
              <h1 className="text-2xl font-bold mb-6">Relabel Queue</h1>
              <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
                <p className="text-slate-400 text-sm mb-4">
                  Predictions flagged by customer feedback or active learning uncertainty
                  sampling. Sent to Label Studio for human correction.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center justify-between bg-slate-700 rounded p-3">
                    <div>
                      <p className="text-sm text-slate-300">No items in relabel queue</p>
                      <p className="text-xs text-slate-500">Items will appear when customers flag incorrect predictions</p>
                    </div>
                    <a
                      href="http://localhost:8080"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3 py-1 bg-orange-500 text-sm rounded hover:bg-orange-600"
                    >
                      Open Label Studio
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
