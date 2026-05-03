"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}`
  : "/api";

interface QueueStats {
  quote_queue: number;
  feedback_queue: number;
  relabel_queue: number;
}

export default function OpsPage() {
  const [activeTab, setActiveTab] = useState<"jobs" | "models" | "drift" | "relabel">("jobs");
  const [queueStats, setQueueStats] = useState<QueueStats | null>(null);

  useEffect(() => {
    // Fetch queue stats
    fetch(`${API_BASE}/health`)
      .then(() => {
        // Stub: in production, fetch from /ops/queue-stats
        setQueueStats({ quote_queue: 0, feedback_queue: 0, relabel_queue: 0 });
      })
      .catch(() => {
        setQueueStats({ quote_queue: 0, feedback_queue: 0, relabel_queue: 0 });
      });
  }, []);

  const tabs = [
    { id: "jobs", label: "Job Queue" },
    { id: "models", label: "Models" },
    { id: "drift", label: "Drift Monitor" },
    { id: "relabel", label: "Relabel Queue" },
  ] as const;

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
              <div className="grid grid-cols-3 gap-4 mb-8">
                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <p className="text-sm text-slate-400">Quote Queue</p>
                  <p className="text-3xl font-bold">{queueStats?.quote_queue ?? "—"}</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <p className="text-sm text-slate-400">Feedback Queue</p>
                  <p className="text-3xl font-bold">{queueStats?.feedback_queue ?? "—"}</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <p className="text-sm text-slate-400">Relabel Queue</p>
                  <p className="text-3xl font-bold text-orange-400">{queueStats?.relabel_queue ?? "—"}</p>
                </div>
              </div>
              <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
                <p className="text-slate-400 text-sm">
                  Job history and real-time status will appear here when connected to the API.
                </p>
                <div className="mt-4 space-y-2">
                  <div className="flex items-center gap-3 text-sm">
                    <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                    <span className="text-slate-300">No jobs in queue</span>
                  </div>
                </div>
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
                      <th className="text-right py-2">Roof IoU</th>
                      <th className="text-right py-2">Corrosion IoU</th>
                      <th className="text-right py-2">Area MAPE</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-slate-700">
                      <td className="py-2">Roof Detector (SegFormer-B3)</td>
                      <td className="py-2 font-mono">v0.1.0-dev</td>
                      <td className="py-2"><span className="px-2 py-0.5 bg-yellow-900 text-yellow-400 text-xs rounded">dev</span></td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                    </tr>
                    <tr className="border-b border-slate-700">
                      <td className="py-2">Corrosion Detector (SegFormer-B2)</td>
                      <td className="py-2 font-mono">v0.1.0-dev</td>
                      <td className="py-2"><span className="px-2 py-0.5 bg-yellow-900 text-yellow-400 text-xs rounded">dev</span></td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                      <td className="py-2 text-right">—</td>
                    </tr>
                  </tbody>
                </table>
                <p className="text-slate-500 text-xs mt-4">
                  Metrics will populate after Phase 1a baseline training completes.
                  Models must pass frozen test set gates before promotion to staging/production.
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
