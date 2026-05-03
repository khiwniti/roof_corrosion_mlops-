import Link from "next/link";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-white">
      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-4 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <svg className="w-8 h-8 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1" />
          </svg>
          <span className="text-xl font-bold">Roof Corrosion AI</span>
        </div>
        <div className="flex gap-4">
          <Link href="/portal" className="px-4 py-2 text-sm hover:text-orange-400 transition">Portal</Link>
          <Link href="/ops" className="px-4 py-2 text-sm hover:text-orange-400 transition">Ops</Link>
          <Link href="/portal" className="px-4 py-2 text-sm bg-orange-500 rounded-md hover:bg-orange-600 transition">
            Get a Quote
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-8 pt-24 pb-16">
        <h1 className="text-5xl font-bold leading-tight mb-6">
          Satellite-Powered<br />
          <span className="text-orange-400">Roof Corrosion Detection</span>
        </h1>
        <p className="text-xl text-slate-300 max-w-2xl mb-8">
          Upload an address and get a precision corrosion analysis and itemized
          repair/replacement quote — powered by high-resolution satellite imagery
          and state-of-the-art AI segmentation.
        </p>
        <div className="flex gap-4">
          <Link
            href="/portal"
            className="px-6 py-3 bg-orange-500 rounded-lg text-lg font-semibold hover:bg-orange-600 transition"
          >
            Analyze My Roof
          </Link>
          <Link
            href="#how-it-works"
            className="px-6 py-3 border border-slate-600 rounded-lg text-lg hover:border-orange-400 transition"
          >
            How It Works
          </Link>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="max-w-6xl mx-auto px-8 py-16">
        <h2 className="text-3xl font-bold mb-12 text-center">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {[
            { step: "1", title: "Enter Address", desc: "Provide your building address or coordinates" },
            { step: "2", title: "Satellite Scan", desc: "We fetch high-res satellite imagery of your roof" },
            { step: "3", title: "AI Analysis", desc: "Our model detects roof boundaries and corrosion areas" },
            { step: "4", title: "Get Quote", desc: "Receive an itemized repair/replacement quote" },
          ].map((item) => (
            <div key={item.step} className="text-center">
              <div className="w-12 h-12 bg-orange-500 rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                {item.step}
              </div>
              <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
              <p className="text-slate-400 text-sm">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-8 py-16">
        <h2 className="text-3xl font-bold mb-12 text-center">Why Roof Corrosion AI</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            { title: "Sub-Pixel Precision", desc: "30cm satellite resolution detects corrosion patches as small as 0.1m²" },
            { title: "Two-Stage Architecture", desc: "Separate roof detection and corrosion segmentation for maximum accuracy" },
            { title: "Confidence Gating", desc: "Low-confidence results are flagged for human review before quoting" },
          ].map((item) => (
            <div key={item.title} className="p-6 bg-slate-800 rounded-lg border border-slate-700">
              <h3 className="text-lg font-semibold mb-2 text-orange-400">{item.title}</h3>
              <p className="text-slate-300 text-sm">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-700 py-8 text-center text-slate-500 text-sm">
        <p>Roof Corrosion AI — Satellite imagery powered by Maxar/Nearmap</p>
      </footer>
    </div>
  );
}
