export default function HomePage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-4">Roof Corrosion AI</h1>
      <p className="text-lg text-gray-600 max-w-xl text-center">
        Satellite-powered roof corrosion detection and quoting. Upload an
        address, get a precision corrosion analysis and itemized
        repair/replacement quote.
      </p>
      <nav className="mt-8 flex gap-4">
        <a
          href="/portal"
          className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
        >
          Customer Portal
        </a>
        <a
          href="/ops"
          className="rounded-md border border-gray-300 px-4 py-2 hover:bg-gray-50"
        >
          Ops Dashboard
        </a>
      </nav>
    </main>
  );
}
