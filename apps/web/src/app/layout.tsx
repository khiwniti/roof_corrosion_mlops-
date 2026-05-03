export const metadata = {
  title: "Roof Corrosion AI — Satellite Roof Analysis & Quoting",
  description:
    "Upload an address and get a satellite-derived corrosion analysis and repair/replacement quote for your roof.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
