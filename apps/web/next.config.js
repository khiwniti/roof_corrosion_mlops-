/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["@roof-corrosion/shared-types"],

  // Proxy API requests to FastAPI backend during development
  // This lets the frontend call /api/* instead of http://localhost:8000/*
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
