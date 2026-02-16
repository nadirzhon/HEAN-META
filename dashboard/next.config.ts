import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Allow API calls to HEAN backend */
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
