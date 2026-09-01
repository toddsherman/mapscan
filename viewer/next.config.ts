import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  basePath: "/mapscan",
  allowedDevOrigins: ["127.0.0.1"],
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
