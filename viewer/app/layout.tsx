import type { Metadata } from "next";
import "mapbox-gl/dist/mapbox-gl.css";
import "./globals.css";
import { siteOrigin, socialMetadata } from "@/lib/social-metadata";

export const metadata: Metadata = {
  metadataBase: siteOrigin,
  title: "MapScan — Compose recovered California map data",
  description:
    "Select and combine individual layers recovered from static California map images.",
  // Canonical and social tags default to the story page at /mapscan, whose
  // title and description (set in app/page.tsx) are repeated here because
  // Next does not derive Open Graph fields from a page's own title.
  // app/map/page.tsx overrides them for the map.
  ...socialMetadata({
    path: "/mapscan",
    title: "MapScan — Recovering data from flat map images",
    description:
      "How I used Claude, Mapbox, and an iterative image-processing pipeline to turn nine static California maps into composable data layers.",
  }),
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
