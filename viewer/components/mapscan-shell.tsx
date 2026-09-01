"use client";

import dynamic from "next/dynamic";
import type { ViewerDataset } from "@/lib/types";

const MapScanMap = dynamic(
  () => import("./mapscan-map").then((module) => module.MapScanMap),
  {
    ssr: false,
    loading: () => <div className="map-loading">Preparing the map…</div>,
  },
);

export function MapScanShell({ datasets }: { datasets: ViewerDataset[] }) {
  return <MapScanMap datasets={datasets} />;
}
