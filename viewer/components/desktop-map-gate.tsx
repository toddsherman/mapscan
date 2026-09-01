"use client";

import { useEffect, useState } from "react";
import type { ViewerDataset } from "@/lib/types";
import { MapScanShell } from "@/components/mapscan-shell";

const desktopQuery = "(min-width: 768px)";

export function DesktopMapGate({ datasets }: { datasets: ViewerDataset[] }) {
  const [desktop, setDesktop] = useState<boolean | null>(null);

  useEffect(() => {
    const media = window.matchMedia(desktopQuery);
    const update = () => setDesktop(media.matches);

    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  if (desktop === null) {
    return <div className="map-loading">Preparing MapScan…</div>;
  }

  if (!desktop) {
    return (
      <main className="map-mobile-gate">
        <p>MapScan</p>
        <h1>The map is build for desktop.</h1>
        <a href="/mapscan">Read about the project</a>
      </main>
    );
  }

  return <MapScanShell datasets={datasets} />;
}
