import quakeDataset from "@/public/data/datasets/quake-shaking-autonomous-v2/dataset.json";
import plantDataset from "@/public/data/datasets/plant-hardiness-autonomous-v3/dataset.json";
import deerDataset from "@/public/data/datasets/deer-distribution-autonomous-v2/dataset.json";
import forestDataset from "@/public/data/datasets/forest-cover-autonomous-v5/dataset.json";
import farmsDataset from "@/public/data/datasets/agricultural-land-use-v2-highres/dataset.json";
import rainfall1981Dataset from "@/public/data/datasets/rainfall-1981-2010-autonomous-v2/dataset.json";
import elevationDataset from "@/public/data/datasets/elevation-bands-autonomous-v2/dataset.json";
import populationDataset from "@/public/data/datasets/population-density/dataset.json";
import fireDataset from "@/public/data/datasets/fire-hazard-responsibility/dataset.json";
import severeWeatherDataset from "@/public/data/datasets/severe-weather-impacts/dataset.json";
import geologicDataset from "@/public/data/datasets/geologic-units-highres/dataset.json";
import { MapScanShell } from "@/components/mapscan-shell";
import type { ViewerDataset } from "@/lib/types";

const datasets = [
  {
    ...quakeDataset,
    public_path: "/mapscan/data/datasets/quake-shaking-autonomous-v2/",
    menu_title: "Earthquake shaking",
  },
  {
    ...plantDataset,
    public_path: "/mapscan/data/datasets/plant-hardiness-autonomous-v3/",
    menu_title: "Plant hardiness zones",
  },
  {
    ...deerDataset,
    public_path: "/mapscan/data/datasets/deer-distribution-autonomous-v2/",
    menu_title: "Deer distribution",
  },
  {
    ...forestDataset,
    public_path: "/mapscan/data/datasets/forest-cover-autonomous-v5/",
    menu_title: "Forest cover",
  },
  {
    ...farmsDataset,
    public_path: "/mapscan/data/datasets/agricultural-land-use-v2-highres/",
    menu_title: "Agricultural land use",
  },
  {
    ...populationDataset,
    public_path: "/mapscan/data/datasets/population-density/",
    menu_title: "Population density, 2020",
  },
  {
    ...fireDataset,
    public_path: "/mapscan/data/datasets/fire-hazard-responsibility/",
    menu_title: "Fire hazard and responsibility areas",
  },
  {
    ...severeWeatherDataset,
    public_path: "/mapscan/data/datasets/severe-weather-impacts/",
    menu_title: "Severe weather impacts",
  },
  {
    ...geologicDataset,
    public_path: "/mapscan/data/datasets/geologic-units-highres/",
    menu_title: "Geologic map, 2010",
  },
  {
    ...rainfall1981Dataset,
    public_path: "/mapscan/data/datasets/rainfall-1981-2010-autonomous-v2/",
    menu_title: "Annual precipitation 1981–2010",
  },
  {
    ...elevationDataset,
    public_path: "/mapscan/data/datasets/elevation-bands-autonomous-v2/",
    menu_title: "Topography and elevation",
  },
] as unknown as ViewerDataset[];

export default function MapPage() {
  return <MapScanShell datasets={datasets} />;
}
