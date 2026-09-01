import elevationDataset from "@/public/data/datasets/elevation-bands-autonomous-v2/dataset.json";
import fireDataset from "@/public/data/staging/fire-autonomous-v2-clipped/dataset.json";
import farmsDataset from "@/public/data/staging/farms-v2-autonomous-v1/dataset.json";
import forestDataset from "@/public/data/staging/forest-autonomous-v5/dataset.json";
import geologicDataset from "@/public/data/staging/geologic-highres-autonomous-v1/dataset.json";
import landslideDataset from "@/public/data/staging/landslide-autonomous-v2-clipped/dataset.json";
import populationDataset from "@/public/data/staging/population-native-fidelity-repair-v2/dataset.json";
import plantDataset from "@/public/data/staging/plant-hardiness-autonomous-v3/dataset.json";
import rainfallDataset from "@/public/data/datasets/rainfall-1981-2010-autonomous-v2/dataset.json";
import { MapScanShell } from "@/components/mapscan-shell";
import type { ViewerDataset } from "@/lib/types";

const datasets = [
  {
    ...forestDataset,
    public_path: "/mapscan/data/staging/forest-autonomous-v5/",
    menu_title: "Forest cover · state-perimeter evidence repair candidate",
  },
  {
    ...plantDataset,
    public_path: "/mapscan/data/staging/plant-hardiness-autonomous-v3/",
    menu_title: "Plant hardiness zones · globally ranked alignment candidate",
  },
  {
    ...farmsDataset,
    public_path: "/mapscan/data/staging/farms-v2-autonomous-v1/",
    menu_title: "Agricultural land use · autonomous candidate",
  },
  {
    ...populationDataset,
    public_path: "/mapscan/data/staging/population-native-fidelity-repair-v2/",
    menu_title: "Population density · native-fidelity candidate",
  },
  {
    ...fireDataset,
    public_path: "/mapscan/data/staging/fire-autonomous-v2-clipped/",
    menu_title: "Fire hazard and responsibility areas · state-clipped candidate",
  },
  {
    ...landslideDataset,
    public_path: "/mapscan/data/staging/landslide-autonomous-v2-clipped/",
    menu_title: "Severe weather impacts · state-clipped candidate",
  },
  {
    ...geologicDataset,
    public_path: "/mapscan/data/staging/geologic-highres-autonomous-v1/",
    menu_title: "Geologic units · high-resolution candidate",
  },
  {
    ...elevationDataset,
    public_path: "/mapscan/data/datasets/elevation-bands-autonomous-v2/",
    menu_title: "Topography and elevation · selectable bands",
  },
  {
    ...rainfallDataset,
    public_path:
      "/mapscan/data/datasets/rainfall-1981-2010-autonomous-v2/",
    menu_title: "Annual precipitation 1981–2010 · newest publication",
  },
] as unknown as ViewerDataset[];

export default function Staging() {
  return <MapScanShell datasets={datasets} />;
}
