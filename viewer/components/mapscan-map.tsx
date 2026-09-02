"use client";

import Image from "next/image";
import mapboxgl, {
  type ExpressionSpecification,
  type Map as MapboxMap,
} from "mapbox-gl";
import {
  compressToEncodedURIComponent,
  decompressFromEncodedURIComponent,
} from "lz-string";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  findThematicRasterInsertionLayer,
  findWaterFillInsertionLayer,
} from "@/lib/mapbox-style";
import {
  indexedRasterCategoryColor,
  indexedRasterCategoryLayerId,
  indexedRasterSourceId,
} from "@/lib/indexed-raster";
import type { CategoryManifest, CategoryStyle, ViewerDataset } from "@/lib/types";

type CompositionStyles = Record<string, Record<string, CategoryStyle>>;

type MapView = {
  center: [number, number];
  zoom: number;
  bearing: number;
  pitch: number;
};

type SharedCompositionStateV3 = {
  v: 3;
  d: string;
  l: [string, string, 0 | 1, string, number][];
  o: string[];
  x: [string, number][];
  m: [number, number, number, number, number];
};

type DatasetPalette = {
  id: string;
  label: string;
  kind: "source" | "single" | "dual";
  colors?: readonly [string, string];
};

const CALIFORNIA_MIN_ZOOM = 5;
const DEFAULT_LAYER_OPACITY = 0.82;
const DEFAULT_ACTIVE_DATASET_IDS = new Set([
  "california-forest-cover",
  "california-agricultural-land-use",
]);
const DEFAULT_FOCUSED_DATASET_ID = "california-forest-cover";
const DATASET_PALETTES = [
  { id: "source", label: "Source", kind: "source" },
  {
    id: "blue",
    label: "Blue",
    kind: "single",
    colors: ["#d9e9f7", "#164d78"],
  },
  {
    id: "green",
    label: "Green",
    kind: "single",
    colors: ["#dcebd5", "#245f3b"],
  },
  {
    id: "purple",
    label: "Purple",
    kind: "single",
    colors: ["#eadcf2", "#5c2d73"],
  },
  {
    id: "orange",
    label: "Orange",
    kind: "single",
    colors: ["#f7e1c9", "#a84b1d"],
  },
  {
    id: "blue-gold",
    label: "Blue / gold",
    kind: "dual",
    colors: ["#246b91", "#e1a43a"],
  },
  {
    id: "teal-coral",
    label: "Teal / coral",
    kind: "dual",
    colors: ["#247f7d", "#df7056"],
  },
] as const satisfies readonly DatasetPalette[];

function mixHexColors(start: string, end: string, amount: number) {
  const startValue = Number.parseInt(start.slice(1), 16);
  const endValue = Number.parseInt(end.slice(1), 16);
  const mixChannel = (shift: number) => {
    const startChannel = (startValue >> shift) & 0xff;
    const endChannel = (endValue >> shift) & 0xff;
    return Math.round(startChannel + (endChannel - startChannel) * amount);
  };
  return `#${[mixChannel(16), mixChannel(8), mixChannel(0)]
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`;
}

function datasetPaletteColors(
  palette: DatasetPalette,
  categories: CategoryManifest[],
) {
  const endpoints = palette.colors;
  if (palette.kind === "source" || !endpoints) {
    return categories.map((category) => rgbToHex(category.display_rgb));
  }
  const denominator = Math.max(1, categories.length - 1);
  return categories.map((_, index) =>
    mixHexColors(
      endpoints[0],
      endpoints[1],
      categories.length === 1 ? 0.5 : index / denominator,
    ),
  );
}

function matchingDatasetPalette(
  datasetStyles: Record<string, CategoryStyle>,
  categories: CategoryManifest[],
) {
  return DATASET_PALETTES.find((palette) => {
    const paletteColors = datasetPaletteColors(palette, categories);
    return categories.every(
      (category, index) =>
        datasetStyles[category.id]?.color.toLowerCase() === paletteColors[index],
    );
  });
}

function datasetPalettePreview(
  palette: DatasetPalette,
  categories: CategoryManifest[],
) {
  const colors = datasetPaletteColors(palette, categories);
  if (colors.length === 0) return "transparent";
  if (colors.length === 1) return colors[0];
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

function sharedStateV3FromUrl() {
  if (typeof window === "undefined") return null;
  const searchParams = new URL(window.location.href).searchParams;
  if (searchParams.get("config") !== "3") return null;
  const encoded = searchParams.get("state");
  if (!encoded) return null;
  try {
    const decompressed = decompressFromEncodedURIComponent(encoded);
    if (!decompressed) return null;
    const parsed = JSON.parse(decompressed) as Partial<SharedCompositionStateV3>;
    return parsed.v === 3 ? parsed : null;
  } catch {
    return null;
  }
}

function validView(
  values: unknown,
  minimumZoom: number,
): MapView | null {
  if (!Array.isArray(values)) return null;
  const [longitude, latitude, zoom, bearing = 0, pitch = 0] = values.map(Number);
  if (
    !Number.isFinite(longitude) ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(zoom) ||
    longitude < -180 ||
    longitude > 180 ||
    latitude < -85 ||
    latitude > 85 ||
    zoom < minimumZoom ||
    zoom > 15 ||
    !Number.isFinite(bearing) ||
    bearing < -180 ||
    bearing > 180 ||
    !Number.isFinite(pitch) ||
    pitch < 0 ||
    pitch > 85
  ) {
    return null;
  }
  return {
    center: [longitude, latitude],
    zoom,
    bearing,
    pitch,
  };
}

function rgbToHex([red, green, blue]: [number, number, number]) {
  return `#${[red, green, blue].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function mapboxKey(value: string) {
  return value.replace(/[^a-zA-Z0-9_-]/g, "-");
}

function categoryLayerId(datasetId: string, category: CategoryManifest) {
  return `mapscan-${mapboxKey(datasetId)}-${mapboxKey(category.id)}`;
}

function indexedKey(datasetId: string, layerId: string) {
  return `${mapboxKey(datasetId)}-${mapboxKey(layerId)}`;
}

function datasetLayerForCategory(dataset: ViewerDataset, categoryId: string) {
  return dataset.layers.find((layer) =>
    layer.categories.some((category) => category.id === categoryId),
  );
}

function renderLayerId(
  dataset: ViewerDataset,
  category: CategoryManifest,
) {
  const datasetLayer = datasetLayerForCategory(dataset, category.id);
  if (datasetLayer?.indexed_raster) {
    return indexedRasterCategoryLayerId(
      indexedKey(dataset.id, datasetLayer.id),
      category.id,
    );
  }
  return categoryLayerId(dataset.id, category);
}

function referenceBoundaryId(datasetId: string) {
  return `mapscan-reference-boundary-${mapboxKey(datasetId)}`;
}

function rasterColor(color: string): ExpressionSpecification {
  return ["interpolate", ["linear"], ["raster-value"], 0, color, 1, color];
}

function isNativeColor(category: CategoryManifest) {
  return category.render_mode === "native_color";
}

function rgb([red, green, blue]: [number, number, number]) {
  return `rgb(${red}, ${green}, ${blue})`;
}

function legendGradient(category: CategoryManifest) {
  const stops = category.legend_stops ?? [];
  if (stops.length < 2) return rgb(category.display_rgb);
  return `linear-gradient(to right, ${stops.map((stop) => rgb(stop.display_rgb)).join(", ")})`;
}

function datasetCategories(dataset: ViewerDataset) {
  return dataset.layers.flatMap((layer) => layer.categories);
}

function bulkSelectableCategories(dataset: ViewerDataset) {
  const categories = datasetCategories(dataset);
  const bands = categories.filter(
    (category) => category.category_role === "continuous_band",
  );
  return bands.length > 0 ? bands : categories;
}

function datasetAssetBase(dataset: ViewerDataset) {
  return dataset.asset_base ?? dataset.public_path;
}

function initialDatasetOrder(datasets: ViewerDataset[]) {
  const fallback = datasets.map((dataset) => dataset.id);
  if (typeof window === "undefined") return fallback;
  const shared = sharedStateV3FromUrl();
  const searchParams = new URL(window.location.href).searchParams;
  const encoded = searchParams.get("datasetOrder");
  const legacyOrder = searchParams.get("order");
  const requestedIds = Array.isArray(shared?.o)
    ? shared.o.filter((id): id is string => typeof id === "string")
    : encoded
      ? encoded.split(",")
      : (legacyOrder ?? "").split(",").flatMap((qualifiedId) => {
        const separator = qualifiedId.indexOf("~");
        return separator > 0 ? [qualifiedId.slice(0, separator)] : [];
      });
  if (requestedIds.length === 0) return fallback;
  const known = new Set(fallback);
  const requested: string[] = [];
  for (const id of requestedIds) {
    if (!known.has(id)) continue;
    const priorIndex = requested.indexOf(id);
    if (priorIndex >= 0) requested.splice(priorIndex, 1);
    requested.push(id);
  }
  const requestedSet = new Set(requested);
  return [...fallback.filter((id) => !requestedSet.has(id)), ...requested];
}

function initialDatasetOpacities(datasets: ViewerDataset[]) {
  const fallback = Object.fromEntries(datasets.map((dataset) => [dataset.id, 1]));
  if (typeof window === "undefined") return fallback;
  const shared = sharedStateV3FromUrl();
  if (Array.isArray(shared?.x)) {
    for (const entry of shared.x) {
      if (!Array.isArray(entry) || entry.length !== 2) continue;
      const [datasetId, rawPercent] = entry;
      const percent = Number(rawPercent);
      if (typeof datasetId !== "string" || !(datasetId in fallback)) continue;
      if (!Number.isFinite(percent)) continue;
      fallback[datasetId] = Math.min(1, Math.max(0, percent / 100));
    }
    return fallback;
  }
  const encoded = new URL(window.location.href).searchParams.get("datasetOpacity");
  if (!encoded) return fallback;
  for (const entry of encoded.split(",")) {
    const separator = entry.lastIndexOf(":");
    if (separator < 1) continue;
    const datasetId = entry.slice(0, separator);
    const percent = Number(entry.slice(separator + 1));
    if (!(datasetId in fallback) || !Number.isFinite(percent)) continue;
    fallback[datasetId] = Math.min(1, Math.max(0, percent / 100));
  }
  return fallback;
}

function moveMapDatasets(
  map: MapboxMap,
  datasets: ViewerDataset[],
  order: string[],
  beforeId?: string,
) {
  for (const datasetId of order) {
    const dataset = datasets.find((candidate) => candidate.id === datasetId);
    if (!dataset) continue;
    for (const datasetLayer of dataset.layers) {
      for (const category of datasetLayer.categories) {
        const id = renderLayerId(dataset, category);
        if (map.getLayer(id)) map.moveLayer(id, beforeId);
      }
    }
  }
}

function reloadVisibleRasterSources(
  map: MapboxMap,
  sourceLayers: Map<string, string[]>,
) {
  for (const [sourceId, layerIds] of sourceLayers) {
    const hasVisibleLayer = layerIds.some(
      (layerId) =>
        map.getLayer(layerId) &&
        map.getLayoutProperty(layerId, "visibility") !== "none",
    );
    if (!hasVisibleLayer) continue;
    const source = map.getSource(sourceId);
    if (source && "reload" in source && typeof source.reload === "function") {
      source.reload();
    }
  }
}

function DatasetSymbol({ dataset }: { dataset: ViewerDataset }) {
  const id = dataset.id;
  if (id.includes("forest")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 7.5 10h2.7L6.5 16h4.2v5h2.6v-5h4.2l-3.7-6h2.7Z" /></svg>;
  }
  if (id.includes("hardiness")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1" /></svg>;
  }
  if (id.includes("deer")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4v5l4 4 4-4V4M8 7 4 5M16 7l4-2M12 13v7M9 17h6" /></svg>;
  }
  if (id.includes("agricultural")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v16H4zM4 10h16M10 4v16M16 10v10" /></svg>;
  }
  if (id.includes("population")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="8" cy="8" r="2.5" /><circle cx="16" cy="8" r="2.5" /><circle cx="12" cy="15" r="3" /><path d="M3 20c.4-3 2-5 5-5M21 20c-.4-3-2-5-5-5" /></svg>;
  }
  if (id.includes("fire")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2c1 5-3 6-1 10 1-2 3-3 4-5 3 3 4 6 3 9a7 7 0 0 1-14 0c0-4 3-7 6-10 0 3 0 4 2 6" /></svg>;
  }
  if (id.includes("severe")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2 6 13h6l-1 9 7-12h-6Z" /></svg>;
  }
  if (id.includes("geologic")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7c4-3 7 3 11 0s6 0 7 1M3 12c4-3 7 3 11 0s6 0 7 1M3 17c4-3 7 3 11 0s6 0 7 1" /></svg>;
  }
  if (id.includes("precipitation")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2S6 10 6 15a6 6 0 0 0 12 0c0-5-6-13-6-13Z" /></svg>;
  }
  if (id.includes("elevation") || id.includes("topography")) {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m2 20 7-12 3 5 3-7 7 14ZM7 20l5-7 4 7" /></svg>;
  }
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 13h4l2-7 4 13 3-10 2 4h5" /></svg>;
}

function DatasetSelectionToggle({
  dataset,
  checked,
  indeterminate,
  onChange,
}: {
  dataset: ViewerDataset;
  checked: boolean;
  indeterminate: boolean;
  onChange: (checked: boolean) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (inputRef.current) inputRef.current.indeterminate = indeterminate;
  }, [indeterminate]);

  return (
    <input
      ref={inputRef}
      className="dataset-selection-toggle"
      type="checkbox"
      checked={checked}
      aria-label={`${checked ? "Clear" : "Select"} all ${datasetShortLabel(dataset)} layers`}
      onChange={(event) => onChange(event.target.checked)}
    />
  );
}

function datasetShortLabel(dataset: ViewerDataset) {
  const id = dataset.id;
  if (id.includes("forest")) return "Forest";
  if (id.includes("hardiness")) return "Hardiness";
  if (id.includes("deer")) return "Deer";
  if (id.includes("agricultural")) return "Farms";
  if (id.includes("population")) return "Population";
  if (id.includes("fire")) return "Fire";
  if (id.includes("severe")) return "Weather";
  if (id.includes("geologic")) return "Geology";
  if (id.includes("precipitation")) return "Rainfall";
  if (id.includes("elevation") || id.includes("topography")) return "Elevation";
  return "Quake";
}

function initialStyles(datasets: ViewerDataset[], focusedDatasetId: string) {
  const explicitDatasetId =
    typeof window === "undefined"
      ? null
      : new URL(window.location.href).searchParams.get("dataset");
  const enabledDatasetIds = explicitDatasetId
    ? new Set([focusedDatasetId])
    : DEFAULT_ACTIVE_DATASET_IDS;

  return Object.fromEntries(
    datasets.map((dataset) => [
      dataset.id,
      Object.fromEntries(
        datasetCategories(dataset).map((category) => [
          category.id,
          {
            enabled:
              enabledDatasetIds.has(dataset.id)
                ? (category.default_enabled ?? true)
                : false,
            color: rgbToHex(category.display_rgb),
            opacity: DEFAULT_LAYER_OPACITY,
          },
        ]),
      ),
    ]),
  ) as CompositionStyles;
}

function stylesFromUrl(datasets: ViewerDataset[], focusedDatasetId: string) {
  const fallback = initialStyles(datasets, focusedDatasetId);
  if (typeof window === "undefined") return fallback;

  const shared = sharedStateV3FromUrl();
  const encoded = new URL(window.location.href).searchParams.get("layers");
  if (!Array.isArray(shared?.l) && !encoded) return fallback;

  for (const datasetStyles of Object.values(fallback)) {
    for (const style of Object.values(datasetStyles)) style.enabled = false;
  }
  if (Array.isArray(shared?.l)) {
    for (const entry of shared.l) {
      if (!Array.isArray(entry) || entry.length !== 5) continue;
      const [datasetId, categoryId, enabled, hex, rawOpacity] = entry;
      const percent = Number(rawOpacity);
      if (
        typeof datasetId !== "string" ||
        typeof categoryId !== "string" ||
        !fallback[datasetId]?.[categoryId] ||
        !(enabled === 0 || enabled === 1) ||
        typeof hex !== "string" ||
        !/^[0-9a-f]{6}$/i.test(hex) ||
        !Number.isFinite(percent)
      ) {
        continue;
      }
      fallback[datasetId][categoryId] = {
        enabled: enabled === 1,
        color: `#${hex}`,
        opacity: Math.min(1, Math.max(0, percent / 100)),
      };
    }
    return fallback;
  }
  if (!encoded) return fallback;
  if (encoded === "none") return fallback;

  for (const entry of encoded.split(",")) {
    const fields = entry.split(":");
    if (fields.length === 4 && fields[0].includes("~")) {
      const [qualifiedId, enabled, hex, opacity] = fields;
      const separator = qualifiedId.indexOf("~");
      const datasetId = qualifiedId.slice(0, separator);
      const categoryId = qualifiedId.slice(separator + 1);
      const percent = Number(opacity);
      if (
        !fallback[datasetId]?.[categoryId] ||
        !/^[0-9a-f]{6}$/i.test(hex) ||
        !Number.isFinite(percent)
      ) {
        continue;
      }
      fallback[datasetId][categoryId] = {
        enabled: enabled === "1",
        color: `#${hex}`,
        opacity: Math.min(1, Math.max(0, percent / 100)),
      };
      continue;
    }

    // Backward compatibility for the first multi-dataset format:
    // dataset-id~category-id:rrggbb:opacity-percent.
    if (fields.length === 3) {
      const [qualifiedId, hex, opacity] = fields;
      const separator = qualifiedId.indexOf("~");
      if (separator < 1) continue;
      const datasetId = qualifiedId.slice(0, separator);
      const categoryId = qualifiedId.slice(separator + 1);
      const percent = Number(opacity);
      if (
        !fallback[datasetId]?.[categoryId] ||
        !/^[0-9a-f]{6}$/i.test(hex) ||
        !Number.isFinite(percent)
      ) {
        continue;
      }
      fallback[datasetId][categoryId] = {
        enabled: true,
        color: `#${hex}`,
        opacity: Math.min(1, Math.max(0, percent / 100)),
      };
      continue;
    }

    // Backward compatibility for the original single-dataset share format:
    // category-id:enabled:rrggbb:opacity-percent.
    const [categoryId, enabled, hex, opacity] = fields;
    const percent = Number(opacity);
    if (
      !fallback[focusedDatasetId]?.[categoryId] ||
      !/^[0-9a-f]{6}$/i.test(hex ?? "") ||
      !Number.isFinite(percent)
    ) {
      continue;
    }
    fallback[focusedDatasetId][categoryId] = {
      enabled: enabled === "1",
      color: `#${hex}`,
      opacity: Math.min(1, Math.max(0, percent / 100)),
    };
  }
  return fallback;
}

function selectedDatasetFromUrl(datasets: ViewerDataset[]) {
  const defaultDatasetId =
    datasets.find((dataset) => dataset.id === DEFAULT_FOCUSED_DATASET_ID)?.id ??
    datasets[0].id;
  if (typeof window === "undefined") return defaultDatasetId;
  const shared = sharedStateV3FromUrl();
  const sharedDataset = typeof shared?.d === "string" ? shared.d : null;
  if (datasets.some((dataset) => dataset.id === sharedDataset)) return sharedDataset!;
  const requested = new URL(window.location.href).searchParams.get("dataset");
  return datasets.some((dataset) => dataset.id === requested) ? requested! : defaultDatasetId;
}

function viewFromUrl(minimumZoom: number) {
  if (typeof window === "undefined") return null;
  const shared = sharedStateV3FromUrl();
  const sharedView = validView(shared?.m, minimumZoom);
  if (sharedView) return sharedView;
  const encoded = new URL(window.location.href).searchParams.get("view");
  if (!encoded) return null;
  return validView(encoded.split(","), minimumZoom);
}

function selectedCount(dataset: ViewerDataset, styles: CompositionStyles) {
  return datasetCategories(dataset).reduce(
    (count, category) => count + (styles[dataset.id]?.[category.id]?.enabled ? 1 : 0),
    0,
  );
}

function compositionUrl(
  datasets: ViewerDataset[],
  focusedDatasetId: string,
  styles: CompositionStyles,
  datasetOpacities: Record<string, number>,
  datasetOrder: string[],
  map: MapboxMap,
) {
  const layers = datasets.flatMap((candidateDataset) =>
    datasetCategories(candidateDataset).flatMap((category) => {
      const style = styles[candidateDataset.id][category.id];
      const defaultColor = rgbToHex(category.display_rgb);
      const opacityPercent = Math.round(style.opacity * 100);
      const hasStyleOverride =
        style.color.toLowerCase() !== defaultColor ||
        opacityPercent !== Math.round(DEFAULT_LAYER_OPACITY * 100);
      return style.enabled || hasStyleOverride
        ? [[
            candidateDataset.id,
            category.id,
            style.enabled ? 1 : 0,
            style.color.slice(1).toLowerCase(),
            opacityPercent,
          ] as SharedCompositionStateV3["l"][number]]
        : [];
    }),
  );
  const datasetOpacity = datasets.flatMap((candidateDataset) => {
    const percent = Math.round((datasetOpacities[candidateDataset.id] ?? 1) * 100);
    return percent === 100
      ? []
      : [[candidateDataset.id, percent] as SharedCompositionStateV3["x"][number]];
  });
  const center = map.getCenter();
  const state: SharedCompositionStateV3 = {
    v: 3,
    d: focusedDatasetId,
    l: layers,
    o: datasetOrder,
    x: datasetOpacity,
    m: [
      Number(center.lng.toFixed(6)),
      Number(center.lat.toFixed(6)),
      Number(map.getZoom().toFixed(4)),
      Number(map.getBearing().toFixed(2)),
      Number(map.getPitch().toFixed(2)),
    ],
  };
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("config", "3");
  url.searchParams.set(
    "state",
    compressToEncodedURIComponent(JSON.stringify(state)),
  );
  return url;
}

export function MapScanMap({ datasets }: { datasets: ViewerDataset[] }) {
  const minimumMapZoom = Math.max(
    CALIFORNIA_MIN_ZOOM,
    ...datasets.map((candidate) => candidate.minimum_zoom),
  );
  const [selectedId, setSelectedId] = useState(() => selectedDatasetFromUrl(datasets));
  const [styles, setStyles] = useState(() =>
    stylesFromUrl(datasets, selectedDatasetFromUrl(datasets)),
  );
  const [datasetOpacities, setDatasetOpacities] = useState(() =>
    initialDatasetOpacities(datasets),
  );
  const [datasetOrder, setDatasetOrder] = useState(() => initialDatasetOrder(datasets));
  const dataset = datasets.find((candidate) => candidate.id === selectedId) ?? datasets[0];
  const categories = useMemo(() => datasetCategories(dataset), [dataset]);
  const [mapReady, setMapReady] = useState(false);
  const [sourceOpen, setSourceOpen] = useState(false);
  const [alignmentInspection, setAlignmentInspection] = useState(false);
  const [copied, setCopied] = useState(false);
  const mapContainer = useRef<HTMLDivElement>(null);
  const datasetListRef = useRef<HTMLDivElement>(null);
  const pendingDatasetPositionsRef = useRef<Map<string, number> | null>(null);
  const datasetAnimationsRef = useRef<Map<string, Animation>>(new Map());
  const mapRef = useRef<MapboxMap | null>(null);
  const initialDatasetRef = useRef(dataset);
  const initialStylesRef = useRef(styles);
  const initialDatasetOpacitiesRef = useRef(datasetOpacities);
  const initialDatasetOrderRef = useRef(datasetOrder);
  const appliedStylesRef = useRef(styles);
  const appliedDatasetOpacitiesRef = useRef(datasetOpacities);
  const preInspectionStylesRef = useRef<CompositionStyles | null>(null);
  const initialViewRef = useRef(viewFromUrl(minimumMapZoom));
  const renderOrderAnchorRef = useRef<string | undefined>(undefined);
  const shareStateRef = useRef({
    selectedId,
    styles,
    datasetOpacities,
    datasetOrder,
  });
  shareStateRef.current = {
    selectedId,
    styles,
    datasetOpacities,
    datasetOrder,
  };
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  const approved = dataset.approval?.status === "approved";
  const boundaryComponentCount = dataset.boundary?.continuous_border_component_count;
  const expectedBoundaryComponentCount =
    dataset.boundary?.expected_boundary_component_count ?? boundaryComponentCount;
  const exactBoundary =
    typeof boundaryComponentCount === "number" &&
    boundaryComponentCount >= 1 &&
    boundaryComponentCount === expectedBoundaryComponentCount &&
    dataset.boundary?.colored_pixel_count_outside_boundary === 0 &&
    dataset.boundary?.unclassified_pixel_count_inside_boundary === 0;
  const selectionCounts = useMemo(
    () =>
      Object.fromEntries(
        datasets.map((candidate) => [candidate.id, selectedCount(candidate, styles)]),
      ) as Record<string, number>,
    [datasets, styles],
  );
  const activeLayerCount = Object.values(selectionCounts).reduce(
    (total, count) => total + count,
    0,
  );
  const activeDatasetCount = Object.values(selectionCounts).filter(
    (count) => count > 0,
  ).length;
  const selectedDatasetLayerCount = selectionCounts[dataset.id] ?? 0;
  const activeDatasetPalette = useMemo(
    () => matchingDatasetPalette(styles[dataset.id], categories),
    [categories, dataset.id, styles],
  );
  const orderedDatasets = useMemo(
    () =>
      datasetOrder.flatMap((datasetId) => {
        const candidateDataset = datasets.find((candidate) => candidate.id === datasetId);
        return candidateDataset ? [candidateDataset] : [];
      }),
    [datasetOrder, datasets],
  );

  useEffect(() => {
    if (!mapContainer.current || !token || mapRef.current) return;
    mapboxgl.accessToken = token;
    const initialView = initialViewRef.current;
    const initialDataset = initialDatasetRef.current;
    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: initialView?.center ?? initialDataset.center,
      zoom: initialView?.zoom ?? 5.25,
      bearing: initialView?.bearing ?? 0,
      pitch: initialView?.pitch ?? 0,
      minZoom: minimumMapZoom,
      maxZoom: 15,
      attributionControl: true,
    });
    const thematicSourceLayers = new Map<string, string[]>();
    const refreshLowZoomThematicSources = () => {
      if (map.getZoom() > minimumMapZoom + 0.75) return;
      reloadVisibleRasterSources(map, thematicSourceLayers);
    };
    const syncMapViewToUrl = () => replaceUrlWithCurrentState(map);
    map.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "bottom-right");
    map.on("zoomend", refreshLowZoomThematicSources);
    map.on("moveend", syncMapViewToUrl);
    map.on("load", () => {
      const styleLayers = map.getStyle().layers;
      const waterMaskLayerId = findWaterFillInsertionLayer(styleLayers);
      const postFillLayerId = findThematicRasterInsertionLayer(styleLayers);
      const thematicInsertionLayerId = postFillLayerId ?? waterMaskLayerId;
      const fallbackInitialStyles = initialStyles(datasets, initialDataset.id);
      if (!waterMaskLayerId) {
        console.warn(
          "MapScan could not find a Mapbox water fill; low-zoom coastal pixels will rely on each dataset's pinned alignment diagnostic.",
        );
      }

      for (const candidateDataset of datasets) {
        const candidateStyles =
          initialStylesRef.current[candidateDataset.id] ??
          fallbackInitialStyles[candidateDataset.id];
        const assetBase = datasetAssetBase(candidateDataset);
        for (const datasetLayer of candidateDataset.layers) {
          if (datasetLayer.indexed_raster) {
            const indexed = datasetLayer.indexed_raster;
            const key = indexedKey(candidateDataset.id, datasetLayer.id);
            const sourceId = indexedRasterSourceId(key);
            const template = `${window.location.origin}${assetBase}${indexed.tile_template}`;
            map.addSource(sourceId, {
              type: "raster",
              tiles: [template],
              tileSize: 256,
              minzoom: candidateDataset.minimum_zoom,
              maxzoom: candidateDataset.maximum_native_zoom,
              bounds: candidateDataset.bounds,
            });
            thematicSourceLayers.set(
              sourceId,
              datasetLayer.categories.map((category) =>
                indexedRasterCategoryLayerId(key, category.id),
              ),
            );
            for (const category of datasetLayer.categories) {
              const style = candidateStyles[category.id];
              map.addLayer(
                {
                  id: indexedRasterCategoryLayerId(key, category.id),
                  type: "raster",
                  source: sourceId,
                  minzoom: candidateDataset.minimum_zoom,
                  paint: {
                    "raster-fade-duration": 0,
                    "raster-resampling": "nearest",
                    "raster-opacity":
                      style.opacity *
                      initialDatasetOpacitiesRef.current[candidateDataset.id],
                    "raster-color-mix": indexed.raster_color_mix,
                    "raster-color-range": indexed.raster_color_range,
                    "raster-color": indexedRasterCategoryColor(category, style.color),
                  },
                  layout: { visibility: style.enabled ? "visible" : "none" },
                },
                thematicInsertionLayerId,
              );
            }
            continue;
          }

          for (const category of datasetLayer.categories) {
            if (!category.tile_template) {
              console.warn(
                `MapScan category ${candidateDataset.id}/${category.id} has no tile template.`,
              );
              continue;
            }
            const id = categoryLayerId(candidateDataset.id, category);
            const style = candidateStyles[category.id];
            const template = `${window.location.origin}${assetBase}${category.tile_template}`;
            map.addSource(id, {
              type: "raster",
              tiles: [template],
              tileSize: 256,
              minzoom: candidateDataset.minimum_zoom,
              maxzoom: candidateDataset.maximum_native_zoom,
              bounds: candidateDataset.bounds,
            });
            thematicSourceLayers.set(id, [id]);
            map.addLayer(
              {
                id,
                type: "raster",
                source: id,
                minzoom: candidateDataset.minimum_zoom,
                paint: {
                  "raster-fade-duration": 0,
                  "raster-resampling": "nearest",
                  "raster-opacity":
                    style.opacity *
                    initialDatasetOpacitiesRef.current[candidateDataset.id],
                  ...(isNativeColor(category)
                    ? {}
                    : {
                        "raster-color-mix": [1, 0, 0, 0],
                        "raster-color-range": [0, 1],
                        "raster-color": rasterColor(style.color),
                      }),
                },
                layout: { visibility: style.enabled ? "visible" : "none" },
              },
              thematicInsertionLayerId,
            );
          }
        }
      }

      if (waterMaskLayerId && postFillLayerId) map.moveLayer(waterMaskLayerId, postFillLayerId);
      const renderOrderAnchor = waterMaskLayerId ?? thematicInsertionLayerId ?? undefined;
      renderOrderAnchorRef.current = renderOrderAnchor;
      moveMapDatasets(
        map,
        datasets,
        initialDatasetOrderRef.current,
        renderOrderAnchor,
      );

      for (const candidateDataset of datasets) {
        const id = referenceBoundaryId(candidateDataset.id);
        const assetBase = datasetAssetBase(candidateDataset);
        if (candidateDataset.boundary?.raster && candidateDataset.boundary.raster_bounds) {
          const [west, south, east, north] = candidateDataset.boundary.raster_bounds;
          map.addSource(id, {
            type: "image",
            url: `${window.location.origin}${assetBase}${candidateDataset.boundary.raster}`,
            coordinates: [
              [west, north],
              [east, north],
              [east, south],
              [west, south],
            ],
          });
          map.addLayer({
            id,
            type: "raster",
            source: id,
            layout: { visibility: "none" },
            paint: {
              "raster-fade-duration": 0,
              "raster-resampling": "nearest",
              "raster-opacity": 1,
            },
          });
        } else if (candidateDataset.boundary?.geojson) {
          map.addSource(id, {
            type: "geojson",
            data: `${window.location.origin}${assetBase}${candidateDataset.boundary.geojson}`,
          });
          map.addLayer({
            id,
            type: "line",
            source: id,
            layout: { visibility: "none" },
            paint: {
              "line-color": "#15c968",
              "line-opacity": 0.96,
              "line-width": ["interpolate", ["linear"], ["zoom"], 4, 1.25, 9, 2.2, 14, 3.2],
            },
          });
        }
      }

      if (!initialView) map.fitBounds(initialDataset.bounds, { padding: 54, duration: 0 });
      setMapReady(true);
    });
    mapRef.current = map;
    return () => {
      map.off("zoomend", refreshLowZoomThematicSources);
      map.off("moveend", syncMapViewToUrl);
      map.remove();
      mapRef.current = null;
    };
  }, [datasets, minimumMapZoom, token]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const previousStyles = appliedStylesRef.current;
    const previousDatasetOpacities = appliedDatasetOpacitiesRef.current;
    const fallbackCurrentStyles = initialStyles(datasets, selectedId);
    for (const candidateDataset of datasets) {
      const candidateStyles =
        styles[candidateDataset.id] ?? fallbackCurrentStyles[candidateDataset.id];
      const previousCandidateStyles = previousStyles[candidateDataset.id] ?? {};
      const datasetOpacity = datasetOpacities[candidateDataset.id] ?? 1;
      const previousDatasetOpacity = previousDatasetOpacities[candidateDataset.id] ?? 1;
      for (const datasetLayer of candidateDataset.layers) {
        if (datasetLayer.indexed_raster) {
          const key = indexedKey(candidateDataset.id, datasetLayer.id);
          for (const category of datasetLayer.categories) {
            const style = candidateStyles[category.id];
            const previous = previousCandidateStyles[category.id];
            const id = indexedRasterCategoryLayerId(key, category.id);
            if (!map.getLayer(id)) continue;
            if (!previous || previous.enabled !== style.enabled) {
              map.setLayoutProperty(id, "visibility", style.enabled ? "visible" : "none");
            }
            if (!previous || previous.color !== style.color) {
              map.setPaintProperty(
                id,
                "raster-color",
                indexedRasterCategoryColor(category, style.color),
              );
            }
            if (
              !previous ||
              previous.opacity !== style.opacity ||
              previousDatasetOpacity !== datasetOpacity
            ) {
              map.setPaintProperty(
                id,
                "raster-opacity",
                style.opacity * datasetOpacity,
              );
            }
          }
          continue;
        }

        for (const category of datasetLayer.categories) {
          const id = categoryLayerId(candidateDataset.id, category);
          const style = candidateStyles[category.id];
          const previous = previousCandidateStyles[category.id];
          if (!map.getLayer(id)) continue;
          if (!previous || previous.enabled !== style.enabled) {
            map.setLayoutProperty(id, "visibility", style.enabled ? "visible" : "none");
          }
          if (
            !previous ||
            previous.opacity !== style.opacity ||
            previousDatasetOpacity !== datasetOpacity
          ) {
            map.setPaintProperty(
              id,
              "raster-opacity",
              style.opacity * datasetOpacity,
            );
          }
          if (!isNativeColor(category) && (!previous || previous.color !== style.color)) {
            map.setPaintProperty(id, "raster-color", rasterColor(style.color));
          }
        }
      }
    }
    appliedStylesRef.current = styles;
    appliedDatasetOpacitiesRef.current = datasetOpacities;
  }, [datasetOpacities, datasets, mapReady, selectedId, styles]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    moveMapDatasets(map, datasets, datasetOrder, renderOrderAnchorRef.current);
  }, [datasetOrder, datasets, mapReady]);

  useEffect(() => {
    if (!mapReady) return;
    replaceUrlWithCurrentState();
  }, [datasetOpacities, datasetOrder, mapReady, selectedId, styles]);

  useLayoutEffect(() => {
    const previousPositions = pendingDatasetPositionsRef.current;
    pendingDatasetPositionsRef.current = null;
    const list = datasetListRef.current;
    if (
      !previousPositions ||
      !list ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      return;
    }

    for (const row of list.querySelectorAll<HTMLElement>("[data-dataset-id]")) {
      const datasetId = row.dataset.datasetId;
      const previousTop = datasetId ? previousPositions.get(datasetId) : undefined;
      if (previousTop === undefined) continue;
      const previousAnimation = datasetId
        ? datasetAnimationsRef.current.get(datasetId)
        : undefined;
      previousAnimation?.cancel();
      const delta = previousTop - row.getBoundingClientRect().top;
      if (Math.abs(delta) < 1) continue;
      const animation = row.animate(
        [
          { transform: `translateY(${delta}px)`, zIndex: 1 },
          { transform: "translateY(0)", zIndex: 1 },
        ],
        {
          duration: 280,
          easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        },
      );
      if (datasetId) {
        datasetAnimationsRef.current.set(datasetId, animation);
        animation.addEventListener("finish", () => {
          if (datasetAnimationsRef.current.get(datasetId) === animation) {
            datasetAnimationsRef.current.delete(datasetId);
          }
        });
      }
    }
  }, [datasetOrder]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    for (const candidateDataset of datasets) {
      const id = referenceBoundaryId(candidateDataset.id);
      if (!map.getLayer(id)) continue;
      map.setLayoutProperty(
        id,
        "visibility",
        alignmentInspection && candidateDataset.id === dataset.id ? "visible" : "none",
      );
    }
  }, [alignmentInspection, dataset.id, datasets, mapReady]);

  function selectDataset(id: string) {
    setSourceOpen(false);
    setSelectedId(id);
  }

  function replaceUrlWithCurrentState(map = mapRef.current) {
    if (!map) return null;
    const current = shareStateRef.current;
    const url = compositionUrl(
      datasets,
      current.selectedId,
      current.styles,
      current.datasetOpacities,
      current.datasetOrder,
      map,
    );
    if (url.toString() !== window.location.href) {
      window.history.replaceState(null, "", url);
    }
    return url;
  }

  function updateCategory(id: string, patch: Partial<CategoryStyle>) {
    setStyles((current) => ({
      ...current,
      [dataset.id]: {
        ...current[dataset.id],
        [id]: { ...current[dataset.id][id], ...patch },
      },
    }));
  }

  function toggleCategory(category: CategoryManifest) {
    setStyles((current) => {
      const datasetStyle = { ...current[dataset.id] };
      const enabling = !datasetStyle[category.id].enabled;
      datasetStyle[category.id] = { ...datasetStyle[category.id], enabled: enabling };
      if (enabling && category.category_role === "continuous_band") {
        for (const candidate of categories) {
          if (candidate.category_role === "continuous_surface") {
            datasetStyle[candidate.id] = { ...datasetStyle[candidate.id], enabled: false };
          }
        }
      } else if (enabling && category.category_role === "continuous_surface") {
        for (const candidate of categories) {
          if (candidate.category_role === "continuous_band") {
            datasetStyle[candidate.id] = { ...datasetStyle[candidate.id], enabled: false };
          }
        }
      }
      return { ...current, [dataset.id]: datasetStyle };
    });
  }

  function resetCategoryColor(category: CategoryManifest) {
    updateCategory(category.id, { color: rgbToHex(category.display_rgb) });
  }

  function applyDatasetPalette(palette: DatasetPalette) {
    const colors = datasetPaletteColors(palette, categories);
    setStyles((current) => ({
      ...current,
      [dataset.id]: Object.fromEntries(
        categories.map((category, index) => [
          category.id,
          {
            ...current[dataset.id][category.id],
            color: colors[index],
          },
        ]),
      ),
    }));
  }

  function moveDataset(datasetId: string, direction: "up" | "down") {
    const list = datasetListRef.current;
    if (list) {
      pendingDatasetPositionsRef.current = new Map(
        [...list.querySelectorAll<HTMLElement>("[data-dataset-id]")].flatMap(
          (row) => {
            const rowDatasetId = row.dataset.datasetId;
            return rowDatasetId
              ? [[rowDatasetId, row.getBoundingClientRect().top] as const]
              : [];
          },
        ),
      );
    }
    setDatasetOrder((current) => {
      const currentIndex = current.indexOf(datasetId);
      const adjacentIndex = direction === "up" ? currentIndex + 1 : currentIndex - 1;
      if (currentIndex < 0 || adjacentIndex < 0 || adjacentIndex >= current.length) {
        return current;
      }
      const next = [...current];
      [next[currentIndex], next[adjacentIndex]] = [
        next[adjacentIndex],
        next[currentIndex],
      ];
      return next;
    });
  }

  function setDatasetSelection(candidateDataset: ViewerDataset, enabled: boolean) {
    setStyles((current) => {
      const candidateStyles = { ...current[candidateDataset.id] };
      const bulkCategoryIds = new Set(
        bulkSelectableCategories(candidateDataset).map((category) => category.id),
      );
      for (const category of datasetCategories(candidateDataset)) {
        candidateStyles[category.id] = {
          ...candidateStyles[category.id],
          enabled: enabled && bulkCategoryIds.has(category.id),
        };
      }
      return { ...current, [candidateDataset.id]: candidateStyles };
    });
  }

  function selectAllInDataset() {
    setDatasetSelection(dataset, true);
  }

  function clearDataset() {
    setStyles((current) => ({
      ...current,
      [dataset.id]: Object.fromEntries(
        Object.entries(current[dataset.id]).map(([id, style]) => [
          id,
          { ...style, enabled: false },
        ]),
      ),
    }));
  }

  function clearComposition() {
    setStyles((current) =>
      Object.fromEntries(
        Object.entries(current).map(([datasetId, datasetStyle]) => [
          datasetId,
          Object.fromEntries(
            Object.entries(datasetStyle).map(([id, style]) => [
              id,
              { ...style, enabled: false },
            ]),
          ),
        ]),
      ),
    );
  }

  function toggleAlignmentInspection() {
    if (!alignmentInspection) {
      preInspectionStylesRef.current = styles;
      clearComposition();
      setAlignmentInspection(true);
      return;
    }
    if (preInspectionStylesRef.current) setStyles(preInspectionStylesRef.current);
    preInspectionStylesRef.current = null;
    setAlignmentInspection(false);
  }

  async function copyView() {
    const url = replaceUrlWithCurrentState();
    if (!url) return;
    await navigator.clipboard.writeText(url.toString());
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <nav className="wordmark" aria-label="MapScan breadcrumb">
          <a href="https://www.todd.sh/">← todd.sh</a>
          <span>/</span>
          <a href="/mapscan">Map Scan</a>
        </nav>
        <div className="experiment-tag"><i /> Multi-map composition lab</div>
      </header>

      <aside className="catalog-panel">
        <p className="eyebrow">Recovered maps</p>
        <h1>Compose California.</h1>
        <p className="catalog-intro">
          Focus a dataset to edit it. Selected layers from every dataset stay on the map.
        </p>
        <div className="composition-summary" aria-label="Map composition summary">
          <span>{activeLayerCount} layer{activeLayerCount === 1 ? "" : "s"}</span>
          <span>{activeDatasetCount} dataset{activeDatasetCount === 1 ? "" : "s"}</span>
          <button type="button" onClick={clearComposition} disabled={activeLayerCount === 0}>Clear map</button>
        </div>
        <div className="dataset-list-heading">
          <p className="eyebrow">Datasets</p>
          <small>Top to base</small>
        </div>
        <div
          className="dataset-symbols"
          aria-label="Recovered datasets, top to bottom"
          ref={datasetListRef}
        >
          {[...orderedDatasets].reverse().map((candidate, index, ordered) => {
            const count = datasetCategories(candidate).length;
            const candidateSelectedCount = selectionCounts[candidate.id] ?? 0;
            const bulkCategories = bulkSelectableCategories(candidate);
            const allSelected =
              bulkCategories.length > 0 &&
              bulkCategories.every(
                (category) => styles[candidate.id]?.[category.id]?.enabled,
              );
            const focused = candidate.id === dataset.id;
            const contributing = candidateSelectedCount > 0;
            return (
              <div
                className={`dataset-row ${focused ? "focused" : ""} ${contributing ? "contributing" : ""}`}
                data-dataset-id={candidate.id}
                key={candidate.id}
              >
                <DatasetSelectionToggle
                  dataset={candidate}
                  checked={allSelected}
                  indeterminate={contributing && !allSelected}
                  onChange={(checked) => {
                    setDatasetSelection(candidate, checked);
                    selectDataset(candidate.id);
                  }}
                />
                <button
                  className={`dataset-symbol-button ${focused ? "active" : ""}`}
                  type="button"
                  title={candidate.menu_title}
                  onClick={() => selectDataset(candidate.id)}
                  aria-label={`Edit ${candidate.menu_title} layers, ${candidateSelectedCount} of ${count} selected`}
                >
                  <span className="dataset-symbol"><DatasetSymbol dataset={candidate} /></span>
                  <span className="dataset-button-copy">
                    <span className="dataset-short-label">{datasetShortLabel(candidate)}</span>
                    <small>{candidateSelectedCount} of {count} layers</small>
                  </span>
                  {contributing ? <span className="dataset-selection-count">{candidateSelectedCount}</span> : null}
                </button>
                <span className="dataset-order-actions">
                  <button
                    type="button"
                    aria-label={`Move ${datasetShortLabel(candidate)} dataset up`}
                    title="Move up"
                    disabled={index === 0}
                    onClick={() => moveDataset(candidate.id, "up")}
                  >↑</button>
                  <button
                    type="button"
                    aria-label={`Move ${datasetShortLabel(candidate)} dataset down`}
                    title="Move down"
                    disabled={index === ordered.length - 1}
                    onClick={() => moveDataset(candidate.id, "down")}
                  >↓</button>
                </span>
              </div>
            );
          })}
        </div>
        <div className="dataset-list-base">Base map</div>
        <div className="pipeline-note">
          <span>Image</span><b>→</b><span>Align</span><b>→</b><span>Extract</span><b>→</b><span>Compose</span>
        </div>
      </aside>

      <section className="map-stage">
        <div className="map-canvas" ref={mapContainer} />
        {!token ? (
          <div className="token-card">
            <span className="token-mark">MB</span>
            <h2>Mapbox token needed</h2>
            <p>Add <code>NEXT_PUBLIC_MAPBOX_TOKEN</code> to display the basemap and exported tiles.</p>
          </div>
        ) : null}
        <div className="map-caption">
          <span className="live-dot" />
          <div>
            <strong>{activeLayerCount > 0 ? `${activeLayerCount} selected layer${activeLayerCount === 1 ? "" : "s"}` : "No data layers selected"}</strong>
            <small>{activeDatasetCount > 0 ? `Composed from ${activeDatasetCount} independent dataset${activeDatasetCount === 1 ? "" : "s"}` : "Choose any layer from any recovered map"}</small>
          </div>
        </div>
      </section>

      <aside className="layer-panel">
        <div className="layer-header">
          <span className="dataset-symbol header-symbol"><DatasetSymbol dataset={dataset} /></span>
          <div><p className="eyebrow">Editing dataset</p><h2>{dataset.menu_title}</h2></div>
          <button
            className="share-button"
            type="button"
            onClick={copyView}
            aria-label="Copy composed map link"
          >
            {copied ? "Copied ✓" : "Copy link ↗"}
          </button>
        </div>
        <div className="dataset-meta">
          <span>California</span><span>{selectedDatasetLayerCount} / {categories.length} selected</span><span>Raster</span>
          {approved ? <span>Approved</span> : null}
        </div>
        <div className="dataset-opacity-control">
          <div className="control-heading">
            <label htmlFor={`dataset-opacity-${dataset.id}`}>Dataset opacity</label>
            <output>{Math.round((datasetOpacities[dataset.id] ?? 1) * 100)}%</output>
          </div>
          <input
            id={`dataset-opacity-${dataset.id}`}
            aria-label={`${datasetShortLabel(dataset)} dataset opacity`}
            type="range"
            min="0"
            max="100"
            value={Math.round((datasetOpacities[dataset.id] ?? 1) * 100)}
            onChange={(event) =>
              setDatasetOpacities((current) => ({
                ...current,
                [dataset.id]: Number(event.target.value) / 100,
              }))
            }
          />
        </div>
        <div className="dataset-palette-control">
          <div className="control-heading">
            <span>Color palette</span>
            <output>{activeDatasetPalette?.label ?? "Custom"}</output>
          </div>
          <div className="dataset-palette-grid" aria-label={`${datasetShortLabel(dataset)} color palette`}>
            {DATASET_PALETTES.map((palette) => (
              <button
                className="dataset-palette-button"
                type="button"
                key={palette.id}
                aria-label={`Apply ${palette.label} palette to ${datasetShortLabel(dataset)}`}
                aria-pressed={activeDatasetPalette?.id === palette.id}
                onClick={() => applyDatasetPalette(palette)}
              >
                <span
                  className="dataset-palette-swatch"
                  style={{ background: datasetPalettePreview(palette, categories) }}
                />
                <span>{palette.label}</span>
              </button>
            ))}
          </div>
          <p>Applies a coordinated range across every layer in this dataset.</p>
        </div>
        <div className="layer-actions" aria-label={`${dataset.menu_title} selection actions`}>
          <button type="button" onClick={selectAllInDataset}>Select all</button>
          <button type="button" onClick={clearDataset} disabled={selectedDatasetLayerCount === 0}>Clear dataset</button>
        </div>
        <div className="category-list" aria-label={`${dataset.menu_title} layers`}>
          {categories.map((category) => {
            const style = styles[dataset.id][category.id];
            return (
              <div className={`category-control ${style.enabled ? "" : "disabled"}`} key={category.id}>
                <button
                  className="visibility-toggle"
                  type="button"
                  aria-label={`${style.enabled ? "Hide" : "Show"} ${category.label}`}
                  aria-pressed={style.enabled}
                  onClick={() => toggleCategory(category)}
                >
                  <span style={{ background: isNativeColor(category) ? legendGradient(category) : style.color }} />
                </button>
                <div className="category-fields">
                  <div className="category-heading">
                    <label>{category.label}</label>
                    <button
                      type="button"
                      onClick={() => resetCategoryColor(category)}
                      disabled={
                        isNativeColor(category) ||
                        style.color === rgbToHex(category.display_rgb)
                      }
                    >Reset</button>
                  </div>
                  <div className="style-row">
                    {isNativeColor(category) ? (
                      <span className="continuous-ramp" aria-label={`${category.label} color ramp`} style={{ background: legendGradient(category) }} />
                    ) : (
                      <input aria-label={`${category.label} color`} type="color" value={style.color} onChange={(event) => updateCategory(category.id, { color: event.target.value })} />
                    )}
                    <input aria-label={`${category.label} opacity`} type="range" min="0" max="100" value={Math.round(style.opacity * 100)} onChange={(event) => updateCategory(category.id, { opacity: Number(event.target.value) / 100 })} />
                    <output>{Math.round(style.opacity * 100)}%</output>
                  </div>
                  {isNativeColor(category) && ((category.legend_stops?.length ?? 0) > 0 || (category.special_values?.length ?? 0) > 0) ? (
                    <div className="continuous-legend" aria-label={`${category.label} legend`}>
                      {(category.legend_stops ?? []).map((stop) => <span key={`${category.id}-${stop.value}`}><i style={{ background: rgb(stop.display_rgb) }} />{stop.label}</span>)}
                      {(category.special_values ?? []).map((special) => <span key={`${category.id}-${special.id}`}><i style={{ background: rgb(special.display_rgb) }} />{special.label}</span>)}
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
        <div className="panel-footer">
          {dataset.boundary?.raster || dataset.boundary?.geojson ? (
            <button type="button" aria-pressed={alignmentInspection} onClick={toggleAlignmentInspection}>
              {alignmentInspection ? "Return to composed data" : "Inspect coast alignment"}<span>{alignmentInspection ? "On" : "↗"}</span>
            </button>
          ) : null}
          <button type="button" onClick={() => setSourceOpen(true)}>View source image <span>↗</span></button>
          <p>{exactBoundary ? "Approved boundary; this dataset can be combined with any other layer." : approved ? "Aligned from a static map image and separately audited." : "Automatically aligned and extracted; awaiting visual review."}</p>
        </div>
      </aside>

      {sourceOpen && dataset.source_image ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSourceOpen(false)}>
          <section className="source-modal" role="dialog" aria-modal="true" aria-label="Original map image" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><p className="eyebrow">Original evidence</p><h2>{dataset.title}</h2></div><button type="button" onClick={() => setSourceOpen(false)} aria-label="Close source image">×</button></header>
            <div className="source-image-wrap"><Image src={`${datasetAssetBase(dataset)}${dataset.source_image}`} alt={`Original ${dataset.title} map`} fill sizes="80vw" loading="eager" unoptimized /></div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
