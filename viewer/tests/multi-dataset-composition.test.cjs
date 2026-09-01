const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const mapSource = fs.readFileSync(
  path.join(viewerRoot, "components", "mapscan-map.tsx"),
  "utf8",
);
const publicPage = fs.readFileSync(
  path.join(viewerRoot, "app", "map", "page.tsx"),
  "utf8",
);

test("keeps one persistent composition state across every dataset", () => {
  assert.match(mapSource, /type CompositionStyles = Record<string, Record<string, CategoryStyle>>/);
  assert.match(mapSource, /for \(const candidateDataset of datasets\)/);
  assert.match(mapSource, /categoryLayerId\(candidateDataset\.id, category\)/);
  assert.match(mapSource, /referenceBoundaryId\(candidateDataset\.id\)/);
  assert.doesNotMatch(mapSource, /key=\{datasetRenderKey\}/);
  assert.doesNotMatch(mapSource, /searchParams\.delete\("layers"\)/);
});

test("defaults a clean map to Forest and Farms while preserving explicit dataset links", () => {
  assert.match(
    mapSource,
    /const DEFAULT_ACTIVE_DATASET_IDS = new Set\(\[\s*"california-forest-cover",\s*"california-agricultural-land-use",\s*\]\)/,
  );
  assert.match(mapSource, /const DEFAULT_FOCUSED_DATASET_ID = "california-forest-cover"/);
  assert.match(mapSource, /enabledDatasetIds\.has\(dataset\.id\)/);
  assert.match(mapSource, /searchParams\.get\("dataset"\)/);
  assert.match(mapSource, /explicitDatasetId\s*\? new Set\(\[focusedDatasetId\]\)\s*: DEFAULT_ACTIVE_DATASET_IDS/);
  assert.match(mapSource, /return datasets\.some\(\(dataset\) => dataset\.id === requested\) \? requested! : defaultDatasetId/);
});

test("can publish approval metadata without duplicating immutable raster assets", () => {
  assert.match(mapSource, /dataset\.asset_base \?\? dataset\.public_path/);
  assert.match(mapSource, /const assetBase = datasetAssetBase\(candidateDataset\)/);
  assert.match(mapSource, /datasetAssetBase\(dataset\).*dataset\.source_image/);
});

test("shares a compact versioned state across multiple datasets", () => {
  assert.match(mapSource, /type SharedCompositionStateV3/);
  assert.match(mapSource, /compressToEncodedURIComponent\(JSON\.stringify\(state\)\)/);
  assert.match(mapSource, /decompressFromEncodedURIComponent\(encoded\)/);
  assert.match(mapSource, /candidateDataset\.id,\s*category\.id,\s*style\.enabled \? 1 : 0/s);
  assert.match(mapSource, /const separator = qualifiedId\.indexOf\("~"\)/);
  assert.match(mapSource, /Copy composed map link/);
  assert.match(mapSource, /url\.searchParams\.set\("config", "3"\)/);
  assert.match(mapSource, /url\.searchParams\.set\(\s*"state"/s);
  assert.match(mapSource, /window\.history\.replaceState\(null, "", url\)/);
  assert.match(mapSource, /Copied ✓/);
});

test("round-trips the full editable composition while accepting older links", () => {
  assert.match(mapSource, /Array\.isArray\(shared\?\.l\)/);
  assert.match(mapSource, /enabled: enabled === 1/);
  assert.match(mapSource, /fields\.length === 4 && fields\[0\]\.includes\("~"\)/);
  assert.match(mapSource, /enabled: enabled === "1"/);
  assert.match(mapSource, /Backward compatibility for the first multi-dataset format/);
  assert.match(mapSource, /Backward compatibility for the original single-dataset share format/);
  assert.match(mapSource, /const hasStyleOverride/);
  assert.match(mapSource, /percent === 100/);
  assert.match(mapSource, /map\.getBearing\(\)\.toFixed\(2\)/);
  assert.match(mapSource, /map\.getPitch\(\)\.toFixed\(2\)/);
  assert.match(mapSource, /bearing: initialView\?\.bearing \?\? 0/);
  assert.match(mapSource, /pitch: initialView\?\.pitch \?\? 0/);
  assert.match(mapSource, /map\.on\("moveend", syncMapViewToUrl\)/);
  assert.match(mapSource, /replaceUrlWithCurrentState\(\)/);
  assert.match(mapSource, /\[datasetOpacities, datasetOrder, mapReady, selectedId, styles\]/);
});

test("keeps transient review UI out of shareable map state", () => {
  assert.doesNotMatch(
    mapSource.match(/const state: SharedCompositionStateV3 = \{([\s\S]*?)\n  \};/)?.[1] ?? "",
    /sourceOpen|alignmentInspection|copied/,
  );
});

test("exposes the requested forest elevation and rainfall sources together", () => {
  assert.match(publicPage, /datasets\/forest-cover/);
  assert.match(publicPage, /datasets\/elevation-bands/);
  assert.match(publicPage, /datasets\/rainfall-1981-2010/);
});

test("offers dataset and whole-map selection actions", () => {
  assert.match(mapSource, /onClick=\{selectAllInDataset\}/);
  assert.match(mapSource, /onClick=\{clearDataset\}/);
  assert.match(mapSource, /onClick=\{clearComposition\}/);
  assert.match(mapSource, /selectedCount\(candidate, styles\)/);
});

test("focuses a dataset when its bulk-selection checkbox changes", () => {
  assert.match(
    mapSource,
    /onChange=\{\(checked\) => \{\s*setDatasetSelection\(candidate, checked\);\s*selectDataset\(candidate\.id\);\s*\}\}/,
  );
});

test("uses symbols for datasets while retaining text labels for layers", () => {
  assert.match(mapSource, /function DatasetSymbol\(/);
  assert.match(mapSource, /className={`dataset-symbol-button/);
  assert.match(mapSource, /title={candidate\.menu_title}/);
  assert.match(mapSource, /className="dataset-short-label">{datasetShortLabel\(candidate\)}/);
  assert.match(mapSource, /if \(id\.includes\("agricultural"\)\) return "Farms"/);
  assert.match(mapSource, /className="category-list"/);
  assert.match(mapSource, /<label>{category\.label}<\/label>/);
  assert.doesNotMatch(mapSource, /function LayerSymbol\(/);
});

test("keeps opacity controls scoped to datasets and layers", () => {
  assert.doesNotMatch(mapSource, /collectionOpacity/);
  assert.doesNotMatch(mapSource, /Collection opacity/);
  assert.doesNotMatch(mapSource, /searchParams\.set\("opacity"/);
  assert.match(mapSource, /url\.search = ""/);
  assert.match(mapSource, /style\.opacity \* datasetOpacity/);
});

test("applies and shares an independent opacity multiplier for each dataset", () => {
  assert.match(mapSource, /function initialDatasetOpacities\(datasets/);
  assert.match(mapSource, /const \[datasetOpacities, setDatasetOpacities\] = useState/);
  assert.match(mapSource, /style\.opacity \* datasetOpacity/);
  assert.match(mapSource, /Dataset opacity/);
  assert.match(mapSource, /datasetShortLabel\(dataset\)} dataset opacity/);
  assert.match(mapSource, /"datasetOpacity"/);
});

test("orders and bulk-toggles datasets from one wide picker", () => {
  assert.match(mapSource, /indexedRasterCategoryLayerId\(key, category\.id\)/);
  assert.match(mapSource, /function moveMapDatasets\(/);
  assert.match(mapSource, /for \(const datasetId of order\)/);
  assert.match(mapSource, /for \(const datasetLayer of dataset\.layers\)/);
  assert.match(mapSource, /map\.moveLayer\(id, beforeId\)/);
  assert.match(mapSource, /function DatasetSelectionToggle\(/);
  assert.match(mapSource, /inputRef\.current\.indeterminate = indeterminate/);
  assert.match(mapSource, /type="checkbox"/);
  assert.match(mapSource, /function setDatasetSelection\(/);
  assert.match(mapSource, /function moveDataset\(/);
  assert.match(mapSource, /useLayoutEffect\(\(\) => \{/);
  assert.match(mapSource, /pendingDatasetPositionsRef/);
  assert.match(mapSource, /row\.animate\(/);
  assert.match(mapSource, /prefers-reduced-motion: reduce/);
  assert.match(mapSource, /aria-label="Recovered datasets, top to bottom"/);
  assert.match(mapSource, /Move \$\{datasetShortLabel\(candidate\)\} dataset up/);
  assert.match(mapSource, /Move \$\{datasetShortLabel\(candidate\)\} dataset down/);
  assert.match(mapSource, /o: datasetOrder/);
  assert.doesNotMatch(mapSource, /Dataset stack/);
  assert.doesNotMatch(mapSource, /className="dataset-stack"/);
  assert.doesNotMatch(mapSource, /draggable/);
  assert.doesNotMatch(mapSource, /function reorderDataset\(/);
  assert.doesNotMatch(mapSource, /function moveActiveDataset\(/);
  assert.doesNotMatch(mapSource, /Global layer stack/);
  assert.doesNotMatch(mapSource, /moveActiveLayer/);
});

test("supports per-layer color editing and a default-color reset", () => {
  assert.match(mapSource, /function resetCategoryColor\(category: CategoryManifest\)/);
  assert.match(mapSource, /rgbToHex\(category\.display_rgb\)/);
  assert.match(mapSource, />Reset<\/button>/);
  assert.match(mapSource, /type="color"/);
});

test("keeps the California-wide zoom floor above every raster cutoff", () => {
  assert.match(mapSource, /const CALIFORNIA_MIN_ZOOM = 5/);
  assert.match(mapSource, /Math\.max\(\s*CALIFORNIA_MIN_ZOOM,\s*\.\.\.datasets\.map/);
  assert.match(mapSource, /function viewFromUrl\(minimumZoom: number\)/);
  assert.match(mapSource, /zoom < minimumZoom/);
  assert.match(mapSource, /minZoom: minimumMapZoom/);
  assert.match(mapSource, /minzoom: candidateDataset\.minimum_zoom/);
  assert.match(mapSource, /function reloadVisibleRasterSources\(/);
  assert.match(mapSource, /map\.getLayoutProperty\(layerId, "visibility"\) !== "none"/);
  assert.match(mapSource, /source\.reload\(\)/);
  assert.match(mapSource, /map\.getZoom\(\) > minimumMapZoom \+ 0\.75/);
  assert.match(mapSource, /map\.on\("zoomend", refreshLowZoomThematicSources\)/);
  assert.match(mapSource, /map\.off\("zoomend", refreshLowZoomThematicSources\)/);
});
