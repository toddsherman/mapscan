const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const publicPage = fs.readFileSync(
  path.join(viewerRoot, "app", "map", "page.tsx"),
  "utf8",
);
const styles = fs.readFileSync(path.join(viewerRoot, "app", "globals.css"), "utf8");
const mapSource = fs.readFileSync(
  path.join(viewerRoot, "components", "mapscan-map.tsx"),
  "utf8",
);
const catalog = JSON.parse(
  fs.readFileSync(path.join(viewerRoot, "public", "data", "catalog.json"), "utf8"),
);
const packageRoot = path.join(
  viewerRoot,
  "public",
  "data",
  "datasets",
  "rainfall",
);
const dataset = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "dataset.json"), "utf8"),
);
const activation = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "public-activation-decision.json"), "utf8"),
);
const rainfall1981Root = path.join(
  viewerRoot,
  "public",
  "data",
  "datasets",
  "rainfall-1981-2010-autonomous-v2",
);
const rainfall1981 = JSON.parse(
  fs.readFileSync(path.join(rainfall1981Root, "dataset.json"), "utf8"),
);
const rainfall1981Activation = JSON.parse(
  fs.readFileSync(
    path.join(rainfall1981Root, "public-activation-decision.json"),
    "utf8",
  ),
);
const rainfall1981Provenance = JSON.parse(
  fs.readFileSync(path.join(rainfall1981Root, "provenance.json"), "utf8"),
);

test("retires historical Rainfall from the picker and keeps the PNG dataset", () => {
  assert.doesNotMatch(publicPage, /\/mapscan\/data\/datasets\/rainfall\//);
  assert.match(
    publicPage,
    /\/mapscan\/data\/datasets\/rainfall-1981-2010-autonomous-v2\//,
  );
  assert.equal(catalog.datasets.length, 11);
  assert.equal(
    catalog.datasets.find(
      (candidate) =>
        candidate.id === "california-average-annual-precipitation-1981-2010",
    ).id,
    "california-average-annual-precipitation-1981-2010",
  );
  assert.equal(
    catalog.datasets.some(
      (candidate) =>
        candidate.id === "california-annual-average-precipitation-1900-1960",
    ),
    false,
  );
});

test("retains all legend entries and the reviewed water-aware contract", () => {
  const categories = dataset.layers.flatMap((layer) => layer.categories);
  assert.equal(categories.length, 35);
  assert.deepEqual(
    categories.filter((category) => category.pixel_count === 0).map(
      (category) => category.label,
    ),
    ["5.5 in", "6.5 in", "17.0 in"],
  );
  assert.equal(dataset.boundary.continuous_border_component_count, 5);
  assert.equal(dataset.boundary.publication_interior_component_count, 15);
  assert.equal(dataset.boundary.publication_interior_exclusion_pixel_count, 3_756);
  assert.equal(dataset.boundary.colored_pixel_count_outside_boundary, 0);
  assert.equal(dataset.boundary.unclassified_pixel_count_inside_boundary, 0);
  assert.equal(activation.status, "approved_public_activation");
  assert.equal(activation.staging_assets_copied_byte_identically, true);
  assert.equal(activation.package_file_count, 14_040);
});

test("publishes the newest accepted 1981-2010 extraction", () => {
  const categories = rainfall1981.layers.flatMap((layer) => layer.categories);
  assert.equal(categories.length, 10);
  assert.equal(rainfall1981.approval.status, "approved");
  assert.equal(rainfall1981.categorical_tile_encoding, "indexed_class_id");
  assert.equal(rainfall1981.boundary.kind, "pinned_mapbox_state_coast_diagnostic");
  assert.equal(rainfall1981.boundary.diagnostic_only, true);
  assert.equal(
    rainfall1981Provenance.accepted_extraction.sha256,
    "005fa2040899fae5a0c8c677e3f8959418a3b498a7056a664e35f3e3a3f6fa2c",
  );
  assert.equal(
    rainfall1981Provenance.publication_clip.accepted_colored_pixel_count_outside_state,
    3_988,
  );
  assert.equal(
    rainfall1981Provenance.publication_coverage.layers[0]
      .colored_pixel_count_outside_state,
    0,
  );
  assert.equal(rainfall1981Activation.status, "approved_public_activation");
  assert.equal(rainfall1981Activation.staging_assets_shared_byte_identically, true);
  assert.equal(rainfall1981Activation.immutable_asset_count, 404);
});

test("uses the newest Rainfall publication on both live and staging pages", () => {
  const stagingPage = fs.readFileSync(
    path.join(viewerRoot, "app", "staging", "page.tsx"),
    "utf8",
  );
  assert.match(stagingPage, /rainfall-1981-2010-autonomous-v2/);
  assert.match(stagingPage, /Annual precipitation 1981–2010 · newest publication/);
  assert.doesNotMatch(stagingPage, /rainfall-1981-2010-mapbox-water-v8-approved/);
  assert.equal(
    rainfall1981.asset_base,
    "/mapscan/data/staging/rainfall-1981-2010-autonomous-v2/",
  );
  assert.equal(
    rainfall1981.layers[0].indexed_raster.tile_file_count,
    400,
  );
  assert.match(styles, /\.app-shell\s*\{[^}]*height:\s*100vh;/s);
  assert.match(styles, /\.catalog-panel, \.layer-panel\s*\{[^}]*overflow:\s*auto;/s);
});

test("updates only changed Mapbox style properties for long legends", () => {
  assert.match(mapSource, /const appliedStylesRef = useRef\(styles\)/);
  assert.match(mapSource, /previous\.enabled !== style\.enabled/);
  assert.match(mapSource, /previous\.opacity !== style\.opacity/);
  assert.match(mapSource, /previous\.color !== style\.color/);
});

test("supports a shareable composed camera without changing raster bounds", () => {
  assert.match(mapSource, /searchParams\.get\("view"\)/);
  assert.match(mapSource, /center: initialView\?\.center \?\? initialDataset\.center/);
  assert.match(mapSource, /if \(!initialView\) map\.fitBounds\(initialDataset\.bounds/);
  assert.match(mapSource, /m:\s*\[/);
  assert.match(mapSource, /validView\(shared\?\.m, minimumZoom\)/);
  assert.match(mapSource, /map\.on\("moveend", syncMapViewToUrl\)/);
});
