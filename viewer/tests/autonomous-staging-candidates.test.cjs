const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const stagingPage = fs.readFileSync(
  path.join(viewerRoot, "app", "staging", "page.tsx"),
  "utf8",
);
const publicPage = fs.readFileSync(
  path.join(viewerRoot, "app", "map", "page.tsx"),
  "utf8",
);

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

const candidates = [
  {
    directory: "population-native-fidelity-repair-v2",
    publicDirectory: "population-density",
    id: "california-population-density-autonomous-v2-native-fidelity",
    categoryCount: 12,
    layerCount: 1,
    maximumZoom: 9,
  },
  {
    directory: "fire-autonomous-v2-clipped",
    publicDirectory: "fire-hazard-responsibility",
    id: "california-fire-hazard-and-responsibility-areas-autonomous-v2-clipped",
    categoryCount: 4,
    layerCount: 2,
    maximumZoom: 9,
  },
  {
    directory: "landslide-autonomous-v2-clipped",
    publicDirectory: "severe-weather-impacts",
    id: "california-severe-weather-impacts-autonomous-v2-clipped",
    categoryCount: 6,
    layerCount: 4,
    maximumZoom: 9,
  },
  {
    directory: "geologic-highres-autonomous-v1",
    publicDirectory: "geologic-units-highres",
    id: "california-geologic-units-autonomous-highres-v1",
    categoryCount: 53,
    layerCount: 1,
    maximumZoom: 11,
  },
];

test("keeps audit candidates on staging and exposes only approved public manifests", () => {
  for (const candidate of candidates) {
    assert.match(stagingPage, new RegExp(candidate.directory));
    assert.doesNotMatch(publicPage, new RegExp(candidate.directory));
    assert.match(publicPage, new RegExp(candidate.publicDirectory));
  }
});

test("keeps autonomous staging manifests selectable and explicitly unapproved", () => {
  for (const candidate of candidates) {
    const datasetDirectory = path.join(
      viewerRoot,
      "public",
      "data",
      "staging",
      candidate.directory,
    );
    const manifest = JSON.parse(
      fs.readFileSync(path.join(datasetDirectory, "dataset.json"), "utf8"),
    );

    assert.equal(manifest.id, candidate.id);
    assert.equal(manifest.status, "needs_visual_review");
    assert.equal(manifest.approval.status, "not_approved");
    assert.equal(manifest.minimum_zoom, 4);
    assert.equal(manifest.maximum_native_zoom, candidate.maximumZoom);
    assert.equal(manifest.layers.length, candidate.layerCount);
    assert.equal(
      manifest.layers.flatMap((layer) => layer.categories).length,
      candidate.categoryCount,
    );
    assert.ok(manifest.source_image);
    assert.ok(fs.existsSync(path.join(datasetDirectory, manifest.source_image)));
    assert.equal(manifest.boundary.kind, "pinned_mapbox_state_coast_diagnostic");
    assert.equal(manifest.boundary.diagnostic_only, true);
    const boundaryPath = path.join(datasetDirectory, manifest.boundary.raster);
    assert.ok(fs.existsSync(boundaryPath));
    assert.equal(sha256(boundaryPath), manifest.boundary.raster_sha256);
    const provenancePath = path.join(datasetDirectory, manifest.provenance.manifest);
    assert.equal(sha256(provenancePath), manifest.provenance.sha256);
    const provenance = JSON.parse(fs.readFileSync(provenancePath, "utf8"));
    assert.equal(provenance.boundary.raster, manifest.boundary.raster);
    assert.equal(provenance.boundary.raster_sha256, manifest.boundary.raster_sha256);
    assert.equal(provenance.boundary.authority, manifest.boundary.authority);
    assert.equal(provenance.publication_coverage.layers.length, candidate.layerCount);
    assert.equal(
      provenance.publication_coverage.layers.every(
        (layer) => layer.colored_pixel_count_outside_state === 0,
      ),
      true,
    );

    for (const layer of manifest.layers) {
      assert.equal(layer.kind, "categorical");
      assert.equal(layer.indexed_raster.encoding, "png_luma_alpha_uint8_class_id_v1");
      assert.equal(layer.indexed_raster.nodata_class_id, 0);
      assert.ok(layer.indexed_raster.tile_file_count > 0);
      assert.match(layer.indexed_raster.tile_template, /\{z\}\/\{x\}\/\{y\}\.png/);
      assert.equal(
        new Set(layer.categories.map((category) => category.class_id)).size,
        layer.categories.length,
      );
    }
  }
});

test("preserves the independently selectable overlapping-map families", () => {
  const fire = JSON.parse(
    fs.readFileSync(
      path.join(
        viewerRoot,
        "public",
        "data",
        "staging",
        "fire-autonomous-v2-clipped",
        "dataset.json",
      ),
      "utf8",
    ),
  );
  const landslide = JSON.parse(
    fs.readFileSync(
      path.join(
        viewerRoot,
        "public",
        "data",
        "staging",
        "landslide-autonomous-v2-clipped",
        "dataset.json",
      ),
      "utf8",
    ),
  );
  assert.equal(fire.layers.length, 2);
  assert.deepEqual(
    landslide.layers.map((layer) => layer.id),
    [
      "maximum-daily-precipitation",
      "landslide-susceptibility",
      "maximum-wind-speed",
      "predicted-flooding",
    ],
  );
});

test("retires the Rivers image result from both map pickers without deleting its evidence", () => {
  const catalog = JSON.parse(
    fs.readFileSync(
      path.join(viewerRoot, "public", "data", "catalog.json"),
      "utf8",
    ),
  );
  const evidencePath = path.join(
    viewerRoot,
    "public",
    "data",
    "staging",
    "rivers-autonomous-v1",
    "dataset.json",
  );

  assert.doesNotMatch(publicPage, /named-hydrography/);
  assert.doesNotMatch(stagingPage, /rivers-autonomous-v1/);
  assert.equal(
    catalog.datasets.some(
      (dataset) => dataset.id === "california-rivers-lakes-and-dry-hydrography",
    ),
    false,
  );
  assert.equal(fs.existsSync(evidencePath), true);
});
