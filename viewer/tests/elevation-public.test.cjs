const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const publicPage = fs.readFileSync(
  path.join(viewerRoot, "app", "map", "page.tsx"),
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
  "elevation-bands-autonomous-v2",
);
const assetRoot = path.join(
  viewerRoot,
  "public",
  "data",
  "staging",
  "elevation-autonomous-v2",
);
const datasetPath = path.join(packageRoot, "dataset.json");
const provenancePath = path.join(packageRoot, "provenance.json");
const activationPath = path.join(packageRoot, "public-activation-decision.json");
const dataset = JSON.parse(fs.readFileSync(datasetPath, "utf8"));
const provenance = JSON.parse(fs.readFileSync(provenancePath, "utf8"));
const activation = JSON.parse(fs.readFileSync(activationPath, "utf8"));

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

test("publishes the newest accepted elevation bands as the active dataset", () => {
  assert.match(
    publicPage,
    /\/mapscan\/data\/datasets\/elevation-bands-autonomous-v2\//,
  );
  assert.equal(catalog.datasets.length, 11);
  assert.equal(catalog.datasets.at(-1).id, "california-topography-elevation");
  assert.equal(catalog.datasets.at(-1).category_count, 8);
  assert.equal(
    catalog.datasets.at(-1).manifest,
    "datasets/elevation-bands-autonomous-v2/dataset.json",
  );
});

test("binds every selectable band to the accepted numeric extraction", () => {
  const categories = dataset.layers.flatMap((layer) => layer.categories);
  assert.equal(dataset.status, "approved_publication");
  assert.equal(dataset.approval.status, "approved");
  assert.equal(dataset.categorical_tile_encoding, "indexed_class_id");
  assert.equal(dataset.asset_base, "/mapscan/data/staging/elevation-autonomous-v2/");
  assert.equal(categories.length, 8);
  assert.deepEqual(
    categories.map((category) => category.class_id),
    [1, 2, 3, 4, 5, 6, 7, 8],
  );
  assert.deepEqual(
    categories.map((category) => category.label),
    [
      "Depression (depth not specified)",
      "0–<100 m",
      "100–<250 m",
      "250–<500 m",
      "500–<1,000 m",
      "1,000–<2,000 m",
      "2,000–<4,000 m",
      "4,000–<5,000 m",
    ],
  );
  assert.equal(
    provenance.accepted_extraction.sha256,
    "d7c78616e656f5749e6daa291c80e65d603f58b205401febb9abd03ea9770fd8",
  );
  assert.equal(provenance.accepted_inputs.length, 3);
  assert.equal(
    provenance.class_raster.sha256,
    sha256(path.join(assetRoot, "accepted-class-id.png")),
  );
  assert.equal(dataset.layers[0].indexed_raster.tile_file_count, 400);
});

test("keeps the Mapbox boundary and publication coverage contracts intact", () => {
  assert.equal(dataset.boundary.kind, "pinned_mapbox_state_coast_diagnostic");
  assert.equal(dataset.boundary.diagnostic_only, true);
  assert.equal(
    provenance.publication_coverage.layers[0].colored_pixel_count_outside_state,
    0,
  );
  assert.equal(
    provenance.publication_coverage.layers[0].classified_pixel_count_inside_state,
    5_494_362,
  );
  assert.equal(activation.status, "approved_public_activation");
  assert.equal(activation.staging_assets_shared_byte_identically, true);
  assert.equal(activation.immutable_asset_count, 404);
  assert.equal(
    activation.author_statement,
    "Check that each map uses the newer version and fix it.",
  );
  assert.equal(dataset.provenance.sha256, sha256(provenancePath));
});

test("renders every accepted elevation interval as an independent control", () => {
  const mapSource = fs.readFileSync(
    path.join(viewerRoot, "components", "mapscan-map.tsx"),
    "utf8",
  );
  assert.match(mapSource, /category\.default_enabled \?\? true/);
  assert.match(mapSource, /onClick=\{\(\) => toggleCategory\(category\)\}/);
  assert.match(mapSource, /\[dataset\.id\]: datasetStyle/);
  assert.match(mapSource, /selected layers from every dataset stay on the map/i);
});
