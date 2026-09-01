const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const stagingPage = fs.readFileSync(
  path.join(viewerRoot, "app", "staging", "page.tsx"),
  "utf8",
);
const packageRoot = path.join(
  viewerRoot,
  "public",
  "data",
  "datasets",
  "elevation-bands-autonomous-v2",
);
const dataset = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "dataset.json"), "utf8"),
);
const provenance = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "provenance.json"), "utf8"),
);

test("serves the newest approved elevation package on staging", () => {
  assert.match(stagingPage, /data\/datasets\/elevation-bands-autonomous-v2/);
  assert.equal(dataset.id, "california-topography-elevation");
  assert.equal(dataset.status, "approved_publication");
  assert.equal(dataset.approval.status, "approved");
  assert.equal(provenance.kind, "autonomous_indexed_publication_provenance");
  assert.equal(dataset.layers[0].kind, "categorical");
  assert.equal(dataset.layers[0].categories.length, 8);
  assert.equal(
    dataset.layers[0].indexed_raster.encoding,
    "png_luma_alpha_uint8_class_id_v1",
  );
});

test("retains numeric authority while publishing selectable intervals", () => {
  assert.deepEqual(
    provenance.accepted_inputs.map((input) => input.kind),
    [
      "accepted_quantized_elevation_band_raster",
      "accepted_depression_mask",
      "accepted_continuous_elevation_raster",
    ],
  );
  assert.equal(dataset.boundary.diagnostic_only, true);
  assert.equal(
    provenance.publication_coverage.layers[0].colored_pixel_count_outside_state,
    0,
  );
  assert.equal(dataset.layers[0].indexed_raster.tile_file_count, 400);
});
