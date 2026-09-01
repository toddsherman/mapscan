const assert = require("node:assert/strict");
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
  "agricultural-land-use-v2-highres",
);
const dataset = JSON.parse(fs.readFileSync(path.join(packageRoot, "dataset.json"), "utf8"));
const activation = JSON.parse(
  fs.readFileSync(path.join(packageRoot, "public-activation-decision.json"), "utf8"),
);
const autonomousStagingRoot = path.join(
  viewerRoot,
  "public",
  "data",
  "staging",
  "farms-v2-autonomous-v1",
);
const autonomousStaging = JSON.parse(
  fs.readFileSync(path.join(autonomousStagingRoot, "dataset.json"), "utf8"),
);
const autonomousProvenance = JSON.parse(
  fs.readFileSync(
    path.join(autonomousStagingRoot, "autonomous-preview-provenance.json"),
    "utf8",
  ),
);

test("keeps Farms v2 in the public catalog", () => {
  assert.match(publicPage, /\/mapscan\/data\/datasets\/agricultural-land-use-v2-highres\//);
  assert.equal(
    catalog.datasets.some((dataset) => dataset.id === "california-agricultural-land-use"),
    true,
  );
});

test("public Farms v2 retains the exact reviewed package contract", () => {
  assert.equal(dataset.id, "california-agricultural-land-use");
  assert.equal(
    dataset.layers.flatMap((layer) => layer.categories).length,
    45,
  );
  assert.equal(activation.status, "approved_public_activation");
  assert.equal(activation.staging_assets_copied_byte_identically, false);
  assert.equal(activation.staging_assets_shared_byte_identically, true);
  assert.equal(activation.immutable_asset_count, 5_628);
  assert.equal(activation.coverage[0].coverage_contract, "sparse_visible_evidence");
  assert.equal(activation.coverage[0].colored_pixel_count_outside_state, 0);
  assert.equal(activation.coverage[0].nodata_pixel_count_inside_state, 46_725_663);
  assert.equal(dataset.maximum_native_zoom, 11);
  assert.equal(dataset.asset_base, "/mapscan/data/staging/farms-v2-autonomous-v1/");
});

test("temporary Farms staging package and entry remain retired", () => {
  assert.equal(
    fs.existsSync(
      path.join(viewerRoot, "public", "data", "staging", "farms-v2-v1"),
    ),
    false,
  );
  const stagingPath = path.join(viewerRoot, "app", "staging", "page.tsx");
  const stagingPage = fs.existsSync(stagingPath)
    ? fs.readFileSync(stagingPath, "utf8")
    : "";
  assert.doesNotMatch(stagingPage, /farms-v2-v1/);
});

test("retains the unapproved Farms audit package behind approved public metadata", () => {
  const stagingPage = fs.readFileSync(
    path.join(viewerRoot, "app", "staging", "page.tsx"),
    "utf8",
  );
  assert.match(stagingPage, /farms-v2-autonomous-v1/);
  assert.match(stagingPage, /Agricultural land use · autonomous candidate/);
  assert.doesNotMatch(publicPage, /farms-v2-autonomous-v1/);
  assert.match(publicPage, /agricultural-land-use-v2-highres/);
  assert.equal(autonomousStaging.status, "needs_visual_review");
  assert.deepEqual(autonomousStaging.approval, { status: "not_approved" });
  assert.equal(
    autonomousStaging.boundary.kind,
    "pinned_mapbox_state_coast_diagnostic",
  );
  assert.equal(
    autonomousStaging.boundary.authority,
    "accepted_alignment_mapbox_reference",
  );
  assert.equal(autonomousStaging.boundary.diagnostic_only, true);
  assert.equal(
    autonomousStaging.boundary.raster_sha256,
    "267a83da36dc75f44bd6daa1f87ce5fffa26bb9cde68c3fa3524947df4d2e115",
  );
  assert.equal(
    autonomousStaging.layers.flatMap((layer) => layer.categories).length,
    45,
  );
  assert.equal(autonomousProvenance.publication_approved, false);
  assert.equal(
    autonomousProvenance.accepted_alignment.automatic_iteration_count,
    11,
  );
  assert.equal(
    autonomousProvenance.accepted_extraction.automatic_iteration_count,
    4,
  );
  assert.equal(autonomousProvenance.nodata.partial_source_extent, true);
  assert.equal(autonomousStaging.id, "california-agricultural-land-use-autonomous-v2-highres");
  assert.equal(autonomousStaging.maximum_native_zoom, 11);
  assert.equal(autonomousStaging.categorical_tile_encoding, "indexed_class_id");
  assert.equal(
    autonomousStaging.layers[0].indexed_raster.tile_file_count,
    5_625,
  );
  assert.deepEqual(
    autonomousStaging.layers[0].indexed_raster.class_id_range,
    [1, 45],
  );
  assert.equal(
    autonomousProvenance.nodata.missing_source_extent_pixel_count,
    25_479_454,
  );
  assert.deepEqual(autonomousProvenance.target_grid, {
    crs: "EPSG:3857",
    bounds: [
      -13_857_273.186886752,
      3_833_019.2460767706,
      -12_705_028.292139662,
      5_162_403.053672604,
    ],
    width: 10_192,
    height: 11_758,
  });
  assert.deepEqual(autonomousProvenance.manual_inputs, {
    manual_arrows: false,
    manual_stamps: false,
    human_approval: false,
  });
  assert.equal(
    autonomousProvenance.publication_coverage.layers[0]
      .colored_pixel_count_outside_state,
    0,
  );
});
