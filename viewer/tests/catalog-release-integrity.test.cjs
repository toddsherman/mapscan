const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const dataRoot = path.join(viewerRoot, "public", "data");
const catalogPath = path.join(dataRoot, "catalog.json");

// This is the one pre-artifact_id alias in the existing catalog. New aliases
// must declare artifact_id in catalog.json instead of extending this map.
const legacyArtifactIds = new Map([
  [
    "california-plant-hardiness-zones",
    "california-plant-hardiness-zones-neighbor-v1",
  ],
]);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function assertInside(root, candidate, label) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  assert.ok(
    relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative),
    `${label} escapes ${root}: ${candidate}`,
  );
}

function resolveRelative(root, value, label) {
  assert.equal(typeof value, "string", `${label} must be a string`);
  assert.ok(value.length > 0, `${label} must not be empty`);
  assert.ok(!path.isAbsolute(value), `${label} must be relative`);
  assert.ok(!value.includes("\\"), `${label} must use URL-style separators`);
  assert.ok(!value.includes("?") && !value.includes("#"), `${label} must not contain URL state`);
  const segments = value.split("/");
  assert.ok(
    segments.every((segment) => segment && segment !== "." && segment !== ".."),
    `${label} contains an unsafe path segment`,
  );
  const resolved = path.resolve(root, ...segments);
  assertInside(root, resolved, label);
  return resolved;
}

function resolveAssetBase(value, label) {
  assert.equal(typeof value, "string", `${label} must be a string`);
  assert.match(
    value,
    /^\/mapscan\/data\/(?:[A-Za-z0-9._~-]+\/)+$/,
    `${label} must be a safe root-relative /mapscan/data/ URL ending in /`,
  );
  const relative = value.slice("/mapscan/data/".length, -1);
  const resolved = resolveRelative(dataRoot, relative, label);
  assert.ok(fs.statSync(resolved).isDirectory(), `${label} does not resolve to a directory`);
  return resolved;
}

function aggregateHash(paths, root) {
  const digest = crypto.createHash("sha256");
  const sorted = [...paths].sort((left, right) => left.localeCompare(right));
  for (const filePath of sorted) {
    const relative = path.relative(root, filePath).split(path.sep).join("/");
    digest.update(relative);
    digest.update("\0");
    digest.update(sha256(filePath));
    digest.update("\n");
  }
  return digest.digest("hex");
}

function walkPngs(root) {
  const output = [];
  const pending = [root];
  while (pending.length) {
    const current = pending.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const candidate = path.join(current, entry.name);
      if (entry.isDirectory()) pending.push(candidate);
      else if (entry.isFile() && entry.name.endsWith(".png")) output.push(candidate);
    }
  }
  return output;
}

function verifyIndexedRaster(indexed, assetRoot, context) {
  const tilejsonPath = resolveRelative(assetRoot, indexed.tilejson, `${context} tilejson`);
  assert.ok(fs.statSync(tilejsonPath).isFile(), `${context} TileJSON is missing`);
  const tilejson = readJson(tilejsonPath);
  assert.equal(tilejson.tilejson, "3.0.0", `${context} TileJSON version is stale`);
  assert.ok(
    Array.isArray(tilejson.tiles) && tilejson.tiles.includes("{z}/{x}/{y}.png"),
    `${context} TileJSON does not describe the indexed XYZ pyramid`,
  );

  const tileRoot = path.dirname(tilejsonPath);
  const tiles = walkPngs(tileRoot);
  assert.equal(tiles.length, indexed.tile_file_count, `${context} tile count differs`);

  const aggregate = aggregateHash([...tiles, tilejsonPath], assetRoot);
  assert.equal(aggregate, indexed.tile_set_sha256, `${context} aggregate hash differs`);

  if (indexed.tile_image_byte_count !== undefined) {
    const tileBytes = tiles.reduce((total, filePath) => total + fs.statSync(filePath).size, 0);
    assert.equal(tileBytes, indexed.tile_image_byte_count, `${context} tile byte count differs`);
  }

  assert.equal(typeof indexed.tile_template, "string", `${context} tile template is missing`);
  assert.ok(!indexed.tile_template.includes("#"), `${context} tile template has a fragment`);
  const [templatePath, query = ""] = indexed.tile_template.split("?");
  assert.equal(
    templatePath,
    `${path.posix.dirname(indexed.tilejson)}/{z}/{x}/{y}.png`,
    `${context} tile template and TileJSON roots differ`,
  );
  assert.equal(
    new URLSearchParams(query).get("v"),
    indexed.tile_set_sha256.slice(0, 16),
    `${context} cache key is not bound to its tile aggregate`,
  );
}

test("catalog entries resolve to complete, isolated, integrity-bound datasets", () => {
  const catalog = readJson(catalogPath);
  assert.equal(catalog.schema_version, 1);
  assert.ok(Array.isArray(catalog.datasets) && catalog.datasets.length > 0);

  const catalogIds = catalog.datasets.map((entry) => entry.id);
  assert.equal(new Set(catalogIds).size, catalogIds.length, "catalog IDs must be unique");

  for (const entry of catalog.datasets) {
    const context = `catalog dataset ${entry.id}`;
    assert.match(entry.id, /^[a-z0-9]+(?:-[a-z0-9]+)*$/, `${context} has an unsafe ID`);

    const manifestPath = resolveRelative(dataRoot, entry.manifest, `${context} manifest`);
    assert.ok(fs.statSync(manifestPath).isFile(), `${context} manifest is missing`);
    const manifest = readJson(manifestPath);

    if (manifest.id !== entry.id) {
      const artifactId = entry.artifact_id ?? legacyArtifactIds.get(entry.id);
      assert.equal(
        artifactId,
        manifest.id,
        `${context} must match its manifest ID or declare the manifest artifact_id`,
      );
    }

    const declaredAssetBases = [entry.asset_base, manifest.asset_base].filter(
      (value) => value !== undefined,
    );
    if (declaredAssetBases.length === 2) {
      assert.equal(
        declaredAssetBases[0],
        declaredAssetBases[1],
        `${context} catalog and manifest asset bases differ`,
      );
    }
    const assetRoot = declaredAssetBases.length
      ? resolveAssetBase(declaredAssetBases[0], `${context} asset_base`)
      : path.dirname(manifestPath);

    const catalogSource = resolveRelative(dataRoot, entry.source_image, `${context} source`);
    assert.ok(fs.statSync(catalogSource).isFile(), `${context} catalog source is missing`);
    assert.equal(typeof manifest.source_image, "string", `${context} manifest source is missing`);
    const manifestSource = resolveRelative(
      assetRoot,
      manifest.source_image,
      `${context} manifest source`,
    );
    assert.ok(fs.statSync(manifestSource).isFile(), `${context} manifest source is missing`);
    assert.equal(
      fs.realpathSync(catalogSource),
      fs.realpathSync(manifestSource),
      `${context} catalog and manifest sources resolve differently`,
    );

    if (entry.manifest.startsWith("datasets/")) {
      assert.equal(manifest.status, "approved_publication", `${context} is not publication-ready`);
      assert.equal(manifest.approval?.status, "approved", `${context} lacks approval`);
    }

    assert.ok(Array.isArray(manifest.layers) && manifest.layers.length > 0);
    const categoryIds = manifest.layers.flatMap((layer) =>
      (layer.categories ?? []).map((category) => category.id),
    );
    assert.ok(categoryIds.every((id) => typeof id === "string" && id.length > 0));
    assert.equal(
      new Set(categoryIds).size,
      categoryIds.length,
      `${context} repeats a category ID across layers`,
    );
    assert.equal(
      categoryIds.length,
      entry.category_count,
      `${context} catalog category count differs from its manifest`,
    );

    for (const layer of manifest.layers) {
      if (layer.indexed_raster) {
        verifyIndexedRaster(layer.indexed_raster, assetRoot, `${context}/${layer.id}`);
      }
    }
  }
});
