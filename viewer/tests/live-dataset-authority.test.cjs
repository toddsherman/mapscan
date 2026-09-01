const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.resolve(__dirname, "..");
const repositoryRoot = path.resolve(viewerRoot, "..");
const pagePath = path.join(viewerRoot, "app", "map", "page.tsx");
const catalogPath = path.join(viewerRoot, "public", "data", "catalog.json");
const authorityPath = path.join(
  repositoryRoot,
  "config",
  "live-dataset-authority-v2.json",
);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

test("every live dataset is pinned to its newest accepted extraction", () => {
  const page = fs.readFileSync(pagePath, "utf8");
  const authority = readJson(authorityPath);
  const catalog = readJson(catalogPath);
  assert.equal(authority.schema_version, 1);

  const importedDirectories = Array.from(
    page.matchAll(/@\/public\/data\/datasets\/([^/]+)\/dataset\.json/g),
    (match) => match[1],
  );
  const expectedDirectories = authority.datasets.map(
    (dataset) => dataset.public_directory,
  );
  assert.deepEqual(
    [...importedDirectories].sort(),
    [...expectedDirectories].sort(),
  );
  assert.equal(new Set(importedDirectories).size, importedDirectories.length);
  assert.doesNotMatch(page, /rivers|hydrography/i);

  for (const record of authority.datasets) {
    const acceptedPath = path.join(repositoryRoot, record.accepted_extraction);
    assert.equal(sha256(acceptedPath), record.accepted_extraction_sha256);

    const publicRoot = path.join(
      viewerRoot,
      "public",
      "data",
      "datasets",
      record.public_directory,
    );
    const dataset = readJson(path.join(publicRoot, "dataset.json"));
    const provenance = readJson(path.join(publicRoot, "provenance.json"));
    assert.equal(dataset.status, "approved_publication");
    assert.equal(dataset.publication_approved, true);
    assert.equal(
      provenance.accepted_extraction.sha256,
      record.accepted_extraction_sha256,
    );
    assert.equal(
      dataset.provenance.sha256,
      sha256(path.join(publicRoot, "provenance.json")),
    );

    const manifest = `datasets/${record.public_directory}/dataset.json`;
    const catalogRecords = catalog.datasets.filter(
      (entry) => entry.manifest === manifest,
    );
    assert.equal(catalogRecords.length, 1, `catalog entry for ${manifest}`);
    assert.match(
      page,
      new RegExp(`/mapscan/data/datasets/${record.public_directory}/`),
    );
  }
});
