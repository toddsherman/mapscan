const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const dataset = JSON.parse(
  fs.readFileSync(
    path.join(
      viewerRoot,
      "public",
      "data",
      "datasets",
      "geologic-units-highres",
      "dataset.json",
    ),
    "utf8",
  ),
);
const descriptions = JSON.parse(
  fs.readFileSync(
    path.join(viewerRoot, "lib", "geologic-unit-descriptions.json"),
    "utf8",
  ),
);
const mapComponent = fs.readFileSync(
  path.join(viewerRoot, "components", "mapscan-map.tsx"),
  "utf8",
);

test("provides a legend description for every published geologic unit", () => {
  const categories = dataset.layers.flatMap((layer) => layer.categories ?? []);
  assert.equal(categories.length, 53);
  assert.deepEqual(
    Object.keys(descriptions).sort(),
    categories.map((category) => category.id).sort(),
  );
  for (const category of categories) {
    assert.equal(typeof descriptions[category.id], "string");
    assert.ok(descriptions[category.id].length > 12, category.id);
  }
});

test("renders geology descriptions beneath their source legend codes", () => {
  assert.match(mapComponent, /geologic-unit-descriptions\.json/);
  assert.match(mapComponent, /categoryDescription\(dataset, category\)/);
  assert.match(mapComponent, /className="category-description"/);
});

test("keeps the landslide code and its full term paired", () => {
  assert.match(descriptions["unit-02-qls"], /landslides/i);
});
