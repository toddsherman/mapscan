const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

const filename = path.join(__dirname, "..", "lib", "mapbox-style.ts");
const source = fs.readFileSync(filename, "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  fileName: filename,
}).outputText;
const loaded = { exports: {} };
new Function("exports", "require", "module", "__filename", "__dirname", compiled)(
  loaded.exports,
  require,
  loaded,
  filename,
  path.dirname(filename),
);
const {
  findThematicRasterInsertionLayer,
  findWaterFillInsertionLayer,
} = loaded.exports;

test("uses a true water fill instead of earlier waterway linework", () => {
  assert.equal(
    findWaterFillInsertionLayer([
      { id: "waterway", type: "line", "source-layer": "waterway" },
      { id: "water", type: "fill", "source-layer": "water" },
    ]),
    "water",
  );
});

test("supports a renamed fill backed by the water source layer", () => {
  assert.equal(
    findWaterFillInsertionLayer([
      { id: "waterway", type: "line" },
      { id: "natural-water", type: "fill", "source-layer": "water" },
    ]),
    "natural-water",
  );
});

test("fails closed when the style exposes only water linework", () => {
  assert.equal(
    findWaterFillInsertionLayer([
      { id: "water", type: "line", "source-layer": "water" },
      { id: "waterway", type: "line", "source-layer": "waterway" },
    ]),
    undefined,
  );
});

test("places thematic rasters above every basemap fill but below linework", () => {
  assert.equal(
    findThematicRasterInsertionLayer([
      { id: "land", type: "fill" },
      { id: "water", type: "fill", "source-layer": "water" },
      { id: "urban", type: "fill" },
      { id: "roads", type: "line" },
      { id: "labels", type: "symbol" },
    ]),
    "roads",
  );
});

test("returns no thematic insertion point when a style ends in fills", () => {
  assert.equal(
    findThematicRasterInsertionLayer([
      { id: "land", type: "fill" },
      { id: "water", type: "fill", "source-layer": "water" },
    ]),
    undefined,
  );
});
