const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");
const { validate } = require("mapbox-gl/dist/style-spec/index.cjs");

const filename = path.join(__dirname, "..", "lib", "indexed-raster.ts");
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
  indexedRasterCategoryColor,
  indexedRasterCategoryLayerId,
  indexedRasterColor,
  indexedRasterLayerId,
  indexedRasterSourceId,
} = loaded.exports;

function fixture(count) {
  const categories = [];
  const styles = {};
  for (let classId = 1; classId <= count; classId += 1) {
    const id = `class-${classId}`;
    categories.push({ id, class_id: classId });
    styles[id] = {
      enabled: classId !== 22,
      color: classId === 53 ? "#123456" : "#abcdef",
      opacity: classId === 53 ? 0.37 : 0.82,
    };
  }
  return { categories, styles };
}

test("one Mapbox raster-color expression independently styles 53 class IDs", () => {
  const { categories, styles } = fixture(53);
  const expression = indexedRasterColor(categories, styles);

  assert.deepEqual(expression.slice(0, 3), ["step", ["raster-value"], "rgba(0, 0, 0, 0)"]);
  assert.equal(expression[3], 0.5 / 255);
  assert.equal(expression[4], "rgba(171, 205, 239, 0.82)");
  assert.equal(expression[46], "rgba(0, 0, 0, 0)");
  assert.equal(expression.at(-2), 53.5 / 255);
  assert.equal(expression.at(-1), "rgba(0, 0, 0, 0)");
  assert.ok(expression.includes("rgba(18, 52, 86, 0.37)"));

  const sourceId = indexedRasterSourceId("geology");
  const layerId = indexedRasterLayerId("geology");
  const style = {
    version: 8,
    sources: {
      [sourceId]: {
        type: "raster",
        tiles: ["https://example.com/tiles/{z}/{x}/{y}.png"],
        tileSize: 256,
      },
    },
    layers: [
      {
        id: layerId,
        type: "raster",
        source: sourceId,
        paint: {
          "raster-resampling": "nearest",
          "raster-opacity": 1,
          "raster-color-mix": [1, 0, 0, 0],
          "raster-color-range": [0, 1],
          "raster-color": expression,
        },
      },
    ],
  };
  assert.deepEqual(validate(style), []);
});

test("rejects duplicate and out-of-range indexed class IDs", () => {
  const styles = { a: { enabled: true, color: "#000000", opacity: 1 } };
  assert.throws(
    () => indexedRasterColor([{ id: "a", class_id: 256 }], styles),
    /1 to 255/,
  );
  assert.throws(
    () =>
      indexedRasterColor(
        [
          { id: "a", class_id: 1 },
          { id: "b", class_id: 1 },
        ],
        { ...styles, b: styles.a },
      ),
    /duplicated/,
  );
});

test("isolates an indexed class into its own reorderable Mapbox layer", () => {
  assert.equal(
    indexedRasterCategoryLayerId("forest", "western hardwoods"),
    "mapscan-indexed-layer-forest-western-hardwoods",
  );
  assert.deepEqual(
    indexedRasterCategoryColor({ id: "western-hardwoods", class_id: 7 }, "#123456"),
    [
      "step",
      ["raster-value"],
      "rgba(0, 0, 0, 0)",
      6.5 / 255,
      "rgba(18, 52, 86, 1)",
      7.5 / 255,
      "rgba(0, 0, 0, 0)",
    ],
  );
});
