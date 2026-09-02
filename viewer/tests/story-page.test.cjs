const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const viewerRoot = path.join(__dirname, "..");
const storyPage = fs.readFileSync(path.join(viewerRoot, "app", "page.tsx"), "utf8");
const mapPage = fs.readFileSync(
  path.join(viewerRoot, "app", "map", "page.tsx"),
  "utf8",
);
const gallery = fs.readFileSync(
  path.join(viewerRoot, "components", "source-map-gallery.tsx"),
  "utf8",
);
const processLoopAnimation = fs.readFileSync(
  path.join(viewerRoot, "components", "process-loop-animation.tsx"),
  "utf8",
);
const desktopMapGate = fs.readFileSync(
  path.join(viewerRoot, "components", "desktop-map-gate.tsx"),
  "utf8",
);
const styles = fs.readFileSync(
  path.join(viewerRoot, "app", "globals.css"),
  "utf8",
);

const sourceMaps = [
  "forest",
  "hardiness",
  "farms",
  "population",
  "fire",
  "weather",
  "geology",
  "elevation",
  "rainfall",
];

test("publishes the project story at the MapScan root and the viewer at /map", () => {
  assert.match(storyPage, /Turning flat map images into data I could actually combine/);
  assert.match(storyPage, /href="\/map"/);
  assert.match(mapPage, /<DesktopMapGate datasets=\{datasets\} \/>/);
  assert.match(desktopMapGate, /<MapScanShell datasets=\{datasets\} \/>/);
  assert.doesNotMatch(storyPage, /<MapScanShell/);
});

test("hides MapScan story links entirely on mobile without an emoji arrow", () => {
  assert.doesNotMatch(storyPage, /The map is build for desktop/);
  assert.doesNotMatch(storyPage, /story-map-link-mobile/);
  assert.match(storyPage, /className="story-map-link-arrow"/);
  assert.doesNotMatch(storyPage, /↗/);
  assert.match(mapPage, /<DesktopMapGate datasets=\{datasets\} \/>/);
  assert.match(desktopMapGate, /window\.matchMedia\(desktopQuery\)/);
  assert.match(desktopMapGate, /The map is build for desktop/);
  assert.match(styles, /@media \(max-width: 767px\)[\s\S]*\.story-map-link-desktop \{ display: none; \}/);
  assert.doesNotMatch(styles, /\.story-map-link-mobile/);
});

test("publishes the population, farms, and elevation composition as a saved example", () => {
  assert.match(storyPage, /Population and farms follow low ground/);
  assert.match(storyPage, /53 layers · 3 datasets/);
  assert.match(storyPage, /population-farms-elevation\.jpg/);
  assert.match(storyPage, /populationFarmsElevationExampleHref/);
  assert.match(storyPage, /config=3&state=/);
  assert.doesNotMatch(storyPage, /Examples coming soon/);
  assert.equal(
    fs.existsSync(
      path.join(
        viewerRoot,
        "public",
        "editorial",
        "mapscan",
        "examples",
        "population-farms-elevation.jpg",
      ),
    ),
    true,
  );
});

test("publishes the forest and elevation composition as a saved example", () => {
  assert.match(storyPage, /Tree types roughly follow elevation/);
  assert.match(storyPage, /6 layers · 2 datasets/);
  assert.match(storyPage, /forest-elevation\.jpg/);
  assert.match(storyPage, /forestElevationExampleHref/);
  assert.match(storyPage, /Fir-spruce, pinyon-juniper, and lodgepole pine/);
  assert.equal(
    fs.existsSync(
      path.join(
        viewerRoot,
        "public",
        "editorial",
        "mapscan",
        "examples",
        "forest-elevation.jpg",
      ),
    ),
    true,
  );
});

test("shows all nine source maps with lightweight editorial previews", () => {
  for (const sourceMap of sourceMaps) {
    assert.match(storyPage, new RegExp(`/mapscan/editorial/mapscan/${sourceMap}\\.jpg`));
    assert.equal(
      fs.existsSync(
        path.join(viewerRoot, "public", "editorial", "mapscan", `${sourceMap}.jpg`),
      ),
      true,
    );
  }
  assert.equal((storyPage.match(/shortTitle:/g) ?? []).length, 9);
  assert.match(
    styles,
    /\.story-source-grid\s*\{[^}]*grid-template-columns:\s*repeat\(3,/s,
  );
  assert.doesNotMatch(
    styles,
    /\.story-source-grid\s*\{[^}]*overflow-x:\s*auto/s,
  );
});

test("opens source maps in an accessible dismissible lightbox", () => {
  assert.match(gallery, /role="dialog"/);
  assert.match(gallery, /aria-modal="true"/);
  assert.match(gallery, /event\.key === "Escape"/);
  assert.match(gallery, /document\.body\.style\.overflow = "hidden"/);
  assert.match(gallery, /Open full resolution/);
  assert.match(
    styles,
    /\.story-lightbox-image\s*\{[^}]*overflow:\s*hidden/s,
  );
});

test("opens the problem story with Todd's interest in maps and data", () => {
  assert.match(
    storyPage,
    /I’ve always loved maps, data, and playing with the union of the/,
  );
});

test("illustrates both gated comparison loops with controllable motion", () => {
  assert.match(storyPage, /<ProcessLoopAnimation \/>/);
  assert.match(processLoopAnimation, /Loop 01 · Geometry/);
  assert.match(processLoopAnimation, /Bring the source to Mapbox/);
  assert.match(processLoopAnimation, /Loop 02 · Data/);
  assert.match(processLoopAnimation, /Make the extraction match the source/);
  assert.match(processLoopAnimation, /aria-pressed=\{paused\}/);
  assert.doesNotMatch(processLoopAnimation, /Process schematic/);
  assert.doesNotMatch(processLoopAnimation, /Each loop had to settle/);
  assert.match(styles, /\.story-loop-animation \{ margin: 24px 0 0; \}/);
  assert.match(styles, /@keyframes story-geometry-converge/);
  assert.match(styles, /@keyframes story-data-cleanup/);
  assert.match(
    styles,
    /@media \(prefers-reduced-motion: reduce\)[\s\S]*\.story-loop-animation \* \{ animation: none !important; \}/,
  );
});

test("keeps the process conclusion in one prose column", () => {
  assert.match(
    storyPage,
    /This was not a single prompt[\s\S]*canvas underneath\./,
  );
  assert.match(
    storyPage,
    /<div className="story-prose story-process-copy">\s*<p>[\s\S]*<\/p>\s*<\/div>/,
  );
  assert.doesNotMatch(
    styles,
    /\.story-process-copy\s*\{[^}]*grid-template-columns/s,
  );
});
