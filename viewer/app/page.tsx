import type { Metadata } from "next";
import Link from "next/link";
import { ProcessLoopAnimation } from "@/components/process-loop-animation";
import { SourceMapGallery, type SourceMap } from "@/components/source-map-gallery";

export const metadata: Metadata = {
  title: "MapScan — Recovering data from flat map images",
  description:
    "How I used Claude, Mapbox, and an iterative image-processing pipeline to turn nine static California maps into composable data layers.",
};

const sourceMaps: SourceMap[] = [
  {
    title: "Forest Cover of California",
    shortTitle: "Forest",
    preview: "/mapscan/editorial/mapscan/forest.jpg",
    original: "/mapscan/data/staging/forest-autonomous-v5/source.jpg",
    width: 871,
    height: 1232,
    alt: "A map of California showing forest types in distinct legend colors.",
  },
  {
    title: "California Plant Hardiness Zones",
    shortTitle: "Hardiness",
    preview: "/mapscan/editorial/mapscan/hardiness.jpg",
    original: "/mapscan/data/staging/plant-hardiness-autonomous-v3/source.avif",
    width: 801,
    height: 694,
    alt: "A map of California plant hardiness zones.",
  },
  {
    title: "California Agricultural Land Use",
    shortTitle: "Farms",
    preview: "/mapscan/editorial/mapscan/farms.jpg",
    original: "/mapscan/data/staging/farms-v2-autonomous-v1/source.png",
    width: 4250,
    height: 5500,
    alt: "A detailed California agricultural land-use map with many crop classes.",
  },
  {
    title: "California Population Density",
    shortTitle: "Population",
    preview: "/mapscan/editorial/mapscan/population.jpg",
    original: "/mapscan/data/staging/population-native-fidelity-repair-v2/source.png",
    width: 2500,
    height: 3000,
    alt: "A California population-density map.",
  },
  {
    title: "California Fire Hazard and Responsibility Areas",
    shortTitle: "Fire",
    preview: "/mapscan/editorial/mapscan/fire.jpg",
    original: "/mapscan/data/staging/fire-autonomous-v2-clipped/source.webp",
    width: 1480,
    height: 1823,
    alt: "A California map showing fire hazard and responsibility areas.",
  },
  {
    title: "California Severe Weather Impacts",
    shortTitle: "Weather",
    preview: "/mapscan/editorial/mapscan/weather.jpg",
    original: "/mapscan/data/staging/landslide-autonomous-v2-clipped/source.png",
    width: 918,
    height: 918,
    alt: "A California map combining precipitation, landslide, wind, and flooding data.",
  },
  {
    title: "Geologic Map of California",
    shortTitle: "Geology",
    preview: "/mapscan/editorial/mapscan/geology.jpg",
    original: "/mapscan/data/staging/geologic-highres-autonomous-v1/source.png",
    width: 7088,
    height: 9375,
    alt: "A high-resolution geologic map of California.",
  },
  {
    title: "California Topography and Elevation",
    shortTitle: "Elevation",
    preview: "/mapscan/editorial/mapscan/elevation.jpg",
    original: "/mapscan/data/staging/elevation-autonomous-v2/source.gif",
    width: 1117,
    height: 1200,
    alt: "A shaded relief and elevation map of California.",
  },
  {
    title: "California Average Annual Precipitation, 1981–2010",
    shortTitle: "Rainfall",
    preview: "/mapscan/editorial/mapscan/rainfall.jpg",
    original: "/mapscan/data/staging/rainfall-1981-2010-autonomous-v2/source.png",
    width: 3204,
    height: 2366,
    alt: "A map of average annual precipitation across California.",
  },
];

const processSteps = [
  {
    number: "01",
    title: "Read the image",
    copy: "Claude finds the map frame, the legend, the thematic marks, and the visual noise—labels, roads, borders, water, and background—that should not become data.",
  },
  {
    number: "02",
    title: "Work backwards from Mapbox",
    copy: "California’s coast, state boundary, and county geometry are rendered from Mapbox as the geographic reference. The source image is projected into that same pixel space.",
  },
  {
    number: "03",
    title: "Align, compare, repeat",
    copy: "Candidate projection, rotation, scale, skew, and local warp settings are scored against the reference. Comparisons run statewide and again at difficult edges such as the Bay Area and Colorado River.",
  },
  {
    number: "04",
    title: "Recover the legend classes",
    copy: "Legend labels and swatches become named classes. Pixels are classified at the source’s native resolution so the output preserves the original distinctions instead of tracing simplified shapes.",
  },
  {
    number: "05",
    title: "Diff, repair, publish",
    copy: "The extraction is flipped against the source and inspected at several zoom levels. Missing data, false colors, text holes, and spill outside California feed another iteration before Web Mercator tiles are published.",
  },
];

function ArrowIcon() {
  return (
    <svg
      className="story-map-link-arrow"
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 16 16"
    >
      <path d="M3.5 12.5 12.5 3.5M6 3.5h6.5V10" />
    </svg>
  );
}

function InteractiveMapLink({
  children,
  dark = false,
}: {
  children: React.ReactNode;
  dark?: boolean;
}) {
  const className = `story-map-link${dark ? " story-map-link-dark" : ""}`;

  return (
    <>
      <Link className={`${className} story-map-link-desktop`} href="/map">
        <span>{children}</span>
        <ArrowIcon />
      </Link>
      <p className={`${className} story-map-link-mobile`}>
        The map is build for desktop
      </p>
    </>
  );
}

function PaperGrain() {
  return (
    <svg
      className="story-paper-grain"
      aria-hidden="true"
      focusable="false"
      preserveAspectRatio="none"
    >
      <filter id="mapscan-paper-noise" x="0" y="0" width="100%" height="100%">
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.84"
          numOctaves="3"
          seed="17"
          stitchTiles="stitch"
        />
        <feColorMatrix type="saturate" values="0" />
        <feComponentTransfer>
          <feFuncA type="table" tableValues="0 0.1" />
        </feComponentTransfer>
      </filter>
      <rect width="100%" height="100%" filter="url(#mapscan-paper-noise)" />
    </svg>
  );
}

export default function MapScanStoryPage() {
  return (
    <main className="story-page">
      <PaperGrain />

      <header className="story-site-header">
        <nav aria-label="Todd dot sh">
          <a href="https://www.todd.sh/">← todd.sh</a>
        </nav>
      </header>

      <article>
        <section className="story-hero story-container">
          <p className="story-kicker">MapScan</p>
          <h1>Turning flat map images into data I could actually combine.</h1>
          <div className="story-hero-grid">
            <p className="story-dek">
              I wanted to see what California looked like when forest, rainfall,
              elevation, geology, population, fire, and agriculture could be
              viewed together. The maps existed. Their data was trapped inside
              pixels.
            </p>
            <InteractiveMapLink>Open the interactive map</InteractiveMapLink>
          </div>
        </section>

        <section className="story-section story-container story-problem">
          <div className="story-section-label">
            <span>01</span>
            The problem
          </div>
          <div className="story-prose story-prose-lead">
            <p>
              I’ve always loved maps, data, and playing with the union of the
              two. In the past, when I wanted to compare maps, I composited them
              by hand in Photoshop. A transparent overlay was rarely enough.
              Each map used a different crop, scale, rotation, or projection,
              so I spent time warping and skewing the images until the
              coastlines looked approximately right.
            </p>
            <p>
              Even then, the result was still a picture. I could not turn one
              tree type off, recolor a rainfall band, move geology behind
              population, or share a precise combination with someone else.
              The information was visible but not manipulable.
            </p>
            <blockquote>
              The question was simple: could AI recover the geography and the
              legend from a flattened image without reducing the fidelity of
              the original map?
            </blockquote>
          </div>
        </section>

        <section className="story-section story-sources">
          <div className="story-container">
            <div className="story-section-label">
              <span>02</span>
              The source material
            </div>
            <div className="story-section-heading">
              <h2>Nine maps, nine different kinds of trouble.</h2>
              <p>
                I found maps I liked and gave them to Claude. Some were clean
                categorical diagrams. Others were dense, partial, textured, or
                layered with labels and boundaries. Click any image to inspect
                it.
              </p>
            </div>
          </div>
          <div className="story-gallery-wrap">
            <SourceMapGallery maps={sourceMaps} />
          </div>
        </section>

        <section className="story-section story-container">
          <div className="story-section-label">
            <span>03</span>
            The process
          </div>
          <div className="story-section-heading story-process-heading">
            <h2>Two loops: first geometry, then data.</h2>
            <p>
              I asked Claude to work backwards from the Mapbox map shape as a
              first step because it represents the final rendering. Alignment
              had to pass before extraction could begin; otherwise a perfect
              set of pixels would still be in the wrong place.
            </p>
          </div>

          <ol className="story-process">
            {processSteps.map((step) => (
              <li key={step.number}>
                <span className="story-process-number">{step.number}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.copy}</p>
                </div>
              </li>
            ))}
          </ol>

          <div className="story-loops" aria-label="The two iterative comparison loops">
            <div>
              <span>Geometry loop</span>
              <strong>Warp → compare to Mapbox → score → adjust ↻</strong>
            </div>
            <div>
              <span>Data loop</span>
              <strong>Extract → compare to source → diff → repair ↻</strong>
            </div>
          </div>

          <ProcessLoopAnimation />

          <div className="story-prose story-process-copy">
            <p>
              This was not a single prompt that produced a finished map. The
              system made a candidate, rendered evidence, measured what was
              wrong, changed its assumptions, and tried again. Fine coastline
              details, partial source extents, gradient colors, city labels,
              water, and overlapping legends all required different tests.
            </p>
            <p>
              The final product keeps each recovered legend item independently
              selectable. Datasets can be combined, reordered, recolored, and
              given their own opacity while Mapbox remains the live geographic
              canvas underneath.
            </p>
          </div>
        </section>

        <section className="story-section story-play">
          <div className="story-container story-play-grid">
            <div>
              <p className="story-kicker">Try the result</p>
              <h2>Build a map the source images could never make.</h2>
            </div>
            <div>
              <p>
                Select individual classes across datasets, change their colors
                and opacity, reorder the dataset stack, then copy a link to the
                exact composition.
              </p>
              <InteractiveMapLink dark>Launch MapScan</InteractiveMapLink>
            </div>
          </div>
        </section>

        <section className="story-section story-container story-insights">
          <div className="story-section-label">
            <span>04</span>
            What the maps reveal
          </div>
          <div className="story-section-heading">
            <h2>The interesting part starts when the layers meet.</h2>
            <p>
              I’m assembling a set of combinations that make relationships in
              the data easier to see. Those examples will appear here next.
            </p>
          </div>
          <div className="story-insight-placeholder">
            <span>Examples coming soon</span>
            <p>Layer combinations, observations, and saved map views coming next.</p>
          </div>
        </section>
      </article>

      <footer className="story-footer story-container">
        <span>MapScan</span>
        <a href="/">Todd Sherman</a>
      </footer>
    </main>
  );
}
