import type { Metadata } from "next";
import Image from "next/image";
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

const grapesElevationExampleHref = "/map?config=3&state=N4IgbiBcDMA0IBMogMYEMA2BLAZgewCcA7LNAWjQHMCsUBXDAFzoMzIzSITLoGcBTEPAxQA2qNSZchEuXwF%2BvRmRR4w-AkJAKEAdzx4ksAAzwEKfgCMEAViEAOAEwBdWBPTZ5ssvMXLV6prwun4aRGQAFmgEegYIvEKmqI4IAOyWOA4ubpKeMqQ%2BhH4qahpaCHh0lBy8PlhBSWipKNBoAJxZru5SXgW%2BSiWBWgAOeFwaeLzkw1hEgibw9ghoOG3QnTke0sR9RQMBZfA49WS8wwR0FonwqdBtlqlGTl252979-qVBIDNEAJ5jMgAKzoJGGhySNja9lSADZUhtunkdnI9p8hsJDJR%2BKMMPwyL95kloPxbjYAIyI169VEKfZfLQoKLDaKsEQLECOAAsNhw0EsVK2NIo1FoDGYbPYnG4fHmIF0EX4aEY1xASvMOEcQnJxmMLyF%2BXIVBo9CYLDYHC4PAEWgAtlheBYMJb%2BJVatQ0LMKFaon9VaSEPZ%2BJlYDq9ZseoaRSbxeaMFKrbKtCaifB%2BFyEFz%2BChtbr9ZGUdGxWbJZaZTbglgMNwU-7jAhjPwOqG8xHkd5jcWJRbpda5faAB78bgspQsVOIcwWWG58NIt4FTum7vxst9u0Op0ut1kD28AQJDnLFBBrmz-Ptxei5dxhPluWYHCPo1W58YJ9kQcSxSqtIoLOUi2c7UlGS6xqWvZJvAAKgpQBIaPwRCyBgh5JAg0AoPW8xhheC5Gte4E9omFaoHgjCMGMv7GJYKAIAKQG4cKYElkR95aJYSpELUCAEH6HK0ZYSz2OebZ4UWN4QcRcq8KCOAYHgIQEKh8DoJYGZ2AxolMQRLGrpBJH2o6-DOpwrp8HUxncCgBB4MMymoOk9Zapp87aTGul3uu8BTDgckKRCKmtCgKAhjhWmgTpK6eVBpHEKchCUBEdC2t63AycsRCqphaD2MFImuRF7lRWuMUUbayp4D%2BHLWGgXLoPlIGFsxxX6XKqh4ioNl2aqlhoGgxgoAiLmNR2kW3iVJF4uRlz4niKx-LuCiIfZliwm0wVnsNBpNWNklsSprJkStdy5YBYUFTtRXja1IxkRViipacIT8MooyMPdK3GG0XIcQ122jVde1eSAtrGWMtS8AAjnQaC8BEj30PQtocUpqpoAgbTUfR50jVegOscDfmKbURAsAIvEIxENBKOVtSMEtPjRLa9ntLlgl-QWANdtdUnsXw8MozQVWNLC9j2JYGk4-9ePc0DMVjFg4OPZQ0TYDmHJoNA9g2JY6xbZzMsSQTMVKKwuiC1gwvwP19jkpYxgc5e%2BH43pvPwOCwzgqjHJtGkyxGFLBvO7LxsGZuxnbuZ9OXAA1p1tn2W0bSpG0fWO2JzU8-tqBYNHifwjYTTp25Ieu9nyyMFbIBrKk5K1cXhWl9FJF4Ng6iJ8YsLLGdrYXVzRtl8DMdYLoDqqrla1tNhve48HA-N-24cmXMO4yZY9O2bQbA4Bcuf2fYbSwlC2Mz9Lc%2BEYPMVoGAeDoBU%2B%2BwrC5JJw3l1NxND4YLaYzxOP0A2BjGc%2BsnbiQvgvLQaBPZ4n3sYKEaxX79zAR-EYGBkq1HOKCB6vZIEmiOqqZoNgoQO2ARnXaoc2qKgIELeyKcKRLAQYbJBN13YOnekyRWNDYQZnsGgBh58PLIPdkqb2SRbhcmhENQOIDM5yxIuCNATIHpzBQO9Ggcx7KwhwDyewm0pGkJduA92eBQYeiIMqKusJ-zkltnw0BAjmEgyXpHbi2YsAIDoG6VUa1oBpFCqfIOdiWpu3lJgUmjANGwjuM0WxMjyFaA9OCQ85Jrb9WML5GJZDL4kWoPwRCEQ3QTl5NAckcIMkGMESANxeJVQ2BQCkW4tiKKjASREBaxl%2BBgGVIrTKZgcR0l4N07UZgp5tBwJI-xICml4BaW0vEnTGCDLTHMrpgJjBkAomQMMn4hkgGsOhfgksJliSmTMsg7T5mLLVMshZgItkbMcDYNZtodloDtrRPWejhQnNYMMVpZzrmXPOSs8IDy1kbMeU8nZ2s0CWJ7sBM%2B6zbLTJ%2BX8oFNyelXI6cCsgELEWbN1JC0M8BYTtGMGgae8KAnfMgaigFlEllYvRfi3UeLHAEu2USkAWZUjtHVp8qM1LfmzMZYCulIL2UbK5Oy55nLoCOEiTwxpSLTlotFSKwEUqWXgulTs8kXC0jCQYq4EAeAxBqmiIwCI0M0Ax3xHDG1sw4JvUQgstgCK2UPLqGY%2BMrTLA0G4GAQCs8CSWmUFEGIsxFC1AAF5jCUfwLAiVLCEDIIG8ouSCBkAQKwmglg6BMrPoyAJoxhgMGxQgZaucFoeuIcG44ChIhoGjdEbgvY6SjC4lgSwVYq0UAULDItICBCBHxCEZUlCyBYFtCyFRh5g3YlbtM2gn5IFkDZTqQdGdAhUHxJwUmbBziuJmO9Jlz9bZruMBug6kzlUouFRc%2Blda0SDEOMG2JWTBDGoHGIY1zzICiE2Y4ckAA6GEEi4Ba2A3KrkotYSwHsFB8kjhUgLD1AAXyAA";

const palmDesertClimateExampleHref = "/map?config=3&state=N4IgbiBcDMA0IBMogMYEMA2BLAZgewCcA7LNAWjTAFMC0BzKioogV0zIAcCqUsOsALmgFY8RMgEYAnAA4JZAEwAGCUpDwMUANpbUmXIRLl83AM4CyKPNQLqQ3BAHc8eJLCXwEKKgCMEAVnUZBQBdWF10bBMjMhMqc0trGjtHeIEacQALNAInFwRTdQ9UBQQAdh8cINDwvSjDUljCNMSbOwQ8FjoMNFNYrFt3eDQylGg0KWqwiP1oxriEqzb4DjEEGjxTcn4iKiL4GQQ0HCloKdrIg2J55sWkwZAcAbJTLhZvfZAy6CkfMrdgtM6lcYgsLEtkissEQAJ5iMgAKxYJA4kOK-lkZQAbGVzjN6tdjLdwfc7BhXAxVhhGDs9kMQNAqN9-BI8cC5kSzCTlqhshwcrRNPSFAAWfw4aA%2BNmXDmcPAcFg9ETw9ZEUyCGGKJTKOyqJRKRT%2BfXqCTwHA4fXG2CAi6zBrbeWK4SicSq9UCTXKHXwBRGg1%2BtSwU0gKiWq02-EgxqrBVKl1kN0arXekAByRhk3wFBhwMR9n2uWx50qqhqpNewMgPVW4NoHPSu2El5UGyMVLCTI0MhYAC2-JQAkK8B7aAAHr2WD2yI5oQgXqiqG5g0cZCgfJNrTVI7LKDR6Iw0Mw2BhONxePwhMrxNI5FrVLqjZIyqz6QgpFQpDhcZugTKC7vaAYJhWHYLgeD4QRi2vWR5GUe94AkX1JCNT4xgQGQPwbAkYgA-dgOPU9wIvKDJBgu9Kz1SRfU%2BNAeCxBApR-W1sMaXCgMPECTzA89IKvUjbzgysyn8dNA2KWQfH8FAAS3fMmzYg8j1As8IMveMb1glRK0fYTPmCNARUqLCo3IBT8OUojePUsjBLsaB-TE%2BB6KkBQfFzWS-3kmw8I4gjuNUkiNPIuxlDIezPn8Mo5DQBRjJ3bz2KUriVOIvigtshCDW9YoRQkMolCkEU4v-BLFM4wieLU%2BF0q0uwRPg4oFBQfwqBkLFiq8vdEvK-zUusgTavgf1MyrLERXKGQOpiAR5TwOhaA4TJNSoakwCg9oqDA0x1TET4vyoCQUA3PNPOm2b5rQRbltW9b4BWlsSINGbRLIHtPh4FA13c39GzO1YLqush7rWq87GBwL9TIZ7fQNN76VDHwsVkKbGhm-6FqWoGbtBu7sfjGGobwMg-Vez4AjQPKqiY7cCzRuaMeuh6cZDPH4RJ57q1J%2Bl0BkRHvxO37UfOhmsaZl0wdZ69LUJrVIbh4pKmxEU6QFljyDpgHMfB5ntfx6XnpFaX5fgHxoCxJQzZR9XhcurXJYlsX4UNyHnrTY2QDQGRDIkGSQjCEA8G0EMcgETIAEc2AAa0YUxskj6E6DldIiBEdhTsaL0RKeIh2CWnwCCwOcwBfOSYg4HoU7IbJcmheI%2BgALzEeIyF2LA6EyHxCDIYuNq7BAsHMAufBYPj07QOwx4oeasBQRUBBYQUyArucWFMOlS-mAZGGyeucjnQ85zMVYyx8LBsA9ChuF6CfBfIBg8HJOgZ9ey7gqzW%2BKFK8zkssqroIG%2BCG9yBr1bNOKgHYuy9n7IOG%2BatCxOj4omC%2BFZYEmUJujW2jMQbi3fnAsErRkj%2B1HNof2b1IBaEkBILEAA6M2sgcSwGgNAahwksSISxEGJQtCsTQAUEMJQIQAC%2BQA";

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
  prominent = false,
}: {
  children: React.ReactNode;
  dark?: boolean;
  prominent?: boolean;
}) {
  const className = `story-map-link${dark ? " story-map-link-dark" : ""}${prominent ? " story-map-link-prominent" : ""}`;

  return (
    <Link className={`${className} story-map-link-desktop`} href="/map">
      <span>{children}</span>
      <ArrowIcon />
    </Link>
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
            <InteractiveMapLink prominent>Open the interactive map</InteractiveMapLink>
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

          <ProcessLoopAnimation />

          <div className="story-prose story-process-copy">
            <p>
              This was not a single prompt that produced a finished map. The
              system made a candidate, rendered evidence, measured what was
              wrong, changed its assumptions, and tried again. Fine coastline
              details, partial source extents, gradient colors, city labels,
              water, and overlapping legends all required different tests. The
              final product keeps each recovered legend item independently
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
              Saved compositions make relationships across the recovered
              datasets easier to see.
            </p>
          </div>
          <div className="story-insight-example">
            <div className="story-insight-heading-row">
              <figure className="story-insight-figure">
                <Image
                  src="/mapscan/editorial/mapscan/examples/grapes-elevation.jpg"
                  alt="A MapScan composition showing purple grape-growing areas over blue elevation bands around the Sacramento–San Joaquin Delta."
                  width={658}
                  height={662}
                  sizes="(max-width: 767px) 96px, 190px"
                />
                <figcaption>9 layers · 2 datasets</figcaption>
              </figure>
              <div className="story-insight-title">
                <p className="story-kicker">Example 01</p>
                <h3>Grapes grow within particular elevation bands.</h3>
              </div>
            </div>
            <div className="story-insight-copy">
              <p>
                California’s grape-growing areas, shown in purple, cluster
                within particular elevation bands, shown in blue. The overlay
                makes the Central Valley, North Coast, and foothill patterns
                easier to compare than either source map alone.
              </p>
              <div className="story-insight-key" aria-label="Map color key">
                <span>
                  <i className="story-insight-swatch story-insight-swatch-grapes" />
                  Grapes
                </span>
                <span>
                  <i className="story-insight-swatch story-insight-swatch-elevation" />
                  Elevation
                </span>
              </div>
              <Link
                className="story-map-link story-map-link-desktop story-insight-link"
                href={grapesElevationExampleHref}
              >
                <span>Open this composition</span>
                <ArrowIcon />
              </Link>
            </div>
          </div>
          <div className="story-insight-example">
            <div className="story-insight-heading-row">
              <figure className="story-insight-figure">
                <Image
                  src="/mapscan/editorial/mapscan/examples/palm-desert-rain-wind-population.jpg"
                  alt="A MapScan composition of population, the driest rainfall band, and the highest wind band around Palm Desert, California."
                  width={658}
                  height={662}
                  sizes="(max-width: 767px) 96px, 190px"
                />
                <figcaption>6 layers · 3 datasets</figcaption>
              </figure>
              <div className="story-insight-title">
                <p className="story-kicker">Example 02</p>
                <h3>Palm Desert sits where dry and windy overlap.</h3>
              </div>
            </div>
            <div className="story-insight-copy">
              <p>
                One of the driest and windiest places where people live appears
                to be the Palm Desert area. The map combines 0–5 inches of
                annual rain, winds above 60 mph, and populated areas to make
                that overlap visible.
              </p>
              <div className="story-insight-key" aria-label="Map color key">
                <span>
                  <i className="story-insight-swatch story-insight-swatch-rain" />
                  0–5 in rain
                </span>
                <span>
                  <i className="story-insight-swatch story-insight-swatch-wind" />
                  &gt;60 mph wind
                </span>
                <span>
                  <i className="story-insight-swatch story-insight-swatch-population" />
                  Population
                </span>
              </div>
              <Link
                className="story-map-link story-map-link-desktop story-insight-link"
                href={palmDesertClimateExampleHref}
              >
                <span>Open this composition</span>
                <ArrowIcon />
              </Link>
            </div>
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
