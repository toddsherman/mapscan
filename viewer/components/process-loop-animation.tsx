"use client";

import { useState } from "react";

const californiaPath =
  "M118 22 L322 66 L315 93 L309 126 L320 151 L337 178 L354 202 L369 228 L385 252 L401 278 L392 301 L407 321 L391 337 L356 333 L325 334 C315 326 305 319 294 316 C281 306 270 298 262 287 C248 273 236 261 224 249 C212 236 205 224 198 210 C193 195 188 182 181 169 C176 160 168 154 164 145 L176 137 L167 132 C158 126 155 119 148 112 C140 104 135 96 130 87 C125 79 122 71 119 64 C115 55 113 47 114 39 C115 32 116 27 118 22 Z";

function LoopStage({
  labels,
  className = "",
}: {
  labels: string[];
  className?: string;
}) {
  return (
    <div
      className={`story-loop-stage${className ? ` ${className}` : ""}`}
      aria-hidden="true"
    >
      {labels.map((label, index) => (
        <span className={`story-loop-stage-${index + 1}`} key={label}>
          {label}
        </span>
      ))}
    </div>
  );
}

function GeometryLoop() {
  const markers = [
    [118, 22],
    [322, 66],
    [309, 126],
    [401, 278],
    [391, 337],
    [294, 316],
    [198, 210],
    [164, 145],
  ];

  return (
    <section className="story-loop-card" aria-labelledby="geometry-loop-title">
      <header>
        <span>Loop 01 · Geometry</span>
        <h4 id="geometry-loop-title">Bring the source to Mapbox.</h4>
        <strong className="story-loop-formula">
          Warp → compare to Mapbox → score → adjust ↻
        </strong>
        <p>Transform, render, measure the perimeter, then try again.</p>
      </header>
      <div className="story-loop-canvas">
        <svg
          viewBox="0 0 520 370"
          role="img"
          aria-labelledby="geometry-animation-title geometry-animation-description"
        >
          <title id="geometry-animation-title">
            A source map repeatedly aligning to the Mapbox California boundary
          </title>
          <desc id="geometry-animation-description">
            The rust source-derived outline begins rotated, skewed, and offset from
            the black Mapbox reference. Across four iterations it converges on the
            reference and passes the alignment gate.
          </desc>
          <defs>
            <pattern
              id="geometry-grid"
              width="18"
              height="18"
              patternUnits="userSpaceOnUse"
            >
              <path d="M18 0H0V18" fill="none" stroke="currentColor" strokeWidth="0.55" />
            </pattern>
          </defs>

          <rect className="story-loop-grid" x="28" y="24" width="464" height="294" />
          <g className="story-loop-legend" aria-hidden="true">
            <line x1="42" y1="344" x2="66" y2="344" className="story-loop-reference-key" />
            <text x="73" y="348">MAPBOX REFERENCE</text>
            <line x1="250" y1="344" x2="274" y2="344" className="story-loop-source-key" />
            <text x="281" y="348">SOURCE-DERIVED EDGE</text>
          </g>

          <path className="story-geometry-reference-fill" d={californiaPath} />
          <path className="story-geometry-reference" d={californiaPath} />

          <g className="story-geometry-candidate">
            <path className="story-geometry-source-fill" d={californiaPath} />
            <path className="story-geometry-source" d={californiaPath} />
            {markers.map(([cx, cy]) => (
              <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="3.2" />
            ))}
          </g>

          <g className="story-geometry-residuals" aria-hidden="true">
            <path d="M121 22l29 -14M322 66l22 4M309 126l26 10M401 278l18 12" />
            <path d="M391 337l12 8M294 316l14 16M198 210l-18 3M164 145l-20 -10" />
          </g>
          <g className="story-loop-pass" aria-hidden="true">
            <circle cx="427" cy="42" r="15" />
            <path d="m420 42 5 5 10-12" />
            <text x="450" y="46">GATE PASSED</text>
          </g>
        </svg>
        <LoopStage labels={["Candidate 01", "Candidate 02", "Candidate 03", "Best fit"]} />
      </div>
    </section>
  );
}

function SourceDataSubstrate() {
  return (
    <g className="story-data-source-substrate" transform="translate(-68 64) scale(.66)">
      <g clipPath="url(#source-data-clip)">
        <rect x="90" y="8" width="340" height="345" />
        <path d="M285 60C316 95 321 132 304 168S302 225 338 263" />
      </g>
      <path className="story-data-boundary" d={californiaPath} />
    </g>
  );
}

function ExtractedDataTarget() {
  return (
    <g
      className="story-data-extracted-target"
      transform="translate(171 64) scale(.66)"
    >
      <g clipPath="url(#extracted-data-clip)" shapeRendering="crispEdges">
        <rect
          className="story-data-target-paper"
          x="90"
          y="8"
          width="340"
          height="345"
        />
      </g>
    </g>
  );
}

function ExtractedDataFrame() {
  return (
    <g className="story-data-target-frame" transform="translate(171 64) scale(.66)">
      <g clipPath="url(#extracted-data-clip)" shapeRendering="crispEdges">
        <rect className="story-data-pixel-grid" x="88" y="8" width="342" height="345" />
      </g>
      <path className="story-data-boundary" d={californiaPath} />
    </g>
  );
}

function SourceCartography() {
  return (
    <g className="story-data-source-cartography" transform="translate(-68 64) scale(.66)">
      <g clipPath="url(#source-data-clip)">
        <path
          className="story-data-source-river"
          d="M285 60C316 95 321 132 304 168S302 225 338 263"
        />
        <g className="story-data-source-noise">
          <path d="M154 91h63M190 185h76M239 267h80" />
          <circle cx="216" cy="142" r="5" />
          <circle cx="305" cy="222" r="5" />
          <circle cx="262" cy="295" r="4" />
        </g>
      </g>
      <path className="story-data-boundary" d={californiaPath} />
    </g>
  );
}

function DataClassTransfers() {
  return (
    <g className="story-data-transfer-plane" transform="translate(-68 64) scale(.66)">
      <g
        className="story-data-transfer-class story-data-transfer-b"
        clipPath="url(#transfer-data-clip)"
      >
        <rect x="90" y="8" width="340" height="345" fill="#c9ba68" />
      </g>
      <g
        className="story-data-transfer-class story-data-transfer-a"
        clipPath="url(#transfer-data-clip)"
      >
        <ellipse cx="260" cy="106" rx="126" ry="72" fill="#6f9b72" />
        <path
          d="M166 70C220 115 212 170 278 205S342 280 370 334H314C280 286 246 260 226 221S187 140 142 112Z"
          fill="#789b4f"
        />
      </g>
      <g
        className="story-data-transfer-class story-data-transfer-c"
        clipPath="url(#transfer-data-clip)"
      >
        <ellipse cx="280" cy="246" rx="120" ry="78" fill="#c77951" />
      </g>
      <g
        className="story-data-transfer-class story-data-transfer-d"
        clipPath="url(#transfer-data-clip)"
      >
        <path
          d="M142 126C184 148 196 184 201 230S235 295 278 324H232C198 292 174 260 168 215S148 160 127 148Z"
          fill="#c58b91"
        />
      </g>
      <path className="story-data-lift-outline" d={californiaPath} />
    </g>
  );
}

function DataLoop() {
  return (
    <section
      className="story-loop-card story-loop-card-data"
      aria-labelledby="data-loop-title"
    >
      <header>
        <span>Loop 02 · Data</span>
        <h4 id="data-loop-title">Make the extraction match the source.</h4>
        <strong className="story-loop-formula">
          Extract → compare to source → diff → repair ↻
        </strong>
        <p>Classify, flip, diff, repair—and preserve the aligned geography.</p>
      </header>
      <div className="story-loop-canvas">
        <svg
          viewBox="0 0 520 370"
          role="img"
          aria-labelledby="data-animation-title data-animation-description"
        >
          <title id="data-animation-title">
            Legend classes lifting from an aligned source into an extracted map
          </title>
          <desc id="data-animation-description">
            The thematic pixels rise away from an aligned source map. Four legend
            classes then cross to a clean California map one at a time. The completed
            extraction is checked against the source before the data gate passes.
          </desc>
          <defs>
            <clipPath id="source-data-clip">
              <path d={californiaPath} />
            </clipPath>
            <clipPath id="extracted-data-clip">
              <path d={californiaPath} />
            </clipPath>
            <clipPath id="transfer-data-clip">
              <path d={californiaPath} />
            </clipPath>
            <pattern
              id="data-pixel-grid"
              width="12"
              height="12"
              patternUnits="userSpaceOnUse"
            >
              <path d="M12 0H0V12" fill="none" stroke="currentColor" strokeWidth="0.7" />
            </pattern>
          </defs>

          <text className="story-data-map-label" x="129" y="32">
            ALIGNED SOURCE
          </text>
          <text className="story-data-map-label" x="366" y="32">
            EXTRACTED MAP
          </text>
          <SourceDataSubstrate />
          <ExtractedDataTarget />

          <g className="story-data-transfer-guide" aria-hidden="true">
            <path d="M211 183h48" />
            <path d="m252 176 8 7-8 7" />
            <text x="235" y="166">CLASS BY CLASS</text>
          </g>
          <DataClassTransfers />
          <ExtractedDataFrame />
          <SourceCartography />

          <g className="story-data-legend" aria-hidden="true">
            <rect
              className="story-data-legend-a"
              x="104"
              y="337"
              width="10"
              height="10"
              fill="#6f9b72"
            />
            <text x="120" y="346">CLASS A</text>
            <rect
              className="story-data-legend-b"
              x="210"
              y="337"
              width="10"
              height="10"
              fill="#c9ba68"
            />
            <text x="226" y="346">CLASS B</text>
            <rect
              className="story-data-legend-c"
              x="316"
              y="337"
              width="10"
              height="10"
              fill="#c77951"
            />
            <text x="332" y="346">CLASS C</text>
            <rect
              className="story-data-legend-d"
              x="422"
              y="337"
              width="10"
              height="10"
              fill="#c58b91"
            />
            <text x="438" y="346">CLASS D</text>
          </g>
          <g className="story-loop-pass story-loop-pass-data" aria-hidden="true">
            <circle cx="427" cy="42" r="15" />
            <path d="m420 42 5 5 10-12" />
            <text x="450" y="46">GATE PASSED</text>
          </g>
        </svg>
        <LoopStage
          className="story-loop-stage-data"
          labels={["Aligned source", "Lift data", "Map classes", "Match verified"]}
        />
      </div>
    </section>
  );
}

export function ProcessLoopAnimation() {
  const [paused, setPaused] = useState(false);

  return (
    <figure
      className={`story-loop-animation${paused ? " is-paused" : ""}`}
      aria-label="Animated geometry and data comparison loops"
    >
      <div className="story-loop-toolbar">
        <button
          className="story-loop-motion-control"
          type="button"
          aria-pressed={paused}
          onClick={() => setPaused((current) => !current)}
        >
          <span aria-hidden="true">{paused ? "▶" : "Ⅱ"}</span>
          {paused ? "Resume motion" : "Pause motion"}
        </button>
      </div>
      <div className="story-loop-animation-grid">
        <GeometryLoop />
        <DataLoop />
      </div>
    </figure>
  );
}
