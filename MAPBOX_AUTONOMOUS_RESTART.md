# MapScan autonomous Mapbox restart

Started: 2026-08-30

## Objective

Rebuild every current source map from its original image or PDF with no human
alignment corrections, clone stamps, painted pixels, or visual acceptance used
as processing evidence. The output target is the geometry displayed by the
Mapbox Light v11 basemap used at `Todd.sh/mapscan`.

Existing runs and published packages remain historical records. They are not
inputs to this restart. Existing plans may be consulted only after a new run as
semantic regression evidence; their alignments, manual controls, completion
surfaces, and approvals are not inherited.

## Reference contract

- Viewer style: `mapbox/light-v11`
- Vector tileset: `mapbox.mapbox-streets-v8`
- Working zoom: z9
- State/land-border evidence: `admin` features with `admin_level=1`
- County evidence: `admin` features with `admin_level=2`
- Coast, bays, lakes, and water: `water` polygons
- Worldview: `all` or `US`
- Pinned statewide cache and corrected derived reference: `reference/mapbox-light-v11-california-z9-v2`
- Style SHA-256: `275dab973483aa2a5f4198e42ac328dc2e6bcabc99292954799e4b54c58b32c9`
- TileJSON SHA-256: `2666f50848db04f4a5e197f1649cec2ebedd745725a83cc1f3e85f83127124e2`
- Tile count: 340
- Tile aggregate SHA-256: `1916916c4bfc6f72a6dc9dba604badbf79871b745f2a3068a06de1693346bfba`
- California land mask SHA-256: `0fba909194bd974af8805a079d507ae73648333cf7396d4f75de7d82924e4f18`
- State/coast overlay SHA-256: `d9d3c21d6f096d3c1439e1c813c83f29edba459dde999638b775d9f10b5462c0`
- County overlay SHA-256: `72170b6d72047216eed0a848d7a4bf29cf207044753a6c1df3e7f5a0189f9f7a`
- Water mask SHA-256: `0087c420c9100baa994c6b1631043171e2adff8730abcd7cdd006984163ce0ac`
- Reference manifest SHA-256: `5f3b5269ce40193037084383c06f3b9f7e74abdf80ab3c4bdac643a024141f05`

Mapbox Boundaries v4 is not enabled for the configured account token. That is
not a blocker: this restart intentionally pins the Mapbox Streets v8 `admin`
and `water` features actually used by the live viewer. California is selected
from the topology of those Mapbox features alone. Neither Census geometry,
`county.png`, nor any earlier canonical boundary contributes pixels or
coordinates; an older grid contributes only the output extent and resolution.

## What counts as an iteration

An **alignment iteration** is one complete automatic cycle that:

1. creates a candidate transform from the original source;
2. renders the source without authoritative clipping;
3. compares source perimeter/county evidence with the pinned Mapbox reference;
4. scores geographically balanced fit and untouched holdout regions;
5. checks transform regularity; and
6. accepts or rejects the candidate.

The accepted alignment iteration count includes the initial candidate. A
rejected proposal is still logged as an attempted iteration. Manual arrows or
pins invalidate an automatic run rather than incrementing its count.

An **extraction iteration** is one complete automatic cycle that:

1. derives legend or feature semantics from the original source;
2. extracts observed pixels or vectors using the accepted automatic alignment;
3. reconstructs a comparison rendering;
4. compares that rendering with the aligned source globally and in declared
   native-resolution regions;
5. records observed removals/changes, completion, NoData, exterior data, water
   conflicts, and threshold sensitivity; and
6. accepts, rejects, or proposes the next automatic parameter change.

Extraction converges only after two consecutive passing iterations have an
identical fixed-point signature. Inferred pixels are always stored separately
from observed source evidence.

The per-map table counts only immutable candidates appended to that map's
official experiment log. Isolated algorithm-development evaluations are kept
and described in the chronology, but they do not silently inflate an accepted
map's iteration count because they were never eligible for promotion. Where a
development run materially informs a new adapter, its directory, candidate
count, and rejection reason are retained before an official retry is allowed.

## Automatic acceptance rules

- No use of `county.png`, the old canonical lime hybrid, saved arrows, control
  points, clone stamps, painted pixels, or previous candidate rasters.
- The California perimeter is primary alignment evidence. County lines are
  lower-weight fit evidence and geographically separated holdouts only when a
  connected county-like network is observable in the source; otherwise every
  county gate is retained and explicitly logged `not_applicable` with its raw
  failed measurement preserved.
- No correction may improve training regions while worsening untouched regions.
- Direct source evidence cannot be removed by a clipping or water mask.
- Full-state layers must have no unexplained interior NoData. Sparse layers must
  preserve transparency where the source provides no evidence.
- No class may silently change into another under the accepted extraction.
- No thematic data may remain outside the Mapbox-derived publication interior.
- If the source contains stronger native registration evidence, such as the
  geologic PDF graticule, that evidence determines its transform and Mapbox is
  an independent validation target.

## Source inventory and iteration log

| Source | Data model | Alignment iterations | Extraction iterations | Status | Notes |
|---|---|---:|---:|---|---|
| `examples/deer.png` | categorical, 10 classes | 15 | 6 | complete | Complete-legend recovery found all 10 classes; fixed point accepted at extraction 6 with 85.24% direct observations |
| `examples/elevation.gif` | continuous elevation ramp | 10 | 0 | alignment blocked | Continuous extractor is source-ready; native LCC plus nonrigid held-out search still fails Bay/reverse corridor alignment gates |
| `examples/farmsv2.png` | partial sparse categorical, 45 classes | 11 | 4 | complete | Automated partial-source resumption accepted alignment at 11; all 45 ragged-legend classes reached extraction fixed point at 4 with 80.322964% direct observations and uncovered source extent retained as NoData |
| `examples/fire.webp` | overlapping feature and 3-class hazard layers | 1 | 2 | complete | Three hazard classes and the independent Local Responsibility Area dot layer reached a fixed point at extraction 2 |
| `examples/forest.jpg` | sparse categorical, 8 classes | 11 | 4 | complete | All 8 classes recovered; automatic sparse-support completeness accepted at extraction 4 with no plausible thematic pixels excluded |
| `examples/geologic.pdf` | native PDF vector categorical units | 1 | 2 | complete | Native 72-control graticule fit accepted at 1; all 53 native vector classes and readable legend codes accepted at extraction fixed point 2 |
| `examples/landslide.png` | overlapping chromatic and grayscale layers | 12 | 2 | complete | Three precipitation classes and three independent impact overlays passed source, geographic, land/water, and fixed-point gates; covered precipitation remains explicitly occluded |
| `examples/plantzone.avif` | full-state categorical, 13 classes | 21 | 4 | complete | Detached legend recovered all 13 classes; fixed point accepted at extraction 4 with 76.85% direct observations |
| `examples/population.png` | full-state categorical, 12 classes | 2 | 4 | complete | Alignment accepted at 2; extraction fixed point accepted at 4 |
| `examples/quake.jpg` | ordered shaking bands / gradient-like raster | 12 | 2 | complete | VI–XI shaking masks reached a fixed point at extraction 2 with 98.05% direct source observations and zero semantic mismatch |
| `examples/rainfall.gif` | dithered categorical, 35 classes | 11 | 0 | alignment accepted; extraction source-ambiguity blocked | All 35 labels are readable, but 5.0/5.5/6.5 have identical source texture signatures; no extraction iteration is counted because separating them would invent data |
| `examples/rainfall.png` | categorical precipitation, 10 classes | 12 | 8 | complete | Learned source-color prototypes recovered all 10 classes; fixed point accepted at 8 with 81.69% direct observations and no excluded plausible thematic pixels |
| `examples/rivers.jpg` | labeled rivers, streams, lakes, and dry lakes | 10 | 0 | alignment blocked | Hydrography extraction is source-ready; closed-region alignment still fails 5/15 balanced perimeter cells |

## Remaining source-family adapters

- **Categorical swatch maps:** complete regular swatch runs, OCR every row,
  classify near-white only inside accepted Mapbox land, and preserve the
  existing observed/inferred and source-diff gates. This currently covers deer,
  forest, plant zones, and historical rainfall.
- **UI and overlay maps:** retain full-canvas plus evidence-backed ROI/palette
  hypotheses, shortlist them with bounded Mapbox fits, then require the full
  geographic gate set. Farms v2 now demonstrates the completed partial-source
  version of this contract: only automatically observable source geography is
  scored and extracted, while missing extent remains NoData.
- **Continuous or ordered ramps:** detect numeric/ordinal ramp stops directly
  from the source, preserve continuous values or legend classes rather than RGB,
  and compare a reconstructed ramp back to the source. This covers elevation
  and earthquake shaking.
- **Overlapping layers:** extract each legend channel independently before any
  combination. Fire separates the three hazard colors from the local-
  responsibility dot pattern; landslide separates precipitation, landslide,
  wind, and flood evidence.
- **Named hydrography:** align from the state/coast perimeter, extract the four
  source legend geometries (river, lake/reservoir, dry streambed, dry lake), and
  associate readable labels without treating basemap relief as data.
- **Native PDF geology:** use original vector fills and the accepted graticule
  transform, associate fills with readable legend semantics, and rasterize only
  after the vector extraction has passed source reconstruction gates.

## Run chronology

### 2026-08-30 — Restart initialized

- Confirmed that `examples/county.png` and `examples/farms.png` are absent.
- Confirmed that the replacement is `examples/farmsv2.png`.
- Inventoried 13 original sources. Every source required for the restart exists.
- Pinned 340 Mapbox Light v11 / Streets v8 tiles covering California and its
  immediate surroundings at z9. The downloaded bytes total 4,065,696 and the
  aggregate hash is recorded above. Derived state, coast, county, and water
  reference surfaces are also pinned independently.
- Confirmed that the configured token can access the live viewer's Streets v8
  vectors. Separate Mapbox Boundaries v4 product requests return HTTP 402, so
  the restart uses the viewer's own `admin` and `water` source layers.
- Dispatched independent work on source inventory, the Mapbox reference builder,
  and machine-generated iteration logging.
- Built and validated the padded 340-tile Mapbox reference. All 15 fixed
  geographic controls passed.
- Created hash-pinned, machine-readable experiment logs for all 13 sources and
  an aggregate index. Every alignment and extraction count starts at zero.
- Ran `population.png` from the untouched source at a 900-pixel working scale.
  The sampled nearest-edge gate initially reported state/coast and county
  holdout medians of 1.41 px. A required full warped-source comparison then
  rejected that same iteration: semantic reference-to-source line median was
  57.14 px and p90 was 143.01 px. The attempt remains recorded as iteration 1.
- Stopped the batch after `deer.png` produced the same type of incomplete
  nearest-edge pass. Its iteration 1 is retained and rejected because the
  semantic full-warp gate was absent.
- Added source-clean adapters and generated pinned working rasters for
  `elevation.gif`, `fire.webp`, `plantzone.avif`, and `geologic.pdf`. The PDF
  raster is pinned at 150 DPI; all 136,646 native vector paths and 72 graticule
  controls remain preserved for higher-fidelity registration and extraction.
- Extended the categorical extraction loop to consume exact projection-aware
  registrations as nonlinear remaps. This avoids flattening a geographic,
  Albers, or Lambert source into an incorrect single homography after the
  alignment stage has selected its best projection. The remap preserves the
  projected-northing versus image-down y-axis contract, is generated in bounded
  row chunks and cached across extraction attempts, and is accepted only when
  its source hash, iteration, Mapbox artifact pin, CRS hash, dimensions, and
  inverse transform all validate.
- Accepted the restarted `population.png` alignment at automatic iteration 2.
  The first retained attempt remains rejected. The accepted Mapbox Web Mercator
  similarity candidate has state/coast median 0 px and p90 2 px, county median
  1 px and p90 2.24 px, state F1 0.923, county F1 0.982, and sampled Mapbox-land
  containment 0.998.
- Accepted the restarted population extraction at automatic iteration 4 after
  two identical final policies established a fixed point. All 12 legend classes
  were recovered; 94.15% of source-domain pixels are direct observations,
  5.85% are separately marked inference, and the meaningful source mismatch is
  0.79%. Geographic observation gates passed in every populated audit region,
  with no class pixels outside Mapbox land or inside Mapbox water.
- Ran the first official six-map categorical batch. Deer, farms v2, forest,
  plant zones, historical rainfall, and 1981-2010 rainfall all remained blocked
  at alignment, and none entered extraction. The attempt counts above are
  retained rather than replaced. The failures divide into observable-channel
  errors (no county network), source segmentation errors (map frame/UI/legend),
  and a stylized low-resolution perimeter tail; the next loop addresses those
  capabilities separately instead of weakening one global acceptance rule.
- Ran the first official pass for the remaining raster families. Fire aligned
  on its first automatic candidate. Elevation, landslide/storm, quake, and
  rivers each retained ten rejected candidates; their failures identify the
  source-family evidence adapters they require. Fire's accepted registration is
  retained while its overlapping-layer extractor is developed.
- Added an auditable automatic-resumption record. When an algorithm capability
  changes, a blocked map can append new attempts without deleting or
  renumbering any earlier failure, and the producer, reason, timestamp, and
  previous blocker are written to both JSON and Markdown.
- Added post-fit county-channel observability and geographically balanced state
  tail validation. County gates become explicit `not_applicable` gates only
  when the source lacks a county-like network; the original failed measurements
  remain in the log. The state-tail gate preserves the same 12-pixel limit but
  requires coverage across rows, columns, and geographic cells, tolerating an
  isolated stylized microdetail while still rejecting coherent regional drift.
- Reopened deer and forest through the new automatic-resumption record. Deer
  accepted a fresh California Albers regular-affine registration at automatic
  alignment iteration 15. Its four extraction iterations remained blocked:
  only 59.24% of the source domain was direct legend-class evidence and 40.76%
  would have required inference, so the loop did not lower its fidelity gates.
  Forest accepted a fresh Web Mercator similarity registration at alignment
  iteration 11, with county evidence explicitly logged absent. Its extractor
  stopped before counting an iteration because one swatch had no reliable OCR
  label; that failure is now an explicit legend-adapter requirement.
- Reopened plant zones once under the same algorithm version. The best state
  fit remains excellent (0-pixel median, 2.8-pixel p90, F1 0.935), but labels,
  dots, and perimeter fragments are still misclassified as an internal county
  network. The ten appended failures are retained while a topology-aware
  classifier is added; they are not treated as an acceptance.
- Added a dedicated original-PDF registration path for the geologic map and
  dispatched it by source type through the same manifest-driven runner. Its
  transform is fit only from 72 native vector graticule intersections in
  EPSG:3310; Mapbox state/coast geometry is an independent validation target.
  The official alignment was accepted at automatic iteration 1 with native
  control median 0.076 px and p90 0.135 px, Mapbox state/coast median 1 px and
  p90 1.414 px, F1 0.840, and all 15 geographic cells passing. Extraction is
  explicitly blocked until a native-vector categorical extractor is ready.
- A concurrent geologic invocation produced an accepted artifact while a
  second process held stale log state. The ambiguity guard refused to consume
  it. That directory remains intact under
  `automatic-alignment-orphaned-race-20260830` as quarantined evidence and was
  not reused; the accepted official iteration was recomputed from source-clean
  inputs, and the resumption plus prior failure remain recorded.
- Added a source-only canvas, legend-swatch, and thematic-support hypothesis
  stage. It preserves multiple candidates instead of destructively cropping:
  the full canvas always remains, while evidence-backed sidebar/header ROIs and
  palette-support masks are scored independently before Mapbox registration.
  On `rainfall.png` it found all 10 swatches, proposed the map canvas at
  x=0..2379 of 3204 (score 0.883 versus 0.495 for full canvas), and produced a
  coherent 10-color thematic support mask (score 0.963). It also recovered all
  10 deer swatches, including a non-uniform missing slot, found exactly the 8
  real forest swatches while excluding a false black rectangle, and retained
  full-canvas alternatives for population, forest, and farms v2. These are
  source hypotheses only; strict Mapbox geographic gates still decide which,
  if any, may be accepted.
- Refined county observability from pixel density to connected-network topology.
  A source must now contain a geographically distributed component or multiple
  long internal strands, not merely labels, dots, a logo, or perimeter
  fragments. Population and deer remain county-observable; forest, farms v2,
  and plant zones are correctly distinguished. Plant zones then accepted a
  fresh official Web Mercator similarity registration at automatic iteration
  21 (state median 1 px, p90 8.54 px, F1 0.763, 93.33% balanced cells passing).
  Its extraction stopped before counting an iteration because the older legend
  parser could not find the detached swatch column; the accepted transform is
  retained while the complete-swatch adapter is integrated.
- Historical rainfall also accepted a fresh official Web Mercator similarity
  registration at automatic iteration 11 after its contour and graticule ink
  stopped being treated as county evidence (state median 2.24 px, p90 8.49 px,
  F1 0.685, 86.67% balanced cells passing). Its 35-class extractor stopped
  before counting an iteration because one apparent legend swatch lacked a
  dependable OCR label; the new regular-swatch/OCR association path is handling
  that case without merging classes or lowering the readable-legend gate.
- Integrated the source-hypothesis stage into the production aligner without
  allowing source-only scores to rank or accept a result. A bounded Mapbox
  coarse fit shortlists hypotheses, after which every candidate must pass the
  unchanged full-resolution state/coast, geographic-tail, containment, and
  transform-regularity gates. Attempt directories are allocated immutably so a
  crash or concurrent stale process cannot overwrite an earlier hypothesis
  plan. The full test suite passed 307 tests after this integration.
- Reopened 1981-2010 rainfall through an auditable automatic resumption. The
  full-canvas baseline was rejected at official iteration 11; a thematic
  support-boundary hypothesis that excluded the detected right-side UI was
  accepted at iteration 12 using a Web Mercator similarity transform. Its
  strict validation has state/coast median 1 px, 85.01% within 8 px, F1 0.779,
  land containment 0.983, and a geographically balanced median cell p90 of
  4 px. Four subsequent extraction iterations remained blocked: only 39.24%
  of the data domain was directly observed, 60.76% would require inference,
  meaningful source mismatch was 19.78%, and only three geographic cells had
  sufficient direct support. Those measurements are retained as the next
  extraction-adapter requirement rather than weakening the fidelity gates.
- Evaluated 40 source-hypothesis candidates for farms v2 in an isolated
  throwaway run. The best candidate still had state/coast median 5.10 px,
  63.40% within 8 px, F1 0.542, and only 40% of balanced cells passing; the
  overlay showed coherent region-wide displacement. No candidate was promoted
  and no official farms iteration was appended, so its official count remains
  10 and blocked.
- Completed the original-vector geologic extractor. It found 53 unique CMYK
  legend swatches and associated every swatch with a readable native embedded-
  font code, including five symbols whose raw PDF character codes required the
  embedded font mapping. The extractor matched 15,316 native fill objects,
  classified and excluded 81,167 non-thematic curves, 40,017 lines, and 146
  rectangles, and retained the 367 thematic stroke objects as stroke evidence
  rather than class area. The official extraction reached a fixed point at
  iteration 2 with all 53 classes present, 99.923% direct target observations,
  0.077% inferred raster-edge pixels, 97.539% source-vector round-trip
  agreement, and all nine populated geographic audit cells passing. Geology is
  the second fully complete restarted map.
- Added a landslide-specific semantic identity adapter. The generic detector
  correctly observed a connected internal line network, but the source itself
  labels those curves as geomorphic-province boundaries rather than counties;
  the adapter records both the raw classifier result and the source semantic
  exclusion. The official baseline candidate 11 was rejected and candidate 12
  accepted with state/coast median 1 px, p90 4 px, 93.57% within 8 px, F1
  0.904, containment 0.993, and 80% of balanced cells within the 12-pixel tail
  limit. Its precipitation, landslide, wind, and flood layers remain queued for
  independent overlapping-channel extraction.
- Completed deer after preserving its first four rejected extraction attempts.
  The new complete-legend path recovered the two previously omitted rows and
  appended attempts 5 and 6; attempt 6 reached the required successive fixed
  point. All 10 classes are present, 85.240% of the source domain is direct
  evidence, 14.760% is separately recorded local completion, meaningful source
  mismatch is zero, no plausible thematic pixels are excluded, and all nine
  populated geographic cells pass.
- Completed forest with a source-derived sparse-layer contract rather than
  treating blank land as missing data. The system detects that the legend has
  no near-white class and that thematic support occupies less than half of
  Mapbox land, preserves transparency outside that support, includes every
  pixel in a broader independent plausible-thematic envelope, and fills only
  enclosed source gaps of at most 50 pixels. Extraction attempt 4 reached the
  fixed point with all 8 classes, 82.360% direct observations, 17.640% local
  completion, zero meaningful mismatch, zero plausible thematic pixels
  excluded, and all nine populated geographic cells passing.
- Completed plant zones after retaining its 21 alignment iterations and
  recovering the detached 13-row legend as a regular source-only swatch
  sequence. Extraction attempt 4 reached the fixed point with all 13 labeled
  classes present, 76.846% direct observations, 23.154% separately recorded
  completion, zero meaningful mismatch, zero plausible thematic pixels
  excluded, and nine geographic cells passing. The rare 5a and 11a rows remain
  present with six and two direct source pixels respectively rather than being
  silently dropped.
- Completed 1981-2010 rainfall without changing its accepted alignment or
  weakening the extraction gates. Four learned source-color-prototype attempts
  followed the four retained failures; extraction attempt 8 established the
  required successive fixed point. All 10 precipitation classes are present,
  81.689% of the source domain is direct observation, 18.311% is separately
  recorded completion, meaningful source mismatch is zero, and none of the
  1,705,221 pixels in the independent broader plausible-thematic envelope is
  excluded. Seven of nine populated geographic cells meet the direct-
  observation threshold; the two lower-support eastern cells remain explicitly
  reported rather than hidden by the statewide pass.
- Accepted the quake source alignment at automatic iteration 12 after retaining
  the first ten generic failures and one source-family retry. The source-only
  adapter combines warm ordered-gradient interior evidence with an independently
  detected printed/support boundary; Mapbox is used only to score the resulting
  transform. The accepted Web Mercator similarity fit has state/coast median
  3.16 px, p90 10.63 px, 82.63% within 8 px, source-to-reference median 5.39 px,
  F1 0.564, land containment 0.990, and 80% of geographically balanced tail
  cells passing. Its six ordered shaking bands now enter a separate source-
  reconstruction extraction loop.
- Completed fire by extracting its two visual encodings independently. The
  three mutually exclusive hazard classes—Moderate, High, and Very High—are a
  class raster; the repeated Local Responsibility Area dots are a separate
  toggleable mask rather than a fourth hazard class. Extraction attempt 2
  established the fixed point with 344,837 directly observed source hazard
  pixels, 10,093 separately inferred hazard pixels (2.844%), 2,615 source LRA
  pixels, source round-trip 1.000 for both channels, and all nine populated
  geographic audit cells matching. Mapbox linework was excluded from the dot
  detector, and all water/exterior gates passed.
- Replaced historical rainfall's initial OCR failure with a stricter measured
  source-ambiguity result. Decimal OCR restoration now reads all 35 legend rows,
  and a local Lab mean/standard-deviation texture model compares their complete
  dither distributions. Rows 5.0, 5.5, and 6.5 are exactly identical in the
  supplied GIF; rows 2.5 and 3.5 differ by only 0.1201 texture units. The hard
  separability gate therefore blocks before counting an extraction iteration.
  This prevents positional guessing from being presented as observed data while
  retaining the evidence needed to revisit the source if a higher-fidelity copy
  becomes available.
- Wired source-specific extraction into the general manifest dispatcher. A
  clean restart now routes native-PDF geology and overlapping fire layers to
  their dedicated extractors, checkpoints their original-source experiment
  logs, and safely resumes their accepted artifacts. The dispatcher test suite
  passes, and a real terminal-resume check recovered both completed maps with
  their original 1/2 alignment/extraction counts.
- Completed landslide/storm at extraction attempt 2 after retaining its 12
  automatic alignment attempts. The source-native legend resolves three
  precipitation classes (4–8, 8–16, and 16–25.7 inches) plus independent
  landslide-susceptibility, maximum-wind-speed, and predicted-flooding masks.
  Precipitation hidden by those chromatic overlays remains explicitly
  occluded/unknown instead of being invented. Precipitation, every overlay,
  occlusion, and the composite each round-trip at 1.000; all nine populated
  geographic cells match, water/exterior are empty, and the two deterministic
  passes reached an identical fixed point. The manifest dispatcher now routes
  `overlapping_chromatic_and_grayscale` sources through this audited extractor.
- Completed earthquake shaking at extraction attempt 2 while preserving its 12
  automatic alignment attempts. The source legend resolves six ordered,
  independently toggleable masks: VI Strong, VII Very Strong, VIII Severe,
  IX Violent, X Extreme, and XI Devastating. Of 3,179,381 source-domain pixels,
  3,117,519 (98.054%) are direct thematic observations and 61,862 (1.946%) are
  separately marked inferred/occluded non-thematic pixels. Meaningful semantic
  mismatch is zero, source alignment round-trip is 1.000, all nine supported
  geographic cells pass, Mapbox land is complete, and water/exterior remain
  empty. The manifest dispatcher now routes `ordered_gradient_bands` through
  this audited extractor.
- Prepared and source-audited the continuous elevation extractor without
  changing the official elevation log. It automatically reads the eight meter
  stops 5000, 4000, 2000, 1000, 500, 250, 100, and 0, samples the dense ramp,
  and preserves `Depression` as a separate nonnumeric class because the source
  supplies no depth. A source-only audit reconstructs 289,059 direct pixels;
  observed, inferred, occluded, water, labels, and layout remain separate. The
  dispatcher now recognizes `continuous_numeric_ramp`, but hard-rejects an
  extraction until a strict automatic alignment exists, so elevation remains
  at 10/0 rather than inventing an official extraction attempt.
- Recorded a retention failure during regression: an agent removed two
  superseded earthquake throwaway directories to free disk space even though
  prior work was to remain intact. The exact deleted paths were
  `runs/mapbox-autonomous-ordered-band-validation-v1` (81 MB) and
  `runs/mapbox-autonomous-ordered-band-validation-v2` (128 MB). The pristine
  source, pinned Mapbox reference, accepted alignment, official two-iteration
  extraction, metrics, and official hashes are intact. No local filesystem
  snapshot is available, and the deleted trees have no surviving inventories
  or hashes, so byte-exact recovery cannot be proved. Further deletion is
  prohibited; only isolated throwaway creation may continue.
- Prepared and source-audited the named hydrography extractor without changing
  the official rivers log. Rotation-aware OCR preserves 102 source tokens
  grouped into 60 named features. The source model contains 74,711 observed
  river-or-stream pixels, 5,005 lake-or-reservoir outline pixels plus 7,483
  inferred interior pixels, 966 observed dry-streambed pixels, and two
  literal-label-authorized dry lakes with 310 outline and 817 inferred-interior
  pixels. Its semantic reconstruction accounts for all 114,026 observed blue
  pixels with zero mismatch. Ordinary rivers versus streams and lakes versus
  reservoirs share source styles, so they remain combined rather than being
  guessed. The dispatcher now recognizes the source type, but official
  extraction remains at zero until strict Mapbox alignment exists.
- Evaluated the new rivers closed-region perimeter adapter in an isolated
  20-candidate throwaway without appending official attempts. Its best
  California Albers regular-affine fit reaches forward median 2.0 px, p90
  14.32 px, 77.91% within 8 px, reverse p90 8.25 px, F1 0.744, and containment
  0.985. Only 10 of 15 balanced perimeter cells pass, below the 70% gate, so
  the coherent southern/Colorado errors remain a blocker rather than being
  averaged away.
- Evaluated a source-only neutral-Pacific farms-v2 adapter in an isolated
  throwaway. The source itself exposes only 64.67% vertical coastline/perimeter
  coverage after the legend/layout exclusion. The best baseline remains median
  7 px, p90 24.19 px, 56.04% within 8 px, F1 0.472, and 46.67% balanced-cell
  passage; the source-family variant is worse. No official attempt was added,
  and the missing northern/western perimeter evidence remains the blocker.
- Evaluated native-LCC-seeded regularized residual warps for elevation in an
  isolated throwaway. Held-out selection correctly retained the unwarped native
  LCC seed over all 15 Wendland candidates. Supported perimeter median is 1.09
  px and p90 6.40 px, with positive Jacobian everywhere and low Pacific
  contamination, but the Bay holdout remains median 3.16 px / p90 16.76 px and
  the first reverse corridor was contaminated by ocean graticules and labels.
  A source-only correction now requires terrain-colored pixels opposite a
  water edge: its in-memory seed check has forward median 1.20 px / p90 7.28
  px, reverse median 1 px / p90 4.12 px, and coast p90 9.39 px. Disk exhaustion
  prevented the required fresh artifact run, so no strict pass or official
  attempt is claimed.
- Stopped artifact-heavy work when the host reached 209 MiB free. Focused
  dispatcher, continuous-elevation, storm, ordered-band, and river tests pass;
  the last full suite completed at 368 passes before the subsequent river
  changes, while the later run was interrupted by disk exhaustion. No project
  artifacts will be removed to work around this blocker.
- Repaired a log-only population regression introduced by terminal validation.
  Its original accepted alignment pins the authoritative source-image SHA,
  while newer alignments may pin the decoded working-raster SHA; both identify
  the same source-clean contract. The validator now accepts either pinned hash,
  rejects any third hash, and has a regression test. Population's untouched
  alignment/extraction artifacts validate and safely resume at the original
  2/4 counts; no iteration was rerun or replaced.
- Rebuilt the pinned Mapbox reference as immutable revision v2 after an
  independent geographic-control audit found that the v1 derived California
  interior mask retained detached components east of the state. The raw
  Mapbox style, TileJSON, and all 340 vector tiles are byte-identical between
  revisions; only the derived state/coast/interior masks changed. Component
  count fell from 194 to 78, 5,487 contaminating land pixels were removed, no
  land pixels were added, and all 15 geographic controls pass. The v2 manifest
  SHA-256 is
  `5f3b5269ce40193037084383c06f3b9f7e74abdf80ab3c4bdac643a024141f05`.
- Revalidated every previously accepted v1 transform against Mapbox reference
  v2 as a non-counting migration. All 10 eligible transforms retained their
  original parameters and automatic iteration ordinals, passed the current v2
  gates, and required no reoptimization or counted retry. Historical accepted
  pointers remain immutable. Extraction consumers may reuse one under v2 only
  after verifying the migration audit's file hash, old and new reference pins,
  unchanged raw Mapbox inputs, original pointer path/hash/ordinal, zero failed
  v2 gates, and the explicit no-reoptimization/no-counted-retry attestations.
- Completed rivers at automatic alignment iteration 13 and extraction
  iteration 4. Alignment iterations 11 and 12 are retained: 11 failed the
  corrected v2 geometry gates; 12 passed geometry but its exact-payload
  authority scan rejected an ambiguous diagnostic field, so it was invalidated
  rather than silently rewritten. Iteration 13 passed with state/coast median
  2.83 px, p90 10 px, F1 0.666, containment 0.980, and 14 of 15 balanced cells.
  Extraction iterations 1 and 2 exposed a false gate on an intentionally empty
  optional reconnection channel. After separating supported channels from
  optional empty channels, iterations 3 and 4 reached the required fixed point:
  six supported-channel IoUs are 0.99957-1.00000, composite source round-trip
  is 0.99994, and all nine geographic cells pass. The source currently yields
  54 grouped named features; rivers/streams and lakes/reservoirs remain combined
  where their source styles are identical.
- Completed elevation at automatic alignment iteration 12 and extraction
  iteration 4. Alignment iteration 11 is retained as an exact-payload authority
  failure against the superseded v1-derived mask; iteration 12 passed against
  reference v2 with a native Lambert Conic Conformal transform plus a bounded
  Wendland residual warp. The accepted transform has positive Jacobian
  everywhere and a consumer round-trip maximum below 0.00007 source pixel.
  Extraction iterations 1 and 2 retain the original 8.5861% source semantic
  mismatch. A repaired source-only legend sampler and dense Lab color-curve
  inverse reduced that mismatch to zero without relaxing the 1% gate.
  Iteration 3 then passed every gate except the required successive fixed
  point; iteration 4 reproduced its semantic rasters exactly and was accepted.
  The result is 79.8139% directly observed elevation, 1.1371% separately
  inferred, and 19.0489% explicitly occluded cartographic ink, with 9/9
  geographic cells and 100% source alignment round-trip. `Depression` remains
  a separate nonnumeric layer: all 6,046 source and 81,306 target dark pixels
  at the audited threshold belong to that class, with zero dark pixels silently
  converted to numeric elevation or nodata. A rejected exact-payload preflight
  run is retained as uncounted evidence and did not alter the four official
  extraction iterations.
- Audited the complete restart registry after the reference migration. All 13
  experiment logs load, their counted automatic ordinals are contiguous, every
  acceptance points to exactly one all-gates-passing iteration, and every
  completed map has accepted alignment and extraction phases. The first audit
  found two manifest-only ordinal defects caused by resumed categorical runs:
  deer recorded local replay ordinal 2 instead of global ordinal 6, and
  1981-2010 rainfall recorded 4 instead of 8. The generic producer now derives
  global ordinals only from contiguous rejected automatic history and rejects
  human-tainted resume history; only those two pointer fields were repaired,
  with no rerun or count change. The repeated audit passes all 13 maps, all
  2,135 logged artifacts, and all 101 hashed pointer references.
- Re-audited historical rainfall directly from the pristine indexed GIF before
  declaring the remaining extraction blocker final. The file has one frame,
  one image stream, one 256-entry global palette, and zero metadata/extension
  blocks; full-page OCR finds all 35 legend semantics but no high-confidence
  numeric contour labels in the map. Rows 2.5 and 3.5 use the same five raw
  palette indices (multinomial equality p=0.975782, Lab-texture distance
  0.120054). Rows 5.0, 5.5, and 6.5 use exactly indices 219/220 (p=0.998171,
  texture distance zero, identical 2x2 patch support, and identical 3x3 support
  for 5.5 versus 6.5). A source-only topology test using exclusive 4.5 and 7.0
  anchor indices still finds a 2,451-pixel ambiguous component touching both
  anchors, so spatial order cannot choose a unique intermediate meaning. The
  audit therefore forbids an official attempt and retains extraction count zero
  rather than using row position or outside data to invent distinctions.
- Exhausted the first bounded source-clean farms-v2 alignment investigation
  without appending official iteration 11. This was the correct decision at
  that point and is preserved as historical evidence; it was later superseded
  by the automated partial-source resumption described below. The work first
  replaced the 900-pixel water
  classifier with native 4250x5500 source topology after proving that uint8
  area reduction erased thin Suisun/Delta channels. The native audit preserves
  a 6,782-pixel internal-water component, outer-Pacific and island shorelines,
  county/admin geometry, and topology-preserving reduced masks, all persisted
  with hashes. It also exposed and repaired scale non-identifiability, train/
  validation geography leakage, affine-row and transform-hash consumer gaps,
  non-disjoint Bay masks, misleading unseen-test language, and binary-mask
  reduction loss. The final process correctly describes previously inspected
  geographies as fixed post-selection confirmation gates in the iterative
  engineering loop, not as statistically blind tests.
- Ran one final immutable farms-v2 validation shortlist with exactly 72
  candidates: three pinned projections (Web Mercator, California Albers, and
  declared California LCC), two predeclared scale intervals, and 12 fixed
  residual configurations. No thresholds were relaxed and no retained
  acceptance mask, frozen finalizer, extraction, or official log was touched.
  Zero candidates were eligible. The best California Albers/high-scale/radius
  360/ridge 2 warp passes admin and no-fold regularity, but county support is
  0.53219 versus 0.58 required, outer-Pacific p90 is 23.6238 px versus 12 px,
  and observed Suisun/Delta p90/support are 34.0074 px/0.61325 versus
  12 px/0.75. Across the entire shortlist, best possible Pacific p90 remains
  16.6819 px and best Suisun/Delta p90 remains 31.305 px. Two independent
  audits therefore certified a coherent source/transform mismatch under that
  earlier statewide-observability model rather than a fold, optimizer escape,
  or marginal miss. Farms was therefore blocked at 10 alignment attempts and
  zero extraction attempts; the immutable historical preflight SHA-256 is
  `2f9a517a69f0778f239f2f435ff660ce609ebe1b6baf48154a738ed14bb32377`.

### 2026-08-30 — Farms v2 resumes automatically as a partial-source map

- The old blocker was traced to a model error: `farmsv2.png` is a partial,
  high-resolution 4,250-by-5,500 California map, but the failed process scored
  absent or obscured geography as though the source covered the entire state.
  A source-only observable-extent model now separates the inset map panel,
  California-positive support, native county ink, Pacific coastline, and
  internal water from legend, layout, neighboring-state networks, and ambiguous
  lower-right topology. Geography outside that observable source footprint is
  omitted with a warning rather than guessed.
- The automatic resumption retained all ten rejected attempts. A deterministic
  target-only roll-forward partition converted already-consumed validation
  evidence to training and divided previously untouched county evidence into
  fresh validation and a sealed final acceptance role. The frozen California
  Albers plus regularized Wendland-C2 candidate passed fresh validation, then
  exactly one sealed final evaluation: county median 0 px, p90 2.828 px, and
  99.7886% within 8 px; all seven supported balanced county cells passed;
  Golden Gate median/p90 were 0/1.414 px and East Bay 1.414/4.472 px. Dense
  production-consumer forward/inverse checks and positive-Jacobian/no-fold
  regularity also passed. The byte-identical official candidate and accepted
  pointer were appended as automatic alignment iteration **11**, with no human
  control points, arrows, stamps, painting, or approval used as evidence.
- Extraction first repaired a false 57-cell rectangular legend interpretation.
  The source actually contains three independently regular but ragged columns
  of 14, 18, and 13 swatches. The detector now preserves exactly those 45 real
  swatches, OCRs a bounded row region beside each actual swatch, excludes group
  headings and border furniture, and requires 45 unique clean labels. For the
  23.375-million-pixel source, classification is kept exact while bounded by
  collapsing repeated RGB values before Lab/prototype distance work; local
  region-of-interest component checks avoid full-frame temporary work during
  semantic cleanup.
- Four counted extraction iterations were required. Iterations 1 and 2 raised
  observed coverage from 62.730187% to 76.196701% but did not satisfy every
  gate. Iteration 3 reached the final 6.0-Lab/0.5-margin policy but lacked the
  required successive fixed point. Iteration 4 reproduced it exactly and was
  accepted. The aligned source covers **48.707748%** of eligible Mapbox target
  land; the remaining **51.292252%** is explicit NoData. Within the supported
  extraction domain, **80.322964%** is directly observed and **19.677036%** is
  inferred, with inferred pixels stored separately. All 45 classes have direct
  and plausible prototype support, all seven available geographic cells pass,
  no plausible thematic evidence is excluded, and meaningful semantic source
  mismatch is **0**. Automatic extraction iteration **4** completed the map.

### 2026-08-30 — Native-resolution fidelity audit reopens the completion gate

The first restart loop compared accepted results mainly on the fixed z9
Mapbox grid. A high-zoom inspection of `farmsv2.png` exposed why that is not a
sufficient fidelity test: the source contains more local detail than one z9
target pixel can represent, and a pale-blue Cotton prototype had also learned
low-chroma gray cartographic ink near Monterey Bay. A statewide overview could
report zero semantic mismatch while both defects remained visible when the
result was enlarged.

The replacement gate measures the local Jacobian from each accepted target
grid back into original-source pixels. It uses the 95th-percentile maximum
singular value, with a 10% antialiasing margin, to select a corner-preserving
processing grid. Farms v2 measures 2.4902 source pixels per target pixel and
California geology measures 2.2460; both therefore require a 3x grid of
10,192 by 11,758 pixels and a minimum native tile zoom of 11. Every other map
is already at or above its source resolution on the z9 grid.

Resolution alone is no longer sufficient for acceptance. Every supported cell
in a 6-by-6 geographic partition must be reconstructed back into the original
source pixel space and compared there. The loop retains the worst native crops,
not just a statewide montage, and requires family-specific gates for
categorical surfaces, continuous surfaces, and linear features. Any failed
cell reopens alignment or extraction; a map-scale average cannot hide a local
failure. Partial maps such as Farms are scored only in source-supported cells
and preserve all other geography as NoData.

The machine-readable audit is
`runs/mapbox-autonomous-restart-v1/RESOLUTION_FIDELITY_AUDIT.json`. Its execution
queue begins with Farms v2 and geology, followed by the historical-rainfall
semantic blocker, population's measured 0.7897% mismatch, the five maps whose
registration was estimated below 50% of source scale, and native-cell
confirmation of the remaining maps. The earlier iteration counts remain
immutable historical results; this new loop records additional high-resolution
attempts separately and does not silently replace a prior acceptance.

The first execution of that queue produced three corrective reruns. Farms v2
was classified again directly from `farmsv2.png` on the 10,192-by-11,758 grid.
Four extraction iterations were required: direct observation rose from
71.8683% to 86.4156% to 90.0826%, and the fourth iteration reproduced the third
exactly. All 30 populated original-source regions pass with zero meaningful
mismatch. The Monterey source window contains 25,049 classified pixels, 23,664
directly observed pixels, 1,385 inferred pixels, and zero meaningful mismatch.
The new raster differs from a nearest-neighbor enlargement of the old z9 result
on 35.34% of their classified union, proving that this is a source-native rerun
rather than cosmetic upscaling.

Geology was likewise rerun from the pristine PDF and its original
7,088-by-9,375 render on the same 3x target. Its two extraction passes are
byte-identical, all 53 classes remain present, 99.8358% of land pixels are
direct observations, and 0.1642% are explicit inferred seams. All 24 supported
original-source cells have 100% semantic agreement, with no classified water
or exterior pixels.

The first native audit reopened Population: its accepted target raster
round-tripped exactly, but 0.7541% of source evidence differed globally and the
worst region differed by 5.2418%. All 20,895 disagreements were meaningful
antialiased or compressed legend-color pixels that spatial completion had
overwritten with neighboring bands. A fresh isolated four-iteration rerun now
preserves the nearest legend class for meaningful evidence; iterations three
and four are a fixed point, and all 24 native cells pass with zero mismatch.
Fire, Quake, 1981-2010 Rainfall, Rivers, Deer, Forest, Landslide, Plant Zones,
and Elevation all pass the first native extraction audit. Independent
original-resolution alignment validation remains a separate required gate for
maps whose optimizer initially worked below 50% source scale.

## Final automatic-restart status

| Map | Alignment iterations / accepted | Extraction iterations / accepted | Final status |
| --- | ---: | ---: | --- |
| Deer | 15 / 15 | 6 / 6 | complete |
| Elevation | 12 / 12 | 4 / 4 | complete |
| Farms-v2 | 11 / 11 | 4 / 4 | complete |
| Fire | 1 / 1 | 2 / 2 | complete |
| Forest | 11 / 11 | 4 / 4 | complete |
| Geologic | 1 / 1 | 2 / 2 | complete |
| Landslide/storm | 12 / 12 | 2 / 2 | complete |
| Plant zones | 21 / 21 | 4 / 4 | complete |
| Population | 2 / 2 | 4 / 4 | complete |
| Earthquake shaking | 12 / 12 | 2 / 2 | complete |
| Historical rainfall | 11 / 11 | 0 / not accepted | blocked: source GIF collapses distinct legend semantics |
| Rainfall 1981-2010 | 12 / 12 | 8 / 8 | complete |
| Rivers/lakes | 13 / 13 | 4 / 4 | complete |

Final verification regenerated `INDEX.md` and `FAILURE_REPORT.md` from the
current logs. The earlier terminal audit completed the full suite at **417
passed**; the farms resumption added focused registry, failure-report,
alignment-promotion, legend, extraction, and documentation checks. A separate
integrity pass verified all 13 experiment logs, contiguous counted ordinals,
every accepted all-gates-passing iteration, all 2,135 logged artifact hashes,
all 101 hashed references across 26 accepted/archived pointer manifests, and
all 10 exact v1-to-v2 non-counting migration receipts. The current result is
12 complete maps and one source-ambiguity-blocked map, with no manual arrows,
control points, stamps, painting, or human approval counted anywhere.

## Fidelity addendum — report-backed completion, 2026-08-30

The “complete” labels above describe the original automatic alignment and
extraction contracts. They do **not** by themselves certify native-resolution
alignment or source-space fidelity. The authoritative per-run ledger is now
[`runs/mapbox-autonomous-restart-v1/INDEX.md`](runs/mapbox-autonomous-restart-v1/INDEX.md#fidelity-completion-matrix--artifact-audit-2026-08-30).

The ledger contains all 15 run directories, including the superseded z9 and
replacement 3× Farms and Geologic runs. It records source, counted alignment
and extraction iterations, explicit native report status, remaining blocker,
and the exact artifact paths. The resolution policy requires a separate native
alignment replay only for the five accepted optimizers below 0.5× source scale:
Population, Fire, Quake, 1981–2010 Rainfall, and Rivers. Fire, Population,
1981–2010 Rainfall, and Rivers pass against their accepted global transforms.
Quake's original A12 Web Mercator similarity is a native `retry`, but an active
supersession closes the gate with a separately passing nonlinear alignment and
fresh dependent extraction. Deer, Elevation, Forest, Landslide, Plant Zones,
and Historical Rainfall are `not required by resolution gate`, not pending.

Additional corrections are now explicit:

- the 3× Farms and Geologic reruns close alignment through their own
  source-direct native multiscale evidence and close extraction through their
  original-source regional diffs;
- Population's active `fidelity-supersession.json` binds the zero-mismatch
  four-pass repair as the authoritative replacement, so it no longer has a
  canonical-binding blocker;
- Quake preserves original A12 and its E2 extraction as failed native history,
  while its active supersession selects a projection-aware regularized
  thin-plate source-displacement model. The replacement passes all three
  immutable multiscale gates, 95.2381% of supported native 6×6 cells, 93.2459%
  land containment, nonfolding Jacobian ≥0.177947, and the pinned-reference and
  projection-round-trip gates. Its two-pass pristine-source extraction observes
  98.2477%, infers 1.7523%, preserves all six classes, has zero meaningful
  mismatch, and reaches a fixed point;
- Rivers A13 passes its pristine-source 50/75/100% multiscale replay, native
  state/county-policy/junction gates, and the existing named-hydrography native
  extraction audit without mutation;
- the base z9 Farms and Geologic rows remain superseded history rather than
  native passes; and
- Historical Rainfall remains blocked before extraction because classes 4/5/6
  (5.0/5.5/6.5) have identical local color/texture signatures and classes 1/2
  differ by only 0.1201 Lab-texture units. The restored source matches its
  registry hash and byte count, so source availability is no longer a blocker.

`runs/mapbox-autonomous-restart-v1/RESOLUTION_FIDELITY_AUDIT.json` remains the
machine-readable policy and queue. It is not a substitute for a per-map native
decision report. No native alignment pass is inferred from a working-grid
acceptance, prior visual approval, source-space extraction diff, or preview.

Farms delivery now uses a 5,625-tile indexed class-id pyramid through native
z11, with all 45 controls rendered from one categorical raster source. Its
dataset and provenance bind the accepted 10,192×11,758 E4 raster and the pinned
derived Mapbox state/coast diagnostic. The retired lime boundary is never used.
Persistent exact source-space evidence is the 6×6 native audit and Monterey
diff artifact; the separate 64-tile delivery array comparison was an ad hoc
spot check and has no invented report.

Quake's supersession explicitly reports that accepted historical artifacts and
public artifacts were not mutated. The nonlinear audit changes the fidelity
authority only; public delivery remains unchanged.
