# MapScan

MapScan is an experimental pipeline for turning categorical California map images into geographically aligned, interactive raster layers. The current implementation includes automatic and assisted alignment, machine-readable semantic extraction plans, categorical pixel classification, limited gradient classification, and limited separation of overlapping variables.

The project story and live composition tool are at
[todd.sh/mapscan](https://www.todd.sh/mapscan) and
[todd.sh/mapscan/map](https://www.todd.sh/mapscan/map). Map compositions update
the browser URL automatically, so a shared link restores the selected datasets
and layers, their order, per-dataset and per-layer opacity, edited layer colors,
and the complete map camera. See [`MAPSCAN_SHARE_URLS.md`](MAPSCAN_SHARE_URLS.md)
for the versioned deep-link format.

## Local setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/mapscan fetch-reference
```

The reference command downloads pinned 2025 U.S. Census Bureau TIGER/Line state and county packages and records their hashes under `reference/census-2025/`.

## Automatic alignment diagnostic

```bash
.venv/bin/mapscan align examples/forest.jpg \
  --output runs/forest-auto
```

The output is diagnostic only. `alignment.json` records every projection/evidence candidate; `alignment-diagnostic.png` overlays the best authoritative outline on the source image. A low numeric score is not an acceptance decision.

The processor evaluates `full_or_most_state` and `partial_state` coverage separately and writes their diagnostics under `best-full_or_most_state/` and `best-partial_state/`. Partial scoring allows the unseen portion of California to remain off-canvas; it is still diagnostic and can be fooled by non-geographic edges.

Refine and independently validate a coarse alignment with distributed perimeter evidence:

```bash
.venv/bin/mapscan auto-refine-perimeter \
  examples/forest.jpg \
  runs/forest-auto-v4/alignment.json \
  --output runs/forest-auto-perimeter

.venv/bin/mapscan auto-refine-perimeter-batch \
  examples/perimeter-refinement-batch.json \
  --output runs/perimeter-refinement-all
```

The closed loop renders the source without authoritative clipping, samples 16 distributed coastline/state-border neighborhoods, uses robust local edge-and-direction matching, fits from eight spatially distributed anchors, and reserves the rest as holdouts. A correction is accepted only when it improves held-out residuals and the independent boundary score. County junctions are a partial-map fallback and cannot override a stronger coast/border result. This unclipped diagnostic is important: clipping first would manufacture a perfect California outline and make the alignment test circular. Native graticule registration remains authoritative when present and is audited rather than replaced.

Before extraction, compare the retained alignment directly with the exact,
hash-verified active lime line at multiple scales and in eight geographic
regions:

```bash
.venv/bin/mapscan audit-canonical-alignment \
  examples/population.png \
  runs/population-perimeter-loop-v1/alignment.json \
  --target-height 2200 \
  --output runs/population-canonical-audit-v4
```

The audit operates on the unclipped warped source, writes source/line and
residual views plus a blink comparison, and evaluates the mainland separately
from generalized island geometry. A candidate is accepted only when the
regional residual gates pass. A fitted correction that improves training
anchors but worsens unused perimeter or county holdouts is retained as rejected
evidence; it does not replace the input alignment.

A styled, high-resolution county reference can be registered once and added as supplemental validation evidence:

```bash
.venv/bin/mapscan register-county-reference \
  examples/county.png \
  --output runs/county-reference-v2 \
  --maximum-dimension 1800 \
  --web-height 3600

.venv/bin/mapscan auto-refine-perimeter \
  examples/farms.png \
  runs/farms-auto-v9/alignment.json \
  --county-reference runs/county-reference-v2/county-reference.json \
  --working-height 2400 \
  --output runs/farms-auto-county
```

Registration separates the supplied map's thicker opaque-black state stroke from its thinner gray county strokes. The state stroke establishes its Web-Mercator transform; the county network is then checked against pinned Census counties. Automatic refinement may use locally visible, globally consistent raster county junctions as partial-map fallback controls and as a veto against a proposed correction. It remains supplemental because county vintages and generalization differ, and it never overrides worse coastline/state-border evidence.

## Active canonical California border

`reference/canonical-california-boundary.json` is the hash-bound active pointer for every future alignment and review. It currently resolves to `california-county-detail-border-v2`, which combines the exact registered `county.png` coastline and four island outlines with the accepted Census inland-border spans. Review decisions and correction files record the active pointer, manifest, and overlay hashes.

Do not rebuild this line from a filled state mask. That would close open bay entrances and discard coastal branches. Clipping interiors are separate evidence and may change by dataset without changing the canonical display/alignment border.

Remaining categorical gaps can be completed as a separate, reviewable neighbor-assumption layer:

```bash
.venv/bin/mapscan complete-neighbor-unknowns \
  examples/completion-plans/plant-hardiness-neighbor-v1.json
```

The command keeps exterior unknowns transparent, fills every retained interior unknown from nearby classified pixels, gives authored nonzero stamp pixels their configured evidence weight, and writes separate assumption/removal/confidence masks. It never publishes the result.

After visual approval, promotion creates a fresh, self-contained materialization and binds the exact review, alignment, canonical border, and clipping evidence:

```bash
.venv/bin/mapscan promote-neighbor-completion \
  runs/plant-hardiness-neighbor-completion-v1/neighbor-completion.json \
  runs/plant-hardiness-fidelity-review-v1/review-session.json \
  --author-statement "lgtm" \
  --output runs/plant-hardiness-neighbor-approved-v1
```

The versioned promotion and activation commands are:

```bash
.venv/bin/mapscan promote-county-detail-border \
  runs/county-detail-boundary-candidate-v1/county-detail-boundary.json \
  --supersedes reference/canonical-california-boundary-v1/canonical-boundary.json \
  --author-statement "Let's use this as the canonical border for all maps going forward." \
  --output reference/canonical-california-boundary-v2

.venv/bin/mapscan activate-canonical-border \
  reference/canonical-california-boundary-v2/canonical-boundary.json
```

For source PDFs that preserve vector longitude/latitude graticules and readable degree labels, use the stronger distributed-control path:

```bash
.venv/bin/mapscan align-pdf-graticule examples/geologic.pdf \
  --crs EPSG:3310 \
  --render tmp/pdfs/geologic/refinement-4500.png \
  --warp-crs EPSG:3857 \
  --warp-resolution 250 \
  --output runs/geologic-pdf-graticule
```

This detects the two intersecting vector-curve families, ties them to native degree labels, and fits the declared map projection directly to all visible intersections. It does not use dense geology or hillshade edges as alignment evidence. The optional warp uses nearest-neighbor sampling, clips to California, and writes both a clean raster and a Web-Mercator inspection overlay with authoritative state and county lines.

The same PDF can expose its legend swatches without estimating colors from rendered pixels:

```bash
.venv/bin/mapscan extract-pdf-legend examples/geologic.pdf \
  --render tmp/pdfs/geologic/refinement-4500.png \
  --output runs/geologic-pdf-legend
```

The fill colors are exact native PDF objects. Swatch unit codes remain provisional OCR until their nearby descriptions and the PDF's custom font encoding are resolved.

The geology itself can then be extracted directly from native vector fills, excluding text, strokes, images, and antialiasing by construction:

```bash
.venv/bin/mapscan extract-pdf-vector-fills \
  examples/geologic.pdf \
  runs/geologic-pdf-graticule-v5/alignment.json \
  runs/geologic-pdf-graticule-v5/legend-swatches.json \
  --output runs/geologic-pdf-vector-v1

.venv/bin/mapscan audit-pdf-vector-diff \
  runs/geologic-pdf-vector-v1 \
  --output runs/geologic-pdf-vector-diff-v1
```

The class geometry and completion mask are usable independently of the provisional labels. Publication remains blocked until custom-font geologic unit glyphs are paired with their descriptions and validated.

## Assisted alignment

```bash
.venv/bin/mapscan assist examples/farmsv2.png \
  --output runs/farms-assisted
```

The local interface asks for matching landmark clicks on the source and a California reference with county boundaries. Four pairs are the minimum; more widely distributed points are preferable. Saving writes:

- `assist-control-points.json`
- `assisted-overlay.png`
- `assisted-warped-inspection.png`

Assisted results are explicitly marked `alignment_mode: assisted` and must never be counted as automatic successes.

## Categorical extraction

Extraction is driven by a reviewed JSON plan that identifies the source alignment, legend-derived classes, representative color samples, and the classification strategy:

```bash
.venv/bin/mapscan extract-plan examples/plans/deer.json \
  --output runs/deer-extract
```

Each logical variable is written separately. Typical outputs include:

- a source-resolution class-ID PNG and visual preview;
- a nearest-neighbor Web-Mercator class-ID PNG and preview, clipped to California;
- an inspection image containing the warped source plus authoritative state and county lines;
- `extraction.json`, containing class counts, warnings, hashes, and review status; and
- `plan.snapshot.json`, preserving the exact semantic plan used for the run.

Pixels that do not match a class confidently remain NoData. The present strategies support mutually exclusive legend colors, multiple color prototypes for an ordinal gradient, high-confidence dot-pattern masks, grayscale bands with hillshade caveats, and separation of at most two chromatic overlays per pixel. All extraction results remain `needs_visual_review` until their alignment and classification have been inspected.

Before opening a candidate for author review, run the source-diff gate. Every layer must declare either `full_state` or `sparse_visible_evidence` in its plan:

```bash
.venv/bin/mapscan audit-source-diff \
  runs/plantzone-extract-v1 \
  --candidate runs/plantzone-extract-v1/materialized-v2 \
  --output runs/plantzone-source-diff-v1
```

For full-state variables, every California cell must contain data. The gate records all internal NoData, reconstructs those source occlusions from the geographically nearest class, writes a separate completion mask, and fails if any holes remain. For sparse variables, transparent background is valid and is never filled; the gate instead fails when a source-derived evidence pixel disappears from the candidate. `audit-feature-diff` independently recomputes an observed-ink gate for feature maps, and `audit-source-diff-batch` runs the declared checks across the working inventory. All reports, input hashes, before/after masks, repaired class rasters, and previews remain versioned review artifacts rather than silently replacing an approved or published candidate.

The batch audit is iterative. Each pass consumes the audited result of the preceding pass and stops only after two consecutive clean signatures are identical. This catches completion or reprojection changes that appear only after the first comparison. `examples/source-diff-batch.json` currently covers categorical, continuous-ramp, feature-ink, and native-PDF-vector sources.

For simple legend-color layers, perturb the classifier before requesting author
review rather than assuming that a larger color radius is more faithful:

```bash
.venv/bin/mapscan audit-categorical-fidelity \
  runs/forest-canonical-extract-v1 \
  --output runs/forest-categorical-fidelity-v1
```

The audit exactly recomputes the stored classes across a three-by-three ensemble
of Lab-distance and runner-up-margin thresholds. It writes strict-drop,
relaxed-addition, pale-neutral-risk, semantic-change, source comparison, and
per-category source/Web-Mercator masks. Relaxed evidence is diagnostic only and
never enters the candidate automatically: pale legend colors can otherwise
turn an uncolored JPEG background into false data. Tiny components are reported
per category and retained when the source itself contains pixel-scale evidence.

After threshold perturbation and the fixed-point source-diff batch, generate a
lossless source-versus-extraction comparison:

```bash
.venv/bin/mapscan compare-categorical-source \
  runs/population-extract-v8 \
  runs/population-fidelity-loop-v6/categorical-fidelity-audit.json \
  runs/population-source-diff-loop-v6/source-diff-batch.json \
  population-density \
  --output runs/population-comparison-loop-v6
```

This final gate flips between the aligned source and reconstructed classes,
marks derived completion separately, and fails on changed observed class IDs,
direct source classes removed by a contextual mask, nearest-class completion
disagreements, interior NoData, exterior color,
internal-water color, semantic threshold instability, or the absence of two
identical source-diff signatures. Plans may declare a narrowly defined
`source_context_exclusion` for contextual source pixels such as blue water;
those pixels cannot seed a class or receive neighbor completion. External
hydrography remains an independent, named-feature exclusion so a modern water
inventory cannot silently punch unsupported holes into a historical source.
Plans may also define `comparison_regions` as named WGS84 bounding boxes. The
comparison command then writes native-resolution source, extracted, mismatch,
completion, montage, and blink artifacts for every region and applies the same
fidelity gates locally. The full-resolution reviewer exposes those regions in a
dropdown, so Source/Extracted flipping can jump directly to known risk areas
without judging a statewide downsample.

Once a categorical case reaches that fixed point, promote the audited surface into a self-contained review materialization instead of reviewing the pre-completion raster:

```bash
.venv/bin/mapscan promote-source-diff-materialization \
  runs/quake-final-hybrid-extract-v1/materialized-v1 \
  runs/source-diff-quake-final-hybrid-v1 \
  quake-shaking-final-hybrid-v1 \
  --output runs/quake-final-hybrid-extract-v1/final-candidate-v1
```

The command verifies the extraction, batch, case, iteration, and artifact hashes; copies the stable class-ID raster and preview; preserves manual, inferred, and enclosed-fill evidence masks; and promotes the first-pass completion mask as a separate artifact. The resulting `materialization.json` remains `needs_visual_review` and is byte-identical across repeated builds.

When the reviewed display perimeter is a hybrid reference, bind the candidate to the exact filled interior of that continuous line before review:

```bash
.venv/bin/mapscan clip-materialization-to-boundary \
  runs/quake-final-hybrid-extract-v1/final-candidate-v1 \
  runs/quake-lower-colorado-v1-a/hybrid-perimeter-audit.json \
  --output runs/quake-final-hybrid-extract-v1/final-candidate-v2
```

This stage requires a one-component closed border, forces every class and evidence raster to zero outside it, completes only the thin newly exposed interior fringe with the same deterministic nearest-class rule, and stores the removed-outside and completed-inside footprints separately. The resulting audit fails unless colored-outside and unclassified-inside counts are both exactly zero.

### Optional rainfall topology interpretation

Some indexed images have already discarded semantic information before MapScan sees them. In `rainfall.gif`, the 5.0, 5.5, and 6.5-inch legend swatches contain exactly the same two GIF colors in exactly the same proportions. Color and local texture therefore cannot distinguish those classes, even though black isohyet boundaries remain visible between some same-color polygons.

Run the optional, non-mutating topology stage only after categorical extraction:

```bash
.venv/bin/mapscan extract-rainfall-topology \
  examples/plans/rainfall-county-refined-v4.json \
  runs/rainfall-county-refined-extract-v1 \
  --output runs/rainfall-topology-v1
```

The command verifies the exact plan, source, extraction manifest, class raster, and completion mask. It refuses to write inside the extraction run. It emits separate source-resolution and Web-Mercator artifacts for direct non-confusable color evidence, confusable palette-family membership, stable topology-derived endpoint assignments, and the still-unresolved remainder. A class survives only when all nine configured dark-line and boundary-neighborhood variants agree and the component has dominant support from the immediately adjacent readable-color class. Numeric OCR is accepted only when it matches a legend value inside the registered state at a strict confidence threshold.

Topology output is diagnostic evidence, not a repaired categorical candidate. It is never merged automatically, never satisfies publication approval, and never assigns an unanchored middle class. If identical palettes, broken contours, absent numeric isohyet labels, or cartographic occlusions leave several interpretations possible, those pixels remain explicitly unresolved.

Continuous legends use a value-preserving 16-bit branch instead of arbitrary categories:

```bash
.venv/bin/mapscan extract-continuous-plan \
  examples/plans/elevation-continuous.json \
  --output runs/elevation-continuous-v6

.venv/bin/mapscan audit-continuous-diff \
  runs/elevation-continuous-v6 \
  --output runs/elevation-continuous-diff-v6
```

It projects source color onto the legend ramp in Lab space, masks detected cartographic ink, stores direct and completed pixels separately, and records the numeric offset/scale needed to reconstruct values from the uint16 raster.

## Full-resolution alignment review

Open any completed extraction run in the local author-only reviewer:

```bash
.venv/bin/mapscan review runs/quake-assisted-extract-v1 --port 8767
```

The reviewer supports wheel zoom, drag pan, and independent opacity controls for the warped source, extracted classes, authoritative state/coast outline, the thin county network registered from `examples/county.png`, and a county residual diagnostic measured against that same network. The high-resolution raster strokes are resampled as subpixel coverage rather than redrawn at a fixed width. New categorical extractions are clipped to the hash-bound canonical publication interior before review, and their manifests retain both the removed-pixel mask and the canonical interior mask. Source-diff completion reuses that exact manifest mask, so later filling cannot restore pixels outside the lime boundary. A full-state categorical plan may opt into nearest-class reconstruction for labels and dark cartographic strokes with `complete_target_state: true`; the resulting pixels are retained in a separate target-completion mask and only in-boundary observed classes may seed them. The state border and coastline are the primary alignment evidence. County residual colors are secondary evidence only: terrain, faults, labels, and other source linework can create false matches, while differing boundary vintage or generalization can make a correct registration look worse.

When verbal feedback is insufficient, click **Add correction arrows**. Start on the authoritative cyan or magenta line and drag to the matching dark feature in the current warped source; the processor moves the source feature back toward the arrow start. A click without movement records a fixed pin for an already-correct area. Use widely distributed arrows, then save them to `alignment-corrections.json`; the file stores review-raster, normalized, and Web-Mercator coordinates without changing the extraction.

Fit the least-flexible correction model that passes held-out residual gates, then rerun extraction with the resulting alignment:

```bash
.venv/bin/mapscan refine-alignment \
  runs/quake-assisted-boundary-v1/assist-control-points.json \
  runs/quake-assisted-extract-v1/alignment-corrections.json \
  --output runs/quake-reviewed-affine-v1
```

The fitter evaluates translation, similarity, affine, and projective models in that order using leave-one-out residuals. It stops at the first passing model, rejects geographically concentrated controls, records every candidate, and preserves the parent alignment. The corrected output remains diagnostic until another full-resolution review.

Fine corrections concentrated in one corridor must not drive another statewide transform. Apply those as a compact local warp instead:

```bash
.venv/bin/mapscan refine-local-alignment \
  runs/quake-reviewed-affine-v2/alignment.json \
  runs/quake-reviewed-corrected-extract-v1/alignment-corrections.json \
  --radius-px 500 \
  --output runs/quake-reviewed-local-v1
```

The local fitter exactly honors the recorded arrows, smoothly decays each displacement to zero outside its support radius, composes with the parent alignment, and rejects a warp whose sampled Jacobian folds over. A global fit now requires both broad axis spans and substantial convex-hull area; broad x/y ranges alone are insufficient.

Legacy correction files written before schema version 2 labeled the direction oppositely. Use `--reverse-arrows` only when recovering one of those explicitly identified sessions; new files store separate `reference` and `source` endpoints and require no override.

An Approve, Revise, or Reject action writes `review-decision.json` beside the extraction manifest, including the author's notes and the exact manifest hash reviewed. Review does not publish or modify the extraction.

Alignment approval does not approve the extracted legend classes. After alignment passes, inspect every category independently in the classification reviewer:

```bash
.venv/bin/mapscan review-classification \
  runs/quake-reviewed-local-extract-v2 \
  --port 8773
```

The classification reviewer renders the full-resolution class-ID raster directly, supports independent category toggles and Solo inspection, preserves crisp nearest-neighbor pixels, and compares the selected masks with the warped source. Its Approve, Revise, or Reject action writes `classification-review-decision.json`. Classification cannot be approved until the alignment decision is approved.

Approved observed classes may receive conservative, separately masked fills for small text or city-symbol occlusions:

```bash
.venv/bin/mapscan detect-map-labels \
  runs/quake-reviewed-local-extract-v2 \
  --output runs/quake-reviewed-local-extract-v2/label-detection

.venv/bin/mapscan detect-map-labels \
  runs/quake-reviewed-local-extract-v2 \
  --output runs/quake-reviewed-local-extract-v2/label-detection-psm11 \
  --psm 11

.venv/bin/mapscan infer-categorical-gaps \
  runs/quake-reviewed-local-extract-v2 \
  --output runs/quake-reviewed-local-extract-v2/inference-v6 \
  --gap-radius-px 6 \
  --minimum-dominance 0.98 \
  --ocr-tsv runs/quake-reviewed-local-extract-v2/label-detection/neutral-dark-labels.tsv \
  --ocr-tsv runs/quake-reviewed-local-extract-v2/label-detection-psm11/neutral-dark-labels.tsv \
  --ocr-min-confidence 20 \
  --ocr-label-padding-px 28 \
  --ocr-context-radius-px 44 \
  --ocr-max-distance-px 38
```

`detect-map-labels` isolates dark, nearly neutral typography within the aligned state mask before running Tesseract, preventing the colored data surface from overwhelming OCR. Multiple `--ocr-tsv` arguments combine complementary OCR page-layout passes; overlapping pixels that propose different classes are rejected. The inference pass never modifies the observed class-ID PNG. It writes inferred source/Web-Mercator rasters and explicit masks under its output directory. Label gaps are reconstructed pixel-by-pixel from the nearest observed class, so one label may cross multiple categories; pixels too far from evidence or nearly equidistant between categories remain transparent. The classification reviewer displays inferred pixels in neon cyan, automatically selects the highest versioned inference artifact, and records a separate inference decision tied to that artifact's hash.

The same reviewer includes a discrete clone-stamp correction tool for remaining occlusions. Choose a radius, hold **A**, and click an intact or previously painted source patch; then click each destination once. The **Set source** button provides the same action without a keyboard shortcut. Each destination click snapshots and copies the complete circular class-ID patch—including its zero-valued pixels—as one manual override, and dragging never paints a trail. Schema-3 operations record `source_mode: composite_at_operation_time`, so sequential replay can reuse earlier author pixels deterministically; legacy schema-2 operations continue to sample the observed raster exactly as they did when authored. Manual patches render with exactly the same legend palette and opacity as observed classes, eliminating reviewer-only color seams. The underlying observed and automatic-inference artifacts remain unchanged. Undo, Clear, and `stamp-corrections.json` preserve the auditable source/target/radius operation history.

If the automatic fill goes too far, enable **Erase inference**, choose a radius, and brush across the cyan false positives. The eraser is subtractive and inference-only: it cannot remove observed classes or manual stamps, and it never mutates the original inference raster. Undo and Clear replay from that immutable source. **Save exclusions** writes the author strokes to `inference-exclusions.json`, tied to both the extraction and inference hashes; publication subtracts the resulting exclusion mask from automatic inference.

After correction review, materialize the exact composed result as a separately masked candidate:

```bash
.venv/bin/mapscan materialize-corrections \
  runs/quake-reviewed-local-extract-v2 \
  --output runs/quake-reviewed-local-extract-v2/materialized-v1
```

The command verifies every input hash, selects the highest versioned inference artifact unless one is specified, and applies the explicit precedence `observed → retained inference → manual override`. It writes the final class-ID raster and preview alongside retained-observed, retained-inference, manual-footprint, and manual-value rasters. `materialization.json` records counts and hashes for every artifact; its initial status remains `needs_visual_review` rather than silently publishing the composition.

When a categorical extraction itself has been approved in both the alignment and per-category reviewers and needs no further inference or authored correction, package those exact reviewed bytes without routing them through correction replay:

```bash
.venv/bin/mapscan promote-reviewed-extraction \
  runs/deer-canonical-extract-v7 \
  runs/source-diff-deer-canonical-v7/source-diff-batch.json \
  deer-canonical-v7 \
  --author-statement "lgtm" \
  --output runs/deer-canonical-extract-v7/materialized-reviewed-v1
```

This narrow path requires matching alignment and classification approvals, a byte-identical fixed-point source-diff result, the active canonical boundary, a fresh output directory, zero exterior color, and zero interior NoData. It copies the approved class raster unchanged while retaining cartographic-completion and speck-reassignment masks as distinct derived evidence. The generated materialization decision binds the original review decisions and explicitly records that no older materialization approval was carried forward.

When corrections move through several accepted child alignments, stamp migration traverses and verifies the complete parent-hash chain. Projective steps use their declared parent-to-child matrices; bounded nonlinear steps are inverted under their validated sampling model. Geographic targets, radii, composite sources, and operation order remain fixed. The final migration reviewer compares the fixed-point candidate, the previous approval, the aligned source, and a changed-pixel diagnostic while displaying the accepted hybrid border and county network.

Automatic inference can be rejected without deleting its diagnostic artifacts. Put `{"schema_version": 1, "enabled": false}` in `inference-selection.json` at the run root, or pass `--without-inference`. The reviewer then hides inference entirely, preserves observed-source clone stamps, and materialization uses the simpler precedence `observed → manual override` with a zero-valued retained-inference mask.

For the narrower author-approved cleanup of tiny black/transparent holes, generate a separately masked deterministic fill:

```bash
.venv/bin/mapscan fill-enclosed-holes \
  runs/quake-reviewed-local-extract-v2 \
  --output runs/quake-reviewed-local-extract-v2/enclosed-fill-v1 \
  --maximum-area-exclusive 50
```

This uses 8-neighbor connected components and fills only zero-class holes with fewer than 50 pixels whose complete boundary is exactly one nonzero class. Edge-connected components, mixed-class boundaries, and explicit zero-valued manual overrides are never filled. `enclosed-hole-fill-selection.json` enables the artifact for review and materialization; its values render with the surrounding legend color while its mask remains independently auditable.

Named line and polygon maps are a separate pipeline branch. `examples/rivers.jpg`, for example, uses a symbol key and on-map labels for rivers, dry streambeds, lakes/reservoirs, and dry lakes. Its diagnostic contract is recorded in `examples/feature-plans/rivers.json`: preserve the observed blue-ink raster, separate text from geometry, derive vectors, attach validated names, and mark all reconnections beneath labels as inferred.

Generate the non-semantic observed-ink evidence before attempting OCR or vectorization:

```bash
.venv/bin/mapscan extract-feature-ink \
  examples/feature-plans/rivers.json \
  --output runs/rivers-feature-ink-v1
```

The resulting mask intentionally includes both blue feature geometry and blue labels. It is an auditable first stage, not a publishable hydrography layer.

## Raster-tile publication and viewer

Only an explicitly approved materialization may be exported. The categorical exporter writes one transparent white-mask XYZ pyramid per legend category so the web viewer can independently toggle, recolor, and adjust the opacity of every class without regenerating tiles:

```bash
.venv/bin/mapscan export-raster-tiles \
  runs/quake-final-hybrid-extract-v1/final-candidate-v2 \
  --output publish/datasets/quake-shaking \
  --minimum-zoom 4 \
  --maximum-zoom 9 \
  --overview-supersampling 4
```

An approved continuous-ramp extraction uses a separate native-color exporter. It verifies the manifest-bound review decision and an independent zero-difference audit, retains the exact 16-bit value raster, and writes one toggleable opacity-adjustable XYZ surface plus the complete numeric legend:

```bash
.venv/bin/mapscan export-continuous-raster-tiles \
  runs/elevation-v34-continuous-pinned-land-v42 \
  runs/elevation-v34-continuous-pinned-land-diff-v42/source-diff-audit.json \
  --output publish/staging/elevation-v5-band-controls
```

The continuous package keeps the approved native-color surface as its default
view and deterministically partitions the exact 16-bit values into selectable
legend intervals. Band tiles begin disabled; selecting one hides the full
surface, and multiple bands can then be combined or opacity-adjusted without
reclassifying rendered RGB pixels. Re-enabling the full surface clears the band
selection and restores the exact approved overview.

Before approval, the same exporter can create an explicitly non-publishable
Mapbox review package. The independent diff audit is still mandatory, but the
dataset and provenance are labeled `needs_visual_review` and `not_approved`:

```bash
.venv/bin/mapscan export-continuous-raster-tiles \
  runs/elevation-v34-continuous-pinned-land-v42 \
  runs/elevation-v34-continuous-pinned-land-diff-v42/source-diff-audit.json \
  --output publish/staging/elevation-v3-review \
  --review-preview
```

The approved quake publication contains six category pyramids and 2,400 PNG tiles. Zooms 4–8 use a four-by-four categorical supersample: each output pixel retains one dominant class while its alpha records the exact covered fraction, preventing a correctly aligned coast from appearing shifted to a single coarse pixel center. Zoom 9 and all closer overscaling remain exact binary nearest-neighbor pixels. The plant-hardiness staging candidate contains 13 category pyramids and 5,200 PNG tiles. Each dataset also includes its original source image, TileJSON files, a public dataset manifest, and a path-free provenance manifest. For boundary-clipped materializations, export revalidates the approval against the exact boundary-audit and continuous-border hashes and always rejects nonzero outside color. Full-state layers must contain zero interior NoData. Layers explicitly reviewed as `sparse_visible_evidence` instead preserve their exact hash-bound interior-NoData count as transparent, so absence in the source is not turned into invented data. When a canonical display overlay is present, export copies that exact hash-bound raster for alignment inspection instead of reconstructing open bays or islands from the fill mask. Machine-local paths are excluded from public manifests; immutable input and tile-set hashes preserve provenance.

After a staging package has been inspected, activate those exact bytes through a
fresh, hash-bound public directory:

```bash
.venv/bin/mapscan activate-staging-dataset \
  publish/staging/deer-distribution-v1 \
  --author-statement "lgtm" \
  --output publish/datasets/deer-distribution
```

Activation recomputes every category tile-set hash, verifies the canonical
boundary raster and GeoJSON, and always requires zero outside color. Full-state
packages require zero interior NoData; sparse-evidence packages must preserve
their exact reviewed interior-NoData count. Activation then copies the complete
staging inventory byte-for-byte. The new
`public-activation-decision.json` records the author statement and exact package,
materialization, approval, boundary, and tile hashes. Existing outputs are never
overwritten, and the original staging package remains available as the review
archive.

An instruction to continue processing, fix a defect, or use a newer run is not
an approval of visual quality and must never be copied into an activation record
as though it were one. Promotion requires an explicit approval bound to the
exact staged package, or an explicitly selected autonomous publication policy
whose alignment, fidelity, and non-regression evidence is serialized. Version
recency is never a quality gate.

Autonomous indexed packages use the same fail-closed activation boundary. Before
activation, a small checked-in plan binds every source layer to its accepted
raster, the exact publication raster, its semantic kind, and either a
`full_state` or `sparse_visible_evidence` contract. The attestation recomputes
the Mapbox-interior coverage counts and rejects even one classified exterior
pixel:

```bash
.venv/bin/mapscan attest-indexed-staging-coverage \
  viewer/public/data/staging/fire-autonomous-v2-clipped \
  config/publication-coverage/fire-v2-clipped.json
```

The indexed activator then reopens those attested bytes, validates every
TileJSON template, tile count, byte count, aggregate tile hash, cache key,
source hash, and boundary hash, and writes a new approved manifest:

```bash
.venv/bin/mapscan activate-indexed-staging-dataset \
  viewer/public/data/staging/fire-autonomous-v2-clipped \
  --author-statement "I approve the exact attested Fire staging package for public activation" \
  --public-id california-fire-hazard-and-responsibility-areas \
  --public-title "California Fire Hazard and Responsibility Areas" \
  --asset-base /mapscan/data/staging/fire-autonomous-v2-clipped/ \
  --output viewer/public/data/datasets/fire-hazard-responsibility
```

`--asset-base` is an optional root-relative deployed URL for immutable verified
assets. It lets an approved directory contain only its manifest, provenance,
and activation decision instead of duplicating thousands of tile files. The
viewer resolves tiles, source imagery, boundaries, and diagnostics through that
base. Activation rejects absolute URLs, traversal, symlinks, stale hashes, and
partial outputs; the catalog integrity test subsequently rehashes the resolved
assets.

The independently deployable Next.js viewer lives in `viewer/` and is configured for the `/mapscan` base path:

```bash
cd viewer
npm install
cp .env.example .env.local
# Replace the placeholder with a public Mapbox token.
npm run dev
```

Open `http://127.0.0.1:3000/mapscan` for approved datasets or
`http://127.0.0.1:3000/mapscan/staging` for isolated candidates. The viewer
builds one persistent composition: focusing a different dataset changes the
visible controls without clearing layers selected from other datasets. Every
category has independent visibility, color, and opacity; dataset-scoped and
map-wide clear actions are explicit. Source-image modals and dataset-qualified
share URLs preserve the evidence and restore the complete cross-dataset
composition. Approved artifacts under `publish/` are copied to
`viewer/public/data/` for static Vercel delivery. Approved indexed autonomous
packages use catalog-visible metadata under `viewer/public/data/datasets/` and
resolve their exact hash-verified asset inventories under
`viewer/public/data/staging/` through `asset_base`; the staging manifests remain
explicitly unapproved review evidence even though the immutable bytes are shared.

Viewer release gates:

```bash
cd viewer
npm test -- --runInBand
npm run typecheck
npm run build
```

Before declaring the complete live corpus ready, bind every public package to
its immutable processing evidence and rerun both corpus gates:

```bash
.venv/bin/python scripts/audit_live_alignment_nonregression.py \
  --output runs/corpus-readiness-audit-v1/live-alignment-nonregression.json
.venv/bin/python scripts/audit_live_dataset_readiness.py \
  --alignment-audit runs/corpus-readiness-audit-v1/live-alignment-nonregression.json \
  --output runs/corpus-readiness-audit-v1/live-dataset-readiness.json
```

The readiness audit fails if a source, alignment, extraction pointer, public
provenance record, catalog entry, schema-specific extraction gate, deterministic
fixed point, approval state, or exterior-color constraint is missing or stale.

## Tests

```bash
.venv/bin/pytest -q
```

Product decisions and experiment history live in `MAPSCAN_REQUIREMENTS.md` and `MAPSCAN_JOURNEY.md`.
Machine-readable experiment summaries live under `benchmarks/`.
