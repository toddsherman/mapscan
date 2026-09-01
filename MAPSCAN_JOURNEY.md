# MapScan Journey and Decision Log

Last updated: 2026-08-21

## Purpose

MapScan is a powerful proof of concept for automatically extracting categorical data from images of California maps, aligning that data geographically, and presenting it as interactive layers over a Mapbox basemap.

The public result will live at `todd.sh/mapscan` as one of the projects linked from the `todd.sh` root. It is not intended to be a general-purpose SaaS product. Todd is the only author/operator; public visitors explore a curated collection of processed datasets.

This document is a living record of the product journey, decisions, assumptions, experiments, rejections, and lessons learned. It should eventually provide the factual basis for a writeup about using AI to build MapScan.

The current formal product and technology specification is in `MAPSCAN_REQUIREMENTS.md`.

## Current Product Model

MapScan has two distinct parts:

1. An offline ingestion pipeline that processes map images from a local folder. Processing may take an hour or more per image.
2. A separately deployable, desktop-first public viewer, exposed specifically at `todd.sh/mapscan` through the existing site.

The pipeline is expected to iterate until it produces a solid collection. Outputs are manually reviewed before publication. The public viewer will initially contain approximately 20 to 100 accepted datasets.

## Terminology

- **Dataset:** One processed source image.
- **Group:** A heading in a source legend, such as “Field Crops.”
- **Category:** An individual legend entry, such as “Cotton.”
- **Classified raster:** A raster in which each accepted data pixel stores a category identifier rather than merely an RGB display color.
- **Data layer:** The classified data belonging to a category.

## Confirmed Product Decisions

### Source images

- Inputs are standard image files found on the web, rather than photographs of physical maps.
- All initial images depict California.
- Many images will show most or all of California; some will show only part of the state.
- Partial-state maps must contain enough recognizable geography for automatic alignment, especially a useful portion of the state border or coastline. Images without enough evidence may be rejected.
- Visual styles and publishers will vary.
- The original image is retained and can be opened from the viewer in a separate window or modal. No source-versus-result comparison view is required in the public viewer.
- Source attribution and provenance are not a product requirement.

### Supported data semantics

- The initial scope is discrete categorical data encoded primarily by color.
- Categories within one image are mutually exclusive.
- Separate images represent separate datasets and may be displayed together.
- Legend color is the primary extraction key, but the stored result uses stable category identifiers so display colors can be changed later.
- Legend group hierarchy should be preserved when it can be read confidently.
- If legend text cannot be read confidently, the image is rejected.
- If only one or more individual categories cannot be extracted confidently, those categories may be omitted with explicit warnings while the remaining dataset proceeds to review.
- Ambiguous pixels are made transparent rather than assigned speculatively.
- Pixels covered or destroyed by labels, borders, legends, or other cartographic elements remain transparent; MapScan does not infer the missing underlying data.
- Extracted output is clipped to California, including its islands; data shown in neighboring states is excluded.

### Fidelity and rendering

- A classified raster is the leading representation because it can preserve source-pixel detail while supporting per-category controls.
- Warping must use categorical-safe resampling, expected to be nearest-neighbor, so it does not invent blended categories.
- Rendering beyond native resolution should remain crisp rather than smoothed.
- High fidelity takes priority over processing speed and cost.
- Classification accuracy and geographic alignment accuracy are separate quality dimensions.
- Precision is favored over coverage: uncertain pixels become transparent.
- Acceptance thresholds will be derived empirically from the image corpus and a manually prepared gold test subset.

### Alignment

- Legend analysis does not align the map geographically.
- California’s state border and coastline are the primary registration anchors for aligning and warping source maps to the Mapbox view.
- A distinctive coastline segment may be sufficient for a partial map, although another visible portion of the state boundary is preferred.
- County boundaries may be attempted as a fallback when the state border is unusable; they may also provide secondary validation.
- The first pipeline may support north-up digital maps with differing projections, rotation, cropping, and nonlinear distortion.
- Perspective photographs, inset maps, and multi-panel figures may be rejected initially.
- A result must include a local, author-only diagnostic inspection capability that overlays both the warped source-detected border/coastline and the authoritative reference boundary on the Mapbox-aligned result.
- The inspection view uses contrasting outlines, classified data beneath them, independent visibility toggles, and mismatch markers or a local-error heatmap.
- Results above an empirically established alignment-error threshold are rejected automatically but remain available for diagnostic inspection.
- Visual approval by Todd is required before a dataset enters the public catalog.

### Diagnostic output

Each processing attempt should generate a diagnostic report containing, at minimum:

- Original image
- Detected map and legend regions
- Extracted category mask
- Warped overlay on the geographic reference
- Border/coast alignment inspection view
- Classification and alignment confidence information
- Per-category detected swatch, label, pixel count, confidence, extracted mask, and ambiguous-pixel count
- Warnings for omitted categories
- Explicit rejection reasons when processing fails

### Public viewer

- The public viewer is read-only and desktop-first.
- It uses a light/minimal Mapbox basemap initially.
- Visitors choose from a simple list of curated datasets; catalog search and tagging are not initially required.
- Visitors may activate up to five datasets simultaneously at first. The limit is configurable and can increase after performance testing.
- Each individual legend category has visibility and transparency controls.
- Visitors can change category colors.
- Legend groups should be preserved and usable for grouped controls when available.
- Dataset ordering can be changed when several datasets are active.
- Titles, legend labels, and hierarchy are controlled by author configuration rather than permanently edited by visitors.
- The complete viewer state—including camera, selected datasets, category visibility, colors, transparency, and ordering—should be shareable through the URL.
- Visitors do not upload images, download GIS outputs, embed the viewer, or use an API.
- Point inspection and value queries are not currently required.

### Authoring and deployment

- New images are added to a local project folder.
- A command-line or batch pipeline processes new or changed images.
- Generated configuration files are an acceptable authoring mechanism.
- Manual authoring may correct titles, OCR mistakes, legend labels, hierarchy, display colors, and other presentation defaults.
- Automatic ingestion does not use per-image manual crop adjustments, color-threshold tuning, or geographic control points.
- An explicitly labeled assisted alignment fallback may collect paired geographic control points after an automatic rejection. Assisted outcomes remain a separate evaluation cohort and never overwrite the failed automatic attempt.
- Review failures should drive improvements to the general pipeline followed by corpus reprocessing, rather than hand-patched extracted outputs.
- TypeScript, React, and Next.js are acceptable for the viewer.
- Python, GDAL, OpenCV, OCR, and vision services are acceptable for processing.
- Local computation, paid APIs, hosted CPUs, and hosted GPUs are all acceptable when they improve fidelity.
- The public viewer is separately deployable, but the exact public path `todd.sh/mapscan` is important.
- Generated assets may use external object storage if repository or Vercel deployment limits make that appropriate.

## Proof-of-Concept Success Criteria

- Build a solid curated collection rather than claim universal map support.
- Automatically accept and correctly process roughly 60 to 80 of 100 varied candidate images; clear, useful rejection reports for the remainder are acceptable.
- Iterate on the processing pipeline before publishing the viewer or treating the two current examples as a completed vertical slice.
- Prepare trusted reference results for roughly 10 to 20 images to develop quantitative classification and alignment thresholds.
- Preserve enough diagnostic evidence to explain honestly how AI and deterministic geospatial/computer-vision techniques contributed to the result.
- The case study is specifically about AI in the processing pipeline, not about AI assisting with software development.

## Current Technical Hypothesis

The current hypothesis, which must be tested rather than treated as settled architecture, is:

1. Detect and separate the legend and map regions.
2. Use OCR and visual analysis to extract legend groups, labels, and color swatches.
3. Convert legend entries into stable category identifiers.
4. Classify map pixels by perceptual similarity to legend colors, with thresholds that leave ambiguous pixels transparent.
5. Detect the California border and coastline in the source map.
6. Register those features against authoritative California reference geometry.
7. Estimate an appropriate affine, projective, polynomial, or nonlinear warp based on the source map’s projection and distortion.
8. Warp the category-index raster using nearest-neighbor resampling.
9. Generate diagnostic artifacts and confidence measurements.
10. After manual approval, generate optimized Mapbox-compatible assets and update the curated dataset manifest.

Potential viewer representations include category masks, an indexed categorical raster rendered with GPU styling, or another Mapbox-compatible raster mechanism. A prototype must compare fidelity, per-category styling, storage cost, and browser performance before the representation is finalized.

The authoritative output must include the original-resolution classified raster and accepted transform, not only a public tile pyramid. A screen cannot display multiple source pixels inside one screen pixel at low zoom without generalization; public overviews are therefore display products rather than the sole copy of the extracted data.

## Evaluation Questions Still Open

- What measurable border/coast alignment error should cause automatic rejection?
- How should confidence combine legend OCR, color separation, map-region detection, and geographic registration?
- Which automatic transformation models work reliably across the initial corpus without overfitting local distortions?
- How should the pipeline distinguish legend-color data pixels from unrelated map graphics that happen to use similar colors?
- What categorical raster encoding and Mapbox rendering approach best balances fidelity, dynamic styling, storage, and browser performance?
- Which object-storage and tile-delivery option is most cost-effective for the accepted corpus?
- What is the processing cost per image for deterministic CPU tools, OCR/vision APIs, and optional hosted GPU stages?
- Validate the proposed Next.js Multi-Zone rewrite and asset routing against the existing `todd.sh` application.

## Example Corpus

- `examples/forest.jpg`: California forest-cover categories with a compact categorical legend and a nearly complete state outline.
- `examples/farms.png`: Many agricultural categories, hierarchical legend groups, fine-grained pixels, terrain shading, county boundaries, and a legend obscuring part of the map.
- `examples/geologic.pdf`: A large hybrid vector/raster geological poster with a declared California Albers projection, native graticules, 53 main-legend swatches, custom font encoding, detailed linework, and a second explanatory time-scale region that must not be treated as map categories.

## Journey Notes

### 2026-08-20 — Initial product interview

- The concept began as a broadly available image-to-interactive-map service.
- The scope was narrowed to a curated proof of concept operated by one author and published as a project on `todd.sh`.
- Classified raster became the leading data model after comparing it with vector polygons. It supports per-category visibility and recoloring while avoiding polygon simplification of pixel-scale details.
- The corpus was narrowed to California, while allowing both complete and partial-state maps with sufficient geographic alignment evidence.
- Full automation means the automatic ingestion path receives only the image and may reject it. Manual review controls publication. A later decision added an explicitly separate assisted alignment path without weakening or relabeling the automatic benchmark.
- The second agricultural example clarified the need for hierarchical legends, many similar colors, tiny regions, transparent treatment of obscured areas, and robust separation of colored data from grayscale cartography.
- Publication should wait until repeated pipeline iteration produces demonstrably good outputs.
- Alignment inspection, especially reference border and coastline overlays, was identified as essential to knowing whether a map was warped correctly.
- The alignment diagnostic was refined to compare the warped source-detected outline with authoritative California geometry, expose local residual errors, and retain automatically rejected attempts for inspection.
- County boundaries were accepted as an automatic alignment fallback when the state outline is unusable.
- Manual editing was initially limited to semantic metadata and presentation. Per-image extraction repairs remain excluded; alignment repairs now exist only in the separately labeled assisted workflow.
- A weak individual category can be omitted with a warning instead of forcing rejection of an otherwise sound dataset.
- Accepted data is clipped to California, and classification diagnostics report evidence and ambiguity separately for every category.

### 2026-08-20 — Reproducibility and requirements research

- The planned case study was narrowed to AI's role inside the processing pipeline. AI may propose map/legend regions, read difficult text, and structure legend hierarchy, while deterministic and numerically validated code remains authoritative for pixel classification and georeferencing.
- Every run will write a machine-readable manifest containing source and output hashes, pipeline and stage versions, model identifiers, prompt versions, parameters, timing, usage, cost, confidence, warnings, and approval state.
- Processing stages will be content-addressed, cached, and resumable so an hour-long job restarts only invalidated work.
- The 100-image corpus will be split into 60 development, 20 validation, and 20 untouched test images. Thresholds will be frozen after validation and before final test evaluation.
- External OCR, multimodal AI, and hosted GPU services are permitted because the source images are already public.
- Current Mapbox GL JS documentation exposes raster-value colorization and nearest-neighbor overscaling. The first renderer spike will test lossless byte-valued class-ID PNG tiles with runtime category color and alpha. This remains a hypothesis until exact lookup, tile seams, and five-dataset performance are verified.
- Mapbox Raster MTS will be tested as a managed alternative because it retains numeric raster values and supports nearest resampling, but its raster-array source is still marked experimental.
- The initial deployment recommendation is a separate Next.js application with a `/mapscan` base path and a Multi-Zone rewrite from the main `todd.sh` deployment. This is simpler for two projects than adopting the full Vercel Microfrontends product.
- Public assets should begin on Vercel Blob because an existing Pro plan is available. Cloudflare R2 is the fallback if measured tile storage or transfer exceeds included usage.
- Current free tiers make the public proof of concept likely to have near-zero incremental hosting cost at portfolio traffic. A provisional $50 processing experiment budget will be revised after ten representative runs produce actual stage costs.
- The first formal product and technology requirements were captured in `MAPSCAN_REQUIREMENTS.md` before pipeline implementation.

### 2026-08-21 — Alignment risk spike and assisted fallback

- Alignment was elevated ahead of legend extraction because a plausible but geographically wrong warp would invalidate every downstream category.
- A project-local Python processor was created with pinned NumPy, OpenCV, SciPy, Shapely, PyProj, PyShp, Pillow, and Matplotlib versions. The processor downloads and hashes the official 2025 Census TIGER/Line state and county packages.
- The first automatic benchmark renders California in four candidate projections, fits a constrained affine-like source transform against multi-scale edges, reserves every fifth outline sample for a holdout score, and writes image/edge/residual diagnostics.
- `forest.jpg` produced a visually correct automatic California-Albers fit. At the 850-pixel working resolution, the first fit had a 1-pixel held-out median residual, a 3-pixel 90th percentile, and 96.3% of held-out samples within 5 pixels.
- `farms.png` exposed a critical false-confidence failure. The first fit reported a 1-pixel held-out median and 91.2% within 5 pixels while visibly matching California's outline to unrelated crop and terrain edges at the wrong scale and location.
- Directional edge agreement, saturated-data containment, expanded partial-map bounds, and county-network evidence improved the scale of the agricultural proposal but did not make it geographically correct. Dense hillshade and crop edges still provide enough accidental local matches to defeat pointwise chamfer scoring.
- The automatic score is therefore diagnostic only and no acceptance threshold has been set. The agricultural run is retained as a counterexample proving that low residual-to-generic-edge distance is not sufficient validation.
- A local assisted-alignment interface was implemented. Todd can click a landmark on the source and the corresponding landmark on a California-Albers reference with county lines, repeat for at least four widely separated pairs, inspect a live cyan outline preview, and save a server-recomputed projective homography.
- Assisted saves produce an explicit `alignment_mode: assisted` manifest, control-point residuals, a source-space outline overlay, and a source image warped into a canonical reference inspection canvas. Browser verification confirmed the page loads, renders both canvases, enables save after four pairs, and completes the save request without an error overlay.
- The browser test used synthetic non-geographic clicks only to verify the interaction and persistence flow; it is not evidence that `farms.png` has been correctly aligned. Correct agricultural control points require author inspection.
- Todd clarified that `farms.png` is a partial California map. The original automatic objective incorrectly assigned a missing-edge penalty to every authoritative outline point outside the image, structurally favoring a smaller false California fit inside the frame.
- A separate `partial_state` coverage model was added. It permits the state center to lie off-canvas, permits the full state to be several times larger than the image, evaluates only visible reference samples, and records the visible-reference fraction. Full/most-state and partial-state diagnostics are now written separately.
- The first partial-state attempt matched California's straight borders to the inset rectangular map frame. After suppressing the outer three percent of layout edges, a second attempt matched a visible California arc to the colored agricultural-data spine rather than a real coastline or border. Both remain rejected.
- This correction narrows the automatic-alignment research problem: partial maps need geography-specific evidence, such as a detected land-ocean boundary or a semantically identified administrative line. Generic edge proximity, even when visibility-aware, remains insufficient.
- Todd identified the decisive source feature in `farms.png`: the California-Nevada border is not just a diagonal line. A near-vertical segment descends beside the legend and intersects the southeast diagonal at the Lake Tahoe bend. Earlier experiments ignored the vertical segment and consequently attached the reference border to unrelated long agricultural lines.
- The next detector required a connected vertical-plus-diagonal Hough-line pair with a shared junction, then matched the corresponding authoritative California-Nevada border and its geographic hinge. This reduced the source evidence from hundreds of plausible straight edges to one piecewise border candidate.
- `farms-auto-v9` is the first strong automatic candidate for the partial agricultural map. Its detected source hinge and transformed authoritative hinge differ by 0.82 pixels at the 900-pixel working resolution; the visible coastline/state-outline holdout has a 1-pixel median residual, and 90.1% of sampled visible county evidence lies within 5 pixels. The overlay follows both the eastern-border junction and the visible coast. It remains a candidate pending Todd's visual confirmation and is not yet counted as an accepted automatic result.
- The geography-specific change did not regress the complete-state example. `forest-auto-v4` retained a visually correct California-Albers full-state fit with a 1-pixel held-out median, 2-pixel 90th percentile, and 99.2% of held-out outline samples within 5 pixels.
- The diagnostic now draws the authoritative eastern border in orange and circles the detected source hinge in magenta, making the exact alignment assumption inspectable instead of hiding it inside a scalar score.
- `geologic.pdf` was added as the first PDF corpus example. It is a 23 MB, single-page, large-format 2010 California Geological Survey poster authored in Adobe Illustrator. It is a hybrid document rather than a flat scan: inspection found 484 raster image objects, 96,483 curves, 40,017 lines, 146 rectangles, 48,470 text characters, and 16 embedded fonts.
- The PDF has no GeoPDF `/Measure`, `/VP`, or `/LGIDict` georeferencing dictionary, so it still requires visual/geometric registration. Its native page text is extractable, including the geological legend, which should allow direct PDF-object legend parsing before OCR fallback. The main raster content is stored in 6,825-pixel-wide JPEG 2000 strips near 150 dpi; the PDF remains the canonical source rather than a low-resolution screenshot.
- A 2,200-pixel diagnostic rendering was run through the unchanged automatic alignment benchmark. `geologic-pdf-preview-v1` looked plausible and reported a 1-pixel held-out median at the 900-pixel working resolution, but Todd correctly rejected that evidence as insufficient: small low-resolution drift becomes materially wrong when zooming into the source. The earlier “strong candidate” characterization is superseded.
- This example adds two pipeline requirements to validate: PDF intake should preserve native raster/vector/text assets where useful, and a poster may contain multiple semantic legend-like regions. Here the main geological legend defines map categories, while the geological time scale is explanatory rather than a second set of map layers.
- The poster text explicitly declares “Projection is Teale Albers, 1983 North American Datum,” corresponding to EPSG:3310. A 4,500-pixel California-Albers similarity refit improved the generic edge metrics, but its detected eastern-border lines came from poster furniture and the dense geology could still fool nearest-edge validation. That run remains rejected as an acceptance method.
- Native PDF inspection found 22 long vector graticule curves: 11 meridians from 125°W through 115°W and 11 parallels from 32°N through 42°N. The curve families yield 72 visible intersections distributed over the poster. Fitting EPSG:3310 directly to them produces 0.044 PDF-point RMS, 0.065-point 90th percentile, and 0.086-point maximum residual; the rendered diagnostic is one pixel per PDF point. Affine scale anisotropy is only 1.00012.
- `geologic-pdf-graticule-v4` is therefore the first subpixel-validated registration for this source. Its transform is based on geographic grid evidence, not state-outline proximity. The 2025 Census state and county overlays remain independent visual diagnostics; local differences can reflect the 2010 map's line generalization, coastline vintage, and stroke thickness rather than registration drift.
- A reusable `align-pdf-graticule` command now detects a bipartite vector curve grid, ties its curves to native degree labels, infers one missing label in an otherwise contiguous sequence when Illustrator splits the text, fits the declared CRS, and records the full residual distribution. Future PDFs without usable graticules still fall back to border/coast alignment or the explicitly assisted path.
- The accepted transform was used to create a 250-unit Web-Mercator inspection grid with nearest-neighbor sampling. `geologic-pdf-graticule-v5/warped-inspection.png` clips the source to 2025 Census California and overlays state and county references, preventing the poster title, legends, and neatline from appearing as map data outside the state. This is the same target CRS family used by Mapbox and is the direct zoom-level alignment check Todd requested.
- Native legend inspection found 53 unique colored swatches in the main geologic legend. Their CMYK fills can be extracted exactly from PDF rectangles rather than estimated from anti-aliased rendered pixels. Tesseract plus embedded text recovers many unit codes, but several codes use custom Illustrator font encodings and remain provisional. Descriptions and hierarchy are not yet attached, so this is a successful color-extraction diagnostic rather than an accepted semantic legend.

### 2026-08-21 — Four new raster cases and multi-variable extraction

- Four additional examples expanded the corpus beyond simple mutually exclusive color classes: `deer.png`, `fire.webp`, `landslide.png`, and the 3,750-by-4,500-pixel `quake.jpg` poster.
- The earlier one-variable-per-image assumption was superseded. A source may contain several independent variables; each variable receives its own categorical raster or binary mask set. Categories remain mutually exclusive only within a single categorical variable.
- A plan-driven extractor was added. An AI-readable JSON plan records the semantic legend interpretation, while deterministic code performs Lab classification, ambiguity rejection, California clipping, nearest-neighbor Web-Mercator warping, per-category masks, hashes, counts, and diagnostics. These first plans contain AI-proposed labels and legend sample regions and are not yet evidence that layout and legend discovery are fully automatic.
- Every extraction now preserves an original-dimension byte-valued class raster and writes a north-up Web-Mercator class raster. A warped-source inspection image overlays the 2025 Census state boundary in cyan and county lines in magenta so drift can be inspected at output resolution.
- `deer.png` is the clean control case: ten legend categories use flat colors, including a white “rare or absent” class. The California-Albers alignment has a 0-pixel source-scale held-out median and 4.81-pixel 90th percentile. County and state diagnostics are visually coherent, and the classified output is ready for author review rather than automatic publication.
- `fire.webp` separates three fire-hazard categories from an independent Local Responsibility Area dot hatch. Hazard extraction is clean. Navy is also reused for county lines, proving color alone cannot recover the hatch; a conservative dot-texture filter retains 1,172 high-confidence visible dot pixels and omits uncertain sparse edge dots. The LRA mask remains provisional even though the Conus-Albers state/county registration is strong (0-pixel median, 1.90-pixel 90th percentile at source scale).
- `landslide.png` is represented as four variables: a three-class maximum-daily-precipitation raster plus binary landslide-susceptibility, maximum-wind-speed, and predicted-flooding masks. Sparse Lab-chroma unmixing inferred 5,011 pixels with two simultaneous transparent overlays and left 5,911 chromatic pixels ambiguous. It explicitly cannot recover opaque or three-way occlusions.
- The precipitation legend swatches are darker than the same classes as rendered transparently over the map. Exact swatch matching therefore under-extracted the data. The revised deterministic stage learns three map-rendered luminance modes (approximately 224, 172, and 142) while preserving the legend's light-to-dark ordering. This produces plausible broad bands but remains provisional because precipitation and hillshade both occupy the luminance channel.
- `quake.jpg` uses six ordered Modified Mercalli classes, each displayed as a within-class color gradient over hillshade. Sampling each legend row into multiple prototypes preserves one stable class ID per MMI level rather than inventing a continuous physical surface. The latest run classified 3,793,119 state pixels and left 1,294,008 eligible pixels ambiguous or NoData, including labels, water, and nonmatching relief colors.
- The earthquake classification is promising, but its alignment is not publishable. The coarse automatic Web-Mercator candidate has a 12.27-pixel source-scale held-out median and 43.54-pixel 90th percentile. A constrained 2,200-pixel local refinement improved the eastern-border median but did not improve global 90th-percentile or county residuals. The warped county inspection makes the localized drift visible, so the result remains rejected pending stronger landmark/county registration or assisted controls.
- The current triage and review states are recorded in `benchmarks/corpus-triage-002.json`. No new dataset has been marked accepted for the public catalog.

### 2026-08-21 — Precipitation background correction

- Author review caught a semantic classification error in the first `landslide.png` precipitation result: the adaptive grayscale calibration shifted the lightest 4–8-inch class from the approximately 171-gray legend swatch to approximately 224, which is the pale terrain basemap rather than precipitation.
- The earlier claim that the precipitation classes were rendered transparently at approximately 224, 172, and 142 is superseded. Direct inspection supports the literal legend values of approximately 171, 139, and 91; the much lighter white-to-230 hillshade background must remain NoData.
- `landslide-extract-v5` disables adaptive center movement, limits accepted luminance to within 20 levels of a literal legend swatch, and reduces median filtering from a 31-by-31 to an 11-by-11 neighborhood. This removes the falsely classified Central Valley and desert background while preserving the narrow darkest precipitation class.
- The corrected source result contains 36,344 light, 15,486 medium, and 452 dark precipitation pixels. It remains provisional because locally dark hillshade can still resemble a legend class, but pale background is no longer intentionally absorbed into 4–8 inches.
- A regression test now verifies that pixels at the three legend grays retain their class IDs while a pale 230-gray basemap pixel remains NoData.

### 2026-08-21 — High-precision assisted quake controls

- The first assisted interface was insufficient for `quake.jpg`: it displayed the 3,750-by-4,500 source at fit-to-panel scale and the reference contained counties but no matching city evidence.
- The reference now includes twelve projected California city landmarks that occur on the quake poster. Reference clicks within 16 canvas pixels snap to the exact projected city coordinate, removing author estimation error on that side of each pair.
- Independent 1x, 2x, 4x, and 8x source zoom controls and up to 4x reference zoom were added. The quake session starts at 4x source and 2x reference zoom so source clicks can target the centers of printed city dots rather than their labels.
- The recommended manual baseline uses 8–12 geographically distributed cities. Its residuals will determine whether a global projective transform is adequate or whether the source needs a projection-aware or nonlinear warp.

### 2026-08-21 — City landmarks demoted to validation evidence

- Todd correctly noted that cartographic city dots are not placed consistently enough across unrelated maps to be authoritative registration evidence. Coastline shapes and state borders remain the primary alignment basis.
- The quake assisted transform therefore continues to use only eight author-selected coastline and state-border anchors. Its control residual is 4.16 source pixels median, 6.59 pixels RMS, and 12.21 pixels maximum; a withheld generic-outline diagnostic has an 11.35-pixel 90th percentile at the 3,750-by-4,500 source scale.
- Redding, Sacramento, Fresno, and Los Angeles were retained only as interior holdouts. Under the boundary-only transform their dot residuals are approximately 9.0, 12.5, 25.2, and 19.5 pixels. Those values warn about possible local distortion but do not directly prescribe a warp because the dot symbols themselves may be displaced.
- Projection-family testing fit each candidate only to the eight boundary controls. California/CONUS Albers and contiguous-US Lambert variants produced similar city holdout medians of roughly 15–16 pixels; Web Mercator was substantially worse at roughly 34 pixels. This suggests the remaining discrepancy is not solved by switching to Mercator and reinforces the Albers/Lambert family.
- Assisted homographies are now a supported extraction input without being relabeled automatic. `quake-assisted-extract-v1` uses the saved eight-point boundary transform, writes a 3,398-by-3,920 Web-Mercator inspection raster, and classifies 3,829,579 pixels across the same six Modified Mercalli categories while leaving 1,171,859 pixels ambiguous or NoData.
- The full-resolution county overlay is a secondary visual check. It is more geographically meaningful than city dots when the source actually depicts counties, but differences in line generalization and vintage still prevent treating nearest-edge distance as ground truth without inspection.

### 2026-08-21 — Full-resolution alignment acceptance gate

- A local extraction reviewer now makes alignment approval an explicit, reproducible gate. It supports wheel zoom, drag pan, fit-to-view, and independent opacity controls for the clean warped source, extracted class preview, authoritative state/coast outline, county boundaries, and a county residual layer.
- State border and coastline remain the primary evidence. County residual samples within six pixels of the authoritative state edge are excluded so the artificial California clipping edge does not inflate agreement. Remaining county residuals are colored green at no more than 3 pixels, amber at 3–8 pixels, and red above 8 pixels.
- County nearest-edge distance remains diagnostic only: earthquake faults, relief, labels, and other linework can create false low values, while differences in county vintage, stroke width, or generalization can create false high values. It cannot automatically approve the transform.
- For `quake-assisted-extract-v1`, 40,368 county samples have a 2.83-pixel median nearest-source-edge distance and 17.12-pixel 90th percentile; 54.8% fall within 3 pixels and 72.8% within 8 pixels. These numbers accompany, but do not supersede, the eight boundary-control residuals of 4.16-pixel median, 6.59-pixel RMS, and 12.21-pixel maximum.
- The reviewer was browser-verified at fit view and 225% zoom with both source and extracted classes. It records Approve, Revise, or Reject in a separate `review-decision.json`, including author notes and the hash of the extraction manifest reviewed. No decision has yet been made for the quake run.
- Todd found that words alone were insufficient to communicate localized county-line drift. The reviewer now has a correction-input mode: click-drag from the current source feature to its intended authoritative position to create a numbered displacement arrow, or click without movement to pin an already-correct area. Arrows can be undone, cleared, and saved.
- Saved `alignment-corrections.json` records the reviewed manifest hash, direction convention, raster and normalized coordinates, Web-Mercator coordinates, and displacement magnitudes. It is input to a subsequent warp refinement rather than an in-place edit, preserving the original assisted extraction and making the author's correction evidence auditable.
- Browser verification exercised a real pointer drag, rendered its arrow at zoom, and then removed the synthetic state without saving it. The quake reviewer was returned to a clean fit view with no correction file present, ready for Todd's distributed correction vectors.

### 2026-08-21 — Author-guided quake refinement

- Todd added 21 full-resolution displacement arrows and marked the preceding assisted extraction `needs_revision` with the note `see arrows`. The arrows span 74.3% of the Web-Mercator review width and 85.1% of its height, with a 23.16-pixel median and 34.70-pixel maximum requested displacement. Most corrections move the source west, with smaller latitude-dependent vertical changes.
- Candidate transforms were evaluated in increasing flexibility. Translation failed with 12.03-pixel leave-one-out P90; similarity failed with 10.30 pixels. Affine passed with 1.91-pixel leave-one-out median, 3.21-pixel P90, and 5.76-pixel maximum. Projective also passed but was rejected as unnecessary because affine is simpler and has a constant positive determinant of 1.0275.
- A reusable `refine-alignment` command now validates control count and geographic spread, evaluates every candidate with leave-one-out residuals, selects the first passing model, records parent and correction hashes, and emits a composable Web-Mercator sampling correction without overwriting the parent alignment.
- `quake-reviewed-affine-extract-v1` regenerates the clean source, categorical output, overlays, and residual diagnostics on the same 3,398-by-3,920 review grid. The correction is applied during resampling from the original 3,750-by-4,500 source rather than by repeatedly warping the previous JPEG or class raster.
- Direct point verification confirms that the new target locations sample the source features identified by Todd's arrows. The generic county nearest-edge diagnostic nevertheless worsened from 2.83 to 4.00 pixels median and from 17.12 to 27.46 pixels at P90. This reinforces that dense fault, relief, and label edges make that scalar metric an unreliable judge; the author-identified correspondences and full-resolution visual overlay remain decisive.
- The refined run remains `not_reviewed`. Its reviewer is served separately so the original `needs_revision` run and all 21 arrows remain auditable.

### 2026-08-21 — Arrow-direction bug found and corrected

- Todd reviewed `quake-reviewed-affine-extract-v1` and correctly reported that the first correction appeared to fix nothing. He added 16 more arrows rather than accepting the output.
- Comparing repeated locations across both rounds exposed the bug. Todd consistently started each drag on the fixed authoritative cyan/magenta line and ended on the corresponding source feature. The schema and UI claimed the reverse. The first fit therefore moved source content away from the reference, approximately doubling the next requested displacement from a 23.16-pixel to a 48.39-pixel median.
- The second-round evidence was decisive: several reference-start coordinates were effectively identical across both outputs while the source endpoints moved farther away. The error was in MapScan's direction convention, not Todd's control selection.
- Correction schema version 2 now stores explicitly named `reference` and `source` endpoints. The UI says “start on reference, end on source,” colors them separately, and explains that processing moves the source back toward the start. Legacy ambiguous sessions require an explicit `--reverse-arrows` recovery flag.
- Refinement composition was also fixed. A new correction round is multiplied into the parent's target-to-sampling transform instead of silently replacing it. A regression test verifies the matrix composition.
- Reversing the original 21 arrows yields an affine correction with 1.88-pixel leave-one-out median, 3.19-pixel P90, and 5.74-pixel maximum. After accounting for the failed intervening warp, this corrected first-round transform predicts the independent 16 second-round arrows with 1.81-pixel median, 2.63-pixel P90, and 3.06-pixel maximum residual.
- `quake-reviewed-corrected-extract-v1` was regenerated directly from the original source. Its generic county diagnostic now agrees: 1.00-pixel median, 3.00-pixel P90, 90.5% within 3 pixels, and 92.6% within 8 pixels. The failed reverse-direction run is preserved rather than overwritten.
- Browser verification at 265% zoom shows the corrected county and state overlays following the source linework. The run still requires Todd's visual approval; these improved metrics are corroboration, not self-approval.

### 2026-08-21 — Fine translation refinement

- Todd described the corrected port 8769 result as much better and added three smaller schema-v2 corrections. The arrows request 5.24–6.55-pixel source displacements, span 51.8% of the review width and 81.6% of its height, and consistently point roughly 4–5 pixels east and 2–5 pixels south.
- Three points cannot support a responsibly held-out affine fit. The refinement gate was generalized to allow small distributed follow-up sets while evaluating only models that retain at least one point for leave-one-out validation. Translation and similarity were evaluated; affine and projective were explicitly skipped.
- Translation is the simplest passing model, with a 1.60-pixel leave-one-out median, 2.56-pixel P90, and 2.80-pixel maximum. Similarity failed with a 9.06-pixel P90 and 10.34-pixel maximum, showing that its slightly lower training error was overfit.
- `quake-reviewed-corrected-extract-v2` composes the translation with the corrected affine baseline and again resamples from the original source. The broad county diagnostic changes from 1/3-pixel median/P90 to 2/6 pixels, while 93.0% remain within 8 pixels. This is a deliberate visual-review tradeoff: three author-identified local correspondences improve, but a thick-line nearest-edge score becomes less favorable.
- Browser verification at 265% zoom found no gross regression, but Todd's direct comparison did: he reported that port 8770 was worse than port 8769. The 8770 decision is therefore recorded as rejected, while 8769 remains the accepted comparison baseline.

### 2026-08-21 — Local fine correction replaces global translation

- Todd ultimately saved five fine arrows, all concentrated along the eastern and northern alignment corridor. Their independent x/y spans looked broad enough for the old global gate, but their normalized convex hull covers only about 4.6% of the review canvas. They were local evidence, not justification for moving the whole state.
- The global refinement gate now requires substantial normalized convex-hull area as well as broad axis spans. The five-arrow set is rejected for a global model even though a translation can score well on leave-one-out residuals.
- A compact Wendland C2 displacement fitter was added for local corrections. It maps each nearby source endpoint back to its authoritative reference start, uses a 500-pixel support radius, decays smoothly to exactly the parent alignment outside those neighborhoods, and composes with the corrected affine transform rather than resampling an intermediate raster.
- The five controls are fit with zero numerical residual. On an 81-by-81 safety grid, the local mapping's Jacobian ranges from 0.959 to 1.032, so it has no sampled foldovers and only modest local scale change.
- `quake-reviewed-local-extract-v1` preserves the 8769 statewide county diagnostic to the precision that matters here: 1-pixel median and 3-pixel P90. The fraction within 3 pixels changes from 90.5% to 90.3%, while the fraction within 8 pixels remains 92.6%. Port 8771 is the new author-review candidate; it is not self-approved by these metrics.

### 2026-08-21 — Second composed local correction

- Todd inspected port 8771 and saved four additional minor displacement arrows in that review state. They are therefore composed on top of 8771 rather than refit against the older 8769 coordinate state.
- The second local correction uses a tighter 350-pixel support radius. Its four controls have zero numerical residual, kernel condition 1.0, and a sampled Jacobian range of 0.945–1.053. The tighter radius limits each adjustment's influence while preserving smooth falloff.
- `quake-reviewed-local-extract-v2` is served on port 8772. Its county diagnostic remains 1-pixel median and 3-pixel P90; the fraction within 3 pixels rises from 90.3% on 8771 to 90.6%, and the fraction within 8 pixels remains 92.6%. These are corroborating diagnostics only; Todd's visual comparison remains the acceptance gate.

### 2026-08-21 — Quake alignment accepted; classification becomes a separate gate

- Todd reported that port 8772 looks good. That statement is recorded as approval of the alignment scope, with the extraction-manifest hash and the established coastline/state/county evidence policy. It does not silently approve the six extracted Mercalli classes.
- A separate full-resolution classification reviewer now reads the lossless Web-Mercator class-ID PNG and recolors it in the browser. Each legend item can be toggled independently or isolated with Solo while source and class opacity, pan, and zoom remain adjustable.
- The current quake extraction classifies 3,854,226 of 5,129,637 eligible pixels, or 75.1%; 1,275,411 remain ambiguous or NoData. Precision remains the stated priority, so the review explicitly calls for checking false positives in relief, labels, water, borders, and pale background as well as missing areas.
- Classification approval is persisted independently in `classification-review-decision.json` and is refused unless alignment is already approved. Port 8773 is the first classification-review session for the accepted quake alignment.
- Browser verification loaded all six category counts, exercised Solo mode, restored the full set, and visually confirmed the crisp combined overlay. The classification remains `not reviewed` pending Todd's inspection.

### 2026-08-21 — Quake classification accepted; optional occlusion inference

- Todd reviewed the six Mercalli categories on port 8773 and reported that they look good. The classification scope is now approved against extraction manifest `8ff690a…`; the observed class-ID rasters remain unchanged and authoritative.
- Todd explicitly allowed assumptions to fill holes created by city dots and names. The earlier blanket prohibition on hidden-data recovery is superseded by a narrower rule: observed pixels stay canonical, while conservative reconstruction is a separate, toggleable, hashed, and independently reviewed inference artifact.
- The first deterministic inference closes only small zero-valued gaps proposed by one class and requires at least 98% same-class dominance in the observed surrounding ring. It limits source components to 1,200 pixels and 120 pixels in either dimension and rejects cross-class conflicts.
- The quake candidate infers 75,553 pixels, 1.96% of the observed classified count, across 13,366 small components with zero cross-class conflicts. The mask visibly includes city/name letterforms but also small compression and relief gaps, so it is not self-approved.
- Port 8774 adds a neon-cyan inferred-fill layer and a third decision gate. Alignment and observed classification remain approved regardless of whether the optional inference is approved, revised, or rejected.

### 2026-08-21 — `rivers.jpg` creates a named-feature branch

- The new 1,600-by-1,929 image is not actually devoid of semantic explanation: a compact lower-left symbol key distinguishes solid rivers, dotted dry streambeds, outlined lakes/reservoirs, and pale dry lakes. Unlike categorical area maps, individual features are named directly on the map.
- Automatic California-Albers alignment is a strong first candidate: 0-pixel median, 2.14-pixel P90 at source resolution, 98.3% of outline holdouts within 3 working pixels, and full visible-reference coverage. The state/coast overlay visually follows the source.
- A broad HSV blue-ink diagnostic identifies 125,849 pixels, about 4.08% of the image. It cannot be the final data mask because rivers, lake outlines, and their labels deliberately use the same ink.
- Baseline Tesseract recognizes many horizontal lake labels but fragments curved and rotated river names. The feature branch therefore needs rotation-aware OCR or vision, controlled California-hydrography name validation, and geometric association between names and nearby lines or polygons.
- A categorical raster is not the highest-fidelity public representation here. The proposed contract preserves an observed blue-ink raster, derives separate solid-line, dotted-line, lake, and dry-lake geometry, attaches validated names, and records label-gap reconnections as inferred. The initial contract is in `examples/feature-plans/rivers.json`.
- `rivers-feature-ink-v1` implements the first non-semantic stage. It retains 125,849 blue pixels before California clipping and 99,526 after clipping, represented by 904 connected components. The Web-Mercator preview cleanly preserves river/lake ink and labels while rejecting most relief and gray administrative borders.
- Text-versus-geometry separation is now the blocking semantic choice. The extraction implementation also needs a product decision on whether names define individually interactive features or merely label four toggleable symbol groups.

### 2026-08-21 — Quake label and city-dot reconstruction strengthened

- Todd reported that the first optional quake inference still left many holes where city names and city dots interrupted the colored shaking surface. The original 75,553-pixel pass only closed narrow same-class gaps; it was structurally unable to repair broad labels or labels crossing several intensity bands.
- A first OCR-aware attempt exposed two separate failures. Requiring one class to dominate an entire label neighborhood accepted only one word. In addition, Python's default TSV quote handling treated a leading quotation mark in an OCR token near Salinas as the start of a multiline field, silently swallowing many southern California rows. Tesseract TSV is now parsed with quoting disabled, and a regression test preserves that behavior.
- The replacement inference assigns each missing pixel in a label neighborhood to its nearest observed class. It keeps pixels transparent when the closest class is too distant or the two closest classes are nearly tied, allowing a city name to cross an intensity boundary without flattening that boundary.
- OCR on the unfiltered poster remained noisy because colored relief and explanatory copy overwhelmed the city labels. A deterministic preprocessing pass now retains only dark, nearly neutral pixels inside the California source mask, producing an unusually clean page of the black city typography and city dots while discarding the colored hazard surface.
- The reusable `detect-map-labels` command runs that isolation and sparse-page Tesseract OCR, hashes its inputs and outputs, and warns that detected labels are occlusion proposals rather than data evidence. Page segmentation mode 3 recognizes 56 city-name tokens on the quake poster, including the labels previously missed in southern California.
- Complementary page-segmentation modes 3 and 11 recognize slightly different multi-line cities, so the inference command now accepts repeated `--ocr-tsv` inputs. Overlapping proposals are deduplicated, and any pixel receiving different class proposals is excluded as a conflict.
- `inference-v6` combines the original narrow-gap morphology with nearest-class reconstruction across both detected-label passes. It marks 177,903 inferred pixels in total, of which 112,780 are proposed by OCR neighborhoods; there are zero cross-class proposal conflicts. This is 4.62% of the 3,854,226 observed classified pixels. The observed raster is still unchanged and authoritative.
- The classification reviewer now selects the highest versioned inference directory and ignores stale inference decisions whose manifest hash does not match the selected artifact. Port 8775 serves `inference-v6`; the cyan mask visibly follows city names and dots throughout California and still requires Todd's independent approval.

### 2026-08-21 — Manual repair becomes a clone stamp, not a color brush

- Todd proposed a rubber-stamp tool instead of painting one selected legend color. That is a materially better fit for map occlusions because one city label can cover several intensity classes and a curved class boundary.
- The classification reviewer now has a clone-stamp mode. The author selects an intact source patch and a radius, then clicks or drags over a target hole. Each stamp copies the source patch's complete class-ID pattern while refusing to overwrite any observed nonzero class pixel.
- Manual values render in their real legend colors above the cyan automatic-inference layer. Source and target circles remain visible while stamping; Clone mode can be toggled off to restore ordinary pan behavior. Undo, clear, source reset, radius, opacity, and save controls are available.
- Saved `stamp-corrections.json` operations record the layer, source point, target point, radius, Web-Mercator inspection grid, extraction hash, and selected inference hash. The policy explicitly keeps observed, automatic inference, and manual repair as separate evidence layers.
- Python replay and validation tests confirm that a single stamp can reproduce a multi-class pattern, observed target pixels cannot be overwritten, invalid radii are rejected, and saved corrections are exposed only with matching artifact hashes. The browser reviewer loads with zero saved synthetic operations; Todd's first real stamp session starts clean.

### 2026-08-21 — Over-inference becomes an auditable negative correction

- Todd found that some OCR-aware reconstruction extended beyond defensible city-label holes. Regenerating the whole candidate for every local false positive would make review slow and would obscure which pixels were rejected by the author.
- Port 8775 now includes a separate inference eraser. Its adjustable circular brush can affect only pixels present in the automatic inference mask; observed class pixels, manual clone stamps, and the original `inference-v6` artifacts remain immutable.
- Undo and Clear rebuild the visible cyan layer from the original inference plus the current rejection operations. Coverage reports rejected and retained inference counts so subtraction is explicit rather than visually silent.
- **Save exclusions** records layer, center, radius, grid dimensions, extraction hash, and inference hash in `inference-exclusions.json`. Stale exclusions are ignored when either source artifact changes, and publication is specified to subtract the replayed exclusion mask from automatic inference.
- Browser verification confirmed the new controls render, tool activation disables clone mode, a real inferred test region can be subtracted and restored, and no synthetic exclusion file was saved. The complete test suite passes with 39 tests.

### 2026-08-21 — Clone brush replaced by a discrete patch stamp

- Todd clarified that the clone tool should behave like a rubber stamp rather than a paintbrush. The earlier drag-to-paint interaction and zero-target-only restriction are superseded.
- Holding **A** now momentarily enters source-selection mode. A click updates the source center; releasing **A** returns to destination placement with Clone active. The existing **Set source** button remains available.
- Each destination pointer-down records exactly one operation. Pointer movement does not generate additional stamps, so a drag cannot smear a repeated source pattern across the map.
- A stamp now copies the complete circular source class-ID patch, including zero-valued pixels, into a separate manual override footprint. That footprint visually replaces observed and inferred classifications at the destination, while their underlying artifacts remain immutable and recoverable.
- The saved correction schema is version 2 and explicitly records discrete-click, circular-patch, last-stamp-wins, and manual-override semantics.

### 2026-08-21 — Quake author corrections materialized

- Todd finished and saved the quake correction pass. The audit contains 2,154 placements from 104 source patches, spanning radii from 13 to 43 pixels. No separate inference-exclusion file was saved; manual footprints still supersede inference wherever they overlap.
- Deterministic replay produces a 737,919-pixel manual override footprint: 637,998 class-valued pixels and 99,921 explicit zero-valued pixels. The latter are part of the solid source patches rather than omitted data.
- `materialize-corrections` is now a reusable pipeline stage. It rejects stale extraction, inference, stamp, or exclusion hashes and applies a documented precedence of observed classification, retained automatic inference, then manual override.
- `materialized-v1` retains 95,973 automatic-inference pixels outside manual footprints and produces 4,679,774 final classified pixels. The exact final class IDs, preview, retained-observed mask, retained-inference mask, manual footprint, and manual values are independently hashed in `materialization.json`.
- The materialized candidate remains `needs_visual_review`. Manual author edits remain inferred evidence and do not rewrite the observed or automatic-inference source artifacts.

### 2026-08-21 — Automatic quake inference rejected

- Todd determined that the automatic inferred fill was not helpful and asked to remove it. This supersedes the earlier plan to retain a reviewed subset of `inference-v6` for the quake publication candidate.
- `inference-selection.json` now disables inference for the quake run. The generated inference directories remain intact as recoverable pipeline evidence, but the classification reviewer no longer loads or displays them.
- Saved clone-stamp operations remain valid because their source pixels come exclusively from the observed class-ID raster. They are still bound to the extraction hash; an inactive inference hash no longer causes the author work to disappear from review.
- `materialize-corrections` now honors the run selection and also exposes `--without-inference`. In this mode, the final precedence is only observed classification followed by manual override, and the retained-inference mask is explicitly all zero.
- The reviewer also removed its independent 90% manual-stamp opacity. Observed and manually stamped pixels now use the same legend RGB and the same Classes opacity, so identical class IDs are visually indistinguishable while the separate manual footprint remains auditable in saved artifacts.

### 2026-08-21 — Author-defined tiny enclosed-hole fill

- Todd replaced the rejected broad inference with a precise rule: a black/zero region may be filled only when it contains fewer than 50 pixels and is fully surrounded by one class color.
- The implementation uses 8-neighbor connected components. It rejects components touching the raster edge, components with more than one boundary class, components of 50 pixels or more, and any component containing an explicit zero-valued manual override.
- On the saved quake corrections, 20,280 components qualify, totaling 55,987 pixels. They are stored as separate class-value and mask PNGs in `enclosed-fill-v1`; the observed raster and 2,154 clone-stamp operations remain unchanged.
- The classification reviewer renders these pixels in the exact surrounding legend color and Classes opacity. `materialized-v3` contains no broad automatic inference, adds the enclosed-hole mask, and produces 4,639,788 classified pixels with precedence `observed → small enclosed zero fill → manual override`.

### 2026-08-21 — Painted corrections become clone sources

- Todd asked that pixels painted with the clone stamp count as source material for later stamps. New placements now snapshot the composed observed-plus-manual patch at operation time rather than sampling only the immutable observed raster.
- Stamp schema 3 records `source_mode: composite_at_operation_time`. Earlier schema-2 operations default to `observed`, so migrating the tool does not reinterpret or alter the 2,154 existing placements.
- Replay is explicitly sequential and snapshots each circular source patch before writing its destination. This prevents overlapping source and destination regions from feeding partially written pixels back into the same stamp.
- Saving stamps regenerates the tiny enclosed-hole artifact against the new correction hash and refreshes the reviewer. Manual zeros still suppress the derived fill, and publication can replay the same ordered operations without relying on browser state.

### 2026-08-21 — Quake correction pass completed

- Todd completed the painting pass with 2,168 saved stamp operations. The final 14 operations were authored after painted pixels became valid sources and are recorded as sequential `composite_at_operation_time` operations; the preceding 2,154 retain observed-source semantics.
- Regenerating the author-defined small-hole rule produces 20,173 components and 55,762 pixels after the final stamps. Broad automatic inference remains disabled.
- `materialized-v4` contains a 766,250-pixel manual footprint and 4,645,803 final classified pixels. Its exact artifacts and dependencies are hashed in `materialization.json`.
- Todd's statement, “ok this is done,” is recorded as an approval in `materialization-review-decision.json`, bound to the materialization manifest hash. The quake dataset is ready for the Mapbox delivery proof rather than further extraction repair.

### 2026-08-21 — First approved dataset reaches the delivery layer

- The approved quake materialization is exported as six independent transparent raster-mask pyramids, one per Modified Mercalli category. This preserves the final classified grid while allowing the browser to toggle, recolor, and change opacity for each legend item independently.
- Native export covers zoom levels 4 through 9 with nearest-neighbor sampling. The result contains 2,400 PNG tiles—400 per category—plus TileJSON, the source image, and a dataset manifest, totaling roughly 14 MB. Mapbox can overscale the highest native level for closer inspection without inventing smoothed class boundaries.
- Publication is approval-gated and deterministic. `export-raster-tiles` rejects missing, unapproved, or hash-mismatched materializations; each category records an aggregate tile-set hash. Public manifests retain provenance hashes but omit absolute machine-local paths.
- A separately deployable Next.js viewer now lives in `viewer/` with the intended `/mapscan` base path. Its desktop interface provides map navigation, independent category visibility, per-category colors and opacity, an original-source modal, and URLs that restore layer styling.
- The viewer shell, source modal, shared-link state restoration, and direct static-tile delivery were browser-verified. TypeScript, the production Next.js build, and all 50 Python tests pass. Displaying the live Mapbox basemap and masks is now gated only by adding Todd's public `NEXT_PUBLIC_MAPBOX_TOKEN`.

### 2026-08-21 — Three new maps broaden the test set

- Todd added `plantzone.avif`, `rainfall.gif`, and `elevation.gif`. They represent three materially different extraction cases rather than more copies of the quake workflow.
- The plant-hardiness image is a 13-class categorical map with a clean California silhouette. Re-running alignment against the original AVIF selects the full-state CONUS Albers candidate with a 0-pixel source median, 1-pixel P90, 97.1% of outline holdouts within 3 pixels, and complete visible-reference coverage. State/coast evidence looks exact in the interactive reviewer; county lines are secondary and are not present in the source.
- `plantzone-extract-v1` keys directly from all 13 legend swatches and conservatively classifies 99,238 of 118,032 in-state source pixels, or 84.1%. Compression-blended pixels, city text and dots, and outline strokes remain NoData. The input is only 801 by 694 pixels, so preserving its original pixel grid is the fidelity ceiling.
- The rainfall image contains 35 precipitation classes. Several legend swatches deliberately use dots, diagonal hatching, or crosshatching in addition to color. A single-color nearest-neighbor classifier would therefore conflate texture-defined classes; this case requires a local palette/texture fingerprint while preserving one mutually exclusive precipitation variable.
- Rainfall's first automatic full-state candidate is plausible but not yet accepted: its outline holdout is 0 pixels median and about 12.8 pixels P90 at source resolution. It is queued for interactive alignment review and, if necessary, correction arrows before classification can be trusted.
- The elevation image is continuous quantitative data mixed with geomorphic boundaries, hydrography, labels, and relief. Its automatic alignment is rejected because it locks onto only a partial outline and the legend inset edge. The source explicitly declares Lambert Conic Conformal and includes labeled graticules, providing stronger future registration evidence than generic image edges. Supporting it would add a continuous-raster branch beyond the current categorical core rather than discretizing the gradient and losing fidelity.

### 2026-08-21 — Live Mapbox delivery proof passes

- Todd added a public Mapbox token locally; the value remains outside tracked project files. Restarting the viewer loads the Mapbox Light basemap and all six independently hosted quake category-mask pyramids.
- Browser verification confirms a real Mapbox map region, working navigation controls, all six enabled categories, visible static masks aligned over California, and no console errors.
- Disabling one category removes only that mask. Reloading restores the default six-layer state and 82% opacity for each category. The original-source modal remains functional.
- Repeated zooming beyond the z9 native tile ceiling produces crisp square class pixels through nearest-neighbor overscaling rather than blurred categorical boundaries. The first public delivery architecture is therefore validated end to end: approved materialization to deterministic tiles to browser styling.

### 2026-08-21 — Plant-zone alignment accepted

- Todd inspected the interactive plant-hardiness alignment and approved it without correction arrows. The decision is bound to extraction manifest `acd8c67e…` and retains the established evidence policy: state border and coastline are primary, counties are secondary only when present, and city markers are validation evidence.
- Alignment approval advances the 13 legend-derived zones to their independent classification gate. `plantzone-extract-v1` remains conservative at 99,238 classified pixels out of 118,032 eligible pixels, or 84.1%; 18,794 uncertain source pixels remain transparent.
- Port 8782 serves the crisp full-resolution classification review. Every zone can be toggled or isolated with Solo, including the rare Zone 5a and Zone 11a candidates, before any category mask is approved or omitted.

### 2026-08-21 — Plant hardiness becomes the second published proof

- Todd saved 58 clone-stamp operations and then approved classification 1.45 seconds later, so the reviewed screen included the complete author correction state. All 58 operations use sequential `composite_at_operation_time` sources.
- Deterministic materialization applies only observed classification followed by manual overrides; automatic inference is absent. The stamps cover 14,868 Web-Mercator pixels: 14,403 class-valued pixels and 465 explicit transparent pixels. The final raster contains 146,033 classified pixels across 13 zones.
- The materialization approval binds the exact manifest, classification decision, stamp-correction hash, timestamps, operation count, and precedence. No unreviewed inferred artifact enters publication.
- Tile export initially revealed that legend-only categories defaulted to magenta when `display_rgb` was omitted. The exporter now falls back to `legend_rgb`, including the first prototype of a multi-color definition, and a regression test protects the behavior. All 50 pipeline tests pass.
- The corrected plant-hardiness export contains 13 independently recolorable mask pyramids and 5,200 PNG tiles from z4 through z9. The viewer catalog now contains both quake and plant hardiness.
- Browser verification switches between the two datasets without errors, loads all 13 plant classes in their true legend colors, displays the approved map on Mapbox, toggles an individual category, opens the original AVIF, and restores the selected dataset from the URL.

### 2026-08-21 — Rainfall forces color-and-texture classification

- The 35-entry precipitation legend cannot be reduced to one RGB value per class. Its indexed GIF uses ordered dithering, dots, and diagonal texture to render several colors, and the original raster reuses some of the same palette values across adjacent precipitation classes.
- Legend sampling was tightened to the true interior of every swatch. Earlier exploratory measurements accidentally included the black rectangle border for several rows, which would have taught the classifier that cartographic linework was category evidence.
- A format-independent patterned-category classifier now combines exact unambiguous legend colors with a five-pixel local Lab descriptor. The local descriptor contains center color, neighborhood mean and standard deviation, and direction-specific neighbor differences, allowing equal-average vertical, horizontal, and diagonal patterns to be separated without depending on GIF palette indices.
- Two regression tests define the conservative behavior: equal-mean textures with different orientation must separate, while two identical rendered swatches must remain NoData rather than being assigned by category order.
- The first texture-only pass retained 184,353 of 804,148 eligible state pixels, or 22.9%. The hybrid pass improves this to 440,192 pixels, or 54.7%: 399,476 are supported by unambiguous exact legend colors and 40,716 by local texture evidence.
- The legend itself exposes an information ceiling. The 2.5/3.5-inch pair has overlapping local evidence, and the 5.0/5.5/6.5-inch group is nearly or completely indistinguishable after rasterization. The extractor reports these pairs and leaves unresolved pixels transparent. Recovering them would require a separate isohyet-line topology and label-interpretation stage, not more aggressive color guessing.
- The first Web-Mercator alignment candidate is open on port 8783. Its diagnostic county-line median is 1 pixel and P90 is 3.16 pixels, and the fit view visually follows the state border and coast. Those metrics are corroboration only; publication still requires full-resolution author inspection and any needed correction arrows.

### 2026-08-21 — Plant hardiness receives a second correction pass

- Todd returned to the plant-hardiness classification after its first publication and expanded the saved clone-stamp audit from 58 to 118 operations. The classification approval was saved 1.27 seconds after the new stamp file, binding the approval to the complete updated composition rather than the older public candidate.
- `materialized-v2` deterministically replays observed classification followed by the 118 manual override patches, with no automatic inference. The manual footprint grows from 14,868 to 16,370 Web-Mercator pixels, including 15,843 class-valued pixels and 527 explicit transparent pixels.
- The final classified total grows from 146,033 to 146,882 pixels. The new materialization approval records the exact classification-decision hash, stamp hash, timestamps, operation count, precedence, and materialization hash.
- A new 5,200-tile plant-hardiness export was generated in staging, approval-gated, and then promoted to both `publish/` and the viewer's static assets. All three dataset manifests share SHA-256 `99a1754e…`.
- Live viewer verification reloads the updated plant dataset with a Mapbox canvas, all 13 category controls, and no browser warnings or errors.

### 2026-08-21 — Rainfall completion replaces broad NoData holes

- Todd rejected the first rainfall classification because large black regions remained even where the source visibly contained a precipitation fill. The complaint was correct: the conservative pass recovered only 440,192 of 804,148 eligible source pixels.
- An exact GIF-palette audit separated the failure into evidence types. Every pixel color belonging to exactly one legend class had already been recovered. The missing set contained 150,056 pixels whose exact palette color is shared by multiple legend entries, plus 213,900 non-legend pixels from boundaries, labels, graticules, anti-aliasing, and other cartographic ink.
- The replacement completion has three deterministic stages. Shared colors first receive high-confidence class seeds from a nine-pixel local palette histogram; remaining shared pixels inherit only from geographically nearest classes that use that exact color in the legend; non-legend pixels inherit the nearest classified region while dark non-legend ink stays transparent.
- Completion is not silently relabeled as observation. The run writes separate source and Web-Mercator masks for all completed pixels and for preserved dark ink, hashes all four artifacts, and reports assignments and counts by evidence stage.
- Todd also saved 12 distributed alignment arrows before approving the preceding review. They cover 79.7% of the review width, 82.1% of its height, and 26.5% of the normalized canvas convex hull. Eleven arrows broadly request an east-and-north translation, while one southeastern arrow requests the opposite horizontal motion.
- The simplest translation moves source content 3.26 pixels east and 4.41 pixels north. Its leave-one-out median is 1.96 pixels, P90 is 3.02 pixels, and maximum is 8.27 pixels. The standard eight-pixel maximum gate was narrowly relaxed to 8.5 rather than fitting a more flexible global model to one disagreeing local arrow.
- Applying review corrections to an automatic alignment exposed two compatibility bugs: the parent transform model lived under `best` rather than at the manifest root, and extraction ignored a root-level review correction for automatic runs. Both paths now support reviewed automatic alignments and have regression coverage.
- `rainfall-reviewed-extract-v1` initially classified 727,451 of 804,671 eligible pixels, or 90.4%. That metric hid a bad assumption: all 77,220 dark non-legend pixels were protected wholesale as if they were useful thin linework. A literal source-to-class diff showed that this bucket also contained city names, city marks, graticules, coordinate labels, and anti-aliased fragments. Todd correctly rejected the result because these decorations appeared as black holes in the data raster.
- `rainfall-reviewed-extract-v3` treats every non-legend source pixel as an occlusion and reconstructs it from the geographically nearest classified precipitation region. The source diff is now exact: 804,671 of 804,671 California-mask pixels have a class, with zero internal NoData. A second post-warp diff found 1,195 edge cells where alignment sampling fell just outside the source mask; those are reconstructed after clipping, leaving zero internal NoData among 1,048,307 Web-Mercator state cells. Completion masks continue to distinguish the 364,779 reconstructed source pixels from direct observations.
- The translated warp improves the broad county-edge diagnostic from a 3.16-pixel P90 to 1.41 pixels, with 97.8% of samples within 3 pixels and 99.4% within 8 pixels. Port 8783 now serves this regenerated candidate and requires new visual approval because the prior approval targeted the pre-translation run.
- A semantic information ceiling remains: the source GIF renders the 5.0, 5.5, and 6.5-inch swatches with the same two palette colors in exactly the same proportions, and 5.5 and 6.5 are pixel-for-pixel equivalent under the ordered dither. Separating those values would require isohyet-line topology and label interpretation; color or texture cannot recover information absent from the raster symbol.

### 2026-08-21 — Source-to-class diff becomes a corpus gate

- Todd approved the repaired rainfall appearance but required the same literal source-to-class comparison on every other source before another author check. This changed the workflow from a map-specific fix into a required layer-aware QA stage.
- Every processed layer now declares either `full_state` or `sparse_visible_evidence`. Full-state categorical surfaces must have zero NoData inside the authoritative California mask after deterministic nearest-class occlusion completion. Sparse layers retain legitimate transparent background and instead fail only when visible source-derived evidence is dropped.
- The first seven-source batch exposed 48,669 remaining Web-Mercator gaps in deer distribution, 20,176 in the manually corrected plant-hardiness candidate, and 1,204,431 in the manually corrected quake candidate. New audited candidates complete all of them and end with zero internal NoData; the changed pixels remain separately masked and hashed.
- Fire hazard and ARkStorm landslide/precipitation passed the sparse contract without filling their intentionally absent background. Rivers reproduced both its source and Web-Mercator blue-ink evidence exactly, with zero missing or extra pixels; semantic separation of labels from geometry remains a later gate rather than being confused with evidence retention.
- Rainfall already passed with zero gaps. Farms, forest, elevation, and the geologic PDF are recorded as pending because they have alignment work but no class or feature extraction to diff yet. They must enter the same gate when semantic extraction exists.
- `audit-source-diff`, `audit-feature-diff`, and `audit-source-diff-batch` write structured reports, lossless before/after masks, audited class rasters, previews, and hashes. The first complete report is `runs/source-diff-all-v1/source-diff-batch.json`; it passes all seven processed sources without mutating the currently published artifacts.

### 2026-08-22 — Alignment becomes an automatic held-out perimeter loop

- Todd clarified that the goal is a no-input system, not a workflow that asks him to provide correction arrows. Visual comparison therefore became part of the processor itself: propose, warp without clipping, compare, validate on withheld geography, and repeat only while validation improves.
- The automatic refiner samples 16 distributed California perimeter neighborhoods, robustly matches source edges with both distance and tangent direction, selects eight geographically distributed fit anchors, and reserves the remainder as holdouts. It retains the least-flexible safe correction and stops when another iteration does not improve independent evidence.
- Rendering an unclipped source proved essential. A warp clipped to California has an exact authoritative outline regardless of whether the source is correctly registered; using that manufactured edge for validation would be circular.
- The batch also formalizes the evidence hierarchy learned during review. Coastline and state border are primary. County junctions are attempted only for partial maps with insufficient visible perimeter, and a county proposal is vetoed whenever it worsens available coast/border evidence. Strong graticule registration is audited but never displaced by a weaker visual fit.
- Across all 11 sources, the second batch passes. Landslide, quake, and rivers accept a validated correction; the remaining raster sources retain their already stronger parent transforms. The farms map produces 28 plausible county fallback anchors, but the proposed change degrades its already 0–1-pixel visible perimeter to roughly 6 pixels and is correctly rejected. The earlier elevation partial-map proposal is discarded in favor of a full-state model.
- `examples/county.png` is not counted as a twelfth thematic dataset because it contains no legend-derived variable. Todd supplied it as a high-resolution alignment reference, and its separate reference role is recorded explicitly.
- The reproducible aggregate is `runs/perimeter-refinement-all-v2/perimeter-refinement-batch.json`. Manual arrows remain an explicitly assisted fallback, not an ingredient in automatic success.

### 2026-08-22 — Every supplied source reaches repeated visual-diff QA

- Forest and farms now have semantic extraction plans keyed from their legends: eight forest categories and 45 mutually exclusive crop categories. Both use sparse-evidence contracts so genuinely blank background remains transparent.
- Elevation receives a value-preserving branch rather than arbitrary bins. The 16-bit raster stores values through an explicit offset/scale; color is projected onto the printed legend ramp in Lab space, and cartographic ink is separately masked and completed. Six visual-diff iterations were needed to prevent dark province boundaries and labels from being interpreted as low elevation. The accepted run directly observes 68.4% of in-state pixels and explicitly marks the remaining 31.6% as completion beneath ink/occlusions.
- The 24.5 MB geologic PDF exposes higher-quality native evidence than its rendered page: 15,316 filled vector objects match all 53 native CMYK legend swatches with no unsupported path operators. Direct fills cover 96.1% of the California state mask, with the remaining 3.9% recorded as deterministic completion. A rendered-source audit finds zero dropped visible geology pixels.
- The geology geometry is sound, but several geologic age glyphs use a custom PDF encoding that OCR renders as characters such as `@`, `|`, or `=`. Those category labels remain a real semantic publication blocker rather than being guessed.
- The corpus diff now runs to a fixed point: each iteration consumes the previous audited candidate, and success requires two consecutive identical clean signatures. All 11 datasets pass `runs/source-diff-all-v3/source-diff-batch.json`; plant hardiness and quake require three comparisons, while the rest stabilize in two.

### 2026-08-22 — High-resolution county reference becomes supplemental evidence

- Todd clarified that `examples/county.png` is a 3600-by-3600 reference map: California's state border is a thicker black stroke and all county borders are thinner strokes. It is not a thematic source to classify.
- The new registration branch reads the original RGBA styling rather than flattening transparency. It isolates 36,078 state-border pixels from 63,672 county-border pixels, rejects disconnected black watermark components, and fits the complete state outline before considering counties.
- The reference selects Web Mercator. On 480 held-out state samples it measures a 2-pixel source-resolution median and about 8.02-pixel P90. Its interior county network is also compared with the pinned Census vintage: 5.83-pixel median and 16.16-pixel P90 on the 3600-pixel canonical grid. The difference is disclosed as vintage/generalization error rather than hidden.
- Automatic refinement now loads the validated county manifest at a 2400-pixel working height. Up to 28 high-resolution county junctions are compared on every source. They can support a partial-map fit or veto a proposal, but state/coast evidence retains priority.
- Farms demonstrates the intended behavior: 27 of 28 raster county junctions are visible, with 1-pixel median and 2.33-pixel P90 zero-shift residual. A proposed new transform fails held-out gates, so the already strong parent alignment is positively verified and retained.
- The final high-resolution corpus run passes all 11 thematic sources at `runs/perimeter-refinement-all-v4/perimeter-refinement-batch.json`. Quake, rainfall, and rivers accept validated corrections; the other parents remain unchanged, and the geologic graticule registration remains authoritative.

### 2026-08-22 — Quake corrections migrate onto the automated v4 alignment

- The accepted v4 quake alignment is a hash-linked child of the author-reviewed local-v2 alignment and retains the same 3398-by-3920 Web-Mercator grid. This makes a provenance-safe correction migration possible without moving the 2,168 authored destination points.
- Clone-stamp targets remain fixed geographic pixels. The 2,154 observed-source points move through the child alignment's declared parent-to-target projective matrix; the 14 sequential `composite_at_operation_time` sources remain fixed because they address earlier author pixels in the target grid. Radii and operation order remain unchanged.
- The new `stamp-correction-migration.json` records both extraction hashes, both alignment hashes, the exact migration matrix, displacement statistics, and an explicit `approval_carried_forward: false`. The old approval and public quake tiles remain untouched.
- The migrated manual target mask is byte-identical to the approved 766,250-pixel mask. Migrated source replay changes 68,305 manual values, or 8.91%, versus 258,024 values, or 33.67%, if stale source coordinates are replayed unchanged. Transforming sources therefore reduces divergence from the approved material by 73.53% while preserving every authored destination.
- An independent alignment-application audit recomputes the target baseline from the source class raster and v4 manifest. It matches all 13,320,160 grid cells exactly and differs from a parent-alignment recomputation at 2,428,718 pixels, proving the accepted top-level correction was actually applied.
- `materialized-v1` remains `needs_visual_review`. It contains no broad automatic inference, replays all 2,168 stamps, recomputes the author-defined under-50-pixel enclosed-hole rule, and contains 4,687,127 classified pixels before corpus-level full-state completion.
- The repeated source-diff loop fills 1,163,107 remaining NoData cells as separately masked deterministic completion, then reaches the same clean signature twice. The final 5,850,234-cell state surface drops zero visible source-evidence pixels. A new author inspection is still required because the earlier “ok this is done” approval covered the previous alignment only.
- A dedicated review page on port 8784 makes that gate explicit. It switches among the new v4 materialization, the previously approved result, the warped v4 source under cyan state and magenta county overlays, and a magenta changed-pixel diagnostic. It cannot save a decision without a written statement and confirmation that all four views were inspected; any decision binds only the new materialization and v4 alignment hashes.

### 2026-08-22 — The quake review must display Todd's county.png, not generic county vectors

- Todd caught that the first v4 review page displayed the generic Census/TIGER county overlay, including neighboring-state linework. That made the visual gate invalid even though `county.png` had participated in automated refinement as registered evidence.
- The review now renders both the cyan California state stroke and magenta California county strokes directly from the registered masks extracted from `examples/county.png`, resampled onto the exact quake review grid. The UI labels that provenance explicitly, and a saved decision is additionally bound to the `county.png` source and rendered-overlay manifest hashes.

### 2026-08-22 — Rainfall's missing semantics become an explicit topology stage

- Propagating the accepted v4 rainfall alignment moves the categorical raster by exactly the declared minus-four, plus-two pixel correction. After compensating for that displacement, 99.9173% of overlapping classified pixels retain the same class. The repeated source-diff gate stabilizes after two identical passes with zero dropped visible evidence and zero internal NoData; the earlier candidate and public assets remain untouched.
- The legend ceiling is now measured rather than merely suspected. The 5.0, 5.5, and 6.5-inch swatches contain the same two GIF colors in exactly the same proportions, giving every pair a total-variation distance of zero. The 2.5/3.5 pair is also confusable, with only a 0.0038 palette-histogram distance caused by raster details rather than a dependable semantic key.
- Two sparse OCR layouts were run against the registered state-only source. Neither found a credible interior numeric isohyet label matching a legend value. The few numeric-looking tokens were low-confidence interpretations of linework, so no numeric semantic anchor was accepted.
- A conservative topology prototype treats each shared palette as family evidence, seals dark contour gaps, and attempts only ordinal endpoint assignments supported by the immediately adjacent readable-color class. Nine combinations of dark-line threshold and boundary-neighborhood radius must agree at every inferred pixel.
- Only 14,958 of the 34,712 visible pixels in the 5.0/5.5/6.5 family survive that stability gate, all as the 6.5-inch endpoint adjacent to the readable 7.0-inch class. The other 19,754 family pixels remain unresolved; no 5.0 or 5.5 pixels are guessed. None of the 60,582 visible 2.5/3.5-family pixels survive all nine variants.
- The prototype is productized as the optional `extract-rainfall-topology` stage. It verifies the exact plan, source, extraction manifest, observed class raster, and completion mask; refuses to write inside the accepted extraction run; and emits independent observed-color, palette-family, topology-inference, and unresolved artifacts at source and Web-Mercator resolution. Its output is diagnostic-only and cannot silently enter an accepted or published candidate.

### 2026-08-22 — Rivers exposes a text-versus-linework ceiling

- The accepted v4 alignment was propagated into a new feature extraction without touching the earlier candidate. The top-level Web-Mercator correction is now applied to automatic feature runs as well as categorical runs, with regression coverage. Repeated source-diff auditing stabilizes with all 99,030 source blue-ink pixels and all 104,200 warped pixels reproduced exactly.
- Rotation-aware OCR runs 18 page orientations through Tesseract and, when available, Apple Vision. The combined pass yields 185 source-space candidates: 102 accepted by confidence or multi-angle consensus and 83 retained as ambiguous. OCR boxes are not treated as pixel truth because they frequently enclose both a curved label and the river or lake it names.
- Early text separation variants demonstrated both failure directions. A bbox-driven candidate left obvious label glyphs in geometry; a connected expansion swallowed real river lines; and a broad OCR-corridor holdout conservatively marked 54,083 of 99,030 observed pixels unresolved. None was promoted or published.
- The v6 diagnostic combines confirmed compact OCR glyphs with rotation-normalized thick-stroke morphology that requires support from at least three compact components. It moves the conspicuous Sacramento and San Joaquin labels out of the cyan geometry candidate while reducing the unresolved set to 22,507 pixels. The exact source partition is 18,513 confirmed text pixels, 22,507 unresolved text-like pixels, and 58,010 observed geometry pixels; the corresponding Web-Mercator partition is also exact.
- The original source and Web-Mercator observed-ink masks are copied byte-for-byte into the semantic run and retain identical SHA-256 hashes. Proposed reconnections are emitted separately: v6 contains 2,264 inferred pixels in 241 diagnostic hypotheses and zero overlap with observed evidence.
- The remaining ceiling is semantic, not a missing-pixel defect. Curved blue labels physically touch blue river strokes, while short river fragments and individual glyph strokes can have the same width and connected-component shape. Orange pixels therefore remain unresolved and withheld rather than silently entering publishable geometry. The v6 run stays `needs_semantic_review`; no public or approved artifact was mutated.

### 2026-08-22 — Quake fine alignment stops trusting ocean-like edges

- Todd rejected the v4 quake review because neither the state nor county lines were sufficiently aligned and the source contains offshore shadows and ocean artifacts that resemble a coastline. The rejection exposed a structural issue: v4 fitted eight perimeter-only points against generic Canny edges. Its `county.png` matches were diagnostic/veto evidence and never entered the fit because enough perimeter candidates already existed.
- The replacement `fine-align-county` stage makes the user-supplied `examples/county.png` the primary fine-registration target. It matches only distinctive interior multi-arm county junctions against a narrow-dark-stroke likelihood image. Coastline and ocean, terrain and hillshade, thematic boundaries, text, city symbols, and generic nearest-edge matches are explicitly excluded from fitting.
- The first versioned candidate finds 27 locally credible junctions and 22 mutually consistent fit points. At the 2080-by-2400 working grid, repeated matching improves from a 5.0-pixel median and 7.03-pixel P90 to 1.0 and 1.0 pixels.
- Validation is spatial rather than self-referential. Each of four fixed north-to-south regions is withheld from RANSAC and fitting in turn. Aggregate held-out residual improves from 5.0-pixel median and 9.53-pixel P90 to 0.87 and 1.57; all four regions pass independently.
- The correction remains a smooth projective transform: a 31-by-31 Jacobian audit finds determinant 0.976–1.036 and local scale 0.982–1.028 with no foldover. The state perimeter is not used to fit, but it acts as an unclipped veto and improves slightly from 1.71/5.0-pixel median/P90 to 1.41/4.90 rather than following the offshore artifacts.
- Two isolated complete runs produce the identical correction matrix, holdout metrics, source warps, overlays, and diagnostics. The candidate remains unpublished and requires Todd's visual judgment in a dedicated before/fine comparison page.

### 2026-08-22 — Reference conflict narrows the quake edge correction

- Todd found the county-primary candidate much better but still saw drift around the southern and southeastern edges. A regional audit separated a real source residual from a reference-registration error instead of treating every cyan mismatch as a quake error.
- Along the Mexico border, the registered `county.png` state outline is roughly 5–6 native pixels outward from the pinned Census boundary. The quake source already follows the Census edge there. `county.png` therefore remains the high-resolution interior county-network reference, but its state outline cannot override Census where the two disagree.
- The repeatable source residual is confined to the lower Colorado edge. Seven precommitted 50-pixel windows compare the authoritative boundary against three independent source signals: the narrow dark border stroke, the saturation transition, and the Lab-chroma transition. Four alternating windows fit one scalar displacement and the other three remain untouched holdouts.
- The fit estimates a six-pixel eastward source displacement with amplitude median 5.66 pixels and MAD 0.34. A smooth x-only correction ramps in over the southeastern corner, reaches a maximum of six pixels, and applies no vertical movement. It is exactly zero through native x=2600, beyond all 26 accepted county controls whose easternmost point is x=2277.74.
- Fit-window residual improves from 5.0-pixel median and 5.7-pixel P90 to 0.5 and 1.0. Untouched holdouts improve from 6.0/6.0 to 0.0/0.8, with a 1-pixel maximum. A second pass proposes zero additional displacement.
- The upper eastern edge and Mexico-border normal residuals pass no-regression gates. The unaffected image prefix and validity mask are byte-identical, and the composed transform has sampled Jacobian determinant 0.985–1.000, maximum shear 0.0254, and no foldover.
- Two complete same-path rebuilds are byte-identical. The new review on port 8786 shows the authoritative Census state boundary in yellow, the supplemental `county.png` state outline in cyan, and its county network in magenta; it starts on a southeastern close-up and provides Refined, Before, Blink, and Evidence modes. The candidate remains unpublished pending Todd's visual judgment.

### 2026-08-22 — One statewide outline becomes a regional hybrid perimeter

- Todd's visual review revealed that neither reference is uniformly best. The yellow Census geometry follows the lower Colorado River with finer and better-aligned detail, while the cyan outline registered from `county.png` follows the physical California coastline more faithfully. Census coastal geometry includes legal-water and island-envelope semantics that do not match the shoreline drawn on the earthquake map.
- The pipeline now assigns authority by feature family: `county.png` controls the land coastline, while pinned Census 2025 geometry controls interstate and international land borders, including the Colorado River and Mexico border. The complete raw references remain available as audit layers; they are not averaged into a compromise transform.
- Twenty-five source-height coastal windows compare both references against the narrow dark shoreline stroke, land/ocean saturation transition, and Lab-chroma transition. All 25 `county.png` windows pass, with 1-pixel median, 2-pixel P90, and 3-pixel maximum residual. The full Census coast comparison is materially worse at 3-pixel median, 7-pixel P90, and 11-pixel maximum, confirming the different geometry rather than motivating a coastal warp.
- The existing Census-anchored lower-Colorado correction remains unchanged: fit windows end at 0.5/1.0-pixel median/P90 and untouched holdouts at 0.0/0.8, while land-border no-regression and transform-regularity gates continue to pass. The regional audit therefore records `pass_no_additional_warp` instead of forcing a transformation where none is justified.
- A deterministic hybrid overlay now renders the detailed coastline in cyan and land borders in yellow, with magenta county lines independently toggleable. The review page starts with the hybrid enabled and both complete reference outlines disabled, while retaining them as explicit audit comparisons. No candidate is published by this diagnostic stage.
- For direct visual judgment, the same hybrid geometry is also emitted as one lime outline. No smoothing, averaging, or new transformation is introduced. The one-color line is the default review layer, while the cyan/yellow provenance view remains one click away.
- Todd's next close inspection exposed two losses hidden by the first implementation. Selecting one coastal x-coordinate per image row discarded concave shoreline geometry around Humboldt Bay, Point Reyes, and San Francisco Bay. The replacement keeps the complete `county.png` coastal span up to a cut placed safely inside California, so those bays and capes survive unchanged while Census continues to supply the land-border arc.
- The Tahoe vertical-to-diagonal hinge is now its own reference family. Nine independent 100-pixel windows put the registered `county.png` hinge at 2.0-pixel median and 3.2-pixel P90 from the earthquake-map stroke; Census measures 7.0 and 8.2 pixels. The combined outline therefore uses `county.png` at Tahoe as well as the coast, while preserving Census at the Colorado River and other land borders.
- A final San Diego close-up exposed that the first hybrid cut the coast near native row 3820, about 100 pixels before the actual San Diego/Mexico junction. The switch is now bound to the versioned WGS84 seed `(-117.12694883725163, 32.53413690413119)`, which projects to pixel `(2413.84, 3917.20)` on the 3398×3920 review grid. Because the junction itself is clipped by the bottom edge, the complete `county.png` shoreline now continues through the last row and an explicit 11-pixel connector closes the measured 10-pixel endpoint gap to the unchanged Census Mexico-border segment.
- The refined single-color outline contains 35,026 exact union pixels: 27,663 from the `county.png` coastline/Tahoe authority (including the reported crop-edge connector) and 7,364 from Census land-border authority, with one overlapping pixel at an authority join. The review adds direct Eureka, Bay, Tahoe, Colorado, and San Diego focus controls so each regional choice can be inspected at source-pixel scale.

### 2026-08-22 — The accepted hybrid alignment reaches a finished quake candidate

- Todd accepted the iteratively corrected hybrid border and instructed the pipeline to proceed. The new isolated extraction binds alignment SHA-256 `7b53c8a…`, hybrid-perimeter audit `9454bb4…`, its deterministic rebuild proof, and the exact extraction manifest. The previously approved and published quake artifacts remain untouched.
- Stamp migration now follows the complete three-child alignment lineage rather than assuming one projective step: reviewed local-v2 → automatic perimeter v4 → county-primary projective → lower-Colorado smoothstep. All 2,168 operations migrate in one hash-verified pass. The 2,154 observed sources move through each incremental model; 14 composite sources, every target, every radius, and operation order remain unchanged.
- The authored target footprint is still byte-identical at 766,250 pixels. Migrated replay changes 67,260 reviewed manual values compared with the old aligned surface, while stale source coordinates would change 261,423. Following the actual alignment chain reduces divergence by 74.27%, and replay matches the stored candidate exactly.
- Independent alignment application recomputes every target cell from the final source class raster and accepted transform with zero mismatches. It differs from the old parent recomputation at 2,486,396 cells, proving the final chained correction is active rather than accidentally ignored.
- The materialized pre-completion candidate contains 4,702,129 classified cells. Repeated source-diff completes 1,148,105 separately masked cells on the first pass, then produces two identical clean signatures. The promoted fixed-point surface contains all 5,850,234 California cells, with zero internal NoData and zero dropped visible source-evidence pixels.
- Fixed-point promotion is now a reusable command rather than an ad hoc copy. It validates every batch, case, report, iteration, and artifact hash; preserves all correction masks; promotes the first-pass completion mask; and produces byte-identical materialization manifests and artifacts across two isolated builds.
- Port 8787 serves the actionable final review. It shows the completed candidate under the accepted lime hybrid border and magenta county network, the previous approved materialization, the original source under the final warp, and a magenta old-to-final change diagnostic. Approval requires a written statement and explicit confirmation of all four views; it binds fixed-point materialization SHA-256 `24438cb…` and does not carry forward the old approval.

### 2026-08-22 — The green review border becomes the exact publication clip

- Todd noticed that a green union of independently rasterized reference spans was not topologically sufficient: the visible border needed to be one continuous line, and nothing outside that line could remain colored.
- The earlier provenance union contained six disconnected raster components. It also carried coastal/internal reference fragments that were useful for alignment evidence but were not a valid publication boundary. Treating those pixels directly as a clip would have left ambiguous gaps and offshore color.
- The replacement composes the same regional authorities as a filled mainland surface: the registered `county.png` interior supplies the physical coast and Tahoe hinge, while the Census mainland interior supplies the remaining land borders. The pipeline selects the largest mainland component, traces its exterior contour, redraws that contour as one closed 8-connected lime ring, and fills the exact same contour to obtain the clipping mask.
- The continuous ring contains 13,094 pixels and exactly one connected component. Its filled mainland interior contains 5,751,336 pixels. Offshore and island components are deliberately excluded for this reviewed mainland-only dataset.
- Applying that mask to fixed-point candidate v1 removes 106,767 previously colored pixels outside the displayed border. The reference change exposes 7,869 thin coastal cells inside the new line; they are completed with the same deterministic nearest-class rule used by source-diff and recorded in a separate boundary-completion mask.
- Candidate v2 contains exactly 5,751,336 classified pixels, zero colored pixels outside the lime border, and zero NoData pixels inside. All observed, manual, inference, enclosed-fill, and source-diff masks are clipped to the same hash-bound geometry. Of 766,250 migrated stamp-target pixels, 576 lie outside the accepted mainland boundary and are omitted only at this final clip stage.
- The updated port 8787 reviewer validates the boundary audit and refuses to load a candidate whose border hash, component count, outside-color count, or inside-NoData count fails. The candidate remains unpublished and `needs_visual_review`; its new materialization SHA-256 is `a5bda6be…`.

### 2026-08-22 — Todd approves the exact-border quake publication

- Todd approved final candidate v2 in the hash-bound port 8787 review. The saved decision records `inspection_confirmed: true`, does not carry forward an earlier approval, and binds materialization SHA-256 `a5bda6be…`, alignment `7b53c8ae…`, continuous-border `7f7db9e7…`, and boundary-audit `4be25973…`.
- Raster-tile export now revalidates all four dependencies whenever a boundary-clipped materialization is published. It refuses a mismatched audit or border, a disconnected border, any colored pixel outside the boundary, or any NoData pixel inside it.
- Two complete z4–z9 exports produced the same dataset manifest SHA-256 `28338a4b…` and the same six aggregate tile-set hashes. The approved package contains 2,400 recolorable PNG masks, 400 per Modified Mercalli class.
- Public metadata now includes a deterministic, machine-path-free `provenance.json`. It records the exact approval, materialization, alignment, boundary, fixed-point source-diff, class-count, and artifact hashes without exposing local filesystem paths.
- The previous reviewed-local-v2 publication is preserved at `publish/staging/quake-shaking-reviewed-local-v2-archive`. The active `publish/datasets/quake-shaking` package and the Next.js viewer both use the approved final-hybrid dataset; the viewer identifies it as approved and states the exact continuous-boundary guarantee.

### 2026-08-22 — Public coast inspection separates alignment from overview aliasing

- Todd correctly reported that the statewide Mapbox view did not look aligned with the coast. A separate 1732-by-2000 render of the exact approved border over the Mapbox Light basemap showed that the native georeferencing follows Eureka, Point Reyes, San Francisco Bay, Point Conception, and San Diego; the visible discrepancy was introduced by reducing the coastline to a single nearest-neighbor sample in each roughly five-kilometer z5 display pixel.
- Zooms 4–8 now use a deterministic four-by-four categorical supersample. Each overview pixel still contains exactly one dominant legend class, while its alpha records the fractional valid-state coverage. Across the complete quake pyramid this creates 458, 904, 1,771, 3,662, and 7,212 fractional edge pixels at z4 through z8 respectively, with zero inter-category alpha overlap and maximum summed alpha 255. Zoom 9 remains fully binary with zero fractional pixels.
- Map rendering uses linear filtering only for the coverage-aware overview zooms and switches to crisp nearest-neighbor at z9 and closer. Every category URL includes its aggregate tile-set hash, preventing the renderer from reusing an older overview under the same path.
- The exporter also converts the exact approved 13,096-vertex closed boundary into a hash-bound `boundary.geojson`. The viewer exposes it through an off-by-default “Inspect coast alignment” control. One click preserves the current styles, hides all six thematic classes, and shows only the green line over Mapbox; “Return to thematic data” restores the prior styles exactly.
- Two complete revised exports produced dataset SHA-256 `a49ec32f…`, provenance `880681a8…`, and boundary GeoJSON `2800cccd…` identically. The canonical approved class raster and materialization hash remain unchanged; this revision changes only low-zoom display generalization and adds diagnostic evidence.

### 2026-08-22 — Mapbox water becomes the final visual coastline mask

- Todd's second public-view inspection showed that fractional overview coverage still did not satisfy the visual contract. Even with the correct approved boundary, a partially covered low-zoom raster cell paints its whole rectangular screen footprint, and linear texture filtering can extend that alpha farther into the ocean.
- The viewer now places every thematic raster immediately below Mapbox Light's `waterway` and opaque `water` layers instead of appending the rasters at the top of the style. Mapbox's own displayed water geometry therefore becomes a hard final mask over the earthquake colors, including the Humboldt and San Francisco bays and the detailed Pacific coastline the visitor is comparing against.
- Raster sampling is nearest-neighbor at every zoom. Coverage-aware tile alpha remains useful for low-zoom area fidelity, but the renderer no longer interpolates it across the land/water edge.
- This is a presentation correction only. It does not modify the approved class raster, the accepted hybrid border, any tile bytes, or their approval/provenance hashes. The full Python suite, viewer typecheck, production build, and live Mapbox browser console all pass after the layer-order change.

### 2026-08-22 — Plant hardiness exposes the difference between completeness and fidelity

- The plant-hardiness alignment does not benefit from another warp. Its source contains no usable county network: a 2400-pixel search found 54 junction candidates, five locally plausible matches, and zero globally consistent matches against the minimum of 12. The retained automatic alignment has a native perimeter median of 0 pixels and P90 of 2.21; independent regional holdouts measure 1-pixel median and 2.46-pixel P90. Identity beats similarity, affine, and projective alternatives, and two isolated builds are byte-identical.
- The validated target alignment is hash-identical to the already reviewed parent. All 118 saved stamp operations use composite-at-operation-time sources, so their replay, targets, radii, and order remain exactly unchanged. A generalized alignment-application audit now accepts this explicit hash-identical no-op while still requiring an exact target recomputation; it reports zero mismatches and an identity Web-Mercator matrix.
- The earlier corpus source-diff filled all 20,176 remaining state pixels, but a deeper semantic audit rejects that result. Although its median propagation distance is one pixel, its P99 is 34.49 and maximum 55.36. It adds 721 pixels of rare class 11a from only 15 approved pixels, with a 20.62-pixel median propagation distance, and adds 3,367 class-10a pixels. Deterministic coverage is therefore not semantic validation.
- A first plant-specific conservative prototype accepts 384 pixels supported by a slight source-color fringe, an under-50-pixel single-class enclosure, or pure local surroundings; all agree with the aggressive surface, remain within 1.414 pixels of approved evidence, and add no 11a pixels. A second perturbation ensemble uses only approved pixels as seeds and agrees across Euclidean, eroded-core, four- and eight-neighbor geodesic, and plus/minus-one-pixel coordinate variants. It accepts another 176 interior pixels, separates 1,375 otherwise stable pixels into an eight-pixel boundary-review band, and leaves 19,616 pixels explicitly unresolved.
- Both conservative experiments rebuild byte-for-byte, but remain prototype-only. Their thresholds are plant-specific, they use private extraction helpers, and only 560 of 20,176 gaps pass the combined gates. Neither surface mutates the approved materialization or public dataset.
- Boundary semantics are now multi-component without allowing manual edits to manufacture geography. The authoritative mainland is always retained. Of six Census island components, two contain raw observed plant-hardiness pixels and are selected; four contain none and are rejected. Manual and inferred pixels are forbidden from selecting an island. Quake retains its separate one-mainland-component contract.
- Port 8788 serves a read-only completion-fidelity review. It switches among aligned source, the approved 118-stamp evidence, the 560-pixel conservative surface, and the rejected full completion. Independent overlays expose observed pixels, manual stamps, both conservative passes, the boundary-risk band, all 19,616 unresolved pixels, and the boundary evidence. Two islands contain source data, while a later diagnostic correction displays all four `county.png` island outlines; those facts remain separate. The page is diagnostic only and cannot save an approval.

### 2026-08-22 — The approved California outline becomes one canonical reference

- Todd noticed that the first plant-hardiness review looked wrong around San Francisco Bay and asked whether its green line was really the previously accepted boundary. It was not the same artifact: it independently rebuilt the same `county.png`-coast/Census-land authority rule directly on the 572-by-660 plant grid. Although the mainland masks overlapped by 99.4%, 985 cells differed, including a 345-cell Bay Area component. Repeating an approved rule at lower resolution had silently discarded approved geometry.
- The accepted quake boundary is now the single canonical mainland reference at `reference/canonical-california-boundary-v1`. Promotion requires Todd's confirmed materialization decision, the exact boundary-clip audit, the high-resolution interior hash, displayed-border hash, and the public closed GeoJSON derived from that border. The canonical GeoJSON has 13,096 vertices.
- The promotion stage rasterizes that geographic ring back onto the original 3398-by-3920 approval grid and requires an exact round trip. Both the interior and border reproduce with zero mismatched pixels; the approved border retains SHA-256 `7f7db9e7…` and the GeoJSON retains `2800cccd…`.
- Other datasets now rasterize the ordered canonical GeoJSON ring instead of re-running hybrid extraction or merely shrinking the high-resolution mask. Drawing the ordered ring matters at coarse resolution: it keeps the one connected route through Humboldt Bay, Point Reyes, San Francisco Bay, Tahoe, the Colorado River, and San Diego even where quantization would turn a narrow water entrance into a detached contour.
- On the plant grid the canonical mainland contains 164,193 cells and its displayed 2,835-pixel line is one connected component. Four coarse enclosed water components remain the direct fill result of the same ordered ring rather than being silently closed. The two observed-source island components are then added separately under the island audit; they do not modify the canonical mainland.
- Boundary clipping now accepts this canonical raster audit as its mainland authority while preserving compatibility with already approved legacy hybrid candidates. Per-dataset coast and state evidence remains useful for evaluating alignment, but it can no longer create a competing publication outline.

### 2026-08-23 — The Bay defect was missing linework, not raster quantization

- Todd rejected the first attempted Golden Gate repair. Rendering the existing 13,096-vertex ring as a subpixel SVG did not help because the ring itself had been traced from the exterior of a filled mainland mask. That operation had already discarded open coastal linework inside San Francisco Bay and all four offshore island components. Vectorizing the wrong geometry merely drew the same error more precisely.
- The corrected review candidate starts from the registered `county.png` state stroke. It replaces the fill-derived coastal span with all 27,663 registered county-coast pixels, retains none of the old shortcut in the Bay Area, and keeps the accepted Census inland-border spans. Four short, measured authority seams make the mainland line one connected component without connecting a coastal span to itself, so the Golden Gate remains open.
- The same county source contains four closed Southern California island components that the former largest-component extraction discarded. They are now transformed through the hash-bound county registration and displayed as four separate lime outlines. The UI distinguishes these four boundary outlines from the two islands that actually contain observed plant-hardiness data.
- The candidate contains one mainland component plus four island components and drops zero county-coast pixels. At this stage it remained `needs_author_review` and changed only the diagnostic view on port 8788; the previously approved clipping mask, classes, and publication packages stayed unchanged pending Todd's decision.

### 2026-08-23 — The county-detail line becomes the active canonical border

- Todd approved the corrected line for every map going forward with the statement, “Let's use this as the canonical border for all maps going forward.” Promotion binds that statement and candidate manifest SHA-256 `60dc920a…` in the versioned `reference/canonical-california-boundary-v2` package.
- The promoted package is byte-for-byte identical to the inspected candidate: its mainland, island, and combined overlays retain SHA-256 values `d4a8e136…`, `47294b21…`, and `9d0451e6…`. It contains one connected mainland line network and four separate island outlines, preserves all 27,663 registered county-coast pixels, and has zero missing or non-county pixels in the precommitted San Francisco Bay audit window.
- `reference/canonical-california-boundary.json` is now the hash-bound active pointer to canonical ID `california-county-detail-border-v2`. Future alignment diagnostics, review interfaces, and publication-border displays must resolve that pointer; the previous `california-mainland-hybrid-v1` package remains preserved as superseded provenance.
- The active border is deliberately not reconstructed from a fill. Its open Bay entrances and detailed coastal branches are line evidence, while a clipping interior answers a different question and remains separately versioned. Existing approved datasets and clipping masks were not silently rewritten by this activation.

### 2026-08-23 — Remaining plant-zone gaps become an explicit neighbor-assumption layer

- Todd requested a complete surface: remaining unknown pixels outside the lime California boundary must disappear, while every remaining unknown inside must be assigned from neighboring classes, including the 118 authored stamps.
- The new `complete-neighbor-unknowns` stage consumes the 19,616-pixel unresolved mask from the conservative audit. It binds the active `california-county-detail-border-v2` display line, uses the separately versioned canonical mainland clipping interior, and fills the four closed active island outlines independently rather than reconstructing the display coast from a polygon.
- Of the 19,616 unknown pixels, 16,676 fall inside the valid interior and receive deterministic class assumptions; 2,940 fall outside and remain transparent. The final candidate has zero unresolved pixels inside and zero classified pixels outside its valid interior. Sixty-four prior classified pixels outside the same interior are clipped and exposed in their own review mask rather than silently retained.
- Each connected unknown component is assigned by an inverse-square vote of its 16 nearest classified boundary neighbors. Authored nonzero stamp pixels have a declared two-times weight. They appear among the neighbors of 1,368 inferred pixels and change the winning class for 95 pixels. The median nearest-seed distance is one pixel and P90 is three pixels; one isolated 59-pixel component requires a global-neighbor fallback with a 36-pixel maximum distance.
- The candidate keeps observed pixels, authored stamps, earlier conservative repairs, neighbor assumptions, exterior unknown removals, clipped prior evidence, authored-stamp influence, confidence, and nearest-seed distance as separate hash-bound rasters. It remains `needs_visual_review` and cannot publish automatically.

### 2026-08-23 — The reviewed plant surface is promoted to an isolated staging dataset

- Todd approved the neighbor-completed review with “lgtm.” Promotion initially refused the candidate because the publication interior contained 484 zero-valued grid-edge cells that were outside the earlier source-visible unresolved audit. The completion contract was tightened rather than weakening the gate: every zero-valued cell inside the valid interior is now an explicit neighbor assumption. The reviewed total is therefore 17,160 inside assumptions—16,676 previously audited unknowns plus 484 additional interior NoData cells.
- The approved candidate has 164,537 classified cells and exactly fills its five retained components: one canonical mainland plus the four author-approved active canonical island interiors. It contains zero colored cells outside and zero unclassified cells inside. All 2,940 audited exterior unknowns and 64 prior exterior class pixels remain transparent and separately masked.
- `promote-neighbor-completion` binds the reviewed class raster, review session, predecessor materialization, accepted alignment, active canonical manifest, exact canonical display overlay, and boundary audit. The decision does not carry forward an older approval. Observed, authored, conservative, neighbor-assumption, confidence, distance, and removal evidence remain separate artifacts in `runs/plant-hardiness-neighbor-approved-v1`.
- Tile export reopens the approved class raster and publication interior, repeats the zero-outside/zero-inside checks, and verifies every canonical-border hash. The z4–z9 staging package contains 13 recolorable pyramids and 5,200 PNG tiles. A second complete export is byte-identical across all 5,218 files.
- The staging dataset copies the inspected high-resolution canonical lime raster byte-for-byte; its SHA-256 remains `9d0451e6…`. The fill-derived five-feature GeoJSON is retained only as an integrity artifact. “Inspect coast alignment” prefers the exact raster, preserving open bays and the four island outlines instead of drawing a simplified polygon exterior.
- The active public plant-hardiness package remains untouched. The Next.js catalog exposes the new result only as **Plant hardiness · staging**. Full Python tests (131), viewer tests, TypeScript, the production build, live Mapbox rendering, source modal, category toggles, boundary mode, served-asset hashes, and the browser console all pass.

### 2026-08-23 — The reviewed plant surface replaces the earlier public package

- The neighbor-completed plant package is now the public `plant-hardiness` dataset in both `publish/datasets` and the Next.js static catalog. Its manifest SHA-256 is `4b036a29…`, status is `approved_publication`, and it remains bound to approval `15832d77…` and materialization `5b5cba69…`.
- The earlier public package was not deleted. It is preserved recoverably as `plant-hardiness-public-archive-pre-neighbor-v1` under both staging roots, retaining the complete pre-neighbor publication for provenance and comparison.
- The temporary **Plant hardiness · staging** catalog entry was removed after activation, leaving one public plant card and the existing quake card. A clean production build and a freshly restarted local server confirm the approved badge, all 5,200 tiles, the exact canonical-border inspection control, the source modal, and zero browser-console or runtime errors.

### 2026-08-23 — Deer distribution becomes the next canonical-border candidate

- The older deer extraction predated the current review assets, so it was not promoted in place. A new isolated plan, `examples/plans/deer-canonical-v1.json`, points to the validated v4 alignment and regenerates the categorical extraction under the active `california-county-detail-border-v2` review contract.
- The regenerated class raster is byte-identical to the prior deer result at SHA-256 `88db57c2…`; the new extraction manifest is `766f8bc5…`. This isolates review/provenance modernization from semantic changes to the ten legend classes.
- A fresh source-diff audit bound to the new manifest reaches a fixed point in two identical passes. All 322,476 target-state pixels are classified, zero visible source-evidence pixels are dropped, and no completion pixels are added.
- Independent visual inspection finds the canonical line tracking the source outline through the north coast, San Francisco Bay, the Tahoe vertical-to-diagonal hinge, the southern border, and the Colorado River. County residuals are diagnostic but supportive: 1.00-pixel median, 3.16-pixel P90, 89.7% within three pixels, and 98.5% within eight. The candidate remains `needs_visual_review`; no approval is inferred from these automatic checks.

### 2026-08-23 — Deer review switches from thick Census vectors to county.png strokes

- Todd caught that the first deer review had repeated an earlier review error: its magenta county layer was a two-pixel redraw of generalized Census polygons rather than the thin county network in `examples/county.png`.
- The shared extraction path now hash-validates the registered `county.png` source, manifest, and Web-Mercator county mask. It resamples the recovered high-resolution raster as subpixel coverage, preserving its original visual weight instead of assigning a fixed-width vector stroke.
- Deer was regenerated into the isolated `deer-canonical-extract-v2` run. The thematic class raster remains unchanged, while visible magenta county-overlay coverage falls from 58,130 pixels to 13,387 pixels. The served overlay is byte-identical to the generated artifact at SHA-256 `784aa2f6…`.
- The residual diagnostic now samples the same `county.png` network displayed in the UI: 1.00-pixel median, 2.24-pixel P90, 95.3% within three pixels, and 98.5% within eight. A fresh source-diff again reaches the identical two-pass fixed point with no additions or dropped evidence. Port 8790 labels both the layer and diagnostic as `County.png`, and the full 132-test suite passes.

### 2026-08-23 — Canonical clipping becomes part of categorical extraction

- Todd found that the deer review still retained colored pixels outside the active lime boundary. The problem was contractual: the display line had been modernized, but extraction and source-diff still used the older Census target-state mask, allowing a narrow exterior fringe to survive or be completed later.
- Categorical extraction now derives a hash-bound publication interior from the approved canonical mainland and the four closed active `county.png` island outlines. The detailed lime line is still a separate artifact and is never reconstructed from this fill. Each run saves the exact interior mask and a removal mask beside the class raster.
- The regenerated deer run removes 4,041 classified exterior pixels. Its final categorical surface contains 317,737 colored interior pixels, zero colored pixels outside, and zero interior NoData cells across five retained components.
- Source-diff reads and verifies the canonical interior declared by the extraction manifest instead of silently recomputing the legacy target mask. Two identical audit passes prove zero exterior color and zero interior NoData at the fixed point, so neighbor-based repair cannot repopulate the removed ocean fringe.
- The source review is clipped by the same contract and reports zero colored pixels outside. The refreshed port-8790 reviewer exposes the five-component clip and still uses the thin registered `county.png` county network. The full suite passes with 133 tests.

### 2026-08-23 — Deer cartographic ink becomes a masked reconstruction layer

- Todd requested that the black strokes between colored deer regions be removed. The prior deer plan had explicitly retained county lines, labels, and other dark cartography as NoData, leaving 47,971 transparent pixels inside the canonical publication interior.
- The deer plan now enables target-state completion. Every formerly ambiguous pixel inherits the geographically nearest observed legend class, which splits a line between different neighboring classes while filling same-class county lines, labels, and city marks from their surrounding region. The 269,766 directly classified pixels are not modified.
- The reconstruction is not relabeled as observation. `web-mercator-target-completion-mask.png` records all 47,971 assumed pixels independently, and the extraction manifest states the completion policy and exact artifact hash.
- A first diagnostic build revealed that exterior colors could influence nearest-neighbor assignment before clipping. The shared fill primitive now clears every pixel outside its valid mask before measuring distance, and a regression test proves an exterior class cannot seed an interior completion. The corrected deer surface is byte-identical to the independently generated prior source-diff result.
- The final surface contains all 317,737 valid pixels, zero interior NoData, zero exterior color, and zero dropped visible evidence. A fresh source-diff reaches the same two-pass fixed point without adding a pixel. Port 8790 now opens this deer run in class-first mode so the reconstructed surface is visible immediately; the source remains available through its opacity control. The full suite passes with 134 tests.
- Todd approved this exact cleaned extraction with “lgtm.” The saved decision is hash-bound to extraction manifest `49d6efeb…`, active canonical border `california-county-detail-border-v2`, and displayed overlay `9d0451e6…`; no future regeneration may inherit it implicitly.

### 2026-08-23 — Per-category comparison catches isolated antialias specks

- The first ten-class Solo audit showed that the class raster was cleanly filled but still contained tiny green, brown, and one red component outside their coherent regions. These were not missing-data lines: each component had only one to three directly classified seed pixels whose color came from antialiased dark boundaries between neighboring fills.
- A conservative post-classification gate now considers only 8-connected components of 16 pixels or fewer. It reassigns a component only when at least half of its immediate nonzero boundary belongs to one other class. Near-white categories are exempt so legitimate `Deer Rare or Absent` islands cannot be erased, and the real 21-pixel Southern Mule Deer component remains above the threshold.
- The gate reassigns 66 components / 358 pixels: 278 false brown pixels, 75 false green pixels, and one five-pixel red speck. Original class IDs and the exact changed-pixel mask are saved separately. The visible result removes scattered dark artifacts without smoothing any major data boundary.
- The regenerated candidate retains 317,737 classified interior pixels, zero interior unknowns, zero exterior color, and zero dropped source evidence. Its source-diff reaches a two-pass fixed point without additions. The classification reviewer now reports final Web-Mercator counts rather than the pre-reconstruction source counts; all ten Solo controls pass browser verification. The previous approval is not inherited because this candidate changes 358 class values. The full suite passes with 135 tests.

### 2026-08-23 — The refined Deer surface enters isolated Mapbox staging

- Todd approved both the refined combined view and all ten per-category views with “lgtm.” The two decisions bind extraction manifest SHA-256 `0c9cd6ff…`; the earlier v6 approval is not inherited. The reviewed v7 class raster is `5a8ea003…`.
- `promote-reviewed-extraction` is a deliberately narrow publication path for categorical surfaces that require no inference or authored stamps. It validates both approval files, the current plan and accepted alignment, the two-pass source-diff fixed point, and the active canonical pointer. It refuses any interior NoData, exterior color, stale artifact, changed class hash, or non-fresh output directory.
- The promoted class raster is byte-for-byte identical to the dual-reviewed extraction. Its materialization is `cc91dedc…`, its new exact-identity approval is `e251810b…`, and it contains 317,737 classified cells: 269,766 directly classified and 47,971 neighbor-derived cartographic reconstruction cells. All 358 isolated-speck reassignments remain separately masked; none of these derived pixels are relabeled as direct observation.
- The publication audit contains exactly five canonical components: one mainland and the four author-approved `county.png` island interiors. Every component is filled, all four islands contain direct source evidence, and there are zero colored cells outside the canonical interior and zero unclassified cells inside. The alignment display copies the exact active canonical raster at SHA-256 `9d0451e6…`; it is not reconstructed from the clipping fill.
- The z4–z9 export at `publish/staging/deer-distribution-v1` contains ten independently recolorable pyramids and 4,000 PNG tiles. Dataset manifest SHA-256 is `dabf4b5c…`, provenance is `4ea0cb11…`, and the five-feature integrity GeoJSON is `a6bd8249…`. The staging package and the viewer copy are byte-identical.
- A separate static route at `/mapscan/staging` exposes only **Deer distribution · staging**; the public `/mapscan` catalog remains unchanged. Live verification confirms the Mapbox canvas, all ten controls, a complete toggle cycle, the exact canonical-border inspection mode, the source modal, served dataset/boundary/tile assets, and no browser or framework errors. All 137 Python tests, three viewer tests, TypeScript, and the Next.js production build pass.

### 2026-08-23 — The reviewed Deer package enters the public catalog

- Todd inspected the isolated Mapbox staging package and approved it with “lgtm.” Public activation does not export the classes again. The new `activate-staging-dataset` gate reopens the reviewed package, verifies the approved publication and provenance records, recomputes every category tile-set hash, and revalidates the canonical boundary raster, five-feature boundary GeoJSON, source image, and exact zero-outside/zero-inside contract.
- All 4,000 PNG tiles and 15 accompanying package files copied byte-for-byte from `publish/staging/deer-distribution-v1` into the fresh `publish/datasets/deer-distribution` directory. The immutable package-inventory SHA-256 is `1fd260b4…`; dataset `dabf4b5c…`, provenance `4ea0cb11…`, materialization `cc91dedc…`, materialization approval `e251810b…`, and canonical display border `9d0451e6…` remain unchanged.
- The new public activation decision is SHA-256 `eb9bff76…`. It binds Todd's statement, all ten category tile-set hashes, the five-component `california-county-detail-border-v2` contract, zero exterior color, and zero interior NoData. Activation refuses an existing destination and records that no earlier public activation approval was carried forward.
- The main `/mapscan` catalog now has three datasets: earthquake shaking, plant hardiness, and Deer distribution. The temporary `/mapscan/staging` route and its duplicate viewer asset root are retired; the original reviewed package remains preserved under `publish/staging` as the immutable staging archive.
- The live public URL `http://127.0.0.1:8780/mapscan?dataset=deer-distribution-california` serves the public dataset, canonical raster, source, and sample tile successfully. Browser verification confirms all ten class controls, a disable/restore cycle, canonical-border inspection, the public source-image modal, one Mapbox canvas, and zero console errors. The full Python suite passes with 139 tests; all three viewer tests, TypeScript, and the clean Next.js production build pass, and the build contains no staging route.

### 2026-08-23 — The original Forest example becomes a no-manual-input pipeline test

- The first map that motivated MapScan, `examples/forest.jpg`, was regenerated from its automatic California Albers alignment rather than its older extraction. The v4 perimeter loop had already tested a projective correction against state/coast and registered `county.png` diagnostics and accepted zero iterations because no candidate improved independent evidence. Its starting boundary residual is one pixel median and 1.83 pixels P90 on the 876-by-1011 working grid; no authored arrows or stamps enter the run.
- `examples/plans/forest-canonical-v1.json` binds that automatic no-op alignment, the active `california-county-detail-border-v2` display contract, and thin registered `county.png` diagnostics. The regenerated extraction manifest is SHA-256 `200db965…`. It retains 156,581 classified forest pixels across all eight legend classes, removes 37 colored exterior pixels, leaves genuinely uncolored land transparent, and contains zero classified pixels outside the five-component canonical interior.
- The sparse-evidence source-diff gate reaches a fixed point in two identical passes with signature `fd7a45ba…`. It drops zero visible source-derived pixels and adds no completion pixels; transparent land remains intentional rather than being mistaken for missing full-state data.
- A new reusable `audit-categorical-fidelity` stage repeatedly recomputes the source classification across nine Lab-distance/margin variants. The stored source classes reproduce exactly. No accepted pixel changes to another nonzero class, every category remains populated, and the Web-Mercator raster retains zero exterior color. The audit is SHA-256 `72ec8584…`.
- Relaxing the maximum Lab distance from 24 to 28 admits 12,769 additional source pixels under the broadest variant, including 9,639 pixels nearest to the pale Douglas-fir class; 5,520 additions meet a strict pale-neutral background test. The visual diagnostic shows these cyan proposals spreading into uncolored JPEG background, so the reviewed conservative threshold is retained. Strict drops and relaxed additions remain separate evidence and never mutate the candidate.
- Per-category component analysis confirms that one- to three-pixel patches are normal source content, not automatically removable specks. All eight isolated masks remain geographically coherent; Deer-style speck suppression would erase legitimate pixel-level forest evidence. Both reviewers run at ports 8792 and 8793, all eight Solo controls pass, and neither browser reports an error. The full Python suite passes with 141 tests. The candidate remains unapproved until Todd separately reviews alignment and classification.

### 2026-08-23 — Forest preserves sparse evidence through isolated staging

- In the classification comparison Todd initially asked to remove blue water pixels. The class-only view showed that the eight-class raster contained no blue category; the visible lake was the original source image beneath transparent NoData. Todd confirmed, “youre right. no blue,” so the proposed blue-channel exclusion was removed rather than discarding legitimate edge pixels.
- Todd then instructed the pipeline to proceed. The saved alignment and classification decisions bind the unchanged extraction manifest and record that the source underlay—not the class raster—caused the apparent blue. Their SHA-256 values are `305f7294…` and `c852a96a…`; no arrows, stamps, completions, or threshold relaxations enter the reviewed surface.
- The former reviewed-extraction promotion gate assumed every categorical map was full-state and would have rejected Forest's 228,333 intentionally transparent interior cells. The contract now branches only on the plan's predeclared coverage expectation. `full_state` still requires zero interior NoData; `sparse_visible_evidence` requires a passing byte-identical fixed point, zero exterior color, and an exact immutable NoData count.
- The promoted materialization is SHA-256 `8b899990…`, its approval is `c792cc48…`, and its class raster remains byte-identical at `8d0a03da…`. It contains 156,581 classified pixels, 228,333 intentional interior NoData cells, zero exterior color, and the same five active canonical components.
- `publish/staging/forest-cover-v1` contains eight independently recolorable z4–z9 pyramids and 3,200 PNG tiles. Dataset manifest SHA-256 is `c1c789dd…`; public provenance is `ddf858ed…`. The viewer copy is byte-identical and `/mapscan/staging` exposes only **Forest cover · staging**, leaving the three-dataset public catalog unchanged.
- Live Mapbox verification confirms the eight controls can each be disabled and restored, the exact canonical-border inspection mode returns to thematic data, the original-source modal opens, and the staging canvas loads. All 142 Python tests, three viewer tests, and TypeScript pass; all 3,213 staging files are byte-identical to the viewer copy. The staging package remains separate from the public catalog until Todd reviews this final Mapbox presentation.

### 2026-08-24 — The reviewed Forest package enters the public catalog

- Todd inspected the isolated Mapbox staging package and approved it with “lgtm.” Activation reopens the already-reviewed files and does not rerun alignment, classification, or tile generation.
- The activation gate now independently enforces the package's declared coverage contract. Zero exterior color remains mandatory for every dataset. A full-state package still requires zero interior NoData, while a sparse-evidence package must retain the exact interior-NoData count agreed by its dataset and provenance manifests. Regression tests cover both paths and reject a full-state package with even one interior NoData cell.
- All 3,213 reviewed staging assets, including 3,200 PNG tiles, copied byte-for-byte from `publish/staging/forest-cover-v1` into the fresh `publish/datasets/forest-cover` directory. Package inventory SHA-256 is `955c97ffc4ab0774b540766a172d6cbd39cc18045850569f04748d5a4128b531`; dataset `c1c789dd5d2341e0711982f49d0fa24233edc85c7d64d6711b69fbbcd305852f`, provenance `ddf858ed481c74c007747f880b55b4f2188b75467dba2bc47c079168f549ebaf`, materialization `8b899990e9328d742c77674014a586455d153bd6c07bba43e6da0a7d6ff1a7ca`, materialization approval `c792cc484e0f8047c438aee13f3015dd28d45eb3831e125ba551a17739000dd0`, and canonical display border `9d0451e6b0532787b710ef1fec59a8115e011dcfd3ffea2951acd78fe1812691` remain unchanged.
- The new public activation decision is SHA-256 `6c41317b0e286bb76cd764586dd19cfef6f11e04faf0894a340ab5772344d773`. It binds Todd's statement, all eight category tile sets, five `california-county-detail-border-v2` components, zero exterior color, the `sparse_visible_evidence` contract, and exactly 228,333 intentional transparent interior cells.
- The main `/mapscan` catalog now contains earthquake shaking, plant hardiness, Deer distribution, and Forest cover. The temporary `/mapscan/staging` route returns 404 and its duplicate viewer asset copy was moved recoverably to Trash; the original reviewed package remains under `publish/staging` as the immutable review archive.
- Live verification at `http://127.0.0.1:8780/mapscan?dataset=california-forest-cover` confirms the Mapbox map, 3,200-tile summary, all eight enabled class controls, a complete disable/restore cycle, canonical-border inspection and return, source-image modal, exact served dataset/boundary/sample-tile hashes, and zero development-log errors. The final suite passes with 144 Python tests, three viewer tests, and a clean TypeScript check.

### 2026-08-24 — Farms v2 turns a low-resolution partial map into a high-resolution automatic candidate

- Todd supplied `examples/farmsv2.png`, a 4,250-by-5,500 replacement for the 1,336-by-1,729 agricultural map. It has 3.181 times the linear resolution and about 10.12 times as many pixels. Area-downsampling the new image to the old dimensions produces 0.9955 luma correlation, 2.05 RGB mean absolute error, and a 95th-percentile absolute channel error of 9, so the pipeline can treat it as the same page, crop, map extent, and legend rather than silently solving a different source.
- The new `lift-equivalent-source-alignment` gate verifies this equivalence before copying the old normalized coarse registration. It binds both source hashes, rejects unequal x/y scale or mismatched content, and records the comparison in `runs/farmsv2-equivalent-lift-v1`. Two fresh standalone searches were retained as diagnostic local minima rather than replacing the visibly stronger verified parent.
- Automatic county refinement then matches the registered thin strokes from `county.png` across the visible partial-state extent. The accepted affine candidate uses 23 strong fit correspondences. Matched-county residuals improve from 5.96/12.35 pixels to 0/1 pixel median/P90 on the working grid; four spatially independent holdouts improve from 6.58/12.63 to 0.70/1.44. Three quadrants pass strict gates and one isolated quadrant remains a bounded outlier, while the state-boundary veto improves P90 from 7.31 to 5.10 pixels and preserves local scale between 0.991 and 1.017.
- A separate repeat run is byte-identical across its selected affine matrix, provenance, holdouts, vetoes, and every core image/JSON artifact. The determinism audit is stored with the candidate, so the review server refuses to present an unrepeatable fine alignment as ready for author inspection.
- The high-resolution 45-class legend is unusually valuable: every original legend RGB occurs exactly in the new image, and all 45 categories now have visible map evidence, including the tiny Greenhouse class. The conservative extraction contains 2,110,420 classified source pixels and 2,746,656 classified Web-Mercator pixels; 39,825,025 canonical-interior cells remain intentionally transparent under the predeclared `sparse_visible_evidence` contract. No completion, inference, or authored stamps enter this candidate, and zero pixels are colored outside the active canonical California border.
- The categorical fidelity audit recomputes the baseline exactly and compares nine stricter and looser Lab-threshold variants. No accepted nonzero class changes to another category, all 45 legend classes survive, relaxed thresholds add no pale-neutral background pixels, and the conservative settings are retained. A two-pass source-diff reaches a byte-identical fixed point with no dropped visible evidence and no completion.
- The isolated alignment and classification reviewers run on ports 8796 and 8797. Automated checks and the full 152-test suite pass, but the Farms v2 candidate remains `needs_author_review`; it is not staged or published until Todd separately accepts the alignment and the 45-class extraction.
- Todd inspected the fine-alignment reviewer on port 8796 and approved it with “lgtm.” The saved alignment-only decision binds extraction manifest SHA-256 `34ef27b3…`, the active `california-county-detail-border-v2` package, the inspected `farmsv2-county-fine-v5` affine candidate, and its byte-identical determinism repeat. It explicitly does not approve the separate 45-class interpretation.
- Todd then inspected the independent category reviewer on port 8797 and approved it with “lgtm.” Classification decision SHA-256 is `3237fe79…`; it binds the same extraction manifest and conservative sparse-evidence policy with no inference, completion, or stamps. Promotion copies the reviewed class raster byte-for-byte into materialization `db2007a4…`; materialization approval SHA-256 is `db08b5b9…`, and the promoted surface retains 2,746,656 colored cells, 39,825,025 intentional interior NoData cells, and zero colored cells outside the canonical boundary.
- The isolated `publish/staging/farms-v2-v1` export contains 45 independently recolorable z4–z9 pyramids and 18,000 PNG tiles. Dataset manifest SHA-256 is `1b7cee52…`, provenance is `6b0f8900…`, and the full 18,050-file viewer copy has the same aggregate inventory SHA-256 `339d81d5…` as the staging source. The high-resolution `farmsv2.png` source and exact canonical boundary are served byte-identically at `ff58a540…` and `9d0451e6…`.
- A dedicated static `/mapscan/staging` route imports only this immutable package; the public four-dataset page does not reference Farms. Live verification confirms the Mapbox canvas, all 45 controls, Wheat and Greenhouse disable/restore cycles, canonical-border inspection and return, source-image modal, loaded staging tile resources, California center/zoom, served asset hashes, and zero browser errors. All 152 Python tests, five viewer tests, TypeScript, and the Next.js production build pass. Farms remains staging-only pending Todd's final Mapbox presentation review.

### 2026-08-24 — Farms v2 becomes the fifth public MapScan dataset

- Todd approved the isolated Mapbox presentation with “lgtm whats next” and then instructed “continue.” The activation gate reopens `publish/staging/farms-v2-v1`, verifies all category tile-set hashes, the materialization and its approval, provenance, canonical display border, sparse-evidence NoData contract, and every package file before copying anything.
- All 18,050 staging files copy byte-for-byte into `publish/datasets/agricultural-land-use`. Public package inventory SHA-256 is `0bd48124…`; the additional public activation decision is `be583eb7…`. Dataset `1b7cee52…`, provenance `6b0f8900…`, materialization `db2007a4…`, materialization approval `db08b5b9…`, source `ff58a540…`, and canonical border `9d0451e6…` remain unchanged.
- The public catalog now contains five datasets and exposes **Agricultural land use** at `/mapscan?dataset=california-agricultural-land-use`. Live verification confirms all five cards, 45 enabled crop controls, Wheat and Greenhouse disable/restore cycles, canonical-boundary mode, source modal, public tile requests, exact served asset hashes, and no browser or development errors.
- The temporary `/mapscan/staging` route now returns 404. Its 90 MB duplicate viewer asset copy was moved recoverably to `/Users/toddsherman/.Trash/MapMap-viewer-farms-v2-staging-20260824-232752`; the immutable reviewed archive remains at `publish/staging/farms-v2-v1`. The final suite passes with 152 Python tests, six viewer tests, TypeScript, and a clean production build containing no staging route.

### 2026-08-24 — Rainfall is regenerated against the active canonical border

- After Farms became the fifth public dataset, the next-source inventory selected `examples/rainfall.gif` as the most mature unpublished categorical source. Its 35-class color-and-texture extraction had already reached a clean source-diff fixed point, but those bytes predated the active `california-county-detail-border-v2` publication contract and had never been approved against the final automatic alignment.
- `runs/rainfall-canonical-extract-v1` is a fresh extraction from the saved semantic plan and accepted v4 alignment. It embeds the active county-detail pointer and exact display overlay, clips to one mainland plus four reviewed island components, classifies all 1,032,986 valid cells, removes 16,514 previously colored exterior cells, leaves zero exterior color, and reconstructs 2,397 target cells hidden by cartographic ink without changing observed cells.
- The new source-diff reaches a fixed point in two identical passes with signature `99b91857…`, zero dropped visible evidence, and zero internal NoData. A completely independent rebuild is byte-for-byte identical for the manifest, plan, source warp, canonical masks, overlays, class raster, preview, and completion mask; extraction manifest SHA-256 is `9f7470db…` and class-raster SHA-256 is `60a4b4f6…`.
- The rainfall-specific topology audit preserves the image's information ceiling instead of inventing precision. The 5.0, 5.5, and 6.5-inch swatches use identical GIF palettes; a nine-variant contour-order gate can conservatively identify 14,958 source pixels as the 6.5-inch endpoint, but 19,754 pixels in that family remain unresolved and no 5.5-inch pixels can be established. The 2.5/3.5-inch family also remains unresolved by topology. These diagnostic pixels are not merged automatically.
- The generic classification reviewer now surfaces extraction warnings and every zero-coverage legend entry in a prominent **Known limitations** section. The eventual classification decision records the exact warnings and empty classes so an approval cannot hide semantic loss. A regression test covers both the review payload and the hash-bound decision disclosure.
- Ports 8798 and 8799 serve the regenerated alignment and per-class reviews. Processor-side inspection finds the canonical coast and state border visually consistent with the warped source; the registered `county.png` diagnostic measures 1-pixel median, 2-pixel P90, 97.0% within 3 pixels, and 99.4% within 8 pixels. All 153 Python tests pass. The candidate remains unpublished until Todd supplies the independent visual decisions.

### 2026-08-24 — Rainfall separates the Delta's internal water from the canonical border

- Todd found a distinct clipping defect in the first Rainfall review: Suisun Bay, Honker Bay, the San Joaquin River, and the Sacramento River lie inside the approved lime state outline, so a state-interior fill alone incorrectly assigned them precipitation classes. The canonical `california-county-detail-border-v2` display line remains unchanged; internal water is now an independent data-exclusion mask.
- The pipeline pins the 58 California county packages from the U.S. Census Bureau's 2025 TIGER/Line Area Hydrography release under `reference/census-2025-areawater`. Its manifest SHA-256 is `d6b090ba…`. Exact-name matching finds 2 Suisun Bay polygons, 1 Honker Bay polygon, 44 San Joaquin River segments, and 50 Sacramento River segments.
- Two broader automatic candidates were rejected by processor-side visual comparison before author review. Treating every hydrography polygon as removable water deleted 99,492 cells statewide; a fractional-coverage variant still deleted 65,926 cells and produced widespread stream speckling. The accepted candidate is deliberately include-only: no hydrography name outside the four reported features can remove data.
- The final mask removes 2,201 cells and is saved separately at SHA-256 `9f42e179…`. The source remains visible in review while the categorical surface is transparent over the selected water. A Delta triptych compares the warped source, prior filled data, and the exact excluded footprint; statewide inspection shows no unrelated water holes.
- Regenerated extraction manifest SHA-256 is `02bcf163…`; the resulting class raster is `f74ec21d…`. Source-diff reaches a fixed point in two identical passes with signature `7dcca886…`, zero dropped source evidence, zero interior NoData relative to the water-aware publication interior, and zero exterior color. An independent rebuild is byte-identical across all 21 core image artifacts, the extraction manifest, and the plan snapshot. All 155 Python tests pass.
- Todd's next inspection found that the water removal stopped at Suisun Bay instead of continuing west to San Francisco Bay. Exact Census feature inspection identifies the missing chain as 2 Carquinez Strait polygons, 4 San Pablo Bay polygons, 1 San Pablo Strait polygon, and 9 San Francisco Bay polygons. Adding only those names removes 1,726 additional canonical-interior cells, producing one continuous audit mask while leaving all unlisted hydrography unable to affect the data.
- The west-complete candidate excludes 3,927 canonical-interior cells in total. Its mask SHA-256 is `74a1b4ed…`, extraction manifest is `bc4d4a0a…`, and class raster is `74d19581…`. Source-diff again reaches a two-pass fixed point with signature `65975cea…`, zero dropped evidence, zero valid-interior NoData, and zero exterior color. All 21 core artifacts, the manifest, and plan snapshot reproduce byte-for-byte in an independent build; all 155 Python tests pass.
- Todd's next visual diff caught that applying the narrow-water any-contact rule to the broad San Francisco and San Pablo Bay polygons removed too much land-side data around Berkeley and East Bay. A processor-side threshold comparison tested 25%, 50%, and 75% cell-area coverage while holding every selected feature and narrow connection constant. The 50% majority rule restores 171 cells, including 130 in the East Bay audit window; 99 of those contain clear colored source evidence, none are white source-water pixels, and the narrow Carquinez/San Pablo connections remain on the any-contact rule.
- The refined broad-bay mask excludes 3,756 canonical-interior cells and has SHA-256 `7d66aff0…`. Extraction manifest SHA-256 is `3835bdf2…` and class raster SHA-256 is `d56105ef…`. Source-diff reaches a two-pass fixed point with signature `beae12a8…`, zero dropped evidence, zero valid-interior NoData, and zero exterior color. The independent build is byte-identical across all 21 core artifacts, manifest, and plan snapshot; all 155 tests pass.

### 2026-08-24 — Rainfall enters isolated water-aware Mapbox staging

- Todd separately approved the final alignment and 35-class interpretation with “lgtm.” Alignment decision SHA-256 is `87ad1e49…`; classification decision SHA-256 is `d74da559…`. Both bind extraction manifest `3835bdf2…`, the active `california-county-detail-border-v2` display reference, the three explicit zero-coverage entries (5.5, 6.5, and 17.0 inches), and the reviewed exact-name internal-water policy. No earlier Rainfall approval is inherited.
- The first promotion attempt correctly refused the water-aware raster because the Delta river cuts divide the data-bearing publication interior into 15 connected land pieces, while the canonical California outline still has five components. Promotion now models these as separate invariants: five canonical outer components for boundary integrity and display, 15 water-aware data components for coverage. It independently reconstructs and hash-checks the canonical interior, verifies the exact 3,756-cell subtraction, retains zero colored exterior cells and zero valid-interior NoData, and uses the canonical interior—not the river-split data footprint—to generate the five-feature boundary integrity GeoJSON. A regression test constructs a water channel that splits a data surface without redefining its outer border.
- The promoted class raster remains byte-identical at `d56105ef…`. Materialization SHA-256 is `36091302…`; its exact-identity approval is `c91ade79…`. The source, completion mask, publication interior, canonical interior, water exclusion, both review decisions, two-pass fixed point, and active canonical display raster remain independently hash-bound.
- `publish/staging/rainfall-v1` contains all 35 independently recolorable z4–z9 pyramids and 14,000 PNG tiles. The three unrecoverable legend classes remain visible controls with zero pixels rather than being silently removed. Dataset manifest SHA-256 is `9889bd0a…`, provenance is `163d736c…`, canonical display raster is `9d0451e6…`, and the 14,040-file package inventory is `5fe9cd82…`. The viewer copy is byte-identical.
- An exhaustive z9 reconstruction check opens all 35 category pyramids across 288 native tiles: their union differs from the approved class raster at zero pixels, no pixel belongs to two categories, every alpha is binary, and zero sampled internal-water pixels are colored. This proves tile export preserves the reviewed San Francisco Bay–Delta exclusions rather than merely carrying their provenance.
- A dedicated `/mapscan/staging` route exposes only **Annual precipitation · staging**; the five-dataset public catalog remains untouched. Live inspection found and fixed a long-legend layout defect by constraining the three-column shell to viewport height so the 35-control panel scrolls without stretching the map. Browser verification confirms the California fit, all 35 controls, a disable/restore cycle, exact canonical-border mode, the original-source modal, served staging assets, and no console warnings or errors. The final checks pass with 156 Python tests, nine viewer tests, TypeScript, and a clean production build. Rainfall remains staging-only pending Todd's Mapbox presentation review.

### 2026-08-24 — Rainfall becomes the sixth public MapScan dataset

- Todd returned to the completed staging handoff and instructed “please proceed.” The activation gate reopens the immutable package and verifies all 35 category tile-set hashes, materialization `36091302…`, materialization approval `c91ade79…`, provenance `163d736c…`, canonical display border `9d0451e6…`, full-state zero-NoData contract, zero exterior color, and the complete package inventory before copying anything.
- The 14,040 staging files copy byte-for-byte into `publish/datasets/rainfall`; public activation decision SHA-256 is `a423114c…`. Dataset `9889bd0a…`, package inventory `5fe9cd82…`, original source `4d883142…`, and the internal-water-aware publication contract remain unchanged. The additional activation decision brings the public directory to 14,041 files.
- The public catalog now contains six datasets and exposes **Annual precipitation** at `/mapscan?dataset=california-annual-average-precipitation-1900-1960`. The temporary staging route returns 404. Its duplicate 56 MB viewer asset copy moved recoverably to `/Users/toddsherman/.Trash/MapMap-viewer-rainfall-staging-20260824-2030`; the immutable reviewed archive remains at `publish/staging/rainfall-v1`.
- Public browser verification surfaced one interaction inefficiency: changing any category reapplied three style properties to every Mapbox layer. The viewer now tracks the previously applied style and sends only changed visibility, opacity, or color properties, keeping the 35-class control panel responsive without changing tile bytes or saved URLs.
- Live verification confirms all six dataset cards, 35 Rainfall controls, disable/restore, exact canonical-border inspection and return, the source modal, byte-identical served dataset/boundary/source/sample-tile assets, viewport-height long-legend scrolling, and zero browser warnings or errors. The final suite passes with 156 Python tests, ten viewer tests, TypeScript, and a clean production build containing no staging route.

### 2026-08-24 — Mixed coastal cells restore Peninsula and Marin land coverage

- Todd found remaining missing precipitation data on the San Francisco Peninsula and around Marin, San Rafael, and the Bay shoreline. A high-zoom comparison of the published surface, warped source, canonical coast, and exact internal-water mask showed that legend extraction was not dropping the data. The broad-bay policy was deleting an entire categorical cell whenever Census water occupied at least 50% of it, even though the remaining land-side fraction was visible at Mapbox scale.
- Broad San Francisco and San Pablo Bay polygons now remove a cell only when all 16 supersamples are water. The deliberately narrow San Pablo and Carquinez straits, Suisun and Honker bays, and the San Joaquin and Sacramento rivers retain their stricter any-contact rule. This keeps the reviewed internal-water network transparent while allowing mixed coastal cells to carry land data; the public Mapbox water fill remains responsible for visually clipping the water-side fraction.
- The fresh candidate restores exactly 193 previously empty cells statewide, including 159 in the requested Peninsula/Marin audit window. It removes zero previously classified cells and reassigns zero classes. The candidate has 1,029,423 classified publication-interior cells, zero valid-interior NoData, and zero color outside the fixed five-component canonical border. Its water mask contains 3,563 excluded interior cells at SHA-256 `cf299165…`.
- `scripts/analyze_water_exclusion.py` is a reusable source/mask/difference diagnostic. Its five-panel Bay audit separately shows the warped source, selected water, direct source classes still excluded, completed source classes still excluded, and cells restored from the prior candidate. The stored diagnostic is under `runs/rainfall-peninsula-marin-coastal-cell-extract-v4/diagnostics/peninsula-marin-water-v1`.
- Plan SHA-256 is `1c110a8c…`, extraction manifest is `29128925…`, and the corrected class raster is `6766cb12…`. Source-diff reaches a two-pass fixed point with identical signature `5fd10d14…`, no added completion, no dropped visible evidence, and no interior holes. A completely independent rebuild reproduces the extraction manifest, class raster, and publication-interior mask byte-for-byte.
- Processor-side high-resolution review at port 8800 shows the restored categorical surface following the canonical Bay shoreline while the named internal water remains transparent. The existing public package is intentionally unchanged until Todd accepts this replacement surface. All 156 Python tests, ten viewer tests, TypeScript, and the production build pass.
- Todd accepted the corrected combined/alignment view with “lgtm.” The saved alignment decision is SHA-256 `123fc573…` and binds extraction manifest `29128925…`, the exact canonical boundary, and the restored mixed-cell policy. The independent 35-class reviewer is exposed at port 8801 because alignment approval does not implicitly approve legend interpretation; all 35 Solo controls pass automated interaction checks with no browser warnings or errors.

### 2026-08-24 — Staging exposes and replaces the obsolete mainland clipping fill

- The corrected 35-class surface was separately approved and promoted, but its next isolated Mapbox staging comparison still showed large empty land patches over central San Francisco, SFO, San Rafael, and Marin City. Moving thematic rasters above Mapbox land-use fills did not change the holes. Direct target-grid probes proved the tiles themselves were empty because the old `california-mainland-hybrid-v1` fill marked those four land controls outside California; the selected Bay-water mask did not remove them.
- The approved lime `california-county-detail-border-v2` remains the sole display and alignment line at unchanged SHA-256 `9d0451e6…`. Its own manifest already declared clipping to be separate evidence because the detailed open bay entrances cannot safely be converted into an exterior fill contour.
- `california-mainland-clipping-v2` is a new pinned clipping-only reference. It subtracts the 25 exact-name `Pacific Ocean` polygons in the pinned 2025 Census Area Hydrography corpus from the California state polygon, retains the largest mainland component, retains internal bays/rivers/lakes for dataset-specific policy, and continues adding only the four author-approved `county.png` island outlines. The manifest is `c4c4d5ac…`, its GeoJSON is `41d85756…`, and all 13 land, retained-internal-water, and exterior controls pass at both vector and canonical-grid resolution.
- Against the old fill on the 3398-by-3920 canonical grid, the derived mainland restores 21,725 authoritative interior cells and removes 102,581 pixels of old fill-only overreach, primarily the seaward strip visible along the Pacific coast. A full-state diagnostic renders restorations in orange, removals in cyan, and the unchanged active lime line on top. San Francisco, SFO, San Rafael, Marin City, Daly City, and Berkeley are inside; Pacific water is outside; San Francisco and San Pablo bays remain inside the outer clipping contract so the Rainfall-specific water mask can remove them separately.
- Fresh extraction `rainfall-canonical-clipping-v2-extract-v5` has 1,018,219 canonical cells before internal water and 1,013,739 after the exact 4,480-cell reviewed Bay/Delta exclusion. Its manifest is `674b0876…`, class raster is `ce84b747…`, publication interior is `3badcdc9…`, and internal-water mask is `a6540c08…`. The key land controls retain classes while San Francisco Bay, San Pablo Bay, Suisun Bay, and the reviewed river chain remain transparent.
- Compared with the previously reviewed mixed-cell candidate, the new surface restores 2,758 cells, removes 18,442 old clipping cells, and reassigns zero classes. Within the Bay audit window it restores 1,514 cells and removes 3,241; the visual diff shows the restored Peninsula/Marin land and the removal of prior Pacific overreach independently. The source-diff loop reaches the same clean signature `3fcc5f8e…` twice with zero dropped evidence and zero valid-interior NoData.
- A completely independent extraction reproduces every core file byte-for-byte, including manifest `674b0876…`, class raster `ce84b747…`, and publication interior `3badcdc9…`. All 156 Python tests pass. Port 8802 serves the new combined review; neither the defective replacement staging package nor the existing public Rainfall package is activated by this work.

### 2026-08-24 — A newer 1981–2010 precipitation source replaces the pending old-map review

- Todd added `examples/rainfall.png` after noticing that the pending precipitation review used an old map. The new 3,204-by-2,366 ArcGIS screenshot is a different semantic dataset, **CA Average Annual Precipitation 1981 to 2010**, rather than a higher-resolution copy of the 1900–1960 source. It has ten solid ranges from 0–5 through greater than 150–171 inches; source SHA-256 is `df2a8802…`. The six-dataset public site and the old public Rainfall package remain unchanged.
- The screenshot's ArcGIS layer is displayed at 25% transparency, so its ten exact map-fill RGB values differ systematically from the opaque sidebar swatches. The plan preserves both: exact rendered map fills are classification prototypes, while the sidebar swatches remain the display palette and legend provenance. This prevents basemap colors and the right-side ArcGIS interface from entering the data surface.
- `mapscan.solid_mask_alignment` is a reusable alignment stage for mutually exclusive solid maps. It unions exact class fills, retains the mainland and four large island components, warps that evidence to the canonical 3,398-by-3,920 Web-Mercator grid, and fits a bounded similarity correction only against nearby pixels of the approved `california-county-detail-border-v2`. The accepted correction is a 0.999729 scale, -0.0143-degree rotation, +1.27-pixel x translation, and -2.77-pixel y translation. Robust symmetric boundary cost improves from 14.22 to 13.44; the source-to-canonical median improves from 4.00 to 2.77 pixels. A complete command-line repeat produces byte-identical alignment `54e09ad8…` and diagnostic `6f3dbc66…` artifacts.
- Extraction `rainfall-1981-2010-extract-v2` observes all ten legend categories and reconstructs 338,715 pixels hidden by roads, labels, relief, and other basemap ink without changing observed class pixels. The final categorical raster contains all 5,662,166 cells in the canonical water-aware publication interior, zero internal NoData, and zero color outside the approved border. Previously reviewed San Francisco Bay–Delta water remains transparent.
- The source-diff gate passes with zero dropped visible evidence and reproduces the class raster exactly at SHA-256 `9493edba…`. A nine-variant categorical threshold ensemble from Lab distance 0–2 and margin 0.5–1.5 produces zero additions, zero drops, and zero nonzero class changes; fidelity audit SHA-256 is `d4a4d60b…`.
- This source does not contain a usable county-boundary network. Its review therefore disables the generic county residual rather than presenting a misleading number; the state/class union and approved coast are the alignment evidence. The alignment reviewer exposes the solid-boundary metrics, and the independent category reviewer exposes ten Solo controls. Automated interaction checks successfully exercise all ten controls. All 160 Python tests pass. The candidate remains `needs_visual_review` and is not staged or published.

### 2026-08-25 — Rainfall gets a coast-only correction with the islands pinned

- Todd's visual review separated three different registration signals: the four offshore islands already lined up, the straight eastern state border already lined up, but the entire mainland Pacific coast needed to move left. The correction therefore does not refit a global transform and does not change the approved canonical border.
- `mapscan refine-solid-west-coast` is a reusable coast-specific stage. It extracts the exact solid thematic components, treats the largest component as the mainland, samples 13 controls along its western row envelope, and fits a compact Wendland-C2 displacement field. The user-directed coast controls move left by 4–18 target pixels and decay inland over a 360-pixel support radius.
- Every island is protected by five zero-displacement controls—its center and west, east, north, and south extrema—rather than by one weak centroid pin. Across the four candidate island components, maximum centroid drift is 0.059 pixels. The eastern border lies outside the compact support. The fitted field has sampled Jacobian determinant 0.903–1.100, so it introduces no foldover.
- The median target-minus-source western-coast residual improves from -8 pixels to -1 pixel. Whole-boundary median distance improves from 3.00 to 2.24 pixels and the within-3-pixel fraction improves from 53.0% to 59.6%. The coast correction remains a child alignment; the canonical county-derived state/coast and island line stay byte-identical.
- Fresh extraction `rainfall-1981-2010-west-coast-extract-v3` retains all 5,662,166 cells in the water-aware publication interior, zero internal NoData, zero color outside the canonical border, all ten legend categories, and the existing Bay/Delta water exclusions. Source-diff passes with no dropped evidence. The nine-variant threshold audit reports zero semantic class changes and zero strict drops; the broadest relaxation adds 18,985 unproven pixels, so the conservative exact-fill threshold remains in force.
- Port 8804 now serves this exact coast-corrected candidate and port 8805 serves its independent class review. The candidate remains `needs_visual_review`; neither the older public Rainfall package nor the approved canonical border has been changed.

- Todd rejected that first coast candidate after observing a continuous white strip immediately inside the lime Pacific boundary. The earlier gate was wrong for this review question: it minimized an unsigned/median boundary distance, allowing left and right errors to cancel, while the canonical clipping could conceal safe seaward overreach. The revised gate measures only the one-sided white gap where the thematic mainland remains east of the canonical coast.
- A second mainland-only child transform adds a further ten-pixel leftward pull at 13 coast controls with 650-pixel compact support. The white-gap median falls from 1 to 0 pixels, P90 from 11 to 2 pixels, and the affected-row fraction from 56.1% to 14.3%. Safe source overreach remains clipped by the immutable canonical publication interior.
- The four island components retain their 20 zero-displacement controls; maximum centroid drift is 0.281 pixels. A new eastern-envelope veto independently verifies that 99.26% of eligible eastern rows are exactly unchanged and its absolute P90 drift is zero. The canonical border itself remains unchanged. This stronger candidate replaces the rejected v2 candidate for review but is still not approved or public.

- A direct source-only/class-only comparison exposed the deeper cause before presenting that stronger stretch as a fix. The thematic source had already crossed the lime line, but extraction subsequently clipped it against the separate Census/Pacific mainland interior. That clipping edge sat a median 26 pixels inside the county-derived lime coast and recreated the white strip after every warp. Further geometric stretching would therefore distort aligned data without closing the visible seam.
- The plan returns to the island-pinned v2 coast alignment and adds a bounded clipping-seam policy instead. For each western row it closes only gaps of 50 pixels or less between the existing mainland clip and the active lime coast; wider openings such as bays are untouched, and the named Bay/Delta water mask still runs afterward. The seam adds 89,647 interior cells across 3,397 rows while leaving the canonical line, eastern border, and four island components unchanged.
- Fresh candidate `rainfall-1981-2010-coastal-seam-extract-v5` contains 5,751,813 water-aware classified cells, zero interior NoData, zero color outside its declared publication interior, and no dropped visible evidence. The categorical audit reproduces the baseline exactly with zero semantic class changes. The full suite passes with 166 tests. Port 8804 now serves this corrected source/alignment review and port 8805 serves the matching independent class review; neither is approved or public.

- Todd accepted the repaired coast and asked to remove data from Monterey Bay and San Francisco Bay. The pinned 2025 Census Area Hydrography corpus contains two exact-name `Monterey Bay` polygons and nine `San Francisco Bay` polygons; Monterey joins the existing explicit include-only water list rather than enabling unrelated hydrography.
- The review source layer previously used only the outer canonical clip, so named water could still look colored there even when the categorical raster was transparent. Source review and class extraction now share the exact same coastal-seam-plus-water publication footprint. The selected-water mask grows from 20,445 to 27,711 cells, removing 7,266 additional Monterey Bay cells while retaining the established Bay–Delta exclusions and mixed shoreline policy.
- Candidate `rainfall-1981-2010-monterey-sf-water-extract-v6` contains 5,744,547 classified land cells, zero valid-interior NoData, zero exterior color, no dropped source evidence, and zero threshold-induced semantic class changes. Port 8804 serves the matching water-aware source review and port 8805 its class review. All 166 tests pass; the candidate remains unapproved and unpublished.

### 2026-08-25 — Named bays use the approved lime shoreline as their cut edge

- Todd rejected the first Monterey/San Francisco Bay exclusion because its white edge followed the offset Census hydrography polygons instead of the visible approved lime line. The semantic and geometric roles are now separate: exact Census names decide which water bodies may be removed, while `california-county-detail-border-v2` decides the shoreline pixels.
- `snap_named_water_to_active_boundary` treats the active hash-verified mainland line as a hard four-connected barrier. It retains only the image-edge-connected side of that line within 40 target pixels of the exact-name `Monterey Bay` and `San Francisco Bay` Census seeds. This reaches the lime shoreline, rejects Census pixels that fell landward of it, and cannot spread into an unlisted water body or the wider Pacific.
- The snapped broad-bay surface retains 6,894 Census seed cells, removes 9,341 landward seed cells, and adds 7,836 water-side cells needed to reach the lime edge. The established unsnapped Bay–Delta names remain independent; the combined internal-water exclusion contains 24,911 cells.
- Candidate `rainfall-1981-2010-lime-water-extract-v7` contains 5,747,340 classified land cells, zero valid-interior NoData, zero exterior color, and uses one identical publication footprint for source and class review. Source-diff passes with zero dropped evidence; the nine-variant categorical audit has zero strict drops and zero semantic class changes. Side-by-side high-resolution inspection confirms both San Francisco Bay and Monterey Bay now terminate at the approved line. Ports 8804 and 8805 serve the corrected reviews. All 168 Python tests pass; the candidate remains unapproved and unpublished.

### 2026-08-25 — The lime-snapped 1981–2010 Rainfall surface enters isolated staging

- Todd instructed the corrected v7 source/alignment and ten-class surface to proceed. Separate saved alignment and classification decisions bind the same reviewed extraction. The classification decision records all ten live Solo checks, zero dropped source evidence, zero categorical semantic changes, and the exact shared lime-shoreline publication footprint.
- The repeated source comparison reaches a byte-identical fixed point in two passes at signature SHA-256 `8aba64d8…`. Immutable promotion initially caught a real reproducibility gap: the promotion verifier reconstructed the pinned mainland clip but not the candidate's declared 3,397-row coastal seam. Promotion now independently reconstructs that seam before applying water exclusions and rejects any mismatch in method, parameters, active boundary manifest, row counts, added cells, or gap statistics. A regression test covers this order of operations.
- The successful promoted materialization is SHA-256 `c6ca1067…`; its exact-identity approval is `b9c979ba…`, and its class raster is `f6e2813f…`. It retains 5,747,340 classified publication-interior cells, removes exactly 24,911 reviewed water cells from the 5,772,251-cell canonical interior, and has zero exterior color and zero publication-interior NoData. The canonical display line remains the five-component `california-county-detail-border-v2` raster at `9d0451e6…`.
- `publish/staging/rainfall-1981-2010-v1` contains ten independently recolorable z4–z9 pyramids and 4,000 PNG tiles. Dataset manifest SHA-256 is `a34c4ee5…`; public provenance is `4be5ab95…`. The 4,015-file viewer copy is byte-identical to the staging source. An exhaustive native-z9 audit checks 18,874,368 tile pixels and finds zero nonbinary alpha pixels, zero overlaps between categories, zero footprint differences, zero class differences, and no category tile-set hash mismatch.
- The isolated `/mapscan/staging` route exposes only **Annual precipitation 1981–2010 · staging**; the six-dataset public catalog and its older 1900–1960 Rainfall package remain unchanged. The route, dataset manifest, five-feature boundary GeoJSON, and sample category tile all return HTTP 200 with the expected content types. Package-level crops confirm the San Francisco Bay–Delta and Monterey Bay exclusions remain transparent at the approved edge. The in-app browser security policy refused to reload its stale pre-server error tab, so final live Mapbox presentation approval remains Todd's next gate rather than being inferred from file-level QA.
- All 169 Python tests, 12 viewer tests, TypeScript, and the Next.js production build pass. No public activation has occurred.

### 2026-08-25 — The live Mapbox water layer restores Bay Area land

- Close Mapbox inspection of the v7 staging package exposed a different failure than the earlier lime-edge problem: the lime-snapped flood treated parts of Marin, San Rafael, the eastern San Francisco Peninsula, and the SFO shoreline as water even though the live Mapbox basemap treats those cells as land. The thematic classes were present before the broad-bay exclusion, so inventing new precipitation values was unnecessary.
- The new rule retains the semantic/geometric split but makes the geometric authority match the final viewer. Exact Census names still restrict the operation to `San Francisco Bay` and `Monterey Bay`; a pinned copy of the `water` source layer from `mapbox.mapbox-streets-v8` under `mapbox/light-v11` decides land versus water within that bounded neighborhood. The 15 z9 vector tiles, style JSON, tile inventory, and aggregate hash are stored without an access token in `reference/mapbox-light-v11-water-sf-monterey-z9-v1`.
- Against v7, the corrected broad-water surface restores 5,760 cells that Mapbox identifies as land and removes 12,864 cells that Mapbox identifies as water. San Rafael, Marin City, Sausalito, Tiburon, San Francisco, SFO, South San Francisco, San Bruno, Millbrae, Burlingame, San Mateo, Foster City, and East Palo Alto retain neighboring precipitation classes; the centers of San Francisco Bay, San Pablo Bay, the Golden Gate, and Monterey Bay remain transparent. The final categorical surface has 5,740,236 classified publication cells, zero interior NoData, and zero exterior color.
- Extraction `rainfall-1981-2010-mapbox-water-extract-v8` is SHA-256 `55895921…`; its class raster is `c532a533…`, publication interior is `13bdb0c6…`, and combined water mask is `df8c3aca…`. An independent extraction reproduces every core hash byte-for-byte. Two source-diff passes reach identical signature `55ceae29…`, and the nine-variant categorical audit reports zero strict drops, zero relaxed additions, and zero semantic class changes.
- `export-extraction-preview-tiles` creates a review-only Mapbox package directly from a hash-bound extraction without manufacturing or carrying forward approval. The 4,014-file v8 preview repeats byte-for-byte, declares `needs_visual_review`, and is exposed only at `/mapscan/staging`; the six-dataset public site and its older public Rainfall package remain unchanged. A shareable `view=longitude,latitude,zoom` camera now opens repeatable shoreline inspections directly and is included by “Copy map link.”
- Live review at `view=-122.38,37.83,9.20` shows continuous precipitation data over the eastern Peninsula and Marin/San Rafael land while Mapbox water remains visible over San Francisco and San Pablo bays. The exact five-component canonical lime raster can be toggled over the same camera. The browser console has no warnings or errors. All 173 Python tests, 13 viewer tests, TypeScript, and the Next.js production build pass. This corrected surface remains unapproved and unpublished pending Todd's visual decision.

### 2026-08-25 — The Mapbox-water Rainfall surface is approved and activated

- Todd inspected the exact v8 Bay camera and approved it with “lgtm. proceed.” Fresh alignment and classification decisions bind extraction manifest `55895921…`, class raster `c532a533…`, the active five-component canonical boundary, the pinned Mapbox water reference, and the reviewed live presentation. Their SHA-256 values are `01d95783…` and `b9d74ba8…`; neither carries forward a prior decision.
- Immutable promotion now treats the reviewed internal-water mask as a first-class artifact instead of validating counts alone. It proves the hash-bound `df8c3aca…` mask is exactly the canonical interior minus the published surface, copies it into materialization, and retains a sanitized chain to Census hydrography, `mapbox/light-v11`, `mapbox.mapbox-streets-v8`, the 15 pinned z9 tiles, style hash, and tile-aggregate hash without storing or publishing the access token.
- The byte-identical promoted class raster remains `c532a533…`. Materialization is `1d9b5ba5…`, its exact approval is `a0cc8535…`, and it retains 5,740,236 classified publication cells with zero interior NoData and zero exterior color. The canonical lime display raster remains `9d0451e6…`.
- Approved staging package `rainfall-1981-2010-mapbox-water-v8-approved` contains 4,016 files. A complete second export is byte-identical, and all 4,000 thematic tile PNGs are byte-identical to the preview Todd inspected. Dataset manifest is `6370b9a9…`; public provenance is `256af01d…`. The package includes the exact reviewed water-mask artifact and its public, path-sanitized Mapbox reference record.
- Public activation copies the staging package without re-exporting a pixel. Package inventory SHA-256 is `76de4d23…`; activation decision is `03f9b57a…`. The semantically distinct 1900–1960 precipitation dataset remains preserved as item 06, while **Annual precipitation 1981–2010** enters the catalog as item 07.
- The live public Bay camera at `?dataset=california-average-annual-precipitation-1981-2010&view=-122.38,37.83,9.20` shows the approved land/water result, all ten controls, and an Approved badge. Served dataset and native-z9 tile bytes match the local public package exactly, and the browser console has no warnings or errors. All 173 Python tests, 14 viewer tests, TypeScript, and the Next.js production build pass.

### 2026-08-25 — Elevation exercises a canonical continuous-value pipeline

- The next-source audit chose `examples/elevation.gif` over Fire, Landslide, Rivers, and Geology. Its eight-stop elevation legend is already machine-readable, its prior continuous extraction reproduced exactly, and it adds a capability absent from the seven public datasets: one numeric color ramp rather than mutually exclusive category layers. Geology still has 53 provisional unit labels, Rivers still mixes label ink with geometry, and Fire/Landslide still depend on older clipping or overlap assumptions.
- The former continuous branch warped against an older Census state fill and could not prove the active lime-border contract. Schema v2 now reconstructs `california-mainland-clipping-v2` plus the four approved county-detail islands, closes only the bounded west-coast seam, removes pinned Census/Mapbox water as cartographic context, zeros every exterior value, and fills only residual in-boundary target cells from in-boundary encoded values. Source completion, target completion, OCR completion, boundary removal, publication interior, and internal water remain separate hash-bound masks.
- The first canonical run exposed why a zero-byte diff is necessary but insufficient: anti-aliased province and place labels could still land close to the dark-green ramp. Lowering a global dark-ink threshold erased real relief before it erased those labels. Apple Vision therefore proposes rotated and widely spaced label boxes, but the accepted mask contains only compact, locally dark glyph-like components inside those boxes plus a three-pixel anti-alias margin. An intermediate whole-box reconstruction was rejected after it produced visible polygonal interpolation artifacts.
- Candidate `runs/elevation-canonical-continuous-v11` directly observes 250,780 of 394,562 source-state pixels and records 143,782 source occlusions, including 28,543 OCR-located glyph pixels. On the 879-by-1,014 Web-Mercator grid it retains 246,316 direct cells, 128,335 source-completed cells, 642 target-edge completions, and exactly 375,293 final publication cells. It removes 17,525 exterior source values and 11,146 internal-water cells, with zero unknown cells inside and zero colored cells outside.
- A fresh independent extraction reproduces both 16-bit value rasters and every source/OCR/target/boundary/water evidence mask with zero differing pixels. The dedicated continuous reviewer on port 8806 can switch among the aligned source, extracted elevation ramp, OCR reconstruction footprint, all source reconstruction, target-edge completion, and water exclusion while keeping the exact active lime line visible. The page loads without console errors and records alignment arrows and review decisions against the continuous manifest rather than a synthetic categorical copy. Elevation remains unpublished pending full-resolution visual alignment review.

### 2026-08-25 — Elevation receives a pinned northward Pacific-coast correction

- Todd's first full-state elevation review separated the residual registration error from the global fit: the eastern state border looked correct, while the Pacific coast needed a small northward nudge. The accepted response is therefore a compact coast-only child correction, not a new global affine transform and not a change to the approved county-derived boundary.
- `mapscan refine-directional-west-coast` samples 13 controls from the active western mainland line and requests a 12-pixel northward motion on the 3,398-by-3,920 canonical grid, equivalent to about 3.1 pixels in the 879-by-1,014 review raster. Nine zero-motion controls pin the eastern envelope. All four approved islands are held by 20 additional center-and-extrema pins.
- The local Wendland-C2 fit has zero control residual, kernel condition 2,218, and sampled Jacobian determinant 0.915–1.064. The complete eastern envelope moves at most 0.688 canonical-grid pixels (about 0.18 review pixels), while the approved island line moves at most 0.291 canonical-grid pixels (about 0.08 review pixels). The canonical border itself remains byte-identical.
- Candidate `runs/elevation-canonical-continuous-v12` preserves the explicit 1,014-pixel target height, contains exactly 375,293 publication cells, 246,239 direct cells, 128,411 source-completed cells, 643 target-edge completions, zero interior unknowns, and zero exterior color. Alignment SHA-256 is `25994454…`; extraction manifest is `70aa8623…`; value raster is `7f19fd7b…`.
- The independent continuous audit passes with zero differing source pixels, zero differing Web-Mercator pixels, and zero differences in every OCR/source/target/boundary/water evidence mask. Port 8806 is replaced with this v12 candidate for visual confirmation before any approval or publication.

- Todd's class-only inspection then identified a remaining westward registration error. The directional stage now accepts independent west and north components, allowing a second child correction to add 12 canonical-grid pixels of westward motion (about 3.1 review pixels) without discarding the accepted northward component. Source and extracted values still share one transform; the pipeline never offsets classes independently from their evidence.
- The composed coast displacement is approximately 12 pixels west and 12 pixels north at the 13 coast controls. Across the complete eastern envelope the median displacement is zero and the worst sampled displacement is 0.970 canonical pixels (about 0.25 review pixels). Across every approved island-line pixel, P90 displacement is 0.151 and maximum displacement is 0.411 canonical pixels (about 0.11 review pixels). The second field's Jacobian remains positive at 0.936–1.065.
- Candidate `runs/elevation-canonical-continuous-v13` retains all 375,293 publication cells, with 246,526 direct cells, 128,129 source-completed cells, 638 target-edge completions, zero interior unknowns, and zero exterior color. Alignment SHA-256 is `87d82525…`; extraction manifest is `555d734d…`; value raster is `61ebded6…`. The independent audit again reports zero differences for both value rasters and every evidence mask.

- Todd's high-zoom Bay screenshot rejected v13 as a publication candidate. A source-boundary comparison showed the westward source motion had actually improved the outer Pacific residual, but the visible black strip was not source NoData: the all-hydrography water pass was subtracting `Pacific Ocean` after the lime-coast seam had re-added the correct narrow landward band. No amount of additional raster warping could survive that later clip.
- Elevation now excludes the exact `Pacific Ocean` name from its dataset water pass because `california-mainland-clipping-v2` already owns that semantic subtraction. Lakes, reservoirs, named bays, and the Bay–Delta network remain independently transparent. The coast seam also interpolates 74 review-grid rows where nearest-neighbor reduction omitted a canonical line pixel; all 819 accepted coastal rows are now continuous rather than alternating between filled and empty scanlines.
- Two processor-side alternatives were rejected before author review. Using the lime line alone with a 40-pixel named-water flood removed large circular regions of valid Peninsula/East Bay land; reducing that radius to ten pixels retained smaller but still invalid circular holes. Candidate v16 therefore keeps the pinned Mapbox water decision for internal San Francisco and Monterey Bay geometry while using the active lime line solely for the Pacific clipping edge.
- `runs/elevation-canonical-continuous-v16` contains 379,924 publication cells, 248,150 direct cells, 131,116 source-completed cells, 658 target completions, zero interior unknowns, and zero exterior color. It removes 6,888 internal-water cells after the Pacific exclusion. Extraction manifest SHA-256 is `0cadfd30…`, value raster is `e45227da…`, and publication interior is `00b18333…`. The independent audit reports zero differences for both value rasters and every evidence mask.

- Todd's browser inspection correctly rejected v16 because the broad class edge remained visibly inland of the lime coast around Marin and the San Francisco Peninsula. The earlier processor check measured connectivity, not coincidence. Row probes proved the repaired mask reached the lime line before water removal—for example x=163 at review row 500—but the final publication mask retreated to x=172.
- The remaining erasure came from the Mapbox-constrained named-bay snap rather than the 25 exact-name Pacific polygons. Its 40-pixel eligibility neighborhood included outer-Pacific Mapbox water near the San Francisco and Monterey Bay entrances and removed 882 cells from the newly repaired coast seam. All water sources are now bounded by the original pinned mainland clipping interior; that polygon deliberately retains internal bays for dataset water removal, while the westward seam is exclusively recovered outer-coast land.
- `runs/elevation-canonical-continuous-v19` restores those 882 coastal cells. Every measured outer-coast row through Marin, San Francisco, and the Peninsula has zero class-to-lime gap; the remaining openings are the intentional internal Monterey Bay water and a separate southern opening. The candidate contains 380,806 publication cells, 248,382 direct cells, 131,681 source completions, 743 target completions, zero interior unknowns, and zero exterior color. It removes 6,006 internal-water cells.
- Extraction manifest SHA-256 is `1036d8c5…`, value raster is `c8ed9725…`, publication interior is `8ebeb3d9…`, and independent audit is `6ac602cb…`. The audit reports zero differences for both value rasters and every evidence mask. All 182 Python tests pass, including regressions proving neither general hydrography nor snapped named bays can erase the repaired outer-coast seam.

- Todd rejected v19 after correctly identifying that its recovered west-coast band was neighbor-filled at the fixed target rather than produced by stretching the source map. The v19 publication change and both water-mask protections were rolled back; v19 is not a review or publication candidate.
- `elevation-west-coast-stretch-v3` applies a real additional 24-canonical-pixel westward displacement (about six pixels at review resolution) to the western mainland with a 650-pixel compact support. It composes on the accepted west/north child alignment, so source and numeric values move together. Nine eastern-envelope controls and 20 four-island controls remain exactly fixed. The sampled Jacobian determinant is 0.908–1.104 with no foldover; alignment SHA-256 is `d994a150…`.
- Continuous extraction now has a source-supported seam gate for this candidate. Every coastal-seam cell is retained only when the geometrically warped encoded source supplies a nonzero value; target completion is forbidden there. Of 5,985 candidate seam cells, 5,689 are source-supported, including 869 real warped-source cells previously clipped by water, while 216 unsupported cells that an interior fill would have retained are removed. This is a geometric stretch, not target-grid outpainting.
- `runs/elevation-coastal-stretch-continuous-v21` contains 380,577 publication cells: 249,040 direct, 131,089 source-completed, and 448 target-completed away from the coast seam. It has zero interior unknowns and zero exterior color. Extraction manifest is `b389d85a…`, value raster is `8ac2dec4…`, publication interior is `d31b0a38…`, and independent audit is `86429fe6…`; both value rasters and every evidence mask reproduce with zero differences. All 181 Python tests pass.

- Todd correctly rejected v21 as not actually reverted: despite removing neighbor fill, its source-supported seam still restored 869 pixels outside the original publication mask. That mechanism, its plan flag, helper, and regression were removed completely. V21 is not a review or publication candidate.
- Strict-revert candidate `runs/elevation-coastal-stretch-continuous-v22` keeps only the real 24-pixel westward geometric stretch. Its publication-interior mask is byte-identical to pre-change v16 at SHA-256 `00b18333…`, and its internal-water mask is likewise restored exactly to `1ff5447b…`; neither source-supported nor neighbor-filled coastal pixels can extend that footprint.
- V22 contains 379,924 publication cells: 248,821 direct, 130,439 source-completed, and 664 target-completed under the pre-existing interior policy. It has zero interior unknowns and zero exterior color. Manifest is `db6da320…`, stretched value raster is `6cb76985…`, and independent audit is `f3f9a32b…` with zero differences in both values and every evidence mask. All 180 Python tests pass.

- After confirming the strict v22 rollback, Todd explicitly requested a stronger leftward stretch. `elevation-west-coast-stretch-v4` therefore increases the real Pacific-side displacement from 24 to 36 canonical-grid pixels, about nine pixels in the review raster. It uses the same 650-pixel compact support; the eastern controls and all four island controls remain pinned. The sampled Jacobian determinant is 0.862–1.157 with no foldover. Alignment SHA-256 is `60a12d8e…`.
- The fixed target footprint would conceal the outer edge of a real stretch, so v23 restores only a **warp-supported extent** gate: a coastal cell can appear only when the newly warped encoded source itself contains a value there. No target completion or neighbor-derived value can enter this extent. Of 5,985 candidate coastal cells, 5,684 contain warped source evidence; 866 of those become newly visible and 218 unsupported cells are removed.
- `runs/elevation-coastal-stretch-continuous-v23` contains 380,572 publication cells: 249,238 direct, 130,888 source-completed, and 446 target-completed away from the coastal extent. It has zero unknown cells inside and zero colored cells outside. Extraction manifest is `643f4e7e…`, stretched value raster is `f7a6ac8e…`, publication interior is `ae665ac5…`, and the independent audit is `63aa8c46…` with zero differences in both value rasters and every evidence mask. All 181 Python tests pass. Port 8806 serves this candidate at the Bay Area inspection view; it remains unapproved and unpublished.

- Todd rejected v23 because it repeated the earlier mask-order mistake: all 866 newly exposed coastal pixels also overrode pixels already classified as water. That simultaneously reintroduced data in San Francisco Bay and produced horizontal protrusions resembling leftward outpainting. V23 is rejected and must not be promoted.
- The warped-source coastal-extent helper, plan flag, and test were deleted rather than conditionally bypassed. Candidate `runs/elevation-coastal-stretch-continuous-v24` retains the 36-pixel geometric transform but cannot enlarge or rewrite the strict post-water publication footprint. Its publication mask is byte-identical to v22 at `00b18333…`, and its internal-water mask is byte-identical to v22 at `1ff5447b…`. Its value raster differs from v22, proving the stronger geometry remains active beneath those immutable masks.
- V24 contains 379,924 publication cells: 248,974 direct, 130,286 source-completed, and 664 target-completed under the existing interior policy. It removes 6,888 water cells and 12,886 warped source cells outside the canonical footprint, with zero unknown cells inside and zero colored cells outside. Manifest is `bae6f02b…`, value raster is `aaef1b8a…`, and the independent audit is `39f748ef…` with zero value or evidence-mask differences. All 180 Python tests pass. Port 8806 serves v24 at the Bay inspection view; it remains unapproved and unpublished.

- A later blank review was operational rather than a raster failure: the turn-scoped review process had exited, leaving no listener on port 8806 and causing every asset request to fail with connection refusal. The preview artifact remained intact at 511,726 bytes. The v24 reviewer now runs as the persistent `com.toddsherman.mapscan.review` launchd job with the MapMap working directory explicitly set; the page, session manifest, and preview asset all return HTTP 200. Browser automation policy blocks automated localhost reloads in the reconnected session, so the final visual refresh remains a manual browser action rather than an inferred approval.

- Todd's next Bay Area screenshot exposed a different v24 failure: the real 36-pixel source stretch was present, but the strict legacy Census publication polygon still clipped the warped raster several review pixels inside the approved lime coast. This produced a black land seam and horizontal gaps even though the source geometry had moved correctly. Another transform or an outpaint would have repeated the rejected v19/v23 mistakes.
- The optional `active_boundary_ring` publication mode now fills the hash-verified high-resolution mainland ring from `california-county-detail-border-v2`, adds its four approved island rings, downsamples that interior to the extraction grid, and then subtracts the established named/Census/Mapbox water mask. The original byte-identical lime raster remains the display authority; it is never reconstructed from this fill. The legacy pinned polygon remains a conservative seed for completion rather than the outer clipping edge.
- A first ring candidate, v25, was automatically rejected because 177 cells in its newly exposed coastal band came from target-grid completion. V26 adds a strict warped-source support gate: of 5,163 candidate cells outside the old polygon, exactly 4,792 contain real warped values and remain visible, while 371 unsupported cells are removed. Target completion is forbidden in that expansion and occurs only in 70 cells inside the legacy interior.
- `runs/elevation-active-boundary-continuous-v26` contains 379,035 publication cells: 249,101 direct, 129,864 source-completed, and 70 target-completed inside the legacy polygon. It has zero unknown cells inside, zero colored cells outside, removes 6,291 water cells, and preserves San Francisco Bay, the Delta, and Monterey Bay as transparent. Side-by-side full-resolution inspection shows the v24 black seam removed around Point Reyes, the Peninsula, and the central coast without seaward protrusions or invented values.
- Extraction manifest SHA-256 is `c12e73b4…`, value raster is `1565f014…`, publication interior is `d453dcb9…`, water mask is `22aff8c3…`, and preview is `858c95b5…`. The independent audit is `b588a689…` and reports zero differences in both value rasters and every evidence mask. All 182 Python tests pass. The persistent port 8806 reviewer now serves v26; it remains unapproved and unpublished pending Todd's live visual confirmation.

- Todd's high-zoom v26 screenshot revealed narrow horizontal black notches immediately inside the lime coast. Their 371 cells matched the active-ring candidates rejected by the source-support gate. The support test was correct, but it exposed an earlier double clip: source classification was bounded by the legacy Census state in source space, and the warped result was bounded by that legacy state again in target space before the canonical ring was applied.
- Active-ring mode now gives source classification an explicit one-source-pixel reconstruction margin around the legacy state seed and disables only the redundant target-state pre-clip. This margin repairs printed coastline ink and subpixel rasterization gaps before the map is warped. The canonical ring remains the sole outer publication authority, named/Census/Mapbox water is still subtracted afterward, and target-grid completion remains forbidden in the ring expansion.
- The one-pixel source margin contains 5,056 source cells: 942 directly match the elevation ramp and 4,114 are recorded source-side cartographic reconstructions. A bounded diagnostic showed that this recovers 368 of the 371 v26 notches. Candidate `runs/elevation-active-boundary-source-margin-continuous-v27` consequently retains 5,160 of 5,163 ring-expansion cells; the three cells with no source-side support remain transparent.
- V27 contains 379,403 publication cells: 249,291 direct and 130,112 source-completed. It uses zero target-grid completions, has zero interior unknowns and zero exterior color, and removes the same 6,291 water cells as v26. Side-by-side high-zoom inspection confirms the horizontal Pacific-coast notches disappear while San Francisco Bay, the Delta, and Monterey Bay remain transparent.
- Extraction manifest SHA-256 is `03500118…`, value raster is `fe0b2b67…`, publication interior is `be3215b4…`, preview is `1e1b9e41…`, and independent audit is `6d2c9330…`. The audit reproduces both value rasters and every evidence mask with zero differences. All 183 Python tests pass. Port 8806 serves the exact v27 preview bytes and remains the visual approval gate; v27 is not published.

- Todd rejected the entire accumulated v27 alignment approach after comparing the original warped source with the lime line. The source made the geometric error clear: northern, southern, and eastern evidence was acceptable, but the previous child-correction chain placed the Pacific coast consistently offshore. V27 is superseded and must not be promoted.
- Direct row-envelope measurement quantified the mistake. The v4 child alignment put the printed source coast a median seven review pixels west of the lime line. The untouched `elevation-auto-perimeter-v2` registration put it a median three pixels east. The replacement therefore starts from that uncorrected base alignment and inherits none of the north/west/stretch Web-Mercator operations.
- `fit-east-anchored-horizontal-scale` is a reproducible source-first fitting stage. It searches one narrow global x-scale range, separates warm/green elevation land from blue ocean with a five-pixel support gate, minimizes signed and absolute Pacific-coast residuals, and analytically translates the source transform so the complete eastern-border control set retains the same mean source x coordinate. Source y parameters are immutable.
- Fresh fit `runs/elevation-east-anchored-global-v28` selects reference-to-source x multiplier `0.991`, which replaces the previously over-expanded west side and yields zero median coast bias, -0.05-pixel mean bias, two-pixel median absolute error, and five-pixel P90 across 869 rows. The eastern anchor's mean displacement is effectively zero, its median absolute source displacement is 1.39 pixels, its maximum is 2.82 pixels, and every y coordinate is unchanged. Alignment SHA-256 is `67012f88…`; source overlay is `93b65593…`.
- The extraction plan now points only to v28 and removes the one-pixel source-evidence margin used by v27. Candidate `runs/elevation-east-anchored-continuous-v29` contains 379,112 publication cells: 248,792 direct, 130,254 source-completed, and 66 target-completed inside the legacy interior. It retains 4,869 source-supported active-ring expansion cells, removes 294 unsupported cells, has zero unknown cells inside and zero colored cells outside, and preserves the same 6,291-cell water exclusion.
- Extraction manifest SHA-256 is `56f5cbfe…`, value raster is `0d13e6e0…`, publication interior is `27f6590c…`, preview is `40796798…`, and independent audit is `4c768ae0…`. Both values and every evidence mask reproduce with zero differences. All 184 Python tests pass. Port 8806 serves the exact v29 source bytes by default; the source alignment, rather than a clipped class-only view, is the next approval gate. V29 is not published.

### 2026-08-27 — The apparent source reset is rejected and replaced by a literal width reduction

- Todd correctly challenged v28/v29 as not being a true implementation of the requested source reset. Although those runs had discarded the accumulated Web-Mercator child corrections and used the original `examples/elevation.gif` bytes, the automatic coast score selected a `1.00908` rendered-width multiplier. In other words, it widened the remaining base registration while the requested operation was to make the source narrower. V28/v29 are rejected review candidates.
- The fitting stage now has a strict `--require-rendered-width-reduction` contract. It rejects any candidate range capable of widening the rendered source, records the original source SHA-256, proves that no prior pixel materialization or Web-Mercator correction is inherited, and retains the source y parameters byte-for-byte.
- `elevation-source-reset-compression-v30` reads the original 1,117-by-1,200 GIF with SHA-256 `6ee9ed7a…`, starts from the correction-free source registration, and applies one global east-anchored source transform. Its rendered-width multiplier is `0.990197`, the mean eastern-border x displacement is effectively zero, source y displacement is zero, and no reconstructed or previously warped raster is an input.
- Fresh extraction `elevation-source-reset-continuous-v31` contains 379,117 publication cells: 245,516 direct, 133,539 source-completed, and 62 target-completed. It has zero unknown cells inside, zero colored cells outside, and preserves the 6,291-cell internal-water exclusion. The independent continuous audit reports zero differing source pixels, zero differing Web-Mercator pixels, and zero differences in every evidence mask. All 185 Python tests pass. Port 8806 now serves v31 with Source at 100%, Classes at 0%, and the canonical line at 100% so the literal source warp is the visual approval gate.

- Todd then made the stage boundary explicit: source geometry must be approved before data extraction begins. V31 is therefore retained only as discarded exploratory evidence and is no longer a review candidate. The v30 alignment run now emits a lossless `web-mercator-source.png` plus a transparent artifact labeled “Data extraction not run.” The reviewer can open an alignment directory directly, hides classification and canonical-clip controls, and binds decisions or correction arrows to the alignment-manifest SHA-256 rather than an extraction manifest. Port 8806 now serves only the original source warped by v30 and the canonical lime line. The alignment directory contains no extracted value raster, completion mask, or classified layer. All 186 Python tests pass.

- Todd's high-zoom Point Reyes screenshot showed the compressed source still slightly east and south of the canonical coast. “45% left and up” is implemented as an approximately 45-degree northwest vector with equal small components, not as a destructive 45% rescale: v33 moves the western-mainland controls 28 native canonical-grid pixels west and 28 north, about 7.2 pixels per axis in the 879-by-1,014 review raster.
- The correction is one compact source-sampling warp over the untouched GIF. Nine eastern-border pins, nine northern-edge pins, nine southern-edge pins, and 20 island pins prevent the northwest adjustment from becoming a statewide translation. Across the fully sampled eastern envelope the maximum displacement is 0.004 native pixels; across island-line pixels it is 0.867. Across the interior 64% of the north and south edge bands, maximum displacement is below 0.75 native pixels, about 0.2 review pixels. The sampled Jacobian remains positive at 0.789–1.151.
- The west-coast signed residual improves from a seven-review-pixel median under v30 to two pixels under v33; absolute median is three and P90 is 8.7 across 864 supported rows. Port 8806 now serves `elevation-source-west-north-v33/fit` as a source-only alignment review. Its only raster inputs are the freshly warped source, a transparent “Data extraction not run” placeholder, and the canonical line. All 186 tests pass; extraction remains blocked pending visual approval.

- Todd's Bay-area inspection found that v33 slightly overshot northwest and requested a 45% nudge back down and right. V34 is regenerated directly from v30 rather than counter-warping v33: it retains 55% of the prior motion, or 15.4 native canonical-grid pixels west and north, about four review pixels per axis. The same 47 fixed border-and-island pins remain, control residual is numerical zero, and the sampled Jacobian improves to 0.884–1.083. Port 8806 now serves v34 as source alignment only; no extraction has run against it.

- Todd accepted v34 as “good enough” and explicitly authorized data extraction. `runs/elevation-v34-continuous-v35` is the first continuous extraction bound to the accepted alignment `d0f72005…`; it does not reuse the discarded v31 exploratory raster. The extraction manifest SHA-256 is `ee6a0d5a…`, the encoded Web-Mercator value raster is `afc41469…`, and the visual preview is `c2aa370d…`.
- V35 contains exactly 379,106 publication cells: 246,619 directly observed, 132,426 completed from recorded source occlusions, and 61 target-edge completions inside the approved publication interior. It has zero unknown cells inside the canonical boundary and zero colored cells outside it. The pinned hydrography pass removes 6,291 water cells, including the San Francisco and Monterey bays and the Bay–Delta waterways, while keeping the Pacific clipping edge under the canonical boundary contract.
- Independent replay `runs/elevation-v34-continuous-diff-v35` passes with zero differing source-value pixels, zero differing Web-Mercator value pixels, and zero differences in every OCR, source-completion, target-completion, publication-interior, boundary-removal, and internal-water evidence mask. All 186 Python tests pass. V35 advances to visual extraction review but remains unpublished.

- Todd's first v35 extraction review identified blue source pixels inside the lime boundary as a likely source of lost detail. The previous classifier treated most of those pixels as generic Telea occlusions, but directly accepted 741 blue-dominant pixels; some could therefore be confused with the dark depression swatch. V36 makes the assumption explicit and auditable: every source pixel whose blue channel exceeds red by at least 20 and green by at least five is excluded from direct color classification and receives the exact value of the nearest spatially observed legend-ramp pixel. This rule runs before the independent geographic water clip, so actual mapped water remains transparent.
- The new evidence mask selects 18,791 source pixels and retains 10,232 reconstructed land pixels after warping and water removal. It changes 44,620 Web-Mercator values relative to v35 while preserving the same 379,106-cell publication interior, 6,291-cell water exclusion, zero unknown interior pixels, and zero colored exterior pixels. A dedicated reviewer layer shows exactly where the blue-nearest-legend rule was applied.
- Candidate `runs/elevation-v34-continuous-blue-fill-v36` has extraction manifest SHA-256 `8cdc4647…`, encoded value raster `00257a94…`, and preview `3eb3e4d2…`. Independent audit `32666bbc…` reports zero differences in both value rasters and every OCR, blue-fill, source-completion, target-completion, publication-interior, boundary-removal, and water mask. All 187 Python tests pass. Port 8806 serves v36 for visual review; it remains unpublished.

- Todd's high-zoom Bay review showed that blue reconstruction alone did not remove every black canonical-edge gap. The remaining non-water holes were exactly the 300 active-boundary expansion cells that v36 still rejected when the warped source had no value. V37 changes that explicit policy: those 300 cells are retained as approved canonical land and inherit the nearest encoded elevation at the target grid. The existing 6,291-cell Census/Mapbox water mask remains excluded, so San Francisco Bay, Monterey Bay, and the Bay–Delta waterways stay transparent.
- `runs/elevation-v34-continuous-nearest-fill-v37` now contains 379,406 publication cells and 361 target completions: the prior 61 plus all 300 unsupported canonical-land cells. It retains zero unknown cells inside the completed non-water interior and zero colored cells outside. Extraction manifest SHA-256 is `e3cbe3fe…`, value raster `8cd47ca2…`, and preview `ccf37b9d…`.
- Independent audit `69eeb87b…` reproduces both value rasters and every OCR, blue-fill, source-completion, target-completion, publication-interior, boundary-removal, and water mask with zero differences. All 188 Python tests pass. Port 8806 serves v37 for visual review; it remains unpublished.

- Todd's 378% Bay-to-central-coast screenshot proved that v37 still contained a continuous black strip inside the lime line. Row-by-row comparison found the active lime-ring fill began exactly on the canonical coast, but the broad water mask subsequently removed 7–27 landward pixels on the affected rows. The source-support gate was not the cause.
- The water audit found two over-broad operations: the plan selected every Census California water polygon except the exact name `Pacific Ocean`, and the named-bay Mapbox snap could still select outer-Pacific water near bay entrances. V39 limits the Census selection to the nine explicitly confirmed bays, straits, and Bay–Delta rivers, then limits all final water removal to the pinned internal-water eligibility polygon. The active lime ring exclusively owns the outer coastal edge.
- The corrected mask restores 951 outer-coast cells that the Mapbox-constrained pass had marked as water. Automated row probes show zero class-to-lime gap across the affected San Francisco and central-coast rows; the only remaining western opening in that band is the intentionally transparent Monterey Bay. The final publication contains 383,708 cells, including 407 nearest-value target completions, with zero unknown cells inside and zero colored cells outside. Confirmed internal water removes 1,989 cells.
- Candidate `runs/elevation-v34-continuous-lime-coast-v39` has extraction manifest SHA-256 `4fb0cbf3…`, encoded value raster `1366f5ef…`, and preview `656f6fa9…`. Independent audit `13728b5c…` reports zero differences in both value rasters and every evidence mask. All 189 Python tests pass. Port 8806 serves v39 for visual review; it remains unpublished.

- Todd's Monterey Bay screenshot identified two remaining artifacts in v39: the separate `Monterey Bay` water snap cut several pixels landward of the approved lime shoreline, and compact printed dark marks survived classification as the special depression value. V40 removes `Monterey Bay` and `San Francisco Bay` from the external named-water snap. Their coastal openings are now clipped exclusively by the active, approved lime ring; the seven confirmed interior Bay–Delta bays, straits, and rivers remain independently transparent.
- A new post-warp cleanup handles only small special-value components that are fully surrounded by publication data. Components of at most 49 target pixels borrow the nearest ordinary legend-ramp value. This replaces 99 components totaling 671 pixels; all 73 larger special-value components remain authoritative, preventing the cleanup from flattening genuine low-elevation or depression regions. The replacement has its own hash-bound mask and reviewer layer.
- `runs/elevation-v34-continuous-monterey-dark-dot-v40` contains 384,413 final publication cells: 246,827 direct observations, 136,859 source completions, 407 target completions, and 671 small-special replacements. It has zero unknown cells inside, zero colored cells outside, removes 1,284 confirmed internal-water cells, and has no class-to-lime gap across the Monterey Bay coast rows. Extraction manifest SHA-256 is `1a370705…`, encoded value raster is `377070e8…`, and preview is `216b70bf…`.
- Independent audit `7b05ebd8…` reports zero differing pixels in both value rasters and every OCR, blue-fill, source-completion, target-completion, small-special replacement, publication-interior, boundary-removal, and water mask. All 190 Python tests pass. Port 8806 serves v40 for final visual confirmation; it remains unpublished.

### 2026-08-27 — Elevation reaches a native continuous Mapbox staging package

- Todd approved v40 after inspecting the Monterey Bay shoreline and small-dark-component cleanup. `review-decision.json` records that approval against extraction manifest SHA-256 `1a370705…`; it does not carry forward any discarded elevation candidate or publish the result.
- Continuous publication now has its own approval-gated exporter rather than pretending a numeric ramp is a categorical mask set. It requires the exact approved extraction and a zero-difference independent audit, copies the exact 16-bit encoded value raster, produces one native-color XYZ surface, and retains all eight numeric legend stops plus the nonnumeric depression swatch. The viewer lets that surface be toggled and opacity-adjusted but intentionally does not replace the continuous ramp with one user color.
- Low-zoom z4–z8 tiles average the native colors only across valid four-by-four subpixel samples and retain fractional boundary coverage. Z9 stores exact nearest-neighbor source pixels and closer zooms remain crisp through nearest overscaling. The canonical display raster remains byte-identical at `9d0451e6…`; the package contains five boundary components, zero unknown land cells, zero exterior color, and the exact 1,284-pixel internal-water exclusion.
- Staging package `publish/staging/elevation-v1` contains 400 tiles and 409 total files. Dataset manifest SHA-256 is `1156b0d2…`, provenance is `373f127a…`, tile set is `a0bad289…`, and aggregate package hash is `947d62c5…`. A complete isolated rebuild is byte-identical for every one of those hashes.
- The Next.js staging viewer preserves the existing rainfall candidate and adds elevation as one continuous layer with the complete legend, source modal, shareable view, toggle, opacity, and canonical-coast inspection. Live Mapbox checks exercised all of those controls at the Bay Area view. All 192 Python tests, 16 viewer tests, TypeScript checking, and the production build pass. No public dataset or catalog was changed; elevation awaits a final staging-map inspection before activation.

### 2026-08-27 — Live staging exposes a Bay Area land-mask topology failure

- The v40 full-resolution reviewer appeared complete because its “unknown inside” metric was measured only against an already-damaged interior. Live Mapbox staging revealed the actual failure: large parts of San Francisco, the Peninsula, Marin, San Rafael, and shoreline land were transparent before tile generation. Direct grid probes confirmed that San Francisco and San Mateo were NoData in the approved value raster itself; Mapbox rendering and the native-color tile sampler were not the cause.
- Root cause was a contract violation in the elevation plan. It filled the detailed lime display outline as a simple mainland polygon. That line is the approved visible boundary, but its open bays and complex shoreline topology do not make it a reliable fill authority. The fill omitted 2,064 cells that the separately pinned California mainland polygon identifies as land. The alignment remains the accepted v34 transform `d0f72005…`; no source warp or class-only shift was applied.
- Candidate `runs/elevation-v34-continuous-pinned-land-v42` instead starts from `california-mainland-clipping-v2`, adds only the bounded row-wise west-coast seam needed to meet the immutable lime line, and subtracts exact-name Census Bay–Delta water. `San Francisco Bay` and `Monterey Bay` are constrained to the pinned z9 Mapbox water reference used by the viewer. The 386,812-cell canonical land-and-island interior loses exactly 3,511 verified water cells and publishes 383,301 elevation cells, including 613 nearest-value completions. It has zero unknown land cells and zero exterior color. Its source-evidence record explicitly names the pinned-polygon-plus-seam publication authority rather than the discarded outline fill.
- The v42 extraction manifest is `615f1c27…`, encoded values are `950f3aaf…`, publication interior is `ab0a93db…`, and water mask is `81f53c31…`. Independent audit `1793043b…` reproduces both value rasters and every evidence mask with zero differences. The test suite now contains 194 passing Python tests.
- Continuous export has a separate `--review-preview` path so an unapproved numeric candidate can be inspected against Mapbox without fabricating or carrying forward approval. `publish/staging/elevation-v3-review` contains 400 tiles and 409 files, declares `needs_visual_review` / `not_approved`, and replaces only the elevation entry on `/mapscan/staging`; the previously approved v40 archive and all public datasets remain untouched. Dataset manifest is `b6caa1b4…` and review provenance is `552d2b3d…`.
- A fresh browser session at `view=-122.38,37.83,9.20` shows elevation over the previously blank Peninsula and Marin/San Rafael land while San Francisco Bay, San Pablo Bay, Suisun Bay, and the Delta chain remain transparent. The map region loads with zero console warnings or errors. All 16 viewer tests, TypeScript checking, and the Next.js production build pass. V42 remains a review candidate pending Todd's visual approval.

### 2026-08-27 — The corrected elevation land mask is approved in isolated staging

- Todd inspected the live v42 Bay Area review at `view=-122.38,37.83,9.20` and approved it with “lgtm.” The new decision is bound directly to extraction manifest `615f1c27…`, active canonical boundary manifest `0bdb6f42…`, and exact display overlay `9d0451e6…`; no v40 approval is carried forward. Decision SHA-256 is `e87a2d06…`.
- Approval-gated package `publish/staging/elevation-v4-approved` contains the same 400 tile PNGs Todd reviewed, one exact 16-bit value raster, the full continuous legend, the corrected publication interior, and its 3,511-cell water exclusion. Its dataset manifest is `801e949f…`, provenance is `adc6b87a…`, and tile set remains `152b7064…`.
- A complete second approved export is byte-identical across all 409 files. The staging viewer now labels the corrected surface **Topography and elevation · approved staging**. The public catalog remains unchanged pending an explicit publication request.

### 2026-08-27 — The approved elevation surface enters the public catalog

- Todd explicitly said “proceed” after approving the corrected Bay Area staging package. Public activation copied the exact 409 reviewed staging assets into fresh directory `publish/datasets/elevation`; no raster, value, mask, tile, or provenance byte was regenerated. `public-activation-decision.json` records that statement and reports the copy as byte-identical.
- The activated dataset manifest remains SHA-256 `801e949f…`, provenance remains `adc6b87a…`, materialization remains `615f1c27…`, approval remains `e87a2d06…`, and the 400-tile set remains `152b7064…`. The activation decision SHA-256 is `dfc07dc1…`.
- The public Next.js catalog now contains eight datasets and adds **Topography and elevation** at `/mapscan?dataset=california-topography-elevation`. Its automated contract preserves 383,301 land pixels, removes 3,511 internal-water pixels, requires zero exterior color and zero interior NoData, and binds the canonical five-component boundary used during review.
- Final verification passes all 194 Python tests, all 19 viewer tests, TypeScript checking, and the Next.js production build. A fresh browser session at the approved Bay Area camera loads the public dataset as entry 08 with no console warnings or errors; the activated package and viewer-served copy compare byte-for-byte across all 410 files, including the activation decision.

### 2026-08-27 — Elevation band controls and population-map intake

- Todd requested selectable elevation layers rather than only one opacity slider. The approved 16-bit value raster is the authority for this presentation change: the exact full-color surface remains the default, while deterministic, initially hidden interval tiles allow any combination of legend-derived elevation bands to be selected without reclassifying rendered RGB pixels.
- New source `examples/population.png` is recorded as SHA-256 `45bb66b2…` in `examples/plans/population.json`. Intake identifies 12 mutually exclusive population-density classes from `< 1` through `> 10,000` people per square mile. The plan explicitly treats pure white as data only inside California, rejects blue water and black/gray contextual ink, prioritizes the state perimeter for alignment, and reserves county lines as a secondary diagnostic.
- Package `publish/staging/elevation-v5-band-controls` contains the unchanged approved full-surface tile set `152b7064…` plus nine mutually exclusive band tile sets derived directly from value raster `950f3aaf…`. Their pixel counts sum to the exact 383,301-cell publication interior. Two complete exports compare byte-for-byte; the new dataset manifest is `5acbc930…` and provenance is `8343abb2…`.
- Todd's layer-selection request activates those 4,018 reviewed package files into fresh public directory `publish/datasets/elevation-bands`; activation decision `140d0492…` binds the existing materialization `615f1c27…` and approval `e87a2d06…`. The viewer starts with the exact full surface, automatically switches to composable band mode when a band is selected, and restores the approved surface when its control is re-enabled.
- Final checks pass all 195 Python tests, all 20 viewer tests, TypeScript checking, and the Next.js production build. A fresh browser session selects and combines the `0–<100 m` and `1,000–<2,000 m` bands, restores the full surface, and records no console warnings or errors.

### 2026-08-28 — Population alignment and extraction become evidence-driven fixed-point loops

- Population alignment begins with `runs/population-auto-v1` and is retained byte-for-byte as `runs/population-perimeter-loop-v1`. The perimeter refiner evaluates 24 distributed anchors, fits from ten, and reserves the remainder as holdouts. It accepts zero corrections because no candidate improves the unused perimeter and independent boundary evidence. A separate county-only fit is also rejected after failing four geographic holdout regions; county geometry remains a diagnostic veto rather than an authority over the coast and state border.
- New generic command `audit-canonical-alignment` compares the unclipped warped source directly with the exact hash-verified active lime overlay. `runs/population-canonical-audit-v4` passes at a 2,200-pixel working height with a two-pixel mainland median residual, 7.62-pixel P90, and 91.57% of mainland evidence within eight pixels. Eight regional octants are reported separately. Generalized islands remain diagnostic-only, and a cropped-source exception is explicit rather than hidden inside the aggregate. The accepted alignment remains SHA-256 `bc32ee77…`; no unvalidated fitted correction replaces it.
- The first categorical extraction correctly reads all 12 legend reds but incorrectly lets contextual blue source water inherit neighboring population classes. That candidate is rejected. A second iteration subtracts the entire modern Census hydrography inventory; visual flipping shows that it removes many tiny present-day water polygons absent from the historical source, so that candidate is also rejected as overreach. A third implementation-check iteration confirms the masking mechanism but retains the same over-removal and is not a candidate.
- Accepted processing candidate `runs/population-extract-v5` separates the two evidence sources. Blue-dominant pixels in the original map are a plan-declared source-context exclusion and can neither seed nor receive a population class. External hydrography is limited to the named major bays, straits, and Sacramento–San Joaquin river features visible in the source. The lossless pre-context-clip aligned source is preserved for comparison, preventing the audit from concealing the pixels it is supposed to judge.
- `audit-categorical-fidelity` runs a 3-by-3 Lab threshold ensemble. Exact palette matching drops 96,104 antialiased pixels, while the relaxed radius adds at most 54,575 uncertain pixels; none of the nine variants changes one nonzero class into another. The conservative reviewed midpoint remains distance 4 / margin 1, and every one of the 12 population classes remains present.
- New generic command `compare-categorical-source` binds that ensemble to the iterative source-diff batch and produces a side-by-side montage, blink GIF, completion mask, and mismatch mask. `runs/population-comparison-loop-v4` preserves all 2,543,220 directly observed class pixels with zero changes, reconstructs 171,844 line/text/context occlusions as a separate derived layer, reports zero nearest-observed-class disagreements, zero interior NoData, zero exterior color, zero colored internal-water pixels, and zero semantic changes across the threshold ensemble. Two consecutive complete passes produce the identical signature `a377e20e…`, so the extraction has reached a fixed point.
- The population candidate remains `needs_visual_review`; confidence gates do not fabricate author approval or publish it. The final comparison montage is `runs/population-comparison-loop-v4/population-density/source-extraction-comparison.png`, the blink is `source-extraction-blink.gif`, and the all-zero mismatch mask is SHA-256 `1742979a…`. The complete Python suite passes with 200 tests.

### 2026-08-28 — Population source/extraction flip review

- The full-resolution reviewer now provides direct **Source** and **Extracted** buttons plus a Space-bar flip shortcut. Switching atomically sets source/classes opacity to 100/0 or 0/100 while leaving zoom, pan, the canonical lime line, county diagnostics, and detailed opacity controls available. This avoids judging a partially blended image by accident.
- Port 8807 serves `runs/population-extract-v5`. Browser verification exercises both states, confirms the corresponding 100/0 slider values, and reports no console warnings or errors. The candidate remains unapproved and unpublished.

### 2026-08-29 — San Francisco cut-off exposes and closes a circular audit

- Todd's source/extraction blink correctly reveals that the San Francisco area is cut off. A direct pre-clip evidence audit finds 4,978 legend-class pixels deleted by the external named/Mapbox water mask, including 4,418 in the Bay Area. The source-blue exclusion deletes none of them. The earlier comparison counted observed evidence only after applying the final publication interior, so pixels removed by that interior were invisible to its “all observed pixels preserved” metric.
- Population plan v7 now explicitly gives direct source pixels matching a legend class precedence over external hydrography. Source-blue pixels remain authoritative water and can never become data; the change restores only the 4,978 conflicting class observations while retaining 51,256 contextual-water cells. The restoration footprint is saved separately as `web-mercator-observed-water-conflict-restored-mask.png`, and the unmasked warped class evidence is preserved as `web-mercator-class-id-before-context-clip.png`.
- The generic categorical comparison now defines observed evidence from that pre-clip raster over the complete land-plus-water candidate domain. It fails if any such pixel is removed by the final context or water mask and paints the omission red in its mismatch view. This prevents the final mask from defining away evidence that disagrees with it.
- Corrected candidate `runs/population-extract-v7` reaches a new two-pass source-diff fixed point at signature `df354d40…`. `runs/population-comparison-loop-v5` passes with 2,548,198 direct observations, zero observations removed, zero changed observations, 171,844 separately marked completions, zero interior NoData, zero exterior color, zero colored water, and zero semantic threshold changes. Its corrected blink is SHA-256 `70c9fed9…`; the candidate remains unapproved and unpublished.

### 2026-08-29 — Native regional comparisons replace statewide-only confidence

- The fixed-point loop now has plan-defined geographic review regions instead of relying on a single statewide montage. `examples/plans/population.json` declares Bay Area, Delta, Monterey Bay, Lake Tahoe, Los Angeles, San Diego, and North Coast WGS84 windows. Each is converted through the candidate raster grid and rendered at native pixel resolution as source, extracted, mismatch, comparison, montage, and blink artifacts.
- Fresh candidate `runs/population-extract-v8`, fidelity audit `runs/population-fidelity-loop-v6`, source-diff batch `runs/population-source-diff-loop-v6`, and comparison `runs/population-comparison-loop-v6` complete the full loop. Globally, all 2,548,198 directly observed class pixels survive unchanged, 171,844 inferred completion pixels remain separately identified, and the failure counts for removed observations, changed observations, nearest-class disagreement, interior NoData, exterior color, colored water, and threshold instability are all zero.
- All seven regional gates pass independently. The Bay Area window evaluates 149,791 observed pixels and 23,554 completion pixels; Delta evaluates 147,161 and 20,645; Monterey Bay 75,996 and 6,937; Lake Tahoe 26,435 and 2,859; Los Angeles 112,680 and 19,655; San Diego 55,054 and 5,800; and North Coast 122,687 and 4,432. Every regional failure count is zero. Visual inspection of the native montages agrees with those metrics, including the previously cut-off San Francisco area.
- The live full-resolution reviewer now includes the same named regions in a **Review region** dropdown. Choosing one preserves the active Source/Extracted state and moves to the native evidence window; the existing direct buttons and Space-bar flip remain available. This makes the repeated visual comparison reproducible rather than dependent on remembering where to pan and zoom.
- The complete Python suite passes with 206 tests. The population candidate remains unapproved and unpublished until explicit author approval.

### 2026-08-30 — The no-human restart completes Farms v2 from its partial high-resolution source

- The autonomous restart deliberately began again from pristine
  `examples/farmsv2.png` rather than the earlier reviewed alignment, extraction,
  published tiles, county reference image, arrows, stamps, or author approvals.
  The 4,250-by-5,500 source is not a statewide California map: its legend and
  adjacent cartography obscure or omit substantial geography. The earlier
  automatic run treated that missing extent as required statewide evidence and
  correctly stopped after ten rejected alignment attempts. That blocker remains
  in the history, but is now explicitly superseded by an automated partial-map
  resumption rather than erased.
- Native-resolution source analysis separates the inset map panel and California
  evidence from the legend, page margins, Pacific background, and neighboring
  state line networks. It preserves source-native county ink, mainland and
  island coast, San Francisco Bay, and thin Suisun/Delta water evidence before
  any reduction. Ambiguous lower-right topology and geography outside the
  source-observable footprint are omitted with a machine-readable warning; the
  pipeline never presents the partial source as statewide coverage.
- Several apparently good registrations were rejected during development.
  Audits exposed scale aliases, full-state pixels leaking into partial-map
  gates, test geography influencing model selection, thin water channels lost
  by uint8 area reduction, adjacent-state topology mistaken for California
  counties, and projective/residual transforms that were insufficiently bound
  to their actual consumer. The final procedure uses source-only visibility,
  native county/water masks, buffered training assignments, deterministic
  target-only partitions, validation-only selection, an immutable frozen
  candidate, and one sealed final acceptance. The shared extraction consumer
  independently verifies the exact projection-aware residual transform over
  the full target and source domains, including convergence and round-trip
  error.
- The accepted automatic transform is California Albers plus a bounded,
  regularized Wendland-C2 residual. Its sealed county evidence has median 0 px,
  p90 2.828 px, and 99.7886% support within 8 px; all seven supported balanced
  county cells pass. Golden Gate is 0/1.414 px median/p90 with full support, and
  East Bay is 1.414/4.472 px with 99.6301% within 8 px. Positive Jacobian,
  no-fold topology, sparse and dense consumer round trips, exact hashes, and
  retry refusal all pass. The previous ten failures remain counted, and the new
  result is official automatic alignment **A11**, with no human input used as
  processing evidence.
- The first source-only legend audit also stopped rather than guessing. The
  generic detector incorrectly forced three ragged columns of 14, 18, and 13
  swatches into a 57-cell rectangle, inventing 12 white rows; a more direct OCR
  pass found all 45 real swatches but contaminated nine labels with group
  headings or the right border. The repaired detector preserves the ragged
  geometry, associates OCR only with a bounded region beside each actual row,
  rejects heading/border furniture, and now reads **45 unique clean labels**.
  No older 45-class plan or reviewed extraction supplies those semantics.
- Processing the 23.375-million-pixel image remained pixel-exact while becoming
  tractable: repeated exact RGB values are classified once and expanded through
  their inverse map, prototype-distance tensors are bounded, and component
  cleanup uses local regions of interest instead of full-frame temporaries.
  These are computational optimizations only; they do not sample away source
  pixels or relax a fidelity gate.
- Extraction took four counted iterations. E1 used Lab distance/margin 2.0/1.0
  and directly observed 62.730187% of its supported source domain. E2 used
  4.0/0.75 and reached 76.196701%. E3 used the final 6.0/0.5 policy and reached
  80.322964%, but remained a retry because no identical predecessor yet proved
  a fixed point. E4 reproduced E3 exactly and was accepted. The supported target
  footprint is **48.707748%**; the missing **51.292252%** remains explicit
  NoData. Inside the supported extraction domain, **80.322964%** is observed and
  **19.677036%** is separately inferred. All 45 classes pass direct/plausible
  prototype support, all seven available geographic cells pass, no plausible
  thematic pixels are excluded, and meaningful source mismatch is **0**.
- The canonical restart registry now reports Farms v2 as **11/11 alignment,
  4/4 extraction, complete**. Its old blocked row is preserved in chronology as
  the evidence that forced the stronger partial-source contract. Generated
  `INDEX.md` and `FAILURE_REPORT.md` now derive the current status from the
  machine-readable log rather than hand-maintained prose.

### 2026-08-30 — The autonomous Farms result enters isolated staging

- A new generic, hash-verified exporter consumes only the accepted automatic
  extraction pointer, its exact A11 alignment, E4 iteration report, 45-entry
  legend, class raster, source, and partial-extent mask. It derives the target
  grid from the accepted Mapbox transform, preserves class zero as transparent
  NoData, and refuses changed hashes, dimensions, class ids, or any classified
  pixel inside the missing source extent. It deliberately imports no lime,
  `county.png`, earlier reviewed materialization, manual approval, arrows, or
  stamps.
- `publish/staging/farms-v2-autonomous-v1` contains 45 independently
  recolorable z4-z9 pyramids, 18,000 PNG tiles, and 18,048 total files. Dataset
  manifest SHA-256 is `e107650f…`, autonomous preview provenance is
  `10d1b305…`, and aggregate package inventory is `19fb66b2…`. The byte-identical
  viewer copy has the same 18,048-file inventory hash. The existing public
  Agricultural land use dataset remains unchanged.
- `/mapscan/staging?dataset=california-agricultural-land-use-autonomous-v1`
  exposes the result as **Agricultural land use · autonomous candidate** with
  all 45 layer toggles, colors, and opacity controls. Browser verification
  confirms Mapbox rendering, a Wheat disable/restore cycle, the original-source
  modal, the explicit automatic/unapproved label, and zero console warnings or
  errors. Eleven adjacent Python exporter tests, all 21 viewer tests, TypeScript
  checking, and the Next.js production build pass. This is a visual-review gate,
  not publication or evidence added back into the processing loop.

### 2026-08-30 — Zoomed Farms review changes the definition of fidelity

- Todd's high-zoom staging review found that the Farms candidate was visibly
  coarser than the 4,250-by-5,500 source and contained light-blue blocks around
  Monterey Bay. The blue was not a Mapbox rendering artifact: it was class 9,
  Cotton (`#bfd1ff`), already present in the extracted class raster. Of 259
  Cotton pixels in the Monterey audit window, 252 were inferred and only seven
  were directly observed; their aligned source color was low-chroma gray map
  ink rather than the legend swatch.
- The generic categorical learner now fails closed on chromaticity as well as
  three-dimensional Lab distance. A chromatic legend class may preserve exact
  and genuinely tinted support, but neutral hillshade, water fill, text, and
  antialiased boundary ink cannot become prototypes or sparse thematic support
  merely because they are close to a pale swatch. A full-source replay reduces
  Cotton in the Monterey window from 259 pixels to six while all 45 class
  support gates continue to pass.
- A batch resolution audit evaluates the accepted transform's local
  target-to-source Jacobian. Farms v2 and the geologic PDF exceed z9 resolution
  and require a corner-preserving 3x processing grid (10,192 by 11,758); all
  other supplied sources fit within the existing z9 grid. The high-resolution
  process samples the pristine source directly and evaluates the accepted
  transform on that denser grid. It never enlarges a previous class raster.
- Every map is now queued for original-source-space regional comparison on a
  6-by-6 grid. Each iteration flips source and extraction at native scale,
  measures the diff in every supported cell, retains the worst regional chips,
  and retries alignment or extraction until the family-specific gates reach a
  repeated fixed point. Statewide overviews remain useful diagnostics but can
  no longer accept a map. The ordered queue and current risk measurements are
  stored in
  `runs/mapbox-autonomous-restart-v1/RESOLUTION_FIDELITY_AUDIT.json`.

### 2026-08-30 — Native reruns expose what overview agreement concealed

- Farms v2 completes a genuine 3x rerun at **10,192 by 11,758** pixels. The
  accepted E4 result is 90.0826% directly observed and 9.9174% separately
  inferred, with zero meaningful source mismatch. All 30 populated native
  6-by-6 regions pass. In the Monterey window, all 25,049 supported pixels are
  classified and none mismatches meaningful source evidence. A direct
  comparison against a nearest-neighbor enlargement of the old z9 class raster
  differs on 1,597,221 pixels, or 35.34% of the classified union; the new detail
  came from the pristine source rather than interpolation.
- The same resolution audit sends California geology through a 3x rerun from
  the pristine PDF and original 7,088-by-9,375 render. Two byte-identical
  passes reach a fixed point with all 53 classes, 99.8358% direct observation,
  0.1642% inferred seams, no water/exterior contamination, and 24 of 24 native
  regional comparisons at 100% semantic agreement.
- The first ten-map native batch passes Fire, Quake, 1981-2010 Rainfall,
  Rivers, Deer, Forest, Landslide, Plant Zones, and Elevation. Population is
  the useful failure: its target-grid round trip was perfect, but original
  source cells reveal 0.7541% global mismatch and 5.2418% in the worst region.
  The 20,895 missed pixels are antialiased or compressed colors that still lie
  within meaningful legend distance; the prior nearest spatial completion had
  replaced their correct band with a neighbor.
- A fresh Population repair preserves the nearest legend class for all
  meaningful color evidence, while keeping weaker occlusions explicitly
  inferred. Its third and fourth extraction passes are byte-identical and all
  24 supported source regions now have zero semantic and round-trip mismatch.
  Thresholds were not relaxed. The old accepted/public artifacts remain intact
  while this repaired candidate and its before/after panels stay isolated in
  `runs/population-native-fidelity-repair-v2`.
- Serving these higher-resolution categorical results creates a product-level
  constraint: separate RGBA pyramids for 45 to 53 selectable classes would
  duplicate hundreds of thousands of mostly empty tiles. The staging work is
  therefore moving to a single indexed class-id pyramid whose Mapbox rendering
  still supports independent class visibility, color, and opacity. Farms needs
  native zoom 11: z10 spans only 7,537 by 8,696 screen pixels over the target
  bounds and would undersample the 10,192-by-11,758 raster; z11 is the first
  non-undersampling zoom.

### 2026-08-30 — Fidelity addendum: completion becomes report-backed

- The restart's original `complete` state and native fidelity are now tracked
  separately. A full 15-row matrix in
  `runs/mapbox-autonomous-restart-v1/INDEX.md` binds each source to its counted
  alignment and extraction iterations, explicit native reports, blockers, and
  authoritative artifact paths.
- The resolution gate is narrower than the first ledger draft stated. A
  separate native alignment replay is required only for accepted optimizer
  scales below 0.5×: Population, Fire, Quake, 1981–2010 Rainfall, and Rivers.
  Fire, Population, 1981–2010 Rainfall, and Rivers pass against their accepted
  transforms. Quake's original global transform is a documented native retry,
  but its active nonlinear supersession now passes. The six other 1× families
  are `not required by resolution gate`, not pending.
- Farms and Geologic close both sides through their source-direct 3× reruns:
  native multiscale alignment evidence plus original-source regional extraction
  diffs. Their earlier z9 rows remain superseded history. Population's active
  `fidelity-supersession.json` now makes the zero-mismatch four-pass repair the
  authoritative replacement, eliminating the earlier canonical-binding gap.
- Quake's A12 Web Mercator similarity and original E2 remain immutable history.
  The active replacement is a projection-aware regularized thin-plate
  source-displacement warp rather than a fictitious A13. It passes all three
  multiscale gates, 20 of 21 supported native cells, silhouette containment,
  nonfolding topology, pinned hashes, and projection round trip. Its fresh
  two-iteration pristine-source extraction preserves all six ordered classes,
  observes 98.2477%, infers 1.7523%, records zero meaningful mismatch, and
  reaches a fixed point. The supersession mutates neither historical nor public
  artifacts, so public delivery stays unchanged.
- Rivers A13 independently passes pristine-source checks at 50%, 75%, and 100%,
  the native 6×6 state and county-observability policy, automatic junctions,
  pinned Mapbox hashes, and its existing native hydrography extraction audit.
- Historical Rainfall remains the principled blocker: classes 4/5/6
  (5.0/5.5/6.5) have identical local color/texture signatures and classes 1/2
  are separated by only 0.1201 Lab-texture units. The exact 623,177-byte source
  is restored at `examples/rainfall.gif` and matches registry SHA-256
  `4d8831424e194d6b462dfc7721c6cce6eb3460d9ee3c3b1f43214beb7b2ca447`;
  source availability is resolved, but semantic completion would still invent
  data.
- `RESOLUTION_FIDELITY_AUDIT.json` defines the queue and thresholds, but only a
  per-map decision report can close a gate. This prevents a planning audit,
  overview comparison, or older author approval from being mistaken for
  current native-resolution evidence.
- Farms staging now uses one indexed class-id pyramid through z11: 5,625 tiles
  instead of 45 duplicated RGBA pyramids. Dataset provenance binds the accepted
  10,192×11,758 E4 raster and the pinned derived Mapbox state/coast diagnostic;
  the retired lime boundary is never used. The source-native 6×6 audit and
  Monterey diff remain the persistent exact evidence. A separate exact 64-tile
  delivery spot check was ad hoc and therefore has no report in the ledger.
- Final verification restored the exact hash-pinned historical-rainfall GIF and
  exposed one generic OCR guard regression: the decimal label `5.5` had been
  mistaken for duplicated prose. The narrowly corrected numeric-label grammar
  now persists all 35 legend rows and then stops at the intended, measured
  color/texture ambiguity rather than accepting invented semantics. The full
  result passes 507 Python tests, 23 viewer tests, the indexed-raster TypeScript
  contract, a production Next.js build, and a clean browser replay of Farms
  staging and its pinned Mapbox coast diagnostic.

### 2026-08-31 — Elevation bands replace the obsolete staging surface

- Todd retired `examples/rainfall.gif` from future processing; the accepted
  `rainfall.png` 1981–2010 dataset is now the precipitation source going
  forward. The previously published historical package was not removed by
  that source-corpus decision.
- The staging route still referenced the older single-surface elevation
  package even though the approved public materialization already contained a
  full continuous surface plus nine independently selectable elevation bands.
  Staging now consumes that same hash-bound `elevation-bands` package rather
  than duplicating or regenerating elevation data.
- Live verification confirms ten controls, automatic removal of the full
  surface when a band is selected, simultaneous selection of multiple bands,
  and zero browser errors. All 23 viewer tests, the TypeScript contract, and a
  clean Next.js production build pass.

### 2026-08-31 — One map becomes a cross-dataset composition

- The viewer previously treated a dataset change as a map replacement. Its
  state is now a persistent `dataset → category → style` composition. Changing
  the focused dataset only changes which controls appear in the right panel;
  selections from Forest, Elevation, Rainfall, or any other loaded dataset stay
  on the same Mapbox canvas. Dataset cards report their selected counts, while
  `Select all`, `Clear dataset`, and `Clear map` operate at explicit scopes.
- Every Mapbox source and layer identity now includes the dataset id and layer
  id, so identical category names from different maps cannot collide. Indexed
  class-id pyramids remain one raster source per recovered source layer, while
  their Mapbox `raster-color` expression independently controls every class.
  Continuous elevation keeps its fidelity rule: selecting any of the nine
  bands disables the full surface, and selecting the full surface disables the
  bands.
- Share links serialize only the active composition as
  `dataset-id~category-id:color:opacity`, plus the focused dataset and camera.
  An old single-dataset link is still readable. A browser reload reproduced a
  four-layer public composition containing Redwood, Western hardwoods,
  1,000–<2,000 m elevation, and >30–50 in precipitation with the same colors,
  opacities, focused dataset, and camera.
- Historical 1900–1960 Rainfall is no longer offered by either active page.
  Its prior public package remains archived on disk. Todd intentionally removed
  `examples/rainfall.gif`; source-specific regression tests now skip when that
  retired fixture is absent, while the generic ambiguity and decimal-label
  safeguards remain covered. `rainfall.png` is the active precipitation map.

### 2026-08-31 — Fidelity-complete results become compact staging layers

- Five accepted autonomous results are now self-contained staging packages;
  none was promoted into the public catalog or given an approval it did not
  have. Each package is explicitly `needs_visual_review` / `not_approved` and
  preserves the source, accepted lineage, hashes, NoData transparency, and
  native-resolution evidence.
- Population uses its active zero-mismatch repair: **A2**, four original
  extraction iterations plus the four-pass active replacement, 12 selectable
  density classes, and a 400-tile z4–z9 indexed pyramid. Every native z9 tile
  reconstructs the accepted raster pixel-for-pixel.
- Fire remains **A1 / E2** and preserves two overlapping source layers as four
  selectable categories. Severe Weather remains **A12 / E2** and preserves
  maximum daily precipitation, landslide susceptibility, maximum wind speed,
  and predicted flooding as four independent layers. Its 94,799 unresolved
  chromatic pixels remain precipitation NoData rather than receiving an
  invented category.
- Geology remains **A1 / E2** but is delivered from the accepted source-direct
  3× result: 10,192×11,758 pixels, 53 PDF-native legend classes, and a z4–z11
  indexed pyramid. Rivers remains **A13 / E4** and exposes six channels for
  observed lines/outlines and separately inferred wet/dry interiors. Its 54
  accepted named-feature records and 89 tokens are retained as evidence, not
  fabricated point labels.
- Staging can now compose one independently selected layer from all eight
  loaded datasets at once: Farms, Population, Fire, Severe Weather, Geology,
  Rivers, Elevation, and 1981–2010 Rainfall. A fresh URL reload preserved all
  eight contributions and the Mapbox camera. The public page independently
  reproduces the requested Forest + Elevation + Rainfall combination.
- Final gates: all **30 viewer tests** pass, TypeScript passes, the Next.js 16.3
  production build succeeds with static `/` and `/staging` routes, both local
  `/mapscan` endpoints return HTTP 200, and the complete processing suite ends
  at **503 passed / 4 skipped**. The four skips are exactly the two
  source-specific historical-Rainfall checks under both fixture-present and
  Tesseract combinations; there are no pipeline failures.

### 2026-08-31 — Delivery diagnostics use the same pinned Mapbox authority

- Population, Fire, and Severe Weather initially had complete native extraction
  evidence but no viewer-level boundary diagnostic. Their staging packages now
  carry the same hash-bound Mapbox state/coast raster already used by Farms,
  Geology, and Rivers. The exact reference SHA-256 is
  `d9d3c21d6f096d3c1439e1c813c83f29edba459dde999638b775d9f10b5462c0`;
  no former lime or `county.png` contour enters these candidates.
- A fresh browser audit checked the diagnostic isolation/restore cycle on all
  five newly packaged candidate families and opened the retained source image
  for every one of the eight staging datasets. Alignment inspection hides the
  thematic composition temporarily and restores its prior styles exactly;
  source dialogs resolve the package-local evidence rather than an external or
  machine-local path.
- The counted no-human iteration ledger remains the machine-readable authority
  in `runs/mapbox-autonomous-restart-v1/INDEX.md`. For the six newly packaged
  categorical/feature candidates it records Farms A11/E4, Population A2/E4
  plus its four-pass active replacement, Fire A1/E2, Severe Weather A12/E2,
  Geology A1/E2 from the 3× source-direct extraction, and Rivers A13/E4.
- `MAPSCAN_REQUIREMENTS.md` advances to v0.3 and now matches the delivered
  viewer: persistent cross-dataset composition, no artificial five-dataset
  cap, deterministic layer order, dataset-qualified share state, and a pinned
  Mapbox reference as the current geographic authority. The old hybrid
  county/lime decisions remain labeled as superseded history instead of being
  silently erased.

### 2026-08-31 — Six autonomous results cross a fail-closed public boundary

- Todd's repeated `Proceed` instruction authorized the next release step, but
  activation did not treat a visually promising preview as sufficient. An
  independent coverage audit reopened every final class-ID raster against the
  pinned Mapbox state interior. The first Fire and Severe Weather packages
  failed that gate: Fire contained 583 classified exterior pixels and Severe
  Weather contained 2,133 across precipitation, susceptibility, wind, and
  flooding. Those v1 packages remain staging history and did not enter the
  catalog.
- One deterministic correction iteration produced
  `fire-autonomous-v2-clipped` and `landslide-autonomous-v2-clipped`. It kept
  every in-state class ID byte-for-byte and changed only exterior classified
  cells to NoData. Automatic source/result diffs, per-layer coverage reports,
  tile inventories, and aggregate tile hashes all pass. Fire now reports zero
  exterior pixels for both layers; Severe Weather reports zero for all four.
- The same generic, plan-driven coverage attestation was then applied to all
  six autonomous families. Farms, Population, Fire, Severe Weather, Geology,
  and Rivers each bind their accepted evidence raster, final publication
  raster, semantic kind, coverage contract, and exact Mapbox interior hash.
  Every layer has zero classified pixels outside California. Sparse layers keep
  their measured interior NoData instead of inventing coverage: Farms retains
  46,725,663 NoData cells; Population 110,572; Fire 3,109,767 and 5,586,476;
  Severe Weather 4,646,166, 4,717,196, 5,083,728, and 5,323,759; Geology
  728,305; and the six Rivers channels retain between 5,064,068 and 5,602,840.
- Public activation is a separate hash-bound operation. It rejects path
  traversal and symlinks, verifies source and boundary identities, requires a
  coverage record for every source layer, and recomputes TileJSON templates,
  file counts, byte counts, tile-set hashes, and cache keys. Failed activation
  leaves no partial public output. The new generic catalog audit repeats these
  checks from the viewer's resolved asset paths.
- Approved packages use small manifest/provenance/decision envelopes with a
  safe root-relative `asset_base` pointing to the already verified immutable
  staging inventory. This avoids duplicating more than 16,000 static asset
  files without weakening provenance: the viewer resolves tiles, source
  images, boundaries, and diagnostics through the same base, and the release
  audit hashes the resolved bytes.
- The public catalog now contains 12 independently selectable datasets. A live
  browser replay composed Geology Qs, Forest Redwood, the 1,000–<2,000 m
  elevation band, 30–50 inches of rainfall, and Fire Very High simultaneously;
  the interface reported five layers from five datasets. All six new public
  manifests and a representative indexed tile returned HTTP 200.
- The no-human iteration ledger remains unchanged for source alignment and
  extraction: Farms A11/E4; Population A2/E4 plus the four-pass active
  replacement; Fire A1/E2; Severe Weather A12/E2; Geology A1/E2; and Rivers
  A13/E4. Publication added one deterministic coverage-attestation pass to each
  family and one additional clip/diff pass only to Fire and Severe Weather.
- Disk pressure was resolved without touching source images, accepted runs,
  active staging packages, or published datasets. Only generated build/test
  caches, one failed partial activation, and explicitly superseded directories
  already named `throwaway` were permanently removed. The accepted v6 Geology,
  v3 Farms, and v4 Fire/Severe outputs remain intact.
- `MAPSCAN_REQUIREMENTS.md` advances to v0.4. The release contract now records
  exact indexed coverage attestation and shared immutable assets as confirmed
  requirements rather than implementation accidents.
- Final gates are green: **526 Python tests passed / 4 intentional skips**,
  **32 viewer tests passed**, TypeScript emitted no errors, and the Next.js
  16.3.2 production build statically prerendered both `/` and `/staging`.

### 2026-08-31 — `rivers.jpg` is retired from the map product

- Todd determined that the Rivers result does not work as a MapScan dataset.
  It is removed from the public catalog, the public picker, and the staging
  picker. A saved URL naming its former public id therefore falls back to an
  available dataset instead of loading the retired result.
- The source image, extraction run, immutable staging evidence, approved
  metadata envelope, hashes, and A13/E4 iteration record are retained. This is
  a product rejection, not an attempt to erase an informative pipeline failure
  or misrepresent the autonomous corpus success rate.
- The active catalog contains 11 datasets. `MAPSCAN_REQUIREMENTS.md` advances
  to v0.5 and explicitly classifies `rivers.jpg` as retained rejected-case
  evidence rather than a viewer layer.
- The retirement regression proves Rivers is absent from both page sources and
  the catalog while its evidence manifest still exists. All **33 viewer tests**
  pass, TypeScript is clean, and the production build succeeds for `/` and
  `/staging`.

### 2026-08-31 — The live catalog is reconciled to newest accepted authority

- A complete public-lineage audit compared all 11 live dataset manifests with
  the accepted-extraction authorities in the autonomous completion ledger.
  Farms 3×, Population's active repair, Fire, Severe Weather, and Geology 3×
  were already current. Deer, Forest, Plant Hardiness, 1981–2010 Rainfall,
  Elevation, and Quake still referenced earlier reviewed publications and were
  replaced with fresh versioned packages; no historical package was deleted.
- The promoted iteration authority is Deer **A15/E6**, Forest **A11/E4**,
  Plant Hardiness **A21/E4**, Rainfall **A12/E8**, Elevation **A12/E4**, and
  Quake **A12 plus the accepted nonlinear refinement / E2 pristine-source
  replacement**. Quake now publishes the replacement VI→XI ordering rather
  than the reversed class numbering in the old hybrid package. Its accepted
  source reconstruction observes 98.2477%, infers 1.7523%, and reports zero
  meaningful mismatch.
- Elevation's newest accepted continuous extraction remains the numeric
  authority, while its public representation is the accepted depression mask
  plus seven mutually exclusive numeric intervals. Those eight classes are
  indexed and independently toggleable, which preserves cross-dataset
  composition instead of restoring the older single-surface slider.
- A fail-closed state-mask audit identified 3,988 accepted edge pixels in each
  of Deer, Plant Hardiness, and Rainfall, and 37 in Forest, outside the current
  pinned Mapbox state interior. A deterministic publication-only clip made
  those pixels transparent. The accepted rasters and runs remain unchanged,
  provenance records both hashes and exact clip counts, and all six new
  packages now report zero classified pixels outside California.
- `config/live-dataset-authority-v2.json` is now the live release registry. A
  viewer regression hashes every declared accepted-extraction file, verifies
  each public provenance record, and requires the page imports, public paths,
  and catalog manifests to resolve to exactly that 11-dataset set. Rivers stays
  excluded by author request; historical Rainfall stays excluded because its
  legend semantics remain ambiguous.
- Final verification passed **529 Python tests / 4 intentional skips** and
  **34 viewer tests**, plus TypeScript and the Next.js 16.3.2 production build.
  A live browser replay confirmed all 11 dataset controls, Quake's VI→XI
  ordering, Elevation's eight independent categories, mixed Quake + Elevation
  composition, a live Mapbox canvas, and no browser warnings or errors.

### 2026-08-31 — Forest autonomous-v2 alignment regression retrospective

- Todd's visual review correctly identified that the newly activated Forest
  result does not register tightly enough to the Mapbox state boundary. No
  Forest artifact was changed during this retrospective; the purpose of this
  entry is to preserve the failure rather than silently replace it.
- The accepted A11 transform is a Web Mercator similarity fit with state-line
  median/P90 residual **2.0/8.06 working pixels**, symmetric overlap F1
  **0.7403**, and land containment **0.9730**. The same run had already produced
  materially stronger regular-affine candidates: A4 California Albers scored
  **0.0/3.0**, **0.9451**, and **0.9876** respectively; A8 CONUS Albers scored
  **0.0/3.0**, **0.9457**, and **0.9878**. Transporting identical source-border
  positions through A11 versus A4/A8 moves them by roughly **46–47 target
  pixels median** and **57–58 pixels P90**, about **16 km / 19–20 km** at this
  grid.
- Root cause: A1–A10 were evaluated before the source-capability update and
  failed county gates even though `forest.jpg` has no county network. The retry
  then re-evaluated A1 as A11, marked county evidence not applicable, and
  immediately accepted the first passing candidate. The loop returns on the
  first pass and does not re-rank the ensemble after applicability changes, so
  the better A4/A8 state fits were never reconsidered under the corrected gate
  set.
- Extraction E4's zero semantic mismatch does not validate registration. It
  verifies that classified pixels reconstruct the same already-warped source.
  The later native regional diff also uses the accepted transform in both
  directions, so it is a fidelity/round-trip check rather than an independent
  Mapbox alignment check.
- The site's **Inspect coast alignment** mode renders the canonical Mapbox
  boundary alone. Because it does not also render a boundary independently
  recovered from the transformed source, it necessarily looks aligned and
  could not expose this regression.
- Activation also over-interpreted “Check that each map uses the newer version
  and fix it” as visual approval, even though the staging provenance still said
  `needs_visual_review`. Version recency is not quality supersession and must
  not replace a previously approved result without a non-regression comparison.
- Required remediation: determine source capabilities before evaluating the
  ensemble; rank every passing candidate globally; compare the winning
  source-derived perimeter against Mapbox in geographic regions and at native
  resolution; make the inspection mode show both lines; rerun extraction only
  after alignment passes; and activate a new immutable Forest version only
  after it beats both A11 and the formerly approved Forest alignment.

### 2026-08-31 — Global candidate ranking repairs Forest and Plant Hardiness

- The alignment loop now evaluates its complete bounded hypothesis/projection/
  transform ensemble before accepting anything. Every candidate is serialized
  in `candidate-ranking.json`, and the deterministic winner is the lowest
  semantic state/coast score among candidates that pass the capability-aware
  gates. A live non-regression audit independently reopens each extraction,
  its bound alignment, and every comparable candidate before publication.
- Forest required **30 alignment iterations / 4 extraction iterations**. The
  globally selected CONUS Albers regular-affine fit improved the former live
  semantic score from **12.9294 to 2.4918**, state P90 from **8.06 to 2.0**
  pixels, state F1 from **0.7403 to 0.9576**, and land containment from
  **0.9730 to 0.9932**. Its source reconstruction reached a fixed point with
  zero meaningful mismatch, and its publication package contains zero colored
  pixels outside the pinned Mapbox state interior.
- Plant Hardiness required **40 alignment iterations**. Its selected CONUS
  Albers regular-affine fit improved the former live semantic score from
  **12.2974 to 2.5532**, state P90 from **8.54 to 2.0** pixels, state F1 from
  **0.7634 to 0.9510**, and land containment from **0.9613 to 0.9937**.
- Plant's first four-pass extraction was correctly blocked because fixed global
  support floors rejected zones `5a` and `11a`, although each had six direct
  source pixels distributed across three independent clusters. The extraction
  gate now distinguishes robust classes from rare spatially corroborated
  classes. A second four-pass immutable run accepted all 13 legend classes,
  with **8 total logged extraction iterations**, 78.61% directly observed
  source coverage, 21.39% inferred label/line completion, and zero meaningful
  source mismatch.
- Forest is activated as `forest-cover-autonomous-v3`; Plant Hardiness is
  activated as `plant-hardiness-autonomous-v3`. Both retain their source files,
  accepted transforms, candidate rankings, extraction evidence, exact coverage
  attestations, and autonomous non-regression decisions. Prior v2 packages are
  preserved as history.
- The corpus audit now reports **11 datasets, 0 failures**. Every live
  extraction hash matches the registry, every alignment hash matches its
  extraction, and every standard alignment is the best passing candidate in
  its available ensemble. A fresh browser load verified both corrected live
  maps over Mapbox with all categories selectable and no console warnings or
  errors.
- Permanent repository instructions now forbid first-pass acceptance,
  reference-only alignment diagnostics, recency-as-approval, and fixed
  thresholds that discard spatially corroborated rare classes. Final checks:
  **533 Python tests passed / 4 intentional skips**, **34 viewer tests passed**,
  TypeScript passed, and the Next.js 16.3.2 production build succeeded.

### 2026-08-31 — Forest coastline setback is diagnosed as extraction loss, not a new warp

- Todd observed that the original Forest map carries thematic color closer to
  the coast than the v3 Mapbox overlay. The existing statewide alignment still
  passes and remains unchanged. A projection-aware local-warp experiment was
  rejected: it could reduce selected perimeter residuals, but it also displaced
  already accepted thematic pixels and reduced source-supported coastal
  coverage. This separated the visible defect from registration.
- The actual loss occurs after registration. `forest.jpg` uses a neutral dark
  cartographic coastline stroke over the thematic surface. Classification
  correctly rejects that stroke as a non-legend color, but formerly left it as
  transparent NoData. When projected to the 3,398 x 3,920 target grid, a one-
  or two-pixel source stroke becomes a conspicuous inward setback from the
  Mapbox coast.
- Extraction iteration **E5** adds a generic source-space coastal-occlusion
  pass. It considers only neutral/dark pixels on a source-derived coastline,
  requires the independently projected pinned Mapbox mask to classify the
  pixel as land, and copies only the nearest existing source legend class from
  at most 3.5 source pixels away. The repair is stored in dedicated source and
  target masks and remains inferred evidence; the accepted **A30** transform is
  byte-identical to v3.
- The Forest pass recovered **964 source pixels**, just **0.702%** of its
  137,338 existing classified source pixels. It added **14,598 target pixels**
  while removing **0** and reclassifying **0**. Coverage within the declared
  eight-pixel inland coast audit band improved from **3.52% to 10.21%**. A
  final pinned-Mapbox clip leaves **0** classified pixels over water or outside
  California, and a repeat run produced identical source repair bytes.
- The immutable candidate is staged as `forest-autonomous-v4`; v3 remains the
  public map pending visual approval. Repository policy and the product
  requirements now require this coastline-occlusion audit so a passing global
  alignment score cannot hide the same extraction setback in another map.
- Final verification passed **538 Python tests / 4 intentional skips** and
  **34 viewer tests**. TypeScript and the Next.js 16.3.2 production build also
  pass. A live browser comparison at native zoom confirms that v4 restores the
  thin coastal legend evidence while retaining the same accepted alignment.

### 2026-08-31 — Forest repair expands from coastline to the complete state perimeter

- Todd's Tahoe review exposed the same artificial setback along the eastern
  California-Nevada line. A source/native comparison confirms that the accepted
  transform places the black state stroke correctly, while classification
  rejects the stroke and leaves its landward width transparent. This is another
  extraction occlusion, not evidence for changing the A30 transform.
- The v4 coast-only rule was therefore too narrow and remains immutable staging
  history. The replacement algorithm requires a neutral/dark candidate to be
  supported simultaneously by the source-derived state perimeter and the
  pinned Mapbox perimeter independently projected into source space. This dual
  support excludes internal black linework. The candidate must also lie inside
  Mapbox land and within 3.5 source pixels of an existing legend class.
- Forest extraction **E5** under the generalized rule recovers **2,132 source
  pixels** (**1.552%** of existing source evidence) and adds **33,200 target
  pixels**. It removes **0** target pixels, reclassifies **0**, and leaves **0**
  over water or outside California. Every addition is within **28.39 target
  pixels** of the Mapbox perimeter, below the declared 32-pixel veto.
- Eight-pixel perimeter-band coverage improves from **3.45% to 14.67%**.
  Coast coverage improves from **3.52% to 12.46%**, and inland-border coverage
  from **3.27% to 21.80%**. Browser inspection at Tahoe shows the thematic
  raster now meeting the California-Nevada line while the sparse Colorado River
  region remains unpainted where the source has no forest class.
- The replacement remains unapproved and is staged as `forest-autonomous-v5`.
  The approved public v3 bytes and the superseded unapproved v4 candidate are
  unchanged.
- Final verification passed **537 Python tests / 4 intentional skips** and
  **34 viewer tests**, plus TypeScript and the Next.js 16.3.2 production build.

### 2026-08-31 — Forest perimeter repair receives explicit public approval

- Todd reviewed the exact `forest-autonomous-v5` staging package at the Tahoe
  eastern-border view and approved it with “lgtm.” No extraction, resampling,
  or tile regeneration occurs during activation.
- The immutable staging dataset and provenance hashes are
  `69823884b7aae0228de8f36999f0500875a22fca02acd347f8f373df4ed382e9`
  and `3e75649d71e07c5efe8254c4aa0177384ae56b7ce4969dce78b027b4e9835334`.
  The public package shares all 404 staging assets byte-identically and records
  activation mode `author_public_activation`.
- The live viewer, catalog, and dataset-authority registry now bind
  `forest-cover-autonomous-v5` to the accepted E5 pointer
  `runs/perimeter-evidence-repair-v1/forest/extraction-run-03/accepted-extraction.json`.
  The prior v3 public package and v4 staging candidate remain immutable history.
- Post-activation verification passed **537 Python tests / 4 intentional
  skips**, **34 viewer tests**, TypeScript, and the Next.js 16.3.2 production
  build. A fresh browser load of the live Tahoe view served the v5 manifest and
  tiles, exposed all eight selectable legend classes as Approved, and showed
  the recovered thematic evidence meeting the eastern California boundary.

### 2026-08-31 — Corpus-wide readiness loop closes with no additional map inflation

- All **11** live datasets were rebound to their exact accepted source,
  alignment, extraction, public provenance, and catalog records. The complete
  alignment audit passes with **0 failures**; its report hash is
  `3470b8abae2f4eb00f30a48cee4c05ba28da13e1a178e90ab7b626a971d34f7b`.
- The Forest perimeter hypothesis was replayed against the other standard
  categorical maps. Plant hardiness, deer, population, and rainfall were all
  vetoed because a replay would remove accepted pixels or produced no coast
  and inland-border gain. The native-resolution Farms probe likewise produced
  no qualifying perimeter gain. These are diagnostic attempts, not extraction
  iterations, and no accepted or public map bytes were changed.
- Farms exposed a diagnostic scaling defect: its 11,758 x 10,192 extraction
  was initially compared through the 3,920 x 3,398 base alignment grid. The
  perimeter audit now applies the same corner-preserving 3x transform
  conjugation as the extractor and rejects any processing reference whose
  dimensions or bounds do not match the declared supersampled grid.
- A new corpus readiness gate verifies source/alignment hash resolution,
  schema-specific extraction gates, deterministic or fixed-point evidence,
  live-alignment non-regression, public provenance binding, explicit approval,
  catalog membership, and zero colored pixels outside California. All **11 of
  11** datasets pass; the report hash is
  `794dc63ca82ddd9908b55e92730355ab3fc8b341894a38f0e376df34480b2303`.
- Fresh browser inspection covered every dataset family at statewide scale and
  Farms at native Monterey Bay zoom. The viewer exposed every approved legend
  class and rendered without visible exterior leakage. A composed-map check
  retained **five selected layers from three independent datasets**: two Forest
  classes, two elevation intervals, and one rainfall interval.
- Final verification passed **540 Python tests / 4 intentional skips** and
  **34 viewer tests**, plus TypeScript and the Next.js 16.3.2 production build.

### 2026-08-31 — Composition controls distinguish datasets from layers

- Todd clarified the navigation hierarchy after an initial symbolic-layer
  interpretation: **datasets receive symbols; individual legend layers retain
  their text labels**. The catalog is now a compact grid of dataset-specific
  line icons paired with concise names such as Forest, Farms, Deer, Rainfall,
  and Elevation, plus full accessible names and selected-layer counts. The focused
  dataset repeats its symbol beside its title, while the layer editor preserves
  the source legend wording.
- The catalog now combines dataset focus, selection, and z-order into one wide
  row per dataset. A tri-state checkbox selects or clears the dataset's legend
  layers in one action, while compact up/down controls move the entire dataset
  from top to base. This replaces the separate compact picker and draggable
  dataset stack, removing a duplicated representation of the same objects.
- Opacity remains independently controllable and shareable at the dataset and
  individual-layer levels. The collection-wide opacity multiplier was removed
  because it duplicated the more useful dataset scopes. Old `opacity` query
  parameters are ignored and stripped when a new composition link is copied.
- Dataset ordering still updates the real Mapbox layer order. All selected
  legend classes within Forest, Rainfall, Elevation, or another dataset move as
  one contiguous block while retaining their natural internal order. Elevation's
  bulk checkbox enables its discrete bands and leaves the mutually exclusive
  continuous surface off, matching the existing layer-selection behavior.
- Up/down actions use a short FLIP transition: each row's pre-update position is
  measured, the dataset order is committed, and the two affected rows animate
  from their former positions into the new layout. This makes the z-order change
  legible without adding persistent decoration, and the animation is skipped
  when the browser requests reduced motion.
- Indexed categorical sources still render every legend class as an independent
  Mapbox raster layer over one shared tile source, preserving layer-specific
  toggles and colors without exposing an unnecessarily granular z-order.
- Every editable layer retains its color input and now has a reset action tied
  to the immutable `display_rgb` value in its dataset manifest. Dataset
  symbols identify provenance in the stack without replacing layer names.
- Styling follows the established Todd.sh restraint: paper and ink tones,
  hairline borders, compact controls, serif display type, and a single orange
  signal color rather than a new decorative theme.
- Verification passed **40 viewer tests**, TypeScript, and the Next.js 16.3.2
  production build. Browser verification exercised an all-layer dataset toggle
  and moved Farms as a complete Mapbox block, then restored the starting state;
  the browser measured equal and opposite live transforms on the Farms and
  Population rows during the transition. The composition counts and top-to-base
  list updated correctly, with no browser warnings or errors.

### 2026-08-31 — California-wide zoom floor

- The map previously allowed zoom 4, where California occupied only a small
  portion of the canvas and sparse classes became easy to mistake for missing
  data. The viewer now uses zoom 5 as its California-wide floor, while raising
  that floor automatically if any future dataset declares a higher minimum tile
  zoom. Shared camera URLs below the supported floor are rejected and fall back
  to the normal California fit. Every current dataset contains zoom-4 tiles, so
  the new floor stays above the declared raster cutoff.
- The first verification checked a fresh page opened directly at zoom 5. That
  was insufficient: Todd's follow-up exposed that the raster could still vanish
  when *arriving* at zoom 5 through repeated zoom-out gestures. Reproduction
  found a deterministic drop between approximately zoom 6.4 and 5.4. The same
  center and zoom rendered on a fresh load, remained blank after waiting, and
  did not recover after an opacity/paint update. This isolated stale Mapbox
  raster tile state during downward tile-pyramid replacement rather than absent
  files, bounds, layer visibility, or paint state.
- A low-zoom `zoomend` recovery now reloads only raster sources that currently
  have a visible MapScan layer; inactive datasets are not fetched. The exact
  five-click reproduction retains all 12 Population layers at the floor. A
  second round trip retains **22 layers across Population and Rainfall**, with
  the zoom-out control disabled at the floor and no browser warnings or errors.
- Final verification passed **41 viewer tests**, TypeScript, and the Next.js
  16.3.2 production build.

### 2026-08-31 — MapScan becomes a project story with a separate map route

- `/mapscan` now explains the original problem in Todd's voice: useful
  California data was trapped in flattened map images, and manual Photoshop
  composites required projection-aware warping, skewing, and rotation while
  still producing an image that could not be queried or restyled.
- The article shows the nine source maps in the current autonomous staging set:
  Forest, Plant Hardiness, Farms, Population, Fire, Severe Weather, Geology,
  Elevation, and Rainfall. Lightweight editorial previews avoid loading the
  53 MB geology source in the thumbnail grid; each preview opens in an
  accessible lightbox with a link to the full-resolution evidence.
- The process narrative documents the two gated iteration loops. Geometry is
  aligned first against Mapbox state, coast, and county reference geometry;
  legend-driven extraction begins only after that alignment passes. Source and
  extraction comparisons then drive classification repairs before indexed
  Web Mercator tiles are published.
- The existing full-screen composition tool moved unchanged to
  `/mapscan/map`; `/mapscan/staging` remains available for candidate review.
  The story links to the map at both the introduction and result sections and
  leaves an explicit placeholder for Todd's forthcoming layer-combination
  insights.
- The article follows the restrained Todd.sh system: warm paper, ink, Georgia
  display type, Arial utility text, hairline dividers, and one muted red signal
  color. It adds no new application dependencies.
- The nine source previews were tightened into a three-by-three desktop tile
  grid and a two-column mobile grid so the gallery no longer requires a
  horizontal scroll. The source lightbox now contains the entire selected map
  inside the viewport without its own scroll area; the separate full-resolution
  link remains available when pixel-level inspection is wanted.

### 2026-08-31 — Complete, versioned composition URLs for article examples

- MapScan share links now use `config=2` and restore the focused dataset,
  visible layers, layer colors, layer opacity, dataset opacity, dataset z-order,
  and the full camera state including bearing and pitch. Clicking **Copy link**
  updates the address bar as well as the clipboard so a configuration is
  visible and immediately reusable.
- The layer encoding now includes an enabled flag and retains style changes on
  hidden layers. Default hidden styles and 100% dataset opacities are omitted
  to keep links compact; dataset order remains explicit and deterministic.
- Parsers remain backward-compatible with the original single-dataset links
  and the first multi-dataset format. Unknown datasets, categories, and invalid
  numeric values continue to fail closed rather than corrupting the map.
- `MAPSCAN_SHARE_URLS.md` documents the parameter contract and the five-step
  workflow Todd can use to prepare the three linked article examples.

### 2026-08-31 — Production publication architecture

- MapScan remains a separately deployable Next.js application with
  `/mapscan` as its base path. Todd.sh proxies `/mapscan` and every descendant
  path to the MapScan production project, so the article, full-screen map,
  JavaScript chunks, source images, and raster tiles all retain one public
  `todd.sh` URL hierarchy.
- The deployment input excludes superseded and diagnostic dataset packages.
  It publishes only the accepted manifests, current source images, editorial
  thumbnails, and the immutable tile sets referenced by the live catalog.
  The processing archive remains local.
- Because the accepted high-resolution corpus contains more than 15,000 small
  tile files, Vercel deployment uses a compressed source archive. Hosted
  verification checks the article and map HTML, a source-map thumbnail, a
  high-resolution source image, and a representative native raster tile before
  the Todd.sh proxy is promoted.
- The Todd.sh homepage includes MapScan as an AI data experiment while the
  article leaves its three insight examples open for Todd's live iteration.
- Pre-publication verification passed **46 viewer tests**, TypeScript, both
  Next.js 16.3.2 production builds, and the hosted asset checks above.

### 2026-08-31 — Compact, live-synced deep links and public repository

- MapScan composition links now use `config=3` with a compressed, versioned
  state payload. The browser address updates automatically as the composition
  changes, so the current URL is always the reproducible artifact rather than
  something that exists only after pressing **Copy link**.
- The encoded state includes the focused dataset, every layer's enabled state,
  custom color and opacity, dataset opacity, complete dataset z-order, and the
  map camera's longitude, latitude, zoom, bearing, and pitch. Presentation-only
  state such as the copy confirmation, source-image lightbox, and diagnostic
  alignment overlay is intentionally excluded because it does not change the
  resulting map composition.
- The decoder validates dataset and category identifiers, colors, and numeric
  ranges, while retaining backward compatibility with the first two link
  formats. Invalid or obsolete entries are ignored without damaging the rest
  of the composition.
- A full browser round trip restored a customized Forest composition with a
  non-default Redwood color, 37% layer opacity, 64% dataset opacity, reordered
  dataset stack, and a changed camera. The compressed URL was 975 characters;
  an equivalent worst-case uncompressed version-two composition was about
  10,396 characters.
- Release verification passed **48 viewer tests**, TypeScript, the Next.js
  16.3.2 production build, and **540 pipeline tests** with four optional tests
  skipped.
- The complete application, processing pipeline, accepted map assets, tests,
  and decision log were prepared as the initial public `toddsherman/mapscan`
  repository. Local credentials, caches, generated runs, duplicate source
  packages, and superseded staging outputs remain excluded.

### 2026-09-01 — Story animation for the two comparison loops

- The story's process section now includes a restrained, high-level schematic
  of the two loops used throughout MapScan. It deliberately illustrates the
  method rather than presenting any one dataset's measured iteration history.
- The geometry panel keeps the Mapbox California perimeter fixed while a
  source-derived perimeter moves through offset, rotation, skew, and scale
  candidates until the alignment gate passes.
- The extraction panel begins with already-aligned geography, then compares an
  illustrative classified map to its source. Missing blocks, cartographic ink,
  and outside-boundary spill are exposed and repaired before the extraction
  gate passes. This preserves the project's required separation between
  alignment and extraction evidence.
- Motion uses the existing warm paper, ink, muted red, Georgia, Arial, and
  hairline visual system. The looping figure includes a pause control and a
  complete reduced-motion end state; it adds no runtime dependency and changes
  no accepted dataset, alignment, tile, or catalog pointer.
- Site verification passed **49 viewer tests**, TypeScript, the Next.js 16.3.2
  production build, a zero-horizontal-overflow browser check, pause/resume
  state inspection, and a clean browser console.
