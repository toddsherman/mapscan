"""Command-line entrypoint for MapScan processing experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .alignment import PROJECTIONS, align_image
from .auto_refinement import auto_refine_perimeter, auto_refine_perimeter_batch
from .boundary_clip import clip_materialization_to_boundary
from .boundary_components import audit_source_supported_boundary_components
from .canonical_boundary import (
    ACTIVE_CANONICAL_POINTER,
    activate_canonical_border,
    build_county_detail_boundary_candidate,
    promote_canonical_boundary,
    promote_county_detail_border,
    rasterize_canonical_boundary_for_run,
)
from .canonical_alignment_audit import audit_canonical_alignment
from .assist import serve_assist
from .classification_review import serve_classification_review
from .categorical_fidelity import audit_categorical_fidelity
from .categorical_comparison import build_categorical_comparison
from .completion_fidelity_review import serve_completion_fidelity_review
from .coastal_occlusion_repair import run_perimeter_occlusion_repair
from .continuous_extraction import audit_continuous_source_diff, extract_continuous_plan
from .continuous_tile_export import export_continuous_tiles
from .directional_coast_refinement import refine_directional_west_coast
from .east_anchored_refinement import fit_east_anchored_horizontal_scale
from .correction_migration import (
    audit_alignment_application,
    audit_stamp_migration,
    compare_materialized_candidates,
    migrate_stamp_corrections,
)
from .county_reference import register_county_reference
from .county_fine_alignment import fine_align_to_county_reference
from .fine_alignment_review import serve_fine_alignment_review
from .fine_alignment_determinism import audit_fine_alignment_determinism
from .hybrid_perimeter import audit_hybrid_perimeter
from .correction_materialization import materialize_review_corrections
from .enclosed_fill import generate_enclosed_fill_artifact
from .equivalent_source_alignment import lift_alignment_to_equivalent_source
from .pdf_registration import align_pdf_graticule
from .pdf_legend import extract_pdf_legend
from .pdf_vector_extraction import audit_pdf_vector_diff, extract_pdf_vector_fills
from .indexed_coverage_attestation import attest_indexed_staging_coverage
from .indexed_publication_activation import activate_indexed_staging_dataset
from .publication_activation import activate_staging_dataset
from .reference import fetch_reference_data
from .water_reference import fetch_california_areawater
from .mapbox_water_reference import fetch_mapbox_water_reference
from .refinement import fit_local_review_corrections, fit_review_corrections
from .extraction import extract_from_plan
from .extraction_preview_export import export_extraction_preview_tiles
from .feature_extraction import extract_observed_feature_ink
from .inference import infer_categorical_run
from .label_detection import detect_run_labels
from .lower_colorado_refinement import refine_lower_colorado
from .migration_review import serve_migration_review
from .neighbor_completion import complete_neighbor_unknowns, promote_neighbor_completion
from .review import serve_review
from .rainfall_topology import extract_rainfall_topology
from .reviewed_extraction_promotion import promote_reviewed_extraction
from .river_semantics import extract_river_semantics
from .source_diff import (
    audit_extraction_source_diff,
    audit_feature_source_diff,
    audit_source_diff_batch,
)
from .source_diff_materialization import promote_source_diff_materialization
from .solid_mask_alignment import refine_solid_mask_alignment
from .solid_coast_refinement import refine_solid_west_coast
from .southern_edge_refinement import refine_southern_edge
from .tile_export import export_categorical_tiles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mapscan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-reference", help="Fetch pinned Census boundaries")
    fetch.add_argument("--output", type=Path, default=Path("reference/census-2025"))
    fetch.add_argument("--force", action="store_true")

    fetch_water = subparsers.add_parser(
        "fetch-water-reference",
        help="Fetch pinned California Census Area Hydrography polygons",
    )
    fetch_water.add_argument(
        "--boundary-reference",
        type=Path,
        default=Path("reference/census-2025"),
    )
    fetch_water.add_argument(
        "--output",
        type=Path,
        default=Path("reference/census-2025-areawater"),
    )
    fetch_water.add_argument("--force", action="store_true")

    fetch_mapbox_water = subparsers.add_parser(
        "fetch-mapbox-water-reference",
        help="Pin Mapbox Streets water tiles for basemap-exact land/water clipping",
    )
    fetch_mapbox_water.add_argument("--output", type=Path, required=True)
    fetch_mapbox_water.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
    )
    fetch_mapbox_water.add_argument("--zoom", type=int, default=9)
    fetch_mapbox_water.add_argument(
        "--token-file", type=Path, default=Path("viewer/.env.local")
    )
    fetch_mapbox_water.add_argument(
        "--token-variable", default="NEXT_PUBLIC_MAPBOX_TOKEN"
    )
    fetch_mapbox_water.add_argument("--force", action="store_true")

    county_reference = subparsers.add_parser(
        "register-county-reference",
        help="Register a styled high-resolution county-line reference raster",
    )
    county_reference.add_argument("image", type=Path)
    county_reference.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    county_reference.add_argument("--output", type=Path, required=True)
    county_reference.add_argument("--maximum-dimension", type=int, default=1800)
    county_reference.add_argument("--web-height", type=int, default=3600)

    canonical_boundary = subparsers.add_parser(
        "promote-canonical-boundary",
        help="Promote an author-approved mainland border into the canonical reference",
    )
    canonical_boundary.add_argument("approved_materialization", type=Path)
    canonical_boundary.add_argument("approved_boundary_geojson", type=Path)
    canonical_boundary.add_argument("--output", type=Path, required=True)

    canonical_boundary_raster = subparsers.add_parser(
        "rasterize-canonical-boundary",
        help="Rasterize the approved canonical mainland on an extraction grid",
    )
    canonical_boundary_raster.add_argument("canonical_manifest", type=Path)
    canonical_boundary_raster.add_argument("run", type=Path)
    canonical_boundary_raster.add_argument("--output", type=Path, required=True)

    county_detail_boundary = subparsers.add_parser(
        "build-county-detail-boundary",
        help="Build a review boundary from county.png coast and island linework",
    )
    county_detail_boundary.add_argument("canonical_manifest", type=Path)
    county_detail_boundary.add_argument("hybrid_perimeter", type=Path)
    county_detail_boundary.add_argument("county_reference", type=Path)
    county_detail_boundary.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    county_detail_boundary.add_argument("--output", type=Path, required=True)

    promote_county_detail = subparsers.add_parser(
        "promote-county-detail-border",
        help="Promote the reviewed county-detail linework as canonical",
    )
    promote_county_detail.add_argument("candidate_manifest", type=Path)
    promote_county_detail.add_argument("--supersedes", type=Path, required=True)
    promote_county_detail.add_argument("--author-statement", required=True)
    promote_county_detail.add_argument("--output", type=Path, required=True)

    activate_border = subparsers.add_parser(
        "activate-canonical-border",
        help="Point all future map-border consumers at an approved canonical package",
    )
    activate_border.add_argument("canonical_manifest", type=Path)
    activate_border.add_argument(
        "--pointer", type=Path, default=ACTIVE_CANONICAL_POINTER
    )

    align = subparsers.add_parser("align", help="Run automatic outline alignment")
    align.add_argument("image", type=Path)
    align.add_argument("--reference", type=Path, default=Path("reference/census-2025"))
    align.add_argument("--output", type=Path, required=True)
    align.add_argument("--max-dimension", type=int, default=900)
    align.add_argument("--seed", type=int, default=42)
    align.add_argument(
        "--projection",
        action="append",
        choices=tuple(PROJECTIONS),
        help="Restrict the candidate projection; may be repeated",
    )
    align.add_argument(
        "--coverage-model",
        action="append",
        choices=("full_or_most_state", "partial_state"),
        help="Restrict the coverage model; may be repeated",
    )
    align.add_argument(
        "--transform-model",
        action="append",
        choices=("similarity", "affine_like"),
        help="Restrict the source transform; may be repeated",
    )

    equivalent_source = subparsers.add_parser(
        "lift-equivalent-source-alignment",
        help="Lift normalized alignment parameters onto a verified same-crop source",
    )
    equivalent_source.add_argument("alignment", type=Path)
    equivalent_source.add_argument("new_source", type=Path)
    equivalent_source.add_argument("--output", type=Path, required=True)

    pdf_align = subparsers.add_parser(
        "align-pdf-graticule",
        help="Register a vector PDF from its geographic graticule",
    )
    pdf_align.add_argument("pdf", type=Path)
    pdf_align.add_argument("--reference", type=Path, default=Path("reference/census-2025"))
    pdf_align.add_argument("--output", type=Path, required=True)
    pdf_align.add_argument("--render", type=Path)
    pdf_align.add_argument("--crs", default="EPSG:3310")
    pdf_align.add_argument("--page", type=int, default=1)
    pdf_align.add_argument(
        "--warp-crs",
        help="Optionally write a clipped north-up inspection warp, for example EPSG:3857",
    )
    pdf_align.add_argument(
        "--warp-resolution",
        type=float,
        default=250.0,
        help="Inspection-warp pixel size in target CRS units",
    )

    pdf_legend = subparsers.add_parser(
        "extract-pdf-legend",
        help="Extract native color swatches and provisional labels from a PDF legend",
    )
    pdf_legend.add_argument("pdf", type=Path)
    pdf_legend.add_argument("--output", type=Path, required=True)
    pdf_legend.add_argument("--render", type=Path)
    pdf_legend.add_argument("--page", type=int, default=1)

    assist = subparsers.add_parser("assist", help="Open the paired-control-point UI")
    assist.add_argument("image", type=Path)
    assist.add_argument("--reference", type=Path, default=Path("reference/census-2025"))
    assist.add_argument("--output", type=Path, required=True)
    assist.add_argument("--host", default="127.0.0.1")
    assist.add_argument("--port", type=int, default=8765)
    assist.add_argument("--no-open", action="store_true")

    extract = subparsers.add_parser(
        "extract-plan", help="Extract classified layers from a semantic legend plan"
    )
    extract.add_argument("plan", type=Path)
    extract.add_argument("--output", type=Path, required=True)

    perimeter_occlusion = subparsers.add_parser(
        "repair-perimeter-occlusions",
        help=(
            "Recover legend classes hidden by neutral source-map perimeter ink "
            "without changing the accepted alignment"
        ),
    )
    perimeter_occlusion.add_argument("base_extraction", type=Path)
    perimeter_occlusion.add_argument("mapbox_reference", type=Path)
    perimeter_occlusion.add_argument("--output", type=Path, required=True)

    rainfall_topology = subparsers.add_parser(
        "extract-rainfall-topology",
        help=(
            "Write separate palette-family, topology-inference, and unresolved "
            "artifacts for a rainfall extraction"
        ),
    )
    rainfall_topology.add_argument("plan", type=Path)
    rainfall_topology.add_argument("extraction", type=Path)
    rainfall_topology.add_argument("--output", type=Path, required=True)

    continuous_extract = subparsers.add_parser(
        "extract-continuous-plan",
        help="Extract a value-preserving raster from a continuous color-ramp legend",
    )
    continuous_extract.add_argument("plan", type=Path)
    continuous_extract.add_argument("--output", type=Path, required=True)

    continuous_diff = subparsers.add_parser(
        "audit-continuous-diff",
        help="Recompute and diff a stored continuous-ramp extraction",
    )
    continuous_diff.add_argument("run", type=Path)
    continuous_diff.add_argument("--output", type=Path, required=True)

    categorical_fidelity = subparsers.add_parser(
        "audit-categorical-fidelity",
        help="Recompute categorical color evidence under a threshold ensemble",
    )
    categorical_fidelity.add_argument("run", type=Path)
    categorical_fidelity.add_argument("--output", type=Path, required=True)
    categorical_fidelity.add_argument("--distance-perturbation", type=float, default=4.0)
    categorical_fidelity.add_argument("--margin-perturbation", type=float, default=1.0)

    categorical_compare = subparsers.add_parser(
        "compare-categorical-source",
        help="Build a fixed-point source/extraction blink and mismatch audit",
    )
    categorical_compare.add_argument("run", type=Path)
    categorical_compare.add_argument("fidelity_audit", type=Path)
    categorical_compare.add_argument("source_diff_batch", type=Path)
    categorical_compare.add_argument("case_id")
    categorical_compare.add_argument("--output", type=Path, required=True)
    categorical_compare.add_argument("--review-height", type=int, default=1400)

    pdf_vector = subparsers.add_parser(
        "extract-pdf-vector-fills",
        help="Rasterize native PDF fills that match a categorical legend",
    )
    pdf_vector.add_argument("pdf", type=Path)
    pdf_vector.add_argument("alignment", type=Path)
    pdf_vector.add_argument("legend", type=Path)
    pdf_vector.add_argument("--reference", type=Path, default=Path("reference/census-2025"))
    pdf_vector.add_argument("--output", type=Path, required=True)

    pdf_vector_diff = subparsers.add_parser(
        "audit-pdf-vector-diff",
        help="Diff a native-PDF fill extraction against its rendered source",
    )
    pdf_vector_diff.add_argument("run", type=Path)
    pdf_vector_diff.add_argument("--output", type=Path, required=True)

    review = subparsers.add_parser(
        "review", help="Open the full-resolution extraction review UI"
    )
    review.add_argument("run", type=Path)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8767)
    review.add_argument("--no-open", action="store_true")

    classification_review = subparsers.add_parser(
        "review-classification",
        help="Open the category-level classification review UI",
    )
    classification_review.add_argument("run", type=Path)
    classification_review.add_argument("--host", default="127.0.0.1")
    classification_review.add_argument("--port", type=int, default=8773)
    classification_review.add_argument("--no-open", action="store_true")

    completion_fidelity_review = subparsers.add_parser(
        "review-completion-fidelity",
        help="Open a read-only observed/manual/inferred/unresolved comparison UI",
    )
    completion_fidelity_review.add_argument("config", type=Path)
    completion_fidelity_review.add_argument("--host", default="127.0.0.1")
    completion_fidelity_review.add_argument("--port", type=int, default=8788)
    completion_fidelity_review.add_argument("--no-open", action="store_true")

    neighbor_completion = subparsers.add_parser(
        "complete-neighbor-unknowns",
        help="Clip remaining unknowns to the canonical border and fill interior gaps from neighbors",
    )
    neighbor_completion.add_argument("config", type=Path)

    promote_neighbor = subparsers.add_parser(
        "promote-neighbor-completion",
        help="Promote an inspected neighbor-completed raster into an approved materialization",
    )
    promote_neighbor.add_argument("neighbor_report", type=Path)
    promote_neighbor.add_argument("review_session", type=Path)
    promote_neighbor.add_argument("--author-statement", required=True)
    promote_neighbor.add_argument("--output", type=Path, required=True)

    migration_review = subparsers.add_parser(
        "review-migration",
        help="Review a migrated, source-diff-complete materialization candidate",
    )
    migration_review.add_argument("approved", type=Path)
    migration_review.add_argument("target_run", type=Path)
    migration_review.add_argument("candidate", type=Path)
    migration_review.add_argument("comparison", type=Path)
    migration_review.add_argument("alignment_audit", type=Path)
    migration_review.add_argument("stamp_audit", type=Path)
    migration_review.add_argument("--host", default="127.0.0.1")
    migration_review.add_argument("--port", type=int, default=8784)

    migrate_stamps = subparsers.add_parser(
        "migrate-stamp-corrections",
        help="Migrate reviewed stamp sources through a hash-linked child alignment",
    )
    migrate_stamps.add_argument("source_run", type=Path)
    migrate_stamps.add_argument("target_run", type=Path)

    alignment_application = subparsers.add_parser(
        "audit-alignment-application",
        help="Prove a regenerated extraction exactly applies its declared child alignment",
    )
    alignment_application.add_argument("source_run", type=Path)
    alignment_application.add_argument("target_run", type=Path)
    alignment_application.add_argument("--output", type=Path, required=True)

    compare_materialized = subparsers.add_parser(
        "compare-materialized-candidates",
        help="Compare an approved materialization with an unpublished candidate",
    )
    compare_materialized.add_argument("approved", type=Path)
    compare_materialized.add_argument("candidate", type=Path)
    compare_materialized.add_argument("--output", type=Path, required=True)

    stamp_migration = subparsers.add_parser(
        "audit-stamp-migration",
        help="Verify migrated stamp replay against its unchanged-coordinate counterfactual",
    )
    stamp_migration.add_argument("source_run", type=Path)
    stamp_migration.add_argument("target_run", type=Path)
    stamp_migration.add_argument("--approved-materialized", type=Path, required=True)
    stamp_migration.add_argument("--candidate-materialized", type=Path, required=True)
    stamp_migration.add_argument("--output", type=Path, required=True)

    materialize = subparsers.add_parser(
        "materialize-corrections",
        help="Create a hashed final candidate from reviewed inference and stamps",
    )
    materialize.add_argument("run", type=Path)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument(
        "--inference",
        type=Path,
        help="Select an inference directory; defaults to the highest versioned artifact",
    )
    materialize.add_argument(
        "--without-inference",
        dest="include_inference",
        action="store_false",
        default=None,
        help="Exclude all automatic inferred fill from the materialized candidate",
    )

    promote_reviewed = subparsers.add_parser(
        "promote-reviewed-extraction",
        help="Package a dual-reviewed extraction without changing its class pixels",
    )
    promote_reviewed.add_argument("run", type=Path)
    promote_reviewed.add_argument("source_diff_batch", type=Path)
    promote_reviewed.add_argument("source_diff_case_id")
    promote_reviewed.add_argument("--author-statement", required=True)
    promote_reviewed.add_argument("--output", type=Path, required=True)
    promote_reviewed.add_argument(
        "--canonical-pointer",
        type=Path,
        default=ACTIVE_CANONICAL_POINTER,
    )

    enclosed_fill = subparsers.add_parser(
        "fill-enclosed-holes",
        help="Fill tiny zero-class holes surrounded by exactly one class",
    )
    enclosed_fill.add_argument("run", type=Path)
    enclosed_fill.add_argument("--output", type=Path, required=True)
    enclosed_fill.add_argument(
        "--maximum-area-exclusive",
        type=int,
        default=50,
        help="Fill only connected holes with fewer than this many pixels",
    )

    export_tiles = subparsers.add_parser(
        "export-raster-tiles",
        help="Export an approved categorical candidate as static XYZ masks",
    )
    export_tiles.add_argument("materialized", type=Path)
    export_tiles.add_argument("--output", type=Path, required=True)
    export_tiles.add_argument("--minimum-zoom", type=int, default=4)
    export_tiles.add_argument("--maximum-zoom", type=int, default=9)
    export_tiles.add_argument("--overview-supersampling", type=int, default=4)

    export_preview_tiles = subparsers.add_parser(
        "export-extraction-preview-tiles",
        help="Export an unapproved extraction as review-only static XYZ masks",
    )
    export_preview_tiles.add_argument("run", type=Path)
    export_preview_tiles.add_argument("--output", type=Path, required=True)
    export_preview_tiles.add_argument("--minimum-zoom", type=int, default=4)
    export_preview_tiles.add_argument("--maximum-zoom", type=int, default=9)
    export_preview_tiles.add_argument("--overview-supersampling", type=int, default=4)

    export_continuous = subparsers.add_parser(
        "export-continuous-raster-tiles",
        help="Export an approved continuous extraction as native-color XYZ tiles",
    )
    export_continuous.add_argument("run", type=Path)
    export_continuous.add_argument("audit", type=Path)
    export_continuous.add_argument("--output", type=Path, required=True)
    export_continuous.add_argument("--minimum-zoom", type=int, default=4)
    export_continuous.add_argument("--maximum-zoom", type=int, default=9)
    export_continuous.add_argument("--overview-supersampling", type=int, default=4)
    export_continuous.add_argument(
        "--review-preview",
        action="store_true",
        help=(
            "Export a non-publishable Mapbox review package before author approval"
        ),
    )

    activate_dataset = subparsers.add_parser(
        "activate-staging-dataset",
        help="Promote an exact reviewed staging tile package into the public catalog",
    )
    activate_dataset.add_argument("staging", type=Path)
    activate_dataset.add_argument("--author-statement", required=True)
    activate_dataset.add_argument("--output", type=Path, required=True)

    attest_indexed_coverage = subparsers.add_parser(
        "attest-indexed-staging-coverage",
        help="Bind an indexed staging package to an exact Mapbox coverage audit",
    )
    attest_indexed_coverage.add_argument("staging", type=Path)
    attest_indexed_coverage.add_argument("plan", type=Path)

    activate_indexed_dataset = subparsers.add_parser(
        "activate-indexed-staging-dataset",
        help="Promote an autonomous indexed-raster staging package",
    )
    activate_indexed_dataset.add_argument("staging", type=Path)
    activate_indexed_dataset.add_argument("--author-statement", required=True)
    activate_indexed_dataset.add_argument("--output", type=Path, required=True)
    activate_indexed_dataset.add_argument("--public-id")
    activate_indexed_dataset.add_argument("--public-title")
    activate_indexed_dataset.add_argument(
        "--asset-base",
        help="Root-relative deployed URL for verified shared staging assets",
    )
    activate_indexed_dataset.add_argument(
        "--autonomous-evidence",
        type=Path,
        help=(
            "Hash-bound passing non-regression evidence for a no-human activation"
        ),
    )

    infer_gaps = subparsers.add_parser(
        "infer-categorical-gaps",
        help="Create separately masked fills for small single-class occlusions",
    )
    infer_gaps.add_argument("run", type=Path)
    infer_gaps.add_argument("--output", type=Path)
    infer_gaps.add_argument("--gap-radius-px", type=int, default=6)
    infer_gaps.add_argument("--max-area-px", type=int, default=1200)
    infer_gaps.add_argument("--max-dimension-px", type=int, default=120)
    infer_gaps.add_argument("--ring-radius-px", type=int, default=3)
    infer_gaps.add_argument("--minimum-ring-pixels", type=int, default=12)
    infer_gaps.add_argument("--minimum-dominance", type=float, default=0.98)
    infer_gaps.add_argument("--ocr-tsv", type=Path, action="append")
    infer_gaps.add_argument("--ocr-min-confidence", type=float, default=75.0)
    infer_gaps.add_argument("--ocr-label-padding-px", type=int, default=18)
    infer_gaps.add_argument("--ocr-context-radius-px", type=int, default=32)
    infer_gaps.add_argument("--ocr-max-distance-px", type=float, default=28.0)
    infer_gaps.add_argument("--ocr-min-distance-margin-px", type=float, default=1.5)

    detect_labels = subparsers.add_parser(
        "detect-map-labels",
        help="Isolate neutral dark map labels and run sparse Tesseract OCR",
    )
    detect_labels.add_argument("run", type=Path)
    detect_labels.add_argument("--output", type=Path)
    detect_labels.add_argument("--psm", type=int, default=3)
    detect_labels.add_argument("--maximum-channel-value", type=int, default=120)
    detect_labels.add_argument("--maximum-channel-spread", type=int, default=38)
    detect_labels.add_argument("--closing-size-px", type=int, default=2)

    feature_extract = subparsers.add_parser(
        "extract-feature-ink",
        help="Extract a clipped observed-ink diagnostic from a feature plan",
    )
    feature_extract.add_argument("plan", type=Path)
    feature_extract.add_argument("--output", type=Path, required=True)

    river_semantics = subparsers.add_parser(
        "extract-river-semantics",
        help="Separate rotation-aware label proposals from observed river geometry",
    )
    river_semantics.add_argument("run", type=Path)
    river_semantics.add_argument("--output", type=Path, required=True)
    river_semantics.add_argument("--ocr-scale", type=float, default=2.0)
    river_semantics.add_argument("--psm", type=int, default=11)
    river_semantics.add_argument("--minimum-candidate-confidence", type=float, default=30.0)
    river_semantics.add_argument("--high-confidence", type=float, default=65.0)
    river_semantics.add_argument("--consensus-confidence", type=float, default=42.0)
    river_semantics.add_argument("--maximum-gap-px", type=int, default=28)

    source_diff = subparsers.add_parser(
        "audit-source-diff",
        help="Diff an extraction candidate against source evidence and coverage rules",
    )
    source_diff.add_argument("run", type=Path)
    source_diff.add_argument("--output", type=Path, required=True)
    source_diff.add_argument("--candidate", type=Path)
    source_diff.add_argument(
        "--no-repair-full-state",
        dest="repair_full_state",
        action="store_false",
        help="Report full-state holes without creating a completed candidate",
    )

    feature_diff = subparsers.add_parser(
        "audit-feature-diff",
        help="Recompute and diff a feature-ink evidence extraction",
    )
    feature_diff.add_argument("run", type=Path)
    feature_diff.add_argument("--output", type=Path, required=True)

    source_diff_batch = subparsers.add_parser(
        "audit-source-diff-batch",
        help="Run source-diff QA for a configured source inventory",
    )
    source_diff_batch.add_argument("config", type=Path)
    source_diff_batch.add_argument("--output", type=Path, required=True)

    promote_source_diff = subparsers.add_parser(
        "promote-source-diff-materialization",
        help="Promote a stable source-diff surface into a review materialization",
    )
    promote_source_diff.add_argument("materialized", type=Path)
    promote_source_diff.add_argument("source_diff_batch", type=Path)
    promote_source_diff.add_argument("case_id")
    promote_source_diff.add_argument("--output", type=Path, required=True)

    boundary_clip = subparsers.add_parser(
        "clip-materialization-to-boundary",
        help="Clip a review materialization to the exact continuous hybrid border",
    )
    boundary_clip.add_argument("materialized", type=Path)
    boundary_clip.add_argument("perimeter_audit", type=Path)
    boundary_clip.add_argument("--output", type=Path, required=True)
    boundary_clip.add_argument(
        "--component-audit",
        type=Path,
        help="Add hash-bound source-supported islands to the required mainland",
    )

    boundary_components = subparsers.add_parser(
        "audit-boundary-components",
        help="Select authoritative islands that contain observed categorical pixels",
    )
    boundary_components.add_argument("run", type=Path)
    boundary_components.add_argument("--output", type=Path, required=True)
    boundary_components.add_argument("--minimum-observed-pixels", type=int, default=1)
    boundary_components.add_argument("--allow-legacy-snapshot", action="store_true")

    refine = subparsers.add_parser(
        "refine-alignment",
        help="Fit a conservative transform from saved review correction arrows",
    )
    refine.add_argument("alignment", type=Path)
    refine.add_argument("corrections", type=Path)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--max-loo-p90", type=float, default=4.0)
    refine.add_argument("--max-loo-max", type=float, default=8.0)
    refine.add_argument(
        "--reverse-arrows",
        action="store_true",
        help="Reverse legacy arrows that were drawn reference-to-source",
    )

    local_refine = subparsers.add_parser(
        "refine-local-alignment",
        help="Fit compact local corrections that decay to the parent alignment",
    )
    local_refine.add_argument("alignment", type=Path)
    local_refine.add_argument("corrections", type=Path)
    local_refine.add_argument("--output", type=Path, required=True)
    local_refine.add_argument("--radius-px", type=float, default=500.0)

    auto_refine = subparsers.add_parser(
        "auto-refine-perimeter",
        help="Iteratively match California perimeter anchors and refine alignment",
    )
    auto_refine.add_argument("image", type=Path)
    auto_refine.add_argument("alignment", type=Path)
    auto_refine.add_argument("--reference", type=Path, default=Path("reference/census-2025"))
    auto_refine.add_argument("--output", type=Path, required=True)
    auto_refine.add_argument("--max-iterations", type=int, default=3)
    auto_refine.add_argument("--working-height", type=int, default=1600)
    auto_refine.add_argument("--candidate-anchors", type=int, default=16)
    auto_refine.add_argument("--fit-anchors", type=int, default=8)
    auto_refine.add_argument("--search-radius-px", type=int, default=36)
    auto_refine.add_argument("--tangent-radius-px", type=int, default=12)
    auto_refine.add_argument(
        "--county-reference",
        type=Path,
        help="Registered high-resolution county-reference manifest",
    )
    auto_refine.add_argument(
        "--preserve-geographic-registration",
        action="store_true",
        help="Audit only when graticule or other stronger controls must remain authoritative",
    )

    canonical_alignment = subparsers.add_parser(
        "audit-canonical-alignment",
        help="Audit an unclipped warped source against the exact active lime line",
    )
    canonical_alignment.add_argument("image", type=Path)
    canonical_alignment.add_argument("alignment", type=Path)
    canonical_alignment.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    canonical_alignment.add_argument(
        "--canonical-pointer", type=Path, default=ACTIVE_CANONICAL_POINTER
    )
    canonical_alignment.add_argument("--output", type=Path, required=True)
    canonical_alignment.add_argument("--target-height", type=int, default=2200)

    auto_refine_batch = subparsers.add_parser(
        "auto-refine-perimeter-batch",
        help="Run iterative perimeter alignment for a configured source inventory",
    )
    auto_refine_batch.add_argument("config", type=Path)
    auto_refine_batch.add_argument("--output", type=Path, required=True)

    solid_refine = subparsers.add_parser(
        "refine-solid-mask-alignment",
        help="Fit California alignment from exact solid thematic class edges",
    )
    solid_refine.add_argument("image", type=Path)
    solid_refine.add_argument("alignment", type=Path)
    solid_refine.add_argument("evidence", type=Path)
    solid_refine.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    solid_refine.add_argument("--canonical-boundary", type=Path, required=True)
    solid_refine.add_argument("--output", type=Path, required=True)

    coast_refine = subparsers.add_parser(
        "refine-solid-west-coast",
        help="Move a solid thematic mainland coast left while pinning islands",
    )
    coast_refine.add_argument("image", type=Path)
    coast_refine.add_argument("alignment", type=Path)
    coast_refine.add_argument("evidence", type=Path)
    coast_refine.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    coast_refine.add_argument("--canonical-boundary", type=Path, required=True)
    coast_refine.add_argument("--output", type=Path, required=True)
    coast_refine.add_argument("--radius-px", type=float, default=360.0)
    coast_refine.add_argument("--minimum-left-shift-px", type=float, default=4.0)
    coast_refine.add_argument("--maximum-left-shift-px", type=float, default=20.0)

    directional_coast = subparsers.add_parser(
        "refine-directional-west-coast",
        help="Apply a compact northward coast correction while pinning east and islands",
    )
    directional_coast.add_argument("alignment", type=Path)
    directional_coast.add_argument("--canonical-boundary", type=Path, required=True)
    directional_coast.add_argument("--output", type=Path, required=True)
    directional_coast.add_argument("--northward-shift-px", type=float, default=0.0)
    directional_coast.add_argument("--westward-shift-px", type=float, default=0.0)
    directional_coast.add_argument("--radius-px", type=float, default=380.0)
    directional_coast.add_argument("--image", type=Path)
    directional_coast.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    directional_coast.add_argument("--target-height", type=int, default=1014)
    directional_coast.add_argument("--horizontal-edge-pin-count", type=int, default=0)

    east_anchored = subparsers.add_parser(
        "fit-east-anchored-horizontal-scale",
        help="Refit one global source x scale while preserving the eastern border",
    )
    east_anchored.add_argument("image", type=Path)
    east_anchored.add_argument("alignment", type=Path)
    east_anchored.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    east_anchored.add_argument("--canonical-boundary", type=Path, required=True)
    east_anchored.add_argument("--output", type=Path, required=True)
    east_anchored.add_argument("--minimum-multiplier", type=float, default=0.985)
    east_anchored.add_argument("--maximum-multiplier", type=float, default=1.005)
    east_anchored.add_argument("--candidate-count", type=int, default=41)
    east_anchored.add_argument("--target-height", type=int, default=1014)
    east_anchored.add_argument(
        "--require-rendered-width-reduction",
        action="store_true",
        help="Reject ranges or results that do not make the rendered source narrower",
    )

    county_fine = subparsers.add_parser(
        "fine-align-county",
        help="Fine-register a warped map using only county.png interior junctions",
    )
    county_fine.add_argument("image", type=Path)
    county_fine.add_argument("alignment", type=Path)
    county_fine.add_argument("county_reference", type=Path)
    county_fine.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    county_fine.add_argument("--output", type=Path, required=True)
    county_fine.add_argument("--working-height", type=int, default=2400)

    fine_determinism = subparsers.add_parser(
        "audit-fine-alignment-determinism",
        help="Compare two complete county fine-alignment runs byte for byte",
    )
    fine_determinism.add_argument("first_run", type=Path)
    fine_determinism.add_argument("second_run", type=Path)
    fine_determinism.add_argument("--output", type=Path, required=True)

    county_fine_review = subparsers.add_parser(
        "review-fine-alignment",
        help="Open a read-only before/fine review for an automatic county alignment",
    )
    county_fine_review.add_argument("run", type=Path)
    county_fine_review.add_argument("--host", default="127.0.0.1")
    county_fine_review.add_argument("--port", type=int, default=8785)

    southern_edge = subparsers.add_parser(
        "refine-southern-edge",
        help="Automatically add a compact south/southeast border correction",
    )
    southern_edge.add_argument("fine_run", type=Path)
    southern_edge.add_argument("image", type=Path)
    southern_edge.add_argument("county_reference", type=Path)
    southern_edge.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    southern_edge.add_argument("--output", type=Path, required=True)
    southern_edge.add_argument("--radius-px", type=float, default=520.0)

    lower_colorado = subparsers.add_parser(
        "refine-lower-colorado",
        help="Automatically correct only the Census-validated lower Colorado edge",
    )
    lower_colorado.add_argument("fine_run", type=Path)
    lower_colorado.add_argument("image", type=Path)
    lower_colorado.add_argument("county_reference", type=Path)
    lower_colorado.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    lower_colorado.add_argument("--output", type=Path, required=True)

    hybrid_perimeter = subparsers.add_parser(
        "audit-hybrid-perimeter",
        help="Audit county.png coast plus Census land-border regional authority",
    )
    hybrid_perimeter.add_argument("run", type=Path)
    hybrid_perimeter.add_argument("county_reference", type=Path)
    hybrid_perimeter.add_argument(
        "--reference", type=Path, default=Path("reference/census-2025")
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fetch-reference":
        result = fetch_reference_data(args.output, force=args.force)
    elif args.command == "fetch-water-reference":
        result = fetch_california_areawater(
            args.boundary_reference,
            args.output,
            force=args.force,
        )
    elif args.command == "fetch-mapbox-water-reference":
        access_token = os.environ.get(args.token_variable, "")
        if not access_token and args.token_file.is_file():
            prefix = f"{args.token_variable}="
            for line in args.token_file.read_text().splitlines():
                if line.startswith(prefix):
                    access_token = line[len(prefix) :].strip().strip("\"'")
                    break
        result = fetch_mapbox_water_reference(
            args.output,
            access_token,
            bounds_wgs84=tuple(args.bounds),
            zoom=args.zoom,
            force=args.force,
        )
    elif args.command == "promote-canonical-boundary":
        result = promote_canonical_boundary(
            args.approved_materialization,
            args.approved_boundary_geojson,
            args.output,
        )
    elif args.command == "rasterize-canonical-boundary":
        result = rasterize_canonical_boundary_for_run(
            args.canonical_manifest,
            args.run,
            args.output,
        )
    elif args.command == "build-county-detail-boundary":
        result = build_county_detail_boundary_candidate(
            args.canonical_manifest,
            args.hybrid_perimeter,
            args.county_reference,
            args.reference,
            args.output,
        )
    elif args.command == "promote-county-detail-border":
        result = promote_county_detail_border(
            args.candidate_manifest,
            args.supersedes,
            args.output,
            author_statement=args.author_statement,
        )
    elif args.command == "activate-canonical-border":
        result = activate_canonical_border(args.canonical_manifest, args.pointer)
    elif args.command == "register-county-reference":
        result = register_county_reference(
            args.image,
            args.reference,
            args.output,
            maximum_dimension=args.maximum_dimension,
            web_height=args.web_height,
        )
    elif args.command == "align":
        result = align_image(
            args.image,
            args.reference,
            args.output,
            max_dimension=args.max_dimension,
            seed=args.seed,
            projection_names=args.projection,
            coverage_models=args.coverage_model,
            transform_models=args.transform_model,
        )
    elif args.command == "lift-equivalent-source-alignment":
        result = lift_alignment_to_equivalent_source(
            args.alignment,
            args.new_source,
            args.output,
        )
    elif args.command == "align-pdf-graticule":
        result = align_pdf_graticule(
            args.pdf,
            args.reference,
            args.output,
            render_path=args.render,
            projection_crs=args.crs,
            page_number=args.page,
            warp_crs=args.warp_crs,
            warp_resolution=args.warp_resolution,
        )
    elif args.command == "extract-pdf-legend":
        result = extract_pdf_legend(
            args.pdf,
            args.output,
            render_path=args.render,
            page_number=args.page,
        )
    elif args.command == "assist":
        serve_assist(
            args.image,
            args.reference,
            args.output,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return
    elif args.command == "extract-plan":
        result = extract_from_plan(args.plan, args.output)
    elif args.command == "repair-perimeter-occlusions":
        result = run_perimeter_occlusion_repair(
            args.base_extraction,
            args.mapbox_reference,
            args.output,
        )
    elif args.command == "extract-rainfall-topology":
        result = extract_rainfall_topology(
            args.plan, args.extraction, args.output
        )
    elif args.command == "extract-continuous-plan":
        result = extract_continuous_plan(args.plan, args.output)
    elif args.command == "audit-continuous-diff":
        result = audit_continuous_source_diff(args.run, args.output)
    elif args.command == "audit-categorical-fidelity":
        result = audit_categorical_fidelity(
            args.run,
            args.output,
            distance_perturbation=args.distance_perturbation,
            margin_perturbation=args.margin_perturbation,
        )
    elif args.command == "compare-categorical-source":
        result = build_categorical_comparison(
            args.run,
            args.fidelity_audit,
            args.source_diff_batch,
            args.case_id,
            args.output,
            review_height=args.review_height,
        )
    elif args.command == "extract-pdf-vector-fills":
        result = extract_pdf_vector_fills(
            args.pdf,
            args.alignment,
            args.legend,
            args.reference,
            args.output,
        )
    elif args.command == "audit-pdf-vector-diff":
        result = audit_pdf_vector_diff(args.run, args.output)
    elif args.command == "refine-alignment":
        result = fit_review_corrections(
            args.alignment,
            args.corrections,
            args.output,
            max_leave_one_out_p90_px=args.max_loo_p90,
            max_leave_one_out_max_px=args.max_loo_max,
            reverse_declared_direction=args.reverse_arrows,
        )
    elif args.command == "refine-local-alignment":
        result = fit_local_review_corrections(
            args.alignment,
            args.corrections,
            args.output,
            radius_px=args.radius_px,
        )
    elif args.command == "auto-refine-perimeter":
        result = auto_refine_perimeter(
            args.image,
            args.alignment,
            args.reference,
            args.output,
            max_iterations=args.max_iterations,
            working_height=args.working_height,
            candidate_anchor_count=args.candidate_anchors,
            fit_anchor_count=args.fit_anchors,
            search_radius_px=args.search_radius_px,
            tangent_radius_px=args.tangent_radius_px,
            preserve_geographic_registration=args.preserve_geographic_registration,
            county_reference_path=args.county_reference,
        )
    elif args.command == "auto-refine-perimeter-batch":
        result = auto_refine_perimeter_batch(args.config, args.output)
    elif args.command == "audit-canonical-alignment":
        result = audit_canonical_alignment(
            args.image,
            args.alignment,
            args.reference,
            args.output,
            active_pointer_path=args.canonical_pointer,
            target_height=args.target_height,
        )
    elif args.command == "refine-solid-mask-alignment":
        evidence = json.loads(args.evidence.read_text())
        result = refine_solid_mask_alignment(
            args.image,
            args.alignment,
            args.reference,
            args.canonical_boundary,
            args.output,
            evidence["solid_map_rgb"],
            minimum_component_area=int(
                evidence.get("minimum_component_area", 500)
            ),
            maximum_components=int(evidence.get("maximum_components", 5)),
            correspondence_gate_px=float(
                evidence.get("correspondence_gate_px", 30.0)
            ),
        )
    elif args.command == "refine-solid-west-coast":
        evidence = json.loads(args.evidence.read_text())
        result = refine_solid_west_coast(
            args.image,
            args.alignment,
            args.reference,
            args.canonical_boundary,
            args.output,
            evidence["solid_map_rgb"],
            maximum_components=int(evidence.get("maximum_components", 5)),
            radius_px=args.radius_px,
            minimum_left_shift_px=args.minimum_left_shift_px,
            maximum_left_shift_px=args.maximum_left_shift_px,
        )
    elif args.command == "refine-directional-west-coast":
        result = refine_directional_west_coast(
            args.alignment,
            args.canonical_boundary,
            args.output,
            northward_shift_px=args.northward_shift_px,
            westward_shift_px=args.westward_shift_px,
            radius_px=args.radius_px,
            image_path=args.image,
            reference_root=args.reference,
            target_height=args.target_height,
            horizontal_edge_pin_count=args.horizontal_edge_pin_count,
        )
    elif args.command == "fit-east-anchored-horizontal-scale":
        result = fit_east_anchored_horizontal_scale(
            args.image,
            args.alignment,
            args.reference,
            args.canonical_boundary,
            args.output,
            minimum_multiplier=args.minimum_multiplier,
            maximum_multiplier=args.maximum_multiplier,
            candidate_count=args.candidate_count,
            target_height=args.target_height,
            require_rendered_width_reduction=args.require_rendered_width_reduction,
        )
    elif args.command == "fine-align-county":
        result = fine_align_to_county_reference(
            args.image,
            args.alignment,
            args.county_reference,
            args.reference,
            args.output,
            working_height=args.working_height,
        )
    elif args.command == "audit-fine-alignment-determinism":
        result = audit_fine_alignment_determinism(
            args.first_run,
            args.second_run,
            args.output,
        )
    elif args.command == "review-fine-alignment":
        serve_fine_alignment_review(args.run, host=args.host, port=args.port)
        return
    elif args.command == "refine-southern-edge":
        result = refine_southern_edge(
            args.fine_run,
            args.image,
            args.county_reference,
            args.reference,
            args.output,
            radius_px=args.radius_px,
        )
    elif args.command == "refine-lower-colorado":
        result = refine_lower_colorado(
            args.fine_run,
            args.image,
            args.county_reference,
            args.reference,
            args.output,
        )
    elif args.command == "audit-hybrid-perimeter":
        result = audit_hybrid_perimeter(
            args.run,
            args.county_reference,
            args.reference,
        )
    elif args.command == "review":
        serve_review(
            args.run,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return
    elif args.command == "review-classification":
        serve_classification_review(
            args.run,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return
    elif args.command == "review-completion-fidelity":
        serve_completion_fidelity_review(
            args.config,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return
    elif args.command == "complete-neighbor-unknowns":
        result = complete_neighbor_unknowns(args.config)
    elif args.command == "promote-neighbor-completion":
        result = promote_neighbor_completion(
            args.neighbor_report,
            args.review_session,
            args.output,
            author_statement=args.author_statement,
        )
    elif args.command == "review-migration":
        serve_migration_review(
            args.approved,
            args.target_run,
            args.candidate,
            args.comparison,
            args.alignment_audit,
            args.stamp_audit,
            host=args.host,
            port=args.port,
        )
        return
    elif args.command == "migrate-stamp-corrections":
        result = migrate_stamp_corrections(args.source_run, args.target_run)
    elif args.command == "audit-alignment-application":
        result = audit_alignment_application(
            args.source_run, args.target_run, args.output
        )
    elif args.command == "compare-materialized-candidates":
        result = compare_materialized_candidates(
            args.approved, args.candidate, args.output
        )
    elif args.command == "audit-stamp-migration":
        result = audit_stamp_migration(
            args.source_run,
            args.target_run,
            args.output,
            approved_materialized_dir=args.approved_materialized,
            candidate_materialized_dir=args.candidate_materialized,
        )
    elif args.command == "materialize-corrections":
        result = materialize_review_corrections(
            args.run,
            args.output,
            inference_dir=args.inference,
            include_inference=args.include_inference,
        )
    elif args.command == "promote-reviewed-extraction":
        result = promote_reviewed_extraction(
            args.run,
            args.output,
            author_statement=args.author_statement,
            source_diff_batch_path=args.source_diff_batch,
            source_diff_case_id=args.source_diff_case_id,
            canonical_pointer_path=args.canonical_pointer,
        )
    elif args.command == "fill-enclosed-holes":
        result = generate_enclosed_fill_artifact(
            args.run,
            args.output,
            maximum_area_exclusive=args.maximum_area_exclusive,
        )
    elif args.command == "export-raster-tiles":
        result = export_categorical_tiles(
            args.materialized,
            args.output,
            minimum_zoom=args.minimum_zoom,
            maximum_zoom=args.maximum_zoom,
            overview_supersampling=args.overview_supersampling,
        )
    elif args.command == "export-extraction-preview-tiles":
        result = export_extraction_preview_tiles(
            args.run,
            args.output,
            minimum_zoom=args.minimum_zoom,
            maximum_zoom=args.maximum_zoom,
            overview_supersampling=args.overview_supersampling,
        )
    elif args.command == "export-continuous-raster-tiles":
        result = export_continuous_tiles(
            args.run,
            args.audit,
            args.output,
            minimum_zoom=args.minimum_zoom,
            maximum_zoom=args.maximum_zoom,
            overview_supersampling=args.overview_supersampling,
            review_preview=args.review_preview,
        )
    elif args.command == "activate-staging-dataset":
        result = activate_staging_dataset(
            args.staging,
            args.output,
            author_statement=args.author_statement,
        )
    elif args.command == "attest-indexed-staging-coverage":
        result = attest_indexed_staging_coverage(args.staging, args.plan)
    elif args.command == "activate-indexed-staging-dataset":
        result = activate_indexed_staging_dataset(
            args.staging,
            args.output,
            author_statement=args.author_statement,
            public_id=args.public_id,
            public_title=args.public_title,
            asset_base=args.asset_base,
            autonomous_evidence_path=args.autonomous_evidence,
        )
    elif args.command == "infer-categorical-gaps":
        result = infer_categorical_run(
            args.run,
            output_dir=args.output,
            max_gap_radius_px=args.gap_radius_px,
            max_component_area_px=args.max_area_px,
            max_component_dimension_px=args.max_dimension_px,
            ring_radius_px=args.ring_radius_px,
            minimum_ring_pixels=args.minimum_ring_pixels,
            minimum_dominance=args.minimum_dominance,
            ocr_tsv_path=args.ocr_tsv[0] if args.ocr_tsv else None,
            ocr_additional_tsv_paths=args.ocr_tsv[1:] if args.ocr_tsv else (),
            ocr_minimum_confidence=args.ocr_min_confidence,
            ocr_label_padding_px=args.ocr_label_padding_px,
            ocr_context_radius_px=args.ocr_context_radius_px,
            ocr_maximum_propagation_distance_px=args.ocr_max_distance_px,
            ocr_minimum_distance_margin_px=args.ocr_min_distance_margin_px,
        )
    elif args.command == "detect-map-labels":
        result = detect_run_labels(
            args.run,
            output_dir=args.output,
            page_segmentation_mode=args.psm,
            maximum_channel_value=args.maximum_channel_value,
            maximum_channel_spread=args.maximum_channel_spread,
            closing_size_px=args.closing_size_px,
        )
    elif args.command == "extract-feature-ink":
        result = extract_observed_feature_ink(args.plan, args.output)
    elif args.command == "extract-river-semantics":
        result = extract_river_semantics(
            args.run,
            args.output,
            ocr_scale=args.ocr_scale,
            page_segmentation_mode=args.psm,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
            high_confidence=args.high_confidence,
            consensus_confidence=args.consensus_confidence,
            maximum_gap_px=args.maximum_gap_px,
        )
    elif args.command == "audit-source-diff":
        result = audit_extraction_source_diff(
            args.run,
            args.output,
            candidate_dir=args.candidate,
            repair_full_state=args.repair_full_state,
        )
    elif args.command == "audit-feature-diff":
        result = audit_feature_source_diff(args.run, args.output)
    elif args.command == "audit-source-diff-batch":
        result = audit_source_diff_batch(args.config, args.output)
    elif args.command == "promote-source-diff-materialization":
        result = promote_source_diff_materialization(
            args.materialized,
            args.source_diff_batch,
            args.case_id,
            args.output,
        )
    elif args.command == "clip-materialization-to-boundary":
        result = clip_materialization_to_boundary(
            args.materialized,
            args.perimeter_audit,
            args.output,
            component_audit_path=args.component_audit,
        )
    elif args.command == "audit-boundary-components":
        result = audit_source_supported_boundary_components(
            args.run,
            args.output,
            minimum_observed_pixels=args.minimum_observed_pixels,
            allow_legacy_snapshot=args.allow_legacy_snapshot,
        )
    else:  # pragma: no cover - argparse guarantees a known command.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
