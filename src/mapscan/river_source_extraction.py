"""Legend-independent, no-human extraction for named hydrography maps.

The source family uses the same blue ink for feature geometry and labels.  This
adapter detects the blue evidence from the pristine source, runs rotation-aware
OCR only to identify/exclude label ink, and keeps uncertain text separate from
publishable geometry.  It supports the distinctions actually visible in the
source: solid river/stream lines, lake/reservoir outlines, dotted dry
streambeds, and dry-lake outlines/interiors.  The source does not encode a
separate ordinary-river versus ordinary-stream style, nor lake versus reservoir,
so those pairs deliberately remain combined semantic classes.

Source preparation can run before alignment.  The publishable Mapbox pass is
strictly unavailable until the experiment log contains an accepted automatic
alignment; it then performs source reconstruction, geographic diff, land/water
clipping, and a two-pass fixed-point replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .automatic_alignment_loop import (
    _large_pale_blue_water,
    load_pinned_mapbox_reference,
)
from .automatic_categorical_extraction import (
    _load_accepted_alignment,
    _reference_to_source_remap,
    _source_to_reference,
    _source_to_reference_remap,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .river_semantics import (
    _build_ocr_candidate,
    _rotate_page,
    detect_rotation_aware_text_like_regions,
    expand_confirmed_text_regions,
    extract_confirmed_glyph_components,
    infer_reconnections,
    select_text_detections,
)
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.named-hydrography-source-extraction.v1"
PRODUCER = "mapscan.river_source_extraction"
FEATURE_IDS = (
    "river-or-stream",
    "lake-or-reservoir",
    "dry-streambed",
    "dry-lake",
)
FORBIDDEN_PATH_TOKENS = (
    "automatic-alignment-orphaned-race",
    "county.png",
    "census",
    "manual",
    "legacy",
    "river-semantics",
    "rivers-extract",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def _save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path, optimize=True)


def _save_ids(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


def _assert_clean_path(path: Path, kind: str) -> None:
    normalized = str(path.resolve()).lower()
    matched = next((token for token in FORBIDDEN_PATH_TOKENS if token in normalized), None)
    if matched:
        raise ValueError(f"{kind} path contains forbidden no-human evidence: {matched}")


@dataclass(frozen=True)
class RiverSourceConfig:
    blue_hue_minimum: int = 85
    blue_hue_maximum: int = 125
    blue_saturation_minimum: int = 60
    blue_value_minimum: int = 55
    ocr_angles: tuple[int, ...] = tuple(range(-90, 91, 15))
    ocr_scale: float = 2.0
    ocr_minimum_candidate_confidence: float = 18.0
    ocr_high_confidence: float = 80.0
    ocr_consensus_confidence: float = 58.0
    minimum_named_label_count: int = 12
    maximum_reconnection_gap_px: int = 34
    required_replay_count: int = 2
    minimum_channel_roundtrip_iou: float = 0.78
    minimum_composite_roundtrip_fraction: float = 0.84
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_match_fraction: float = 0.80
    minimum_passing_geographic_cells: int = 7

    def __post_init__(self) -> None:
        if self.required_replay_count != 2:
            raise ValueError("named hydrography extraction requires exactly two replays")
        if not self.ocr_angles:
            raise ValueError("rotation-aware OCR requires at least one angle")
        for value in (
            self.minimum_channel_roundtrip_iou,
            self.minimum_composite_roundtrip_fraction,
            self.minimum_geographic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fraction gates must be between zero and one")


@dataclass(frozen=True)
class RiverSourceEvidence:
    source_rgb: np.ndarray
    source_scope: np.ndarray
    observed_blue: np.ndarray
    text_mask: np.ndarray
    ambiguous_text_mask: np.ndarray
    river_or_stream: np.ndarray
    lake_or_reservoir_outline: np.ndarray
    lake_or_reservoir_interior: np.ndarray
    dry_streambed: np.ndarray
    dry_lake_outline: np.ndarray
    dry_lake_interior: np.ndarray
    inferred_reconnections: np.ndarray
    labels: tuple[Mapping[str, Any], ...]
    named_features: tuple[Mapping[str, Any], ...]
    ocr_candidates: tuple[Mapping[str, Any], ...]
    reports: Mapping[str, Any]


@dataclass(frozen=True)
class RiverPreparationResult:
    status: str
    manifest_path: Path
    diagnostic_path: Path
    label_count: int
    feature_pixel_counts: Mapping[str, int]
    irreducible_ambiguities: tuple[str, ...]


@dataclass(frozen=True)
class RiverExtractionIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path


@dataclass(frozen=True)
class RiverExtractionResult:
    status: str
    stop_reason: str
    iterations: tuple[RiverExtractionIteration, ...]
    accepted_extraction_path: Path | None


def _validate_source_adapter(path: Path) -> WorkingRasterArtifact:
    _assert_clean_path(path, "source adapter")
    working = load_working_raster_artifact(path)
    if working.source_path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("named hydrography extraction requires the source JPEG")
    if working.source_path.name.casefold() != "rivers.jpg":
        raise ValueError("named hydrography adapter requires authoritative rivers.jpg")
    if working.manifest.get("authority", {}) != {
        "manual_input_used": False,
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
    }:
        raise ValueError("source-clean authority record is not pristine")
    return working


def _blue_ink(
    rgb: np.ndarray, source_scope: np.ndarray, config: RiverSourceConfig
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (
        (hsv[:, :, 0] >= config.blue_hue_minimum)
        & (hsv[:, :, 0] <= config.blue_hue_maximum)
        & (hsv[:, :, 1] >= config.blue_saturation_minimum)
        & (hsv[:, :, 2] >= config.blue_value_minimum)
        & source_scope
    )


def _rotation_aware_ocr(
    source_rgb: np.ndarray,
    observed_blue: np.ndarray,
    config: RiverSourceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Tesseract is required for named hydrography extraction")
    version = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    blue_only = np.where(observed_blue, 0, 255).astype(np.uint8)
    candidates: list[dict[str, Any]] = []
    raw_word_count = 0
    raw_word_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="mapscan-source-river-ocr-") as temporary:
        temporary_dir = Path(temporary)
        for view_name, page in (
            ("source-grayscale", gray),
            ("source-blue-mask", blue_only),
        ):
            raw_word_counts[view_name] = 0
            for angle in config.ocr_angles:
                rotated, matrix = _rotate_page(page, float(angle), config.ocr_scale)
                image_path = temporary_dir / f"{view_name}-angle-{angle:+04d}.png"
                Image.fromarray(rotated, mode="L").save(image_path, optimize=True)
                output_base = temporary_dir / f"{view_name}-angle-{angle:+04d}"
                subprocess.run(
                    [
                        executable,
                        str(image_path),
                        str(output_base),
                        "-l",
                        "eng",
                        "--psm",
                        "11",
                        "tsv",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                inverse = cv2.invertAffineTransform(matrix)
                with output_base.with_suffix(".tsv").open(newline="") as handle:
                    for row in csv.DictReader(
                        handle, delimiter="\t", quoting=csv.QUOTE_NONE
                    ):
                        text = str(row.get("text", "")).strip()
                        try:
                            confidence = float(row.get("conf", "-1"))
                            bbox = tuple(
                                float(row.get(key, "0"))
                                for key in ("left", "top", "width", "height")
                            )
                        except (TypeError, ValueError):
                            continue
                        if confidence >= 0:
                            raw_word_count += 1
                            raw_word_counts[view_name] += 1
                        candidate, rejection = _build_ocr_candidate(
                            text,
                            confidence,
                            bbox,
                            int(angle),
                            inverse,
                            observed_blue,
                            config.ocr_scale,
                            config.ocr_minimum_candidate_confidence,
                            f"tesseract-{view_name}",
                        )
                        if candidate is not None:
                            candidates.append(candidate)
                        elif rejection is not None:
                            rejection_counts[rejection] = (
                                rejection_counts.get(rejection, 0) + 1
                            )
    selected, ambiguous = select_text_detections(
        candidates,
        high_confidence=config.ocr_high_confidence,
        consensus_confidence=config.ocr_consensus_confidence,
        minimum_consensus_rotations=2,
    )
    return selected, ambiguous, {
        "engine": version,
        "angles_degrees": list(config.ocr_angles),
        "scale": config.ocr_scale,
        "raw_word_count": raw_word_count,
        "raw_word_counts_by_source_view": raw_word_counts,
        "candidate_count": len(candidates),
        "selected_detection_count": len(selected),
        "ambiguous_detection_count": len(ambiguous),
        "rejection_counts": rejection_counts,
    }


def _normalize_word(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _deduplicate_labels(
    detections: Sequence[Mapping[str, Any]], maximum_center_distance: float = 14.0
) -> tuple[dict[str, Any], ...]:
    clusters: list[list[Mapping[str, Any]]] = []
    for detection in sorted(detections, key=lambda item: -float(item["confidence"])):
        center = np.asarray(detection["source_center"], dtype=np.float64)
        for cluster in clusters:
            other = np.asarray(cluster[0]["source_center"], dtype=np.float64)
            if float(np.linalg.norm(center - other)) <= maximum_center_distance:
                cluster.append(detection)
                break
        else:
            clusters.append([detection])
    labels = []
    for index, cluster in enumerate(clusters, 1):
        best = max(cluster, key=lambda item: float(item["confidence"]))
        spellings: dict[str, int] = {}
        for item in cluster:
            normalized = _normalize_word(str(item["text"]))
            if normalized:
                spellings[normalized] = spellings.get(normalized, 0) + 1
        alternatives = sorted(spellings, key=lambda value: (-spellings[value], value))
        labels.append(
            {
                "id": f"source-label-{index:03d}",
                "text_verbatim_ocr": str(best["text"]),
                "normalized_token": _normalize_word(str(best["text"])),
                "confidence": float(best["confidence"]),
                "source_center": [float(v) for v in best["source_center"]],
                "source_polygon": best["source_polygon"],
                "source_word_length_px": float(best["source_word_length_px"]),
                "source_word_height_px": float(best["source_word_height_px"]),
                "source_orientation_degrees": float(best["source_orientation_degrees"]),
                "rotation_support_count": len(
                    {int(item["ocr_angle_degrees"]) for item in cluster}
                ),
                "ocr_spelling_alternatives": alternatives,
                "spelling_ambiguous": len(alternatives) > 1,
            }
        )
    return tuple(labels)


def _orientation_distance(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _group_named_features(
    labels: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Join neighbouring OCR tokens without silently rewriting their text."""

    adjacency = [set() for _ in labels]
    for left_index, left in enumerate(labels):
        left_center = np.asarray(left["source_center"], dtype=np.float64)
        left_angle = float(left["source_orientation_degrees"])
        left_length = float(left["source_word_length_px"])
        left_height = float(left["source_word_height_px"])
        radians = np.deg2rad(left_angle)
        along_axis = np.asarray([np.cos(radians), np.sin(radians)])
        across_axis = np.asarray([-along_axis[1], along_axis[0]])
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            right_center = np.asarray(right["source_center"], dtype=np.float64)
            right_length = float(right["source_word_length_px"])
            right_height = float(right["source_word_height_px"])
            coordinate_offset = np.abs(right_center - left_center)
            visually_stacked = (
                coordinate_offset[0] <= max(15.0, 0.3 * max(left_length, right_length))
                and coordinate_offset[1]
                <= max(24.0, 2.0 * max(left_height, right_height))
            )
            if visually_stacked:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                continue
            if _orientation_distance(
                left_angle, float(right["source_orientation_degrees"])
            ) > 18.0:
                continue
            offset = right_center - left_center
            along = abs(float(np.dot(offset, along_axis)))
            across = abs(float(np.dot(offset, across_axis)))
            longitudinal_gap = along - (left_length + right_length) / 2.0
            same_line = (
                across <= max(8.0, 0.8 * max(left_height, right_height))
                and longitudinal_gap <= 20.0
            )
            stacked_line = (
                along <= max(left_length, right_length) * 0.45
                and across <= max(18.0, 1.7 * max(left_height, right_height))
            )
            if same_line or stacked_line:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    groups: list[list[int]] = []
    unseen = set(range(len(labels)))
    while unseen:
        seed = unseen.pop()
        group = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = adjacency[current] & unseen
            unseen -= neighbours
            group.extend(sorted(neighbours))
            frontier.extend(neighbours)
        groups.append(group)
    output = []
    for feature_index, indices in enumerate(groups, 1):
        tokens = [labels[index] for index in indices]
        orientation = float(np.median([item["source_orientation_degrees"] for item in tokens]))
        centers = np.asarray([item["source_center"] for item in tokens], dtype=np.float64)
        if np.ptp(centers[:, 0]) >= np.ptp(centers[:, 1]):
            tokens.sort(key=lambda item: (item["source_center"][0], item["source_center"][1]))
        else:
            tokens.sort(key=lambda item: (item["source_center"][1], item["source_center"][0]))
        normalized = [str(item["normalized_token"]) for item in tokens]
        if any(value in {"dry", "lakebed"} for value in normalized):
            semantic = "dry-lake"
        elif any(
            value in {"lake", "reservoir", "res", "sea"} or "lake" in value
            for value in normalized
        ):
            semantic = "lake-or-reservoir"
        else:
            types = [str(item["associated_feature_type"]) for item in tokens]
            semantic = max(sorted(set(types)), key=types.count)
        output.append(
            {
                "id": f"named-feature-{feature_index:03d}",
                "name_verbatim_ocr": " ".join(
                    str(item["text_verbatim_ocr"]) for item in tokens
                ),
                "normalized_tokens": normalized,
                "source_token_ids": [str(item["id"]) for item in tokens],
                "source_center": [
                    float(np.mean([item["source_center"][axis] for item in tokens]))
                    for axis in (0, 1)
                ],
                "source_orientation_degrees": orientation,
                "associated_feature_type": semantic,
                "minimum_ocr_confidence": min(float(item["confidence"]) for item in tokens),
                "token_grouping_ambiguous": (
                    len(tokens) > 3
                    or sum(
                        value in {"lake", "reservoir", "res", "sea"}
                        for value in normalized
                    )
                    > 1
                ),
                "ocr_ambiguity_preserved": any(
                    bool(item["spelling_ambiguous"]) for item in tokens
                ),
            }
        )
    return tuple(output)


def _detect_dotted_chains(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    candidate_ids = []
    for component_id in range(1, count):
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if 2 <= area <= 90 and width <= 20 and height <= 20:
            candidate_ids.append(component_id)
    selected_ids: list[int] = []
    points = centroids[candidate_ids] if candidate_ids else np.empty((0, 2))
    for local_index, component_id in enumerate(candidate_ids):
        offsets = points - points[local_index]
        distances = np.linalg.norm(offsets, axis=1)
        nearby = offsets[(distances >= 3) & (distances <= 42)]
        if len(nearby) < 2:
            continue
        covariance = nearby.T @ nearby / len(nearby)
        eigenvalues = np.linalg.eigvalsh(covariance)
        anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], 0.25))
        if anisotropy >= 3.0:
            selected_ids.append(component_id)
    selected = np.isin(labels, selected_ids) if selected_ids else np.zeros(mask.shape, bool)
    return selected, {
        "candidate_component_count": len(candidate_ids),
        "selected_component_count": len(selected_ids),
        "selected_pixel_count": int(np.count_nonzero(selected)),
        "method": "small-component-local-collinearity",
    }


def _compact_ink_in_review_regions(
    observed: np.ndarray, review_regions: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Withhold only glyph-sized components from morphology review corridors.

    A corridor can cross a legitimate river.  Claiming every blue pixel in the
    corridor as text would erase that river, so connected components that look
    like long linework remain geometry evidence.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        observed.astype(np.uint8), 8
    )
    accepted: list[int] = []
    rejected: list[int] = []
    for component_id in range(1, count):
        component = labels == component_id
        if not np.any(component & review_regions):
            continue
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        aspect = max(width, height) / max(min(width, height), 1)
        extent = area / max(width * height, 1)
        overlap = np.count_nonzero(component & review_regions) / max(area, 1)
        if (
            3 <= area <= 240
            and max(width, height) <= 32
            and aspect <= 6.0
            and extent >= 0.08
            and overlap >= 0.7
        ):
            accepted.append(component_id)
        else:
            rejected.append(component_id)
    mask = np.isin(labels, accepted) if accepted else np.zeros(observed.shape, bool)
    return mask, {
        "accepted_compact_component_count": len(accepted),
        "rejected_long_or_large_component_count": len(rejected),
        "accepted_pixel_count": int(np.count_nonzero(mask)),
    }


def _closed_water_polygons(
    complete_solid: np.ndarray,
    source_rgb: np.ndarray,
    dry_label_centers: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    closed = cv2.morphologyEx(
        complete_solid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    wet_outline = np.zeros(complete_solid.shape, dtype=bool)
    wet_interior = np.zeros(complete_solid.shape, dtype=bool)
    dry_outline = np.zeros(complete_solid.shape, dtype=bool)
    dry_interior = np.zeros(complete_solid.shape, dtype=bool)
    candidates: list[dict[str, Any]] = []
    if hierarchy is None:
        return wet_outline, wet_interior, dry_outline, dry_interior, []
    for contour_id, contour in enumerate(contours):
        parent = int(hierarchy[0, contour_id, 3])
        area = float(abs(cv2.contourArea(contour)))
        x, y, width, height = cv2.boundingRect(contour)
        if parent < 0 or area < 28 or area > complete_solid.size * 0.025:
            continue
        if width < 7 or height < 5 or max(width, height) / max(min(width, height), 1) > 14:
            continue
        interior = np.zeros(complete_solid.shape, dtype=np.uint8)
        cv2.drawContours(interior, [contour], -1, 1, thickness=cv2.FILLED)
        interior_bool = interior.astype(bool) & ~complete_solid
        if np.count_nonzero(interior_bool) < 18:
            continue
        outline = cv2.dilate(interior, np.ones((3, 3), np.uint8)).astype(bool) & complete_solid
        center = np.asarray([x + width / 2, y + height / 2], dtype=np.float64)
        candidates.append(
            {
                "source_bbox": [x, y, width, height],
                "source_center": center,
                "outline": outline,
                "interior": interior_bool,
                "interior_pixel_count": int(np.count_nonzero(interior_bool)),
                "outline_pixel_count": int(np.count_nonzero(outline)),
                "interior_median_rgb": [
                    int(v) for v in np.median(source_rgb[interior_bool], axis=0).round()
                ],
            }
        )
    dry_candidate_ids: set[int] = set()
    unused_candidate_ids = set(range(len(candidates)))
    for label_center in dry_label_centers:
        label_center_array = np.asarray(label_center, dtype=np.float64)
        distances = sorted(
            (
                float(np.linalg.norm(candidate["source_center"] - label_center_array)),
                index,
            )
            for index, candidate in enumerate(candidates)
            if index in unused_candidate_ids
        )
        if distances and distances[0][0] <= 125.0:
            _distance, index = distances[0]
            dry_candidate_ids.add(index)
            unused_candidate_ids.remove(index)
    records = []
    for index, candidate in enumerate(candidates):
        nearest_dry = min(
            (
                float(
                    np.linalg.norm(
                        candidate["source_center"] - np.asarray(value, dtype=np.float64)
                    )
                )
                for value in dry_label_centers
            ),
            default=float("inf"),
        )
        is_dry = index in dry_candidate_ids
        if is_dry:
            dry_outline |= candidate["outline"]
            dry_interior |= candidate["interior"]
        else:
            wet_outline |= candidate["outline"]
            wet_interior |= candidate["interior"]
        records.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"source_center", "outline", "interior"}
            }
            | {
                "semantic_type": "dry-lake" if is_dry else "lake-or-reservoir",
                "nearest_literal_dry_label_distance_px": nearest_dry,
            }
        )
    return wet_outline, wet_interior, dry_outline, dry_interior, records


def _assign_label_semantics(
    labels: Sequence[Mapping[str, Any]],
    river: np.ndarray,
    dry_stream: np.ndarray,
    wet_lake: np.ndarray,
    dry_lake: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    def distances_to(feature: np.ndarray) -> np.ndarray:
        if not np.any(feature):
            return np.full(feature.shape, np.inf, dtype=np.float32)
        return cv2.distanceTransform((~feature).astype(np.uint8), cv2.DIST_L2, 5)

    distance = {
        "river-or-stream": distances_to(river),
        "dry-streambed": distances_to(dry_stream),
        "lake-or-reservoir": distances_to(wet_lake),
        "dry-lake": distances_to(dry_lake),
    }
    output = []
    for label in labels:
        x, y = (int(round(v)) for v in label["source_center"])
        x = int(np.clip(x, 0, river.shape[1] - 1))
        y = int(np.clip(y, 0, river.shape[0] - 1))
        normalized = str(label["normalized_token"])
        if normalized in {"dry", "lakebed"}:
            semantic = "dry-lake"
        elif "lake" in normalized or normalized in {"reservoir", "res"}:
            semantic = "lake-or-reservoir"
        else:
            values = {key: float(value[y, x]) for key, value in distance.items()}
            semantic = min(values, key=values.get)
        output.append({**dict(label), "associated_feature_type": semantic})
    return tuple(output)


def _source_diagnostic(evidence: RiverSourceEvidence) -> np.ndarray:
    faded = np.rint(evidence.source_rgb.astype(np.float32) * 0.32 + 255 * 0.68).astype(np.uint8)
    semantic = np.zeros_like(evidence.source_rgb)
    semantic[evidence.river_or_stream] = (0, 125, 255)
    semantic[evidence.lake_or_reservoir_outline] = (0, 220, 255)
    semantic[evidence.lake_or_reservoir_interior] = (100, 210, 255)
    semantic[evidence.dry_streambed] = (255, 165, 0)
    semantic[evidence.dry_lake_outline] = (255, 210, 0)
    semantic[evidence.dry_lake_interior] = (255, 235, 120)
    semantic[evidence.inferred_reconnections] = (0, 255, 80)
    review = faded.copy()
    present = np.any(semantic > 0, axis=2)
    review[present] = semantic[present]
    review[evidence.text_mask] = (235, 0, 210)
    review[evidence.ambiguous_text_mask] = (255, 70, 0)
    panels = [evidence.source_rgb, semantic, review]
    maximum_height = 1500
    scale = min(1.0, maximum_height / evidence.source_rgb.shape[0])
    if scale < 1:
        size = (
            round(evidence.source_rgb.shape[1] * scale),
            round(evidence.source_rgb.shape[0] * scale),
        )
        panels = [cv2.resize(panel, size, interpolation=cv2.INTER_AREA) for panel in panels]
    output = np.concatenate(panels, axis=1)
    canvas = Image.fromarray(output)
    draw = ImageDraw.Draw(canvas)
    names = (
        "pristine source",
        "source-native hydrography classes",
        "audit: magenta OCR text, orange unresolved, green inference",
    )
    width = panels[0].shape[1]
    for index, name in enumerate(names):
        x = index * width + 12
        draw.rectangle((x - 4, 8, x + 390, 36), fill=(0, 0, 0))
        draw.text((x, 12), name, fill=(255, 255, 255))
    return np.asarray(canvas)


def _extract_source_evidence(
    source_rgb: np.ndarray,
    source_scope: np.ndarray,
    config: RiverSourceConfig,
) -> RiverSourceEvidence:
    observed = _blue_ink(source_rgb, source_scope, config)
    selected, ambiguous_detections, ocr_report = _rotation_aware_ocr(
        source_rgb, observed, config
    )
    confirmed_glyphs, glyph_report = extract_confirmed_glyph_components(
        observed, selected
    )
    text_regions, expansion_report = expand_confirmed_text_regions(
        observed, confirmed_glyphs, selected
    )
    text_regions &= observed
    ambiguous_regions, ambiguous_glyph_report = extract_confirmed_glyph_components(
        observed, ambiguous_detections
    )
    ambiguous_regions &= observed
    ambiguous_regions &= ~text_regions
    _text_like, morphology_regions, morphology_report = (
        detect_rotation_aware_text_like_regions(observed)
    )
    # The rotation-aware morphology is review-only. It removes an unrecognised
    # word (for example a curved river name) from publishable geometry without
    # claiming that the glyphs are a successfully read label.
    morphology_glyphs, morphology_glyph_report = _compact_ink_in_review_regions(
        observed, morphology_regions
    )
    ambiguous_regions |= morphology_glyphs & ~text_regions
    remaining = observed & ~text_regions & ~ambiguous_regions
    dry_stream, dry_report = _detect_dotted_chains(remaining)
    solid_candidate = remaining & ~dry_stream
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        solid_candidate.astype(np.uint8), 8
    )
    solid_ids = [
        component_id
        for component_id in range(1, count)
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= 8
        and max(
            int(stats[component_id, cv2.CC_STAT_WIDTH]),
            int(stats[component_id, cv2.CC_STAT_HEIGHT]),
        )
        >= 5
    ]
    solid = np.isin(component_labels, solid_ids)
    inference, inferred_geometries, inference_rejected = infer_reconnections(
        solid, observed, text_regions, maximum_gap_px=config.maximum_reconnection_gap_px
    )
    labels = _deduplicate_labels(selected)
    dry_label_centers = [
        label["source_center"]
        for label in labels
        if label["normalized_token"] in {"dry", "lakebed"}
    ]
    # Ambiguous OCR boxes are not allowed to become ordinary line geometry, but
    # their pixels may close a source-observed lake outline. The closed-polygon
    # detector is therefore given every non-confirmed-text blue pixel; only its
    # closed outlines are promoted out of the unresolved evidence channel.
    polygon_candidate = (observed & ~text_regions & ~dry_stream) | inference
    wet_outline, wet_interior, dry_outline, dry_interior, polygon_records = (
        _closed_water_polygons(polygon_candidate, source_rgb, dry_label_centers)
    )
    # A source pixel has exactly one observed semantic owner. Nested contour
    # candidates can otherwise assign the same outline to a wet and dry basin.
    wet_outline &= ~dry_outline
    wet_interior &= ~dry_interior
    ambiguous_regions &= ~wet_outline & ~dry_outline
    river = solid & ~wet_outline & ~dry_outline
    labels = _assign_label_semantics(
        labels,
        river,
        dry_stream,
        wet_outline | wet_interior,
        dry_outline | dry_interior,
    )
    named_features = _group_named_features(labels)
    claimed = text_regions | ambiguous_regions | river | dry_stream | wet_outline | dry_outline
    unresolved_blue = observed & ~claimed
    ambiguous_regions |= unresolved_blue
    partition = text_regions | ambiguous_regions | river | dry_stream | wet_outline | dry_outline
    partition_valid = bool(np.array_equal(partition, observed))
    return RiverSourceEvidence(
        source_rgb=source_rgb,
        source_scope=source_scope,
        observed_blue=observed,
        text_mask=text_regions,
        ambiguous_text_mask=ambiguous_regions,
        river_or_stream=river,
        lake_or_reservoir_outline=wet_outline,
        lake_or_reservoir_interior=wet_interior,
        dry_streambed=dry_stream,
        dry_lake_outline=dry_outline,
        dry_lake_interior=dry_interior,
        inferred_reconnections=inference,
        labels=labels,
        named_features=named_features,
        ocr_candidates=tuple([*selected, *ambiguous_detections]),
        reports={
            "ocr": ocr_report,
            "confirmed_ocr_glyphs": glyph_report,
            "confirmed_ocr_expansion": expansion_report,
            "ambiguous_ocr_glyphs": ambiguous_glyph_report,
            "unresolved_rotation_aware_text": morphology_report,
            "unresolved_rotation_aware_glyphs": morphology_glyph_report,
            "dotted_streambed": dry_report,
            "closed_water_polygons": polygon_records,
            "inferred_reconnections": {
                "pixel_count": int(np.count_nonzero(inference)),
                "geometry_count": len(inferred_geometries),
                "geometries": inferred_geometries,
                "rejected": inference_rejected,
            },
            "partition_valid": partition_valid,
            "source_partition_reconstruction": {
                "observed_blue_pixel_count": int(np.count_nonzero(observed)),
                "reconstructed_blue_pixel_count": int(np.count_nonzero(partition)),
                "mismatch_pixel_count": int(np.count_nonzero(observed ^ partition)),
                "match_fraction": float(np.mean(observed == partition)),
            },
            "irreducible_semantics": [
                "The source uses one solid-line style for ordinary rivers and streams; they cannot be separated without external data.",
                "The source uses one closed-outline style for lakes and reservoirs; they cannot be separated without label interpretation or external data.",
                "OCR labels are retained verbatim with alternatives; ambiguous or truncated source lettering is never silently corrected.",
                "Dense neighbouring label tokens can have more than one reading order or grouping; those cases are explicitly flagged rather than corrected from outside data.",
            ],
        },
    )


def prepare_river_source_semantics(
    source_adapter_manifest_path: Path,
    output_dir: Path,
    *,
    config: RiverSourceConfig = RiverSourceConfig(),
) -> RiverPreparationResult:
    """Create source-only readiness evidence without consuming an alignment."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("river source preparation requires a fresh output directory")
    working = _validate_source_adapter(source_adapter_manifest_path.resolve())
    output_dir.mkdir(parents=True)
    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))
    # Source-only preparation excludes the automatically detected Pacific
    # component and its printed legend. Exact California clipping is deferred to
    # the accepted Mapbox transform.
    pacific = _large_pale_blue_water(source_rgb).astype(np.uint8)
    pacific = cv2.morphologyEx(
        pacific,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    ).astype(bool)
    source_scope = ~pacific
    evidence = _extract_source_evidence(source_rgb, source_scope, config)
    paths = {
        "observed_blue": output_dir / "source-observed-blue.png",
        "text": output_dir / "source-ocr-text-mask.png",
        "ambiguous_text": output_dir / "source-unresolved-blue-mask.png",
        "river_or_stream": output_dir / "source-river-or-stream.png",
        "lake_outline": output_dir / "source-lake-or-reservoir-outline.png",
        "lake_interior": output_dir / "source-lake-or-reservoir-inferred-interior.png",
        "dry_streambed": output_dir / "source-dry-streambed.png",
        "dry_lake_outline": output_dir / "source-dry-lake-outline.png",
        "dry_lake_interior": output_dir / "source-dry-lake-inferred-interior.png",
        "reconnections": output_dir / "source-inferred-reconnections.png",
        "reconstruction": output_dir / "source-blue-semantic-reconstruction.png",
        "reconstruction_diff": output_dir / "source-blue-partition-diff.png",
    }
    for key, values in (
        ("observed_blue", evidence.observed_blue),
        ("text", evidence.text_mask),
        ("ambiguous_text", evidence.ambiguous_text_mask),
        ("river_or_stream", evidence.river_or_stream),
        ("lake_outline", evidence.lake_or_reservoir_outline),
        ("lake_interior", evidence.lake_or_reservoir_interior),
        ("dry_streambed", evidence.dry_streambed),
        ("dry_lake_outline", evidence.dry_lake_outline),
        ("dry_lake_interior", evidence.dry_lake_interior),
        ("reconnections", evidence.inferred_reconnections),
        (
            "reconstruction",
            evidence.text_mask
            | evidence.ambiguous_text_mask
            | evidence.river_or_stream
            | evidence.lake_or_reservoir_outline
            | evidence.dry_streambed
            | evidence.dry_lake_outline,
        ),
        (
            "reconstruction_diff",
            evidence.observed_blue
            ^ (
                evidence.text_mask
                | evidence.ambiguous_text_mask
                | evidence.river_or_stream
                | evidence.lake_or_reservoir_outline
                | evidence.dry_streambed
                | evidence.dry_lake_outline
            ),
        ),
    ):
        _save_mask(paths[key], values)
    labels_path = output_dir / "source-native-labels.json"
    labels_path.write_text(
        json.dumps(
            {"label_tokens": evidence.labels, "named_features": evidence.named_features},
            indent=2,
        )
        + "\n"
    )
    diagnostic_path = output_dir / "source-separation-diagnostic.png"
    _save_rgb(diagnostic_path, _source_diagnostic(evidence))
    feature_counts = {
        "river_or_stream_observed": int(np.count_nonzero(evidence.river_or_stream)),
        "lake_or_reservoir_outline_observed": int(
            np.count_nonzero(evidence.lake_or_reservoir_outline)
        ),
        "lake_or_reservoir_interior_inferred": int(
            np.count_nonzero(evidence.lake_or_reservoir_interior)
        ),
        "dry_streambed_observed": int(np.count_nonzero(evidence.dry_streambed)),
        "dry_lake_outline_observed": int(np.count_nonzero(evidence.dry_lake_outline)),
        "dry_lake_interior_inferred": int(np.count_nonzero(evidence.dry_lake_interior)),
    }
    manifest_path = output_dir / "source-preparation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ready_for_accepted_alignment",
                "source": {"path": str(working.source_path), "sha256": working.source_sha256},
                "source_adapter": {
                    "path": str(working.manifest_path),
                    "sha256": _sha256(working.manifest_path),
                },
                "alignment_used": False,
                "manual_inputs_used": False,
                "prior_run_artifacts_used": False,
                "feature_pixel_counts": feature_counts,
                "named_feature_count": len(evidence.named_features),
                "ocr_label_token_count": len(evidence.labels),
                "labels": labels_path.name,
                "reports": evidence.reports,
                "artifacts": [_artifact(path, output_dir) for path in [*paths.values(), labels_path, diagnostic_path]],
            },
            indent=2,
        )
        + "\n"
    )
    return RiverPreparationResult(
        "ready_for_accepted_alignment",
        manifest_path,
        diagnostic_path,
        len(evidence.named_features),
        feature_counts,
        tuple(evidence.reports["irreducible_semantics"]),
    )


def _warp_target_to_source(
    target: np.ndarray, source_to_target_remap: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    return cv2.remap(
        target,
        source_to_target_remap[0],
        source_to_target_remap[1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _load_state_interior(
    mapbox_manifest_path: Path, expected_shape: tuple[int, int]
) -> np.ndarray:
    """Load the pinned California footprint including its internal waters."""

    manifest_path = mapbox_manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest.get("artifacts", {}).get("state_interior_mask")
    if not isinstance(artifact, Mapping):
        raise ValueError("Mapbox reference lacks the pinned state interior mask")
    relative = Path(str(artifact.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Mapbox state interior artifact path is unsafe")
    path = (manifest_path.parent / relative).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("Mapbox state interior artifact escapes its reference") from error
    expected_hash = str(artifact.get("sha256", ""))
    if not expected_hash or _sha256(path) != expected_hash:
        raise ValueError("Pinned Mapbox state interior hash mismatch")
    values = np.asarray(Image.open(path).convert("L")) > 0
    if values.shape != expected_shape or not np.any(values):
        raise ValueError("Pinned Mapbox state interior shape or content is invalid")
    return values


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union_count = int(np.count_nonzero(left | right))
    if union_count == 0:
        # An optional inferred channel can legitimately be absent.  Replaying an
        # empty channel as empty is an exact round trip, not a zero-overlap
        # failure.  Non-empty-vs-empty remains zero through the normal formula.
        return 1.0
    return float(np.count_nonzero(left & right) / union_count)


def _roundtrip_channel_gates(
    channel_names: Sequence[str],
    source_channels: Sequence[np.ndarray],
    reconstructed_channels: Sequence[np.ndarray],
    minimum_iou: float,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Gate supported channels and independently prove optional empties stay empty."""

    ious = [
        _iou(source, reconstructed)
        for source, reconstructed in zip(source_channels, reconstructed_channels)
    ]
    supported = [bool(np.any(channel)) for channel in source_channels]
    evaluated = [name for name, present in zip(channel_names, supported) if present]
    empty = [name for name, present in zip(channel_names, supported) if not present]
    supported_values = [value for value, present in zip(ious, supported) if present]
    empty_stays_empty = all(
        not bool(np.any(reconstructed))
        for present, reconstructed in zip(supported, reconstructed_channels)
        if not present
    )
    return (
        ious,
        {
            "passed": bool(evaluated)
            and all(value >= minimum_iou for value in supported_values),
            "values": ious,
            "minimum": minimum_iou,
            "evaluated_source_supported_channels": evaluated,
            "excluded_zero_support_optional_channels": empty,
        },
        {
            "passed": empty_stays_empty,
            "channels": empty,
        },
    )


def _signature(channels: Sequence[np.ndarray]) -> np.ndarray:
    signature = np.zeros(channels[0].shape, dtype=np.uint16)
    for index, channel in enumerate(channels):
        signature[channel] |= np.uint16(1 << index)
    return signature


def _regional_metrics(
    source_signature: np.ndarray,
    reconstructed_signature: np.ndarray,
    remap: tuple[np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    map_x, map_y = remap
    target_height, target_width = target_shape
    support = source_signature > 0
    output = []
    for row in range(rows):
        for column in range(columns):
            cell = (
                support
                & (map_x >= column * target_width / columns)
                & (map_x < (column + 1) * target_width / columns)
                & (map_y >= row * target_height / rows)
                & (map_y < (row + 1) * target_height / rows)
            )
            count = int(np.count_nonzero(cell))
            if not count:
                continue
            matched = int(np.count_nonzero(cell & (source_signature == reconstructed_signature)))
            output.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "expected_pixel_count": count,
                    "matched_pixel_count": matched,
                    "match_fraction": matched / count,
                }
            )
    return output


def _resumable_prior_extraction_count(experiment_log: NoHumanExperimentLog) -> int:
    """Return the next base ordinal while rejecting accepted or non-automatic history."""

    extraction = experiment_log.data["extraction"]
    if extraction.get("accepted_automatic_iteration_count") is not None:
        raise ValueError("rivers extraction is already accepted")
    iterations = extraction.get("iterations", [])
    for expected, item in enumerate(iterations, start=1):
        provenance = item.get("provenance", {})
        if (
            item.get("decision") not in {"retry", "blocked"}
            or item.get("counts_toward_automatic_iteration_count") is not True
            or item.get("automatic_iteration") != expected
            or provenance.get("actor_kind") != "automated"
            or provenance.get("manual_arrows") is not False
            or provenance.get("manual_stamps") is not False
            or provenance.get("human_approval") is not False
        ):
            raise ValueError(
                "rivers extraction can resume only prior rejected automatic attempts"
            )
    return len(iterations)


def run_named_hydrography_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: RiverSourceConfig = RiverSourceConfig(),
) -> RiverExtractionResult:
    """Run the publishable pass; reject immediately without accepted alignment."""

    if experiment_log.data.get("map_id") != "rivers":
        raise ValueError("named hydrography extractor requires the rivers experiment log")
    if experiment_log.data["source"].get("source_type") != "named_linear_and_polygon_features_without_legend":
        raise ValueError("rivers source data model is not named_linear_and_polygon_features_without_legend")
    accepted_count = experiment_log.data["alignment"]["accepted_automatic_iteration_count"]
    if accepted_count is None:
        raise ValueError("named hydrography extraction requires accepted automatic alignment")
    prior_iteration_count = _resumable_prior_extraction_count(experiment_log)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("named hydrography extraction requires a fresh output directory")
    for path, kind in (
        (accepted_alignment_path, "accepted alignment"),
        (mapbox_manifest_path, "Mapbox manifest"),
    ):
        _assert_clean_path(path, kind)
    working = _validate_source_adapter(source_adapter_manifest_path.resolve())
    if experiment_log.data["source"].get("sha256") != working.source_sha256:
        raise ValueError("experiment and source-clean source hashes disagree")
    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    alignment = _load_accepted_alignment(
        accepted_alignment_path,
        working.working_raster_path,
        reference.grid,
        reference.pin,
        accepted_iteration_count=int(accepted_count),
        map_id=str(experiment_log.data["map_id"]),
        reference_revisions=experiment_log.data.get("mapbox_reference_revisions", []),
        alignment_iterations=experiment_log.data["alignment"]["iterations"],
    )
    output_dir.mkdir(parents=True)
    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))
    transform = alignment["transform"]
    source_to_target = _source_to_reference_remap(transform, source_rgb.shape[:2])
    reference_to_source = _reference_to_source_remap(transform)
    target_domain = _load_state_interior(
        mapbox_manifest_path, reference.state_land.shape
    )
    source_scope = _warp_target_to_source(
        target_domain.astype(np.uint8), source_to_target
    ) > 0
    evidence = _extract_source_evidence(source_rgb, source_scope, config)
    source_channels = [
        evidence.river_or_stream,
        evidence.lake_or_reservoir_outline,
        evidence.lake_or_reservoir_interior,
        evidence.dry_streambed,
        evidence.dry_lake_outline,
        evidence.dry_lake_interior,
        evidence.inferred_reconnections,
    ]
    channel_names = [
        "river-or-stream-observed",
        "lake-or-reservoir-outline-observed",
        "lake-or-reservoir-interior-inferred",
        "dry-streambed-observed",
        "dry-lake-outline-observed",
        "dry-lake-interior-inferred",
        "line-reconnection-inferred",
    ]
    previous_source: np.ndarray | None = None
    previous_target: np.ndarray | None = None
    iterations = []
    accepted_path = None
    for replay_number in range(1, config.required_replay_count + 1):
        iteration_number = prior_iteration_count + replay_number
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        target_channels = [
            _source_to_reference(
                channel.astype(np.uint8), transform, cv2.INTER_NEAREST, 0, reference_to_source
            )
            > 0
            for channel in source_channels
        ]
        target_channels = [channel & target_domain for channel in target_channels]
        reconstructed = [
            _warp_target_to_source(channel.astype(np.uint8), source_to_target) > 0
            for channel in target_channels
        ]
        (
            ious,
            channel_roundtrip_gate,
            optional_empty_roundtrip_gate,
        ) = _roundtrip_channel_gates(
            channel_names,
            source_channels,
            reconstructed,
            config.minimum_channel_roundtrip_iou,
        )
        source_signature = _signature(source_channels)
        target_signature = _signature(target_channels)
        reconstructed_signature = _warp_target_to_source(target_signature, source_to_target)
        support = source_signature > 0
        composite = float(
            np.count_nonzero(support & (source_signature == reconstructed_signature))
            / max(np.count_nonzero(support), 1)
        )
        regional = _regional_metrics(
            source_signature,
            reconstructed_signature,
            source_to_target,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            item["match_fraction"] >= config.minimum_geographic_match_fraction
            for item in regional
        )
        stable = (
            previous_source is not None
            and previous_target is not None
            and np.array_equal(previous_source, source_signature)
            and np.array_equal(previous_target, target_signature)
        )
        feature_nonempty = {
            "river-or-stream": bool(np.any(target_channels[0])),
            "lake-or-reservoir": bool(np.any(target_channels[1])),
            "dry-streambed": bool(np.any(target_channels[3])),
            "dry-lake": bool(np.any(target_channels[4])),
        }
        gates: dict[str, Any] = {
            "minimum_source_native_label_count": {
                "passed": len(evidence.named_features) >= config.minimum_named_label_count,
                "value": len(evidence.named_features),
                "minimum": config.minimum_named_label_count,
            },
            "source_partition_exact": bool(evidence.reports["partition_valid"]),
            "all_source_supported_feature_types_nonempty": all(feature_nonempty.values()),
            "text_excluded_from_geometry": not bool(
                np.any(
                    evidence.text_mask
                    & (
                        evidence.river_or_stream
                        | evidence.lake_or_reservoir_outline
                        | evidence.dry_streambed
                        | evidence.dry_lake_outline
                    )
                )
            ),
            "channel_source_roundtrip": channel_roundtrip_gate,
            "optional_empty_channels_stay_empty": optional_empty_roundtrip_gate,
            "source_composite_roundtrip": {
                "passed": composite >= config.minimum_composite_roundtrip_fraction,
                "value": composite,
                "minimum": config.minimum_composite_roundtrip_fraction,
            },
            "geographic_source_diff": {
                "passed": passing_cells >= config.minimum_passing_geographic_cells,
                "value": passing_cells,
                "minimum": config.minimum_passing_geographic_cells,
                "supported_cells": len(regional),
            },
            "mapbox_exterior_empty": not any(
                np.any(channel[~target_domain]) for channel in target_channels
            ),
            "successive_all_channel_fixed_point": stable,
        }
        passed = all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        )
        decision = "accept" if passed else ("retry" if replay_number == 1 else "blocked")
        source_paths = []
        target_paths = []
        for name, source_channel, target_channel in zip(channel_names, source_channels, target_channels):
            source_path = iteration_dir / f"source-{name}.png"
            target_path = iteration_dir / f"mapbox-{name}.png"
            _save_mask(source_path, source_channel)
            _save_mask(target_path, target_channel)
            source_paths.append(source_path)
            target_paths.append(target_path)
        text_path = iteration_dir / "source-ocr-text-mask.png"
        ambiguous_path = iteration_dir / "source-unresolved-blue-mask.png"
        labels_path = iteration_dir / "source-native-labels.json"
        diff_path = iteration_dir / "source-semantic-roundtrip-diff.png"
        diagnostic_path = iteration_dir / "source-separation-diagnostic.png"
        _save_mask(text_path, evidence.text_mask)
        _save_mask(ambiguous_path, evidence.ambiguous_text_mask)
        labels_path.write_text(
            json.dumps(
                {
                    "label_tokens": evidence.labels,
                    "named_features": evidence.named_features,
                },
                indent=2,
            )
            + "\n"
        )
        _save_mask(diff_path, support & (source_signature != reconstructed_signature))
        _save_rgb(diagnostic_path, _source_diagnostic(evidence))
        scores = {
            "named_feature_count": len(evidence.named_features),
            "ocr_label_token_count": len(evidence.labels),
            "feature_nonempty": feature_nonempty,
            "source_feature_pixel_counts": {
                name: int(np.count_nonzero(channel))
                for name, channel in zip(channel_names, source_channels)
            },
            "mapbox_feature_pixel_counts": {
                name: int(np.count_nonzero(channel))
                for name, channel in zip(channel_names, target_channels)
            },
            "channel_source_roundtrip_ious": dict(zip(channel_names, ious)),
            "source_composite_roundtrip_fraction": composite,
            "geographic_cells": regional,
            "successive_source_signature_equal": stable,
            "successive_mapbox_signature_equal": stable,
            "irreducible_semantics": evidence.reports["irreducible_semantics"],
        }
        report_path = iteration_dir / "iteration.json"
        artifacts = [*source_paths, *target_paths, text_path, ambiguous_path, labels_path, diff_path, diagnostic_path]
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "iteration": iteration_number,
                    "decision": decision,
                    "scores": scores,
                    "gates": gates,
                    "provenance": {
                        "source": {"path": str(working.source_path), "sha256": working.source_sha256},
                        "source_adapter": {"path": str(working.manifest_path), "sha256": _sha256(working.manifest_path)},
                        "accepted_alignment": {"path": str(accepted_alignment_path.resolve()), "sha256": _sha256(accepted_alignment_path.resolve())},
                        "mapbox_manifest": {"path": str(mapbox_manifest_path.resolve()), "sha256": _sha256(mapbox_manifest_path.resolve())},
                        "manual_inputs_used": False,
                        "prior_run_artifacts_used": False,
                    },
                    "artifacts": [_artifact(path, output_dir) for path in artifacts],
                },
                indent=2,
            )
            + "\n"
        )
        experiment_log.record_extraction_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=automatic_provenance(
                PRODUCER,
                [
                    "authoritative_original_jpeg_pixels",
                    "rotation_aware_source_native_ocr_labels",
                    "source_blue_geometry_text_partition",
                    "accepted_automatic_mapbox_alignment",
                    "pinned_mapbox_land_and_water",
                    "source_reconstruction_and_geographic_diff",
                    "deterministic_fixed_point_replay",
                ],
            ),
            method=(
                "automatic legend-independent named hydrography separation into solid "
                "river/stream, lake/reservoir, dry-streambed, and dry-lake channels"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in [*artifacts, report_path]
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iterations.append(RiverExtractionIteration(iteration_number, decision, scores, gates, report_path))
        previous_source = source_signature.copy()
        previous_target = target_signature.copy()
        if decision == "accept":
            accepted_path = output_dir / "accepted-extraction.json"
            accepted_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "accepted",
                        "automatic_iteration_count": iteration_number,
                        "source": {"path": str(working.source_path), "sha256": working.source_sha256},
                        "source_adapter": {
                            "path": str(working.manifest_path),
                            "sha256": _sha256(working.manifest_path),
                        },
                        "alignment": {
                            "path": str(accepted_alignment_path.resolve()),
                            "sha256": _sha256(accepted_alignment_path.resolve()),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "labels": f"extraction-{iteration_number:02d}/source-native-labels.json",
                        "layers": {
                            name: f"extraction-{iteration_number:02d}/mapbox-{name}.png"
                            for name in channel_names
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            experiment_log.finalize("complete")
            experiment_log.write(experiment_markdown_path, experiment_json_path)
            break
    if accepted_path is not None:
        return RiverExtractionResult(
            "accepted",
            "source labels, feature partitions, reconstruction, geographic, and fixed-point gates passed",
            tuple(iterations),
            accepted_path,
        )
    blocker = "Named hydrography extraction did not pass every source semantic, reconstruction, geographic, and fixed-point gate"
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return RiverExtractionResult("blocked", blocker, tuple(iterations), None)
