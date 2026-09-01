"""Run the no-human Farms v2 extraction on a vector-rasterized 3x grid."""

from __future__ import annotations

import copy
from pathlib import Path

from mapscan.automatic_categorical_extraction import (
    ExtractionLoopConfig,
    run_automatic_categorical_extraction,
)
from mapscan.experiment_log import NoHumanExperimentLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "examples/farmsv2.png"
BASE_RUN = PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1/farms-v2"
OUTPUT_RUN = PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1/farms-v2-highres-3x"
ALIGNMENT = BASE_RUN / "automatic-alignment/accepted-alignment.json"
BASE_REFERENCE = (
    PROJECT_ROOT / "reference/mapbox-light-v11-california-z9-v2/manifest.json"
)
PROCESSING_REFERENCE = (
    PROJECT_ROOT / "reference/mapbox-light-v11-california-z9-v2-3x/manifest.json"
)


def _fresh_highres_log() -> NoHumanExperimentLog:
    base = NoHumanExperimentLog.load(BASE_RUN / "EXPERIMENT.json")
    data = copy.deepcopy(base.data)
    data["map_id"] = "farms-v2-highres-3x"
    data["extraction"] = {
        "iterations": [],
        "accepted_automatic_iteration_count": None,
    }
    data["final"] = {"status": "in_progress", "blocker": None}
    result = NoHumanExperimentLog.__new__(NoHumanExperimentLog)
    result.data = data
    return result


def main() -> None:
    if OUTPUT_RUN.exists():
        raise FileExistsError(f"High-resolution run already exists: {OUTPUT_RUN}")
    OUTPUT_RUN.mkdir(parents=True)
    log = _fresh_highres_log()
    experiment_markdown = OUTPUT_RUN / "EXPERIMENT.md"
    experiment_json = OUTPUT_RUN / "EXPERIMENT.json"
    log.write(experiment_markdown, experiment_json)
    result = run_automatic_categorical_extraction(
        SOURCE,
        ALIGNMENT,
        BASE_REFERENCE,
        OUTPUT_RUN / "automatic-extraction",
        log,
        experiment_markdown,
        experiment_json,
        config=ExtractionLoopConfig(
            target_supersampling=3,
            compact_rejected_artifacts=True,
            compact_target_artifacts=True,
        ),
        processing_reference_manifest_path=PROCESSING_REFERENCE,
    )
    print(
        {
            "status": result.status,
            "stop_reason": result.stop_reason,
            "accepted_iteration": (
                result.accepted.iteration if result.accepted is not None else None
            ),
            "target_grid": {"width": 10192, "height": 11758},
            "minimum_nonundersampling_zoom": 11,
        }
    )


if __name__ == "__main__":
    main()
