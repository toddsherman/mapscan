"""Run compact native/multiscale alignment validation for low-scale maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from mapscan.native_alignment_validation import (
    SOURCE_FAMILIES,
    run_native_alignment_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_id", choices=(*SOURCE_FAMILIES.keys(), "all"))
    arguments = parser.parse_args()
    map_ids = list(SOURCE_FAMILIES) if arguments.map_id == "all" else [arguments.map_id]
    for map_id in map_ids:
        result = run_native_alignment_validation(
            map_id,
            PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1" / map_id,
            PROJECT_ROOT / "reference/mapbox-light-v11-california-z9-v2/manifest.json",
            PROJECT_ROOT / "runs/mapbox-native-alignment-validation-v1" / map_id,
        )
        print(
            {
                "map_id": result.map_id,
                "status": result.status,
                "report": str(result.report_path),
            }
        )


if __name__ == "__main__":
    main()
