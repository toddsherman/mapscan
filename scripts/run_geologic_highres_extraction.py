"""Run the source-native geologic PDF extraction on the pinned 3x grid."""

from __future__ import annotations

from pathlib import Path

from mapscan.geologic_pdf_highres import run_geologic_pdf_highres_extraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = run_geologic_pdf_highres_extraction(
        PROJECT_ROOT
        / "runs/mapbox-autonomous-restart-v1/geologic/source-clean/source-adapter.json",
        PROJECT_ROOT
        / "runs/mapbox-autonomous-restart-v1/geologic/automatic-alignment/accepted-alignment.json",
        PROJECT_ROOT / "reference/mapbox-light-v11-california-z9-v1/manifest.json",
        PROJECT_ROOT
        / "reference/mapbox-light-v11-california-z9-v2-3x/manifest.json",
        PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1/geologic/EXPERIMENT.json",
        PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1/geologic-highres-3x",
    )
    print(
        {
            "status": result.status,
            "output_root": str(result.output_root),
            "accepted_path": str(result.accepted_path) if result.accepted_path else None,
            "iterations": result.iterations,
        }
    )


if __name__ == "__main__":
    main()
