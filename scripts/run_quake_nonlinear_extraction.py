from pathlib import Path

from mapscan.quake_nonlinear_extraction import run_quake_nonlinear_extraction

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(run_quake_nonlinear_extraction(
        ROOT / "runs/mapbox-autonomous-restart-v1/quake",
        ROOT / "reference/mapbox-light-v11-california-z9-v2/manifest.json",
        ROOT / "runs/mapbox-native-alignment-validation-v1/quake-nonlinear/nonlinear-refinement.json",
        ROOT / "runs/mapbox-native-alignment-validation-v1/quake-nonlinear-extraction",
    ))
