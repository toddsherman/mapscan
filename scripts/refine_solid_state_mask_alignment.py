#!/usr/bin/env python3
"""Refine a California map alignment from exact solid thematic colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapscan.solid_mask_alignment import refine_solid_mask_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--canonical-boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    refine_solid_mask_alignment(
        args.image,
        args.alignment,
        args.reference,
        args.canonical_boundary,
        args.output,
        evidence["solid_map_rgb"],
        minimum_component_area=int(evidence.get("minimum_component_area", 500)),
        maximum_components=int(evidence.get("maximum_components", 5)),
        correspondence_gate_px=float(evidence.get("correspondence_gate_px", 30.0)),
    )


if __name__ == "__main__":
    main()
