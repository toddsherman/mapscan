#!/usr/bin/env python3
"""Compatibility wrapper for the productized rainfall topology stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapscan.rainfall_topology import extract_rainfall_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("extraction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract_rainfall_topology(args.plan, args.extraction, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
