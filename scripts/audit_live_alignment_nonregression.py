#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapscan.live_alignment_nonregression import audit_live_alignment_nonregression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/live-dataset-authority-v2.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    report = audit_live_alignment_nonregression(args.registry, args.output)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
