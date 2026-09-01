#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapscan.live_dataset_readiness import audit_live_dataset_readiness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/live-dataset-authority-v2.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("viewer/public/data/catalog.json"),
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("viewer/public/data/datasets"),
    )
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    report = audit_live_dataset_readiness(
        args.registry,
        args.catalog,
        args.public_root,
        args.alignment_audit,
        args.output,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
