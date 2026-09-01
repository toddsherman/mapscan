import json
from pathlib import Path

import numpy as np
import pytest

from mapscan.cli import _parser
from mapscan.rainfall_topology import (
    _confusable_groups,
    _infer_family_variant,
    _total_variation,
    extract_rainfall_topology,
)


def test_confusable_pairs_form_transitive_palette_families() -> None:
    groups = _confusable_groups(
        [
            {"first_category": "5.0", "second_category": "5.5"},
            {"first_category": "5.5", "second_category": "6.5"},
            {"first_category": "2.5", "second_category": "3.5"},
        ]
    )

    assert sorted(groups) == [["2.5", "3.5"], ["5.0", "5.5", "6.5"]]


def test_identical_rendered_swatches_have_zero_information_distance() -> None:
    first = {(255, 222, 181): 0.5625, (247, 222, 181): 0.4375}
    second = dict(first)
    distinct = {(255, 222, 181): 1.0}

    assert _total_variation(first, second) == 0.0
    assert _total_variation(first, distinct) > 0.0


def test_topology_assigns_only_the_endpoint_with_a_dominant_adjacent_anchor() -> None:
    visible = np.zeros((48, 72), dtype=bool)
    visible[12:36, 20:45] = True
    gray = np.full(visible.shape, 255, dtype=np.uint8)
    source_class = np.zeros(visible.shape, dtype=np.uint8)
    source_class[12:36, 46:50] = 7  # Immediately above [4, 5, 6].
    direct_known = source_class > 0

    inferred, report = _infer_family_variant(
        visible,
        gray,
        source_class,
        direct_known,
        [4, 5, 6],
        10,
        dark_threshold=160,
        boundary_radius=3,
    )

    assert np.all(inferred[visible] == 6)
    assert {item["assigned_index"] for item in report} == {6}
    assert not np.any(inferred[~visible])


def test_topology_leaves_an_unanchored_family_unresolved() -> None:
    visible = np.zeros((48, 72), dtype=bool)
    visible[12:36, 20:45] = True
    gray = np.full(visible.shape, 255, dtype=np.uint8)
    source_class = np.zeros(visible.shape, dtype=np.uint8)

    inferred, report = _infer_family_variant(
        visible,
        gray,
        source_class,
        source_class > 0,
        [4, 5, 6],
        10,
        dark_threshold=160,
        boundary_radius=3,
    )

    assert not np.any(inferred)
    assert {item["assigned_index"] for item in report} == {None}


def test_stage_refuses_to_write_inside_the_extraction_run(tmp_path: Path) -> None:
    extraction = tmp_path / "accepted-extraction"

    with pytest.raises(ValueError, match="separate from the accepted extraction"):
        extract_rainfall_topology(
            tmp_path / "plan.json",
            extraction,
            extraction / "topology",
        )


def test_stage_rejects_a_plan_that_does_not_match_the_extraction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rainfall.gif"
    source.write_bytes(b"not opened because the plan hash fails first")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"source": str(source), "alignment": str(alignment)}) + "\n"
    )
    extraction = tmp_path / "accepted-extraction"
    extraction.mkdir()
    (extraction / "extraction.json").write_text(
        json.dumps(
            {
                "plan": {"sha256": "not-the-plan-hash"},
                "source": {"sha256": "not-reached"},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="does not belong"):
        extract_rainfall_topology(plan, extraction, tmp_path / "topology-run")


def test_cli_exposes_optional_rainfall_topology_stage() -> None:
    args = _parser().parse_args(
        [
            "extract-rainfall-topology",
            "plan.json",
            "accepted-run",
            "--output",
            "topology-run",
        ]
    )

    assert args.command == "extract-rainfall-topology"
    assert args.plan == Path("plan.json")
    assert args.extraction == Path("accepted-run")
    assert args.output == Path("topology-run")
