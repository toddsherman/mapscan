from __future__ import annotations

import cv2
import numpy as np

from mapscan.county_reference import _extract_styled_lines


def test_styled_county_reference_separates_thick_state_and_thin_county_lines() -> None:
    rgba = np.zeros((120, 140, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (20, 10), (110, 105), (0, 0, 0, 255), 5)
    cv2.line(rgba, (22, 55), (108, 55), (51, 51, 51, 255), 2)
    cv2.line(rgba, (65, 12), (65, 103), (51, 51, 51, 255), 2)
    # A black watermark-like component must not be mistaken for the state.
    cv2.putText(rgba, "x", (120, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1)

    state, counties, interior = _extract_styled_lines(rgba)

    assert state[10, 50]
    assert state[60, 20]
    assert not state[110, 125]
    assert counties[55, 50]
    assert counties[50, 65]
    assert not counties[10, 50]
    assert interior[50, 50]
    assert not interior[5, 5]
