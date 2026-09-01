import numpy as np

from mapscan.label_detection import isolate_neutral_dark_labels


def test_neutral_label_isolation_rejects_colored_data_and_outside_pixels():
    rgb = np.full((8, 10, 3), 240, dtype=np.uint8)
    rgb[2, 2] = [40, 45, 42]
    rgb[2, 4] = [30, 30, 130]
    rgb[2, 6] = [35, 35, 35]
    valid = np.ones((8, 10), dtype=bool)
    valid[2, 6] = False
    page = isolate_neutral_dark_labels(rgb, valid, closing_size_px=1)
    assert page[2, 2] == 0
    assert page[2, 4] == 255
    assert page[2, 6] == 255
    assert page[0, 0] == 255
