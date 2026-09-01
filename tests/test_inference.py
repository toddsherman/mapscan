import numpy as np

from mapscan.inference import (
    infer_ocr_label_occlusions,
    infer_ocr_label_occlusions_by_nearest_class,
    infer_small_categorical_gaps,
)


def test_inference_fills_small_single_class_occlusion_and_marks_it():
    values = np.zeros((80, 100), dtype=np.uint8)
    values[10:70, 10:90] = 1
    values[35:40, 45:52] = 0
    valid = np.ones_like(values, dtype=bool)
    inferred, mask, report = infer_small_categorical_gaps(
        values,
        valid,
        class_count=1,
        max_gap_radius_px=6,
        minimum_dominance=0.95,
    )
    assert np.all(inferred[35:40, 45:52] == 1)
    assert np.all(mask[35:40, 45:52])
    assert report["inferred_pixel_count"] == 35
    assert np.array_equal(values[35:40, 45:52], np.zeros((5, 7), dtype=np.uint8))


def test_inference_rejects_gap_between_competing_classes():
    values = np.zeros((80, 100), dtype=np.uint8)
    values[10:70, 10:49] = 1
    values[10:70, 51:90] = 2
    valid = np.ones_like(values, dtype=bool)
    inferred, mask, report = infer_small_categorical_gaps(
        values,
        valid,
        class_count=2,
        max_gap_radius_px=4,
        minimum_dominance=0.90,
    )
    assert not np.any(mask[:, 49:51])
    assert np.all(inferred[:, 49:51] == 0)
    assert report["inferred_pixel_count"] == 0


def test_inference_does_not_fill_large_void():
    values = np.ones((100, 100), dtype=np.uint8)
    values[20:80, 20:80] = 0
    valid = np.ones_like(values, dtype=bool)
    inferred, mask, report = infer_small_categorical_gaps(
        values,
        valid,
        class_count=1,
        max_gap_radius_px=35,
        max_component_area_px=500,
    )
    assert not np.any(mask)
    assert np.array_equal(inferred, values)
    assert report["rejected_component_counts"]["too_large"] == 1


def test_ocr_label_inference_fills_full_padded_occlusion(tmp_path):
    values = np.ones((100, 140), dtype=np.uint8)
    values[35:65, 45:95] = 0
    valid = np.ones_like(values, dtype=bool)
    tsv = tmp_path / "labels.tsv"
    tsv.write_text(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t50\t40\t40\t20\t96.0\tRedding\n"
    )
    inferred, mask, report = infer_ocr_label_occlusions(
        values,
        valid,
        tsv,
        label_padding_px=6,
        minimum_dominance=0.95,
    )
    assert np.all(inferred[35:65, 45:95] == 1)
    assert np.all(mask[35:65, 45:95])
    assert report["accepted_label_count"] == 1
    assert report["accepted_labels"][0]["text"] == "Redding"


def test_ocr_label_inference_rejects_mixed_category_context(tmp_path):
    values = np.ones((100, 140), dtype=np.uint8)
    values[:, 70:] = 2
    values[35:65, 45:95] = 0
    valid = np.ones_like(values, dtype=bool)
    tsv = tmp_path / "labels.tsv"
    tsv.write_text(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t50\t40\t40\t20\t96.0\tBoundary\n"
    )
    inferred, mask, report = infer_ocr_label_occlusions(
        values,
        valid,
        tsv,
        label_padding_px=6,
        minimum_dominance=0.90,
    )
    assert not np.any(mask)
    assert np.array_equal(inferred, values)
    assert report["rejected_label_counts"]["mixed_context"] == 1


def test_nearest_class_ocr_inference_preserves_boundary_under_label(tmp_path):
    values = np.ones((100, 140), dtype=np.uint8)
    values[:, 70:] = 2
    values[35:65, 45:95] = 0
    valid = np.ones_like(values, dtype=bool)
    tsv = tmp_path / "labels.tsv"
    tsv.write_text(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t50\t40\t40\t20\t96.0\tBoundary\n"
    )
    inferred, mask, report = infer_ocr_label_occlusions_by_nearest_class(
        values,
        valid,
        tsv,
        label_padding_px=6,
        context_radius_px=20,
        minimum_distance_margin_px=1.0,
    )
    assert np.count_nonzero(mask[35:65, 45:70]) > 0
    assert np.count_nonzero(mask[35:65, 70:95]) > 0
    assert np.all(inferred[mask & (np.indices(values.shape)[1] < 70)] == 1)
    assert np.all(inferred[mask & (np.indices(values.shape)[1] >= 70)] == 2)
    assert report["accepted_label_count"] == 1
    assert report["inferred_pixels_by_class_id"]["1"] > 0
    assert report["inferred_pixels_by_class_id"]["2"] > 0


def test_nearest_class_ocr_reader_does_not_swallow_rows_after_quote(tmp_path):
    values = np.ones((100, 180), dtype=np.uint8)
    values[25:75, 15:165] = 0
    valid = np.ones_like(values, dtype=bool)
    tsv = tmp_path / "labels.tsv"
    tsv.write_text(
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        '5\t1\t1\t1\t1\t1\t20\t40\t50\t20\t96.0\t"Salinas\n'
        "5\t1\t2\t1\t1\t1\t100\t40\t50\t20\t96.0\tHanford\n"
    )
    _, _, report = infer_ocr_label_occlusions_by_nearest_class(
        values,
        valid,
        tsv,
        label_padding_px=4,
        context_radius_px=20,
    )
    assert [label["text"] for label in report["accepted_labels"]] == [
        '"Salinas',
        "Hanford",
    ]
