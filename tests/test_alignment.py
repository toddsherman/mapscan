import numpy as np

from mapscan.alignment import transform_points
from mapscan.assist import _fit_homography
from mapscan.vision import _connected_eastern_border_evidence
from mapscan.pdf_registration import apply_affine, fit_affine
from mapscan.pdf_legend import cmyk_to_rgb


def test_transform_points_centers_reference_origin() -> None:
    points = np.array([[0.0, 0.0], [0.0, 0.5]])
    parameters = [0.25, 0.75, 0.5, 1.0, 0.0, 0.0]
    transformed = transform_points(points, parameters, (200, 400))
    assert np.allclose(transformed[0], [100, 150])
    assert np.allclose(transformed[1], [100, 200])


def test_transform_points_applies_horizontal_scale() -> None:
    points = np.array([[0.25, 0.0]])
    parameters = [0.5, 0.5, 0.5, 2.0, 0.0, 0.0]
    transformed = transform_points(points, parameters, (200, 400))
    assert np.allclose(transformed[0], [250, 100])


def test_assisted_homography_recovers_projective_transform() -> None:
    reference = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.4, 0.7]], dtype=float)
    expected = np.array([[800, 30, 120], [-45, 1100, 90], [0.08, -0.04, 1]], dtype=float)
    homogeneous = np.column_stack((reference, np.ones(len(reference)))) @ expected.T
    source = homogeneous[:, :2] / homogeneous[:, 2:]
    actual, metrics = _fit_homography(reference, source)
    actual /= actual[2, 2]
    assert np.allclose(actual, expected, atol=1e-6)
    assert metrics["control_point_rms_px"] < 1e-3


def test_eastern_border_evidence_requires_vertical_diagonal_hinge() -> None:
    lines = np.array(
        [
            [[425, 50, 425, 223]],
            [[410, 212, 669, 441]],
            [[558, 35, 558, 263]],  # long legend column without a nearby junction
            [[260, 330, 548, 800]],  # long agricultural alignment in the interior
        ],
        dtype=np.int32,
    )
    evidence, hinge = _connected_eastern_border_evidence(lines, (900, 695))
    assert hinge is not None
    assert np.allclose(hinge, [425, 225.3], atol=1.0)
    assert evidence[100, 425] > 0
    assert evidence[350, 565] > 0
    assert evidence[100, 558] == 0


def test_pdf_graticule_affine_fit_recovers_distributed_transform() -> None:
    source = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 200.0], [100.0, 200.0], [40.0, 90.0]]
    )
    expected = np.array([[1.25, 0.12, 30.0], [-0.08, 0.95, 75.0]])
    target = apply_affine(source, expected)
    actual, residuals = fit_affine(source, target)
    assert np.allclose(actual, expected)
    assert np.max(residuals) < 1e-10


def test_pdf_cmyk_swatch_conversion() -> None:
    assert cmyk_to_rgb((0.0, 0.0, 1.0, 0.0)) == (255, 255, 0)
    assert cmyk_to_rgb((0.0, 0.0, 0.0, 1.0)) == (0, 0, 0)
