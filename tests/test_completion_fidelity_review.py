import numpy as np
import pytest

from mapscan.completion_fidelity_review import compose_conservative_surface


def test_composition_preserves_approved_and_adds_only_masked_stability_values():
    approved = np.asarray([[1, 0, 3], [0, 2, 0]], dtype=np.uint8)
    pass_one = np.asarray([[1, 4, 3], [0, 2, 0]], dtype=np.uint8)
    pass_two_values = np.asarray([[0, 0, 0], [5, 0, 6]], dtype=np.uint8)
    pass_two_mask = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.uint8)

    result = compose_conservative_surface(
        approved, pass_one, pass_two_values, pass_two_mask
    )

    assert result.tolist() == [[1, 4, 3], [5, 2, 0]]


def test_composition_rejects_a_pass_that_changes_approved_evidence():
    approved = np.asarray([[1, 0]], dtype=np.uint8)
    pass_one = np.asarray([[2, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="changes approved evidence"):
        compose_conservative_surface(
            approved,
            pass_one,
            np.zeros_like(approved),
            np.zeros_like(approved),
        )


def test_composition_rejects_stability_pixels_over_approved_evidence():
    approved = np.asarray([[1, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="overlaps approved evidence"):
        compose_conservative_surface(
            approved,
            approved.copy(),
            np.asarray([[2, 0]], dtype=np.uint8),
            np.asarray([[1, 0]], dtype=np.uint8),
        )


def test_composition_allows_an_explicit_boundary_clear_of_approved_evidence():
    approved = np.asarray([[1, 0]], dtype=np.uint8)
    pass_one = np.asarray([[0, 2]], dtype=np.uint8)

    result = compose_conservative_surface(
        approved,
        pass_one,
        np.zeros_like(approved),
        np.zeros_like(approved),
        np.asarray([[1, 0]], dtype=np.uint8),
    )

    assert result.tolist() == [[0, 2]]
