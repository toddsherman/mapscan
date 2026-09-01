"""Safety checks shared by unpublished review-candidate stages."""

from __future__ import annotations

from pathlib import Path


REVIEW_DECISION_FILENAME = "materialization-review-decision.json"


def require_fresh_review_output(output_dir: Path, stage: str) -> None:
    """Refuse to reuse an output directory that already carries an approval.

    Promotion and clipping intentionally create a new candidate whose hashes
    have not been reviewed. Keeping an old decision beside regenerated pixels
    is misleading even when a later exporter would reject a hash mismatch.
    """

    decision_path = output_dir / REVIEW_DECISION_FILENAME
    if decision_path.exists():
        raise ValueError(
            f"{stage} requires a fresh versioned output directory; remove neither "
            f"nor carry forward the existing approval at {decision_path}"
        )
