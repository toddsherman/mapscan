from mapscan.cli import _parser


def test_versioned_migration_commands_are_exposed_by_cli():
    parser = _parser()

    migrate = parser.parse_args(["migrate-stamp-corrections", "old", "new"])
    assert migrate.source_run.name == "old"
    assert migrate.target_run.name == "new"

    audit = parser.parse_args(
        [
            "audit-stamp-migration",
            "old",
            "new",
            "--approved-materialized",
            "approved-v2",
            "--candidate-materialized",
            "candidate-v1",
            "--output",
            "audit-v1",
        ]
    )
    assert audit.approved_materialized.name == "approved-v2"
    assert audit.candidate_materialized.name == "candidate-v1"
