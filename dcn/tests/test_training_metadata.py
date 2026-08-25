from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    has_current_generation_semantics,
)


def test_early_stopping_protocol_invalidates_fixed_epoch_artifacts() -> None:
    assert GENERATION_TRAINING_SEMANTICS_REVISION == 2
    assert not has_current_generation_semantics({"training_semantics_revision": 1})
    assert has_current_generation_semantics({"training_semantics_revision": 2})
