import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils import report_file_facts as report_file_facts_module
from utils.report_file_facts import current_report_file_facts


def test_public_tuning_verifier_reuses_report_file_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"value": "expected"}))
    run_name = "g1_rqtune_cache_test_500m"
    directory = tmp_path / run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"value": "expected", "transfer_invariants": {}})
    )
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.1})
    )
    experiment = SimpleNamespace(run_name=run_name, base_path=tmp_path)
    reads = 0

    def expected_metadata(
        _experiment: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal reads

        def load() -> str:
            nonlocal reads
            reads += 1
            return json.loads(source.read_text())["value"]

        facts = current_report_file_facts()
        value = (
            load()
            if facts is None
            else facts.load_or_compute("test_tuning_metadata", (source,), load)
        )
        return {"value": value}, {}

    monkeypatch.setattr(
        verify_artifact, "_tuning_experiment", lambda *_args: experiment
    )
    monkeypatch.setattr(verify_artifact, "_expected_metadata", expected_metadata)
    monkeypatch.setattr(
        verify_artifact, "has_current_generation_semantics", lambda _metadata: True
    )
    monkeypatch.setattr(
        verify_artifact, "_with_legacy_accumulation_defaults", lambda value: value
    )
    monkeypatch.setattr(
        verify_artifact, "_valid_dynamic_metadata", lambda _metadata: True
    )
    database = tmp_path / "report-file-facts.sqlite3"
    monkeypatch.setenv("DCN_REPORT_FILE_FACTS", str(database))
    assignments = [
        "G1_TUNE_RUN=cache_test",
        "G1_TUNE_SOURCE_VARIANT=baseline",
        "G1_TUNE_TRANSFORMER_FIELDS=",
        "G1_TUNE_EXPERIMENT_FIELDS=",
        "G1_TUNE_EMBEDDING_LR=0.064",
        "G1_TUNE_DEEP_LR=0.012",
    ]

    assert verify_artifact.verify(directory, "500m", assignments)
    report_file_facts_module.report_file_facts(tmp_path).close()
    report_file_facts_module._report_file_facts.cache_clear()
    assert verify_artifact.verify(directory, "500m", assignments)
    assert reads == 1
