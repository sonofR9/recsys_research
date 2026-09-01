from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment
from dcn.config.query_retrieval_training import FirstStageCheckpointExportExperiment
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    RQ15_SOURCE_CHECKPOINT_NAME,
    source_candidate_by_run,
    source_checkpoint_metadata,
)


candidate = source_candidate_by_run(os.environ["G1_RQ15_SOURCE_RUN"])
previous = os.environ.get("G1_RQ8_RUN")
try:
    os.environ["G1_RQ8_RUN"] = candidate.source_recipe_run_name
    source = runpy.run_path(
        str(Path(__file__).with_name("rq8_reinvestigation_variant.py"))
    )["experiment"]
finally:
    if previous is None:
        os.environ.pop("G1_RQ8_RUN", None)
    else:
        os.environ["G1_RQ8_RUN"] = previous

common = {
    field.name: getattr(source, field.name)
    for field in fields(GenerationExperiment)
    if field.init
}
common["run_name"] = candidate.run_name
experiment = FirstStageCheckpointExportExperiment(
    **common,
    mup_base_dim=source.mup_base_dim,
    mup_delta_dim=source.mup_delta_dim,
    mup_base_ffn_dim=source.mup_base_ffn_dim,
    mup_delta_ffn_dim=source.mup_delta_ffn_dim,
    checkpoint_export_metadata=source_checkpoint_metadata(candidate),
    checkpoint_export_history_positions=128,
    checkpoint_export_filename=RQ15_SOURCE_CHECKPOINT_NAME,
)
