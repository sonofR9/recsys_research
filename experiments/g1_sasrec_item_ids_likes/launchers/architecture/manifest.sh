#!/usr/bin/env bash

G1_MANIFEST_AXES="control ffn position dimension depth heads normalization sequence window dropout bos cls schedule time item_embeddings"
G1_DEFAULT_AXES="control ffn position dimension depth heads normalization window dropout bos cls schedule time item_embeddings"

g1_run_stem() {
    case $1 in
        control/control) echo architecture_control ;;
        position/rope_forward) echo position_rope ;;
        position/rope_reverse) echo position_rope_reverse ;;
        position/rope_forward_alibi) echo position_rope_alibi ;;
        position/rope_reverse_alibi) echo position_rope_reverse_alibi ;;
        position/learned_forward_alibi) echo position_learned_alibi ;;
        position/learned_reverse_alibi) echo position_learned_reverse_alibi ;;
        position/rope_forward_learned_forward) echo position_rope_learned ;;
        position/rope_forward_learned_reverse) echo position_rope_learned_reverse ;;
        position/rope_reverse_learned_forward) echo position_rope_reverse_learned ;;
        position/rope_reverse_learned_reverse) echo position_rope_reverse_learned_reverse ;;
        position/rope_forward_learned_forward_alibi) echo position_all ;;
        position/rope_forward_learned_reverse_alibi) echo position_rope_learned_reverse_alibi ;;
        position/rope_reverse_learned_forward_alibi) echo position_rope_reverse_learned_alibi ;;
        position/rope_reverse_learned_reverse_alibi) echo position_reverse_all ;;
        normalization/layer_pre_input_rms_final_rms) echo normalization_all_rms ;;
        normalization/rms_pre) echo normalization_rms ;;
        normalization/batch_pre) echo normalization_batch ;;
        normalization/layer_pre_input_layer) echo normalization_input_layer ;;
        normalization/layer_pre_input_rms) echo normalization_input_rms ;;
        normalization/layer_pre_no_final) echo normalization_no_final ;;
        normalization/layer_post) echo normalization_post ;;
        *) echo "${1//\//_}" ;;
    esac
}

g1_verify_control_artifact() {
    if [[ -n "${G1_TEST_CONTROL_ARTIFACT_VERIFIER:-}" ]]; then
        "$G1_TEST_CONTROL_ARTIFACT_VERIFIER" "$@"
        return
    fi
    python - "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import json
import math
from pathlib import Path
import sys

from dcn.training_metadata import (
    TIMESTAMP_BIN_SEMANTICS_REVISION,
    has_current_generation_semantics,
)
from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import (
    has_unaccumulated_batch_contract,
)

directory = Path(sys.argv[1])
dataset_size = sys.argv[2]
batch_size = int(sys.argv[3])
embedding_lr = float(sys.argv[4])
deep_lr = float(sys.argv[5])
max_epochs = int(sys.argv[6])
run_revision = int(sys.argv[7])
cap = "" if max_epochs == 20 else f"_cap{max_epochs}"
expected_suffix = f"{cap}_ts2_r{run_revision}_{dataset_size}"
try:
    metrics_path = directory / "final_metrics.json"
    metadata_path = directory / "training_metadata.json"
    metrics = json.loads(metrics_path.read_text())
    metadata = json.loads(metadata_path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
invariants = metadata.get("transfer_invariants", {})
transformer = invariants.get("transformer", {})
schedule = invariants.get("lr_schedule", {})
valid = (
    isinstance(metrics, dict)
    and bool(metrics)
    and directory.name.endswith(expected_suffix)
    and has_current_generation_semantics(metadata)
    and metrics_path.stat().st_mtime_ns >= metadata_path.stat().st_mtime_ns
    and metadata.get("seed") == 42
    and metadata.get("num_epochs") == max_epochs
    and metadata.get("max_epochs") == max_epochs
    and isinstance(metadata.get("epochs_trained"), int)
    and 1 <= metadata.get("epochs_trained", 0) <= metadata.get("num_epochs")
    and metadata.get("stopped_epoch") == metadata.get("epochs_trained")
    and metadata.get("stopped_epoch") < metadata.get("num_epochs")
    and isinstance(metadata.get("best_epoch"), int)
    and 1 <= metadata.get("best_epoch", 0) <= metadata.get("stopped_epoch", 0)
    and metadata.get("early_stopped") is True
    and metadata.get("best_epoch_at_cap") is False
    and metadata.get("selection_resolved") is True
    and metadata.get("val_batch_size") == 8192
    and metadata.get("num_workers") == 4
    and metadata.get("prefetch_factor") == 4
    and metadata.get("weight_decay") == 0.0
    and metadata.get("initializer_std") == 0.02
    and metadata.get("runtime_dtype") == "torch.bfloat16"
    and metadata.get("runtime_compile") is False
    and metadata.get("gradient_clip_norm") is None
    and isinstance(metadata.get("targets_per_epoch"), int)
    and metadata.get("targets_per_epoch", 0) > 0
    and metadata.get("training_horizon")
        == metadata.get("targets_per_epoch") * metadata.get("epochs_trained")
    and isinstance(metadata.get("tokens_per_epoch"), int)
    and metadata.get("tokens_per_epoch", 0) > 0
    and metadata.get("token_horizon")
        == metadata.get("tokens_per_epoch") * metadata.get("epochs_trained")
    and metadata.get("tokens_seen") == metadata.get("token_horizon")
    and isinstance(metadata.get("optimizer_steps"), int)
    and metadata.get("optimizer_steps", 0) > 0
    and invariants.get("experiment_class") == "MuTransferGenerationExperiment"
    and invariants.get("mup_base_dim") == 16
    and invariants.get("mup_delta_dim") == 32
    and invariants.get("dataset_size") == dataset_size
    and invariants.get("user_sample") is None
    and invariants.get("event_type_filter") == "like"
    and invariants.get("min_item_interactions_per_item") == 5
    and invariants.get("drop_unmapped_items") is True
    and invariants.get("validation_interval_seconds") == 604800
    and invariants.get("day_range") == {"start_day": 0, "end_day": 300}
    and invariants.get("batch_size") == batch_size
    and has_unaccumulated_batch_contract(metadata, batch_size)
    and invariants.get("model_dim") == 64
    and invariants.get("item_embedding_dim") == 64
    and invariants.get("max_seq_len") == 128
    and invariants.get("window") == "next_item"
    and invariants.get("bos") is False
    and invariants.get("cls_token") is False
    and invariants.get("timestamp_delta") == "bins"
    and invariants.get("timestamp_combination") == "add"
    and invariants.get("timestamp_num_bins") == 16
    and invariants.get("timestamp_bin_semantics_revision")
        == TIMESTAMP_BIN_SEMANTICS_REVISION
    and invariants.get("per_layer_item_embeddings") is False
    and invariants.get("negative_sampling") == "random"
    and invariants.get("num_in_batch_negatives") == 512
    and invariants.get("logq_correction") == "yi2019"
    and invariants.get("random_negative_fraction") == 0.5
    and invariants.get("logq_alpha") == 0.01
    and invariants.get("correct_positive_logq") is False
    and invariants.get("mask_false_negatives") is False
    and invariants.get("exclude_own_group_negatives") is False
    and invariants.get("dense_random_negative_scores") is True
    and invariants.get("eval_ks") == [10, 50, 100]
    and invariants.get("eval_max_users") == 20000
    and invariants.get("eval_every_n_epochs") == 1
    and invariants.get("early_stopping_patience") == 3
    and invariants.get("early_stopping_min_delta") == 0.0
    and invariants.get("early_stopping_metric") == "recall@100"
    and invariants.get("early_stopping_metric_prefix") == "epoch/val_true"
    and invariants.get("selection_k") == 100
    and invariants.get("evaluation_catalog") == "all"
    and invariants.get("exclude_seen_from_evaluation") is False
    and invariants.get("restore_best_weights") is True
    and transformer == {
        "alibi": False,
        "attention_window": 50,
        "dim": 64,
        "dropout": 0.1,
        "ffn": "swiglu",
        "ffn_dropout": 0.1,
        "ffn_intermediate_dim": 171,
        "final_norm": "layer",
        "input_dropout": 0.1,
        "input_norm": None,
        "learned_positions": "forward",
        "nhead": 2,
        "norm": "layer",
        "norm_place": "pre",
        "num_kv_heads": 1,
        "num_layers": 2,
        "rope": None,
    }
    and schedule == {
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "shape": "linear",
        "timescale_fraction": None,
        "timescale_steps": None,
        "warmup_fraction": 0.0,
    }
    and math.isclose(metadata.get("embedding_learning_rate", -1), embedding_lr)
    and math.isclose(metadata.get("deep_learning_rate", -1), deep_lr)
)
raise SystemExit(0 if valid else 1)
PY
}

g1_manifest_rows() {
    printf '%s\n' \
        'control|control|selected_quality_b1280||||' \
        'ffn|gelu128|baseline|ffn||G1_TUNE_FFN_DIM=128|' \
        'ffn|gelu171|baseline|ffn||G1_TUNE_FFN_DIM=171|' \
        'ffn|gelu256|baseline|ffn||G1_TUNE_FFN_DIM=256|' \
        'ffn|gelu384|baseline|ffn||G1_TUNE_FFN_DIM=384|' \
        'ffn|swiglu16|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=16|' \
        'ffn|swiglu32|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=32|' \
        'ffn|swiglu64|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=64|' \
        'ffn|swiglu96|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=96|' \
        'ffn|swiglu114|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=114|' \
        'ffn|swiglu128|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=128|' \
        'ffn|swiglu171|selected_quality_b1280||||control/control' \
        'ffn|swiglu224|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=224|' \
        'position|none|pos_none|alibi,rope,learned_positions|||' \
        'position|learned_forward|selected_quality_b1280||||control/control' \
        'position|learned_reverse|pos_learned_reverse|alibi,rope,learned_positions|||' \
        'position|learned_forward_reverse|pos_learned_forward_reverse|alibi,rope,learned_positions|||' \
        'position|rope_forward|pos_rope|alibi,rope,learned_positions|||' \
        'position|rope_reverse|pos_rope_reverse|alibi,rope,learned_positions|||' \
        'position|alibi|pos_alibi|alibi,rope,learned_positions|||' \
        'position|rope_forward_alibi|pos_rope_alibi|alibi,rope,learned_positions|||' \
        'position|rope_reverse_alibi|pos_rope_reverse_alibi|alibi,rope,learned_positions|||' \
        'position|learned_forward_alibi|pos_learned_alibi|alibi,rope,learned_positions|||' \
        'position|learned_reverse_alibi|pos_learned_reverse_alibi|alibi,rope,learned_positions|||' \
        'position|rope_forward_learned_forward|pos_rope_learned|alibi,rope,learned_positions|||' \
        'position|rope_forward_learned_reverse|pos_rope_learned_reverse|alibi,rope,learned_positions|||' \
        'position|rope_reverse_learned_forward|pos_rope_reverse_learned|alibi,rope,learned_positions|||' \
        'position|rope_reverse_learned_reverse|pos_rope_reverse_learned_reverse|alibi,rope,learned_positions|||' \
        'position|rope_forward_learned_forward_alibi|pos_all|alibi,rope,learned_positions|||' \
        'position|rope_forward_learned_reverse_alibi|pos_rope_learned_reverse_alibi|alibi,rope,learned_positions|||' \
        'position|rope_reverse_learned_forward_alibi|pos_rope_reverse_learned_alibi|alibi,rope,learned_positions|||' \
        'position|rope_reverse_learned_reverse_alibi|pos_reverse_all|alibi,rope,learned_positions|||' \
        'dimension|16|dim_16|dim||G1_TUNE_FFN_DIM=43|' \
        'dimension|32|dim_32|dim||G1_TUNE_FFN_DIM=86|' \
        'dimension|64|selected_quality_b1280||||control/control' \
        'dimension|128|dim_128|dim||G1_TUNE_FFN_DIM=342|' \
        'dimension|256|dim_256|dim||G1_TUNE_FFN_DIM=684|' \
        'depth|1|depth_1|num_layers|||' \
        'depth|2|selected_quality_b1280||||control/control' \
        'depth|4|depth_4|num_layers|||' \
        'heads|mha1|heads_1|nhead,num_kv_heads|||' \
        'heads|mha2|baseline|nhead,num_kv_heads|||' \
        'heads|mha4|heads_4|nhead,num_kv_heads|||' \
        'heads|mha8|heads_8|nhead,num_kv_heads|||' \
        'heads|gqa2q1kv|selected_quality_b1280||||control/control' \
        'normalization|layer_pre|selected_quality_b1280||||control/control' \
        'normalization|rms_pre|norm_rms|norm,norm_place,input_norm,final_norm|||' \
        'normalization|batch_pre|norm_batch|norm,norm_place,input_norm,final_norm|||' \
        'normalization|layer_pre_input_rms_final_rms|norm_all_rms|norm,norm_place,input_norm,final_norm|||' \
        'normalization|layer_pre_input_layer|norm_input_layer|norm,norm_place,input_norm,final_norm|||' \
        'normalization|layer_pre_input_rms|norm_input_rms|norm,norm_place,input_norm,final_norm|||' \
        'normalization|layer_pre_no_final|norm_no_final|norm,norm_place,input_norm,final_norm|||' \
        'normalization|layer_post|norm_post|norm,norm_place,input_norm,final_norm|||' \
        'sequence|12|seq_12||max_seq_len||' \
        'sequence|25|seq_25||max_seq_len||' \
        'sequence|50|seq_50||max_seq_len||' \
        'sequence|100|baseline||max_seq_len||' \
        'sequence|128|selected_quality_b1280||||control/control' \
        'sequence|200|seq_200||max_seq_len||' \
        'sequence|256|seq_256||max_seq_len||' \
        'sequence|512|seq_512||max_seq_len||' \
        'window|none|window_none|attention_window|||' \
        'window|10|window_10|attention_window|||' \
        'window|25|window_25|attention_window|||' \
        'window|50|selected_quality_b1280||||control/control' \
        'window|75|window_75|attention_window|||' \
        'window|100|window_100|attention_window|||' \
        'dropout|0|dropout_0|dropout,input_dropout,ffn_dropout|||' \
        'dropout|5|dropout_5|dropout,input_dropout,ffn_dropout|||' \
        'dropout|10|selected_quality_b1280||||control/control' \
        'dropout|20|dropout_20|dropout,input_dropout,ffn_dropout|||' \
        'dropout|30|dropout_30|dropout,input_dropout,ffn_dropout|||' \
        'dropout|50|dropout_50|dropout,input_dropout,ffn_dropout|||' \
        'bos|off|selected_quality_b1280||||control/control' \
        'bos|on|bos||bos||' \
        'cls|off|selected_quality_b1280||||control/control' \
        'cls|on|cls||cls_token||' \
        'schedule|constant|baseline||lr_schedule||' \
        'schedule|linear|selected_quality_b1280||||control/control' \
        'schedule|cosine|lr_cosine||lr_schedule||' \
        'schedule|polynomial|lr_polynomial||lr_schedule||' \
        'schedule|exponential|lr_exponential||lr_schedule||' \
        'schedule|wsd_warmup5|lr_wsd||lr_schedule||' \
        'schedule|step|lr_step||lr_schedule||' \
        'schedule|inverse_sqrt|lr_inverse_sqrt||lr_schedule||' \
        'schedule|constant_warmup5|lr_warmup||lr_schedule||' \
        'schedule|cosine_warmup5_cycles1|lr_cosine_warmup||lr_schedule||' \
        'schedule|cosine_warmup5_cycles2|lr_cosine_cycles2||lr_schedule||' \
        'schedule|cosine_warmup5_cycles4|lr_cosine_cycles4||lr_schedule||' \
        'schedule|inverse_sqrt_warmup5|lr_inverse_sqrt_warmup||lr_schedule||' \
        'time|none|baseline|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|raw_rope_forward|time_rope|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|raw_rope_reverse|time_rope_reverse|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|log_rope_forward|time_log_rope|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|log_rope_reverse|time_log_rope_reverse|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|plain_delta_add|time_plain_add|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|log_delta_add|time_log_add|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins8_add|time_bins_8|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins16_add|selected_quality_b1280||||' \
        'time|bins32_add|time_bins_add|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins64_add|time_bins_64|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins32_add_raw_rope_reverse|time_bins_reverse_rope|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|log_delta_concat|time_log_concat|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins32_concat|time_bins_concat|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'time|bins32_add_log_rope_forward|time_bins_log_rope|rope|timestamp_delta,timestamp_combination,timestamp_num_bins||' \
        'item_embeddings|shared|selected_quality_b1280||||control/control' \
        'item_embeddings|per_layer|per_layer_embeddings||per_layer_item_embeddings||'
}
