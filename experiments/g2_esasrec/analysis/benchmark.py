from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from dcn.data.features import FeatureValues
from neuralrec.data.transforms import move_to_device

if TYPE_CHECKING:
    from experiments.g2_esasrec.protocol.manifest import CompiledJob

WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 100
QUERY_BATCH_SIZE = 256
QUERY_SELECTION_SEED = 42
QUERY_POPULATION_SIZE = 3414
QUERY_POPULATION_SHA256 = (
    "d2beb685bb9afb25f24f150413e93845cf2639c0020322d69f4784ed82a613a0"
)
QUERY_USER_IDS_SHA256 = (
    "05be59fc94ff2fba26cf6ed5d39c37057e0ed20e3b71f59a5527f171debbcbfa"
)
QUERY_USER_IDS = (
    616200,
    732800,
    967200,
    910400,
    80800,
    615600,
    452500,
    947100,
    883300,
    621400,
    457100,
    172600,
    881100,
    444000,
    942900,
    753600,
    299200,
    85400,
    362900,
    475500,
    68200,
    654700,
    350200,
    765100,
    812800,
    865100,
    611700,
    816300,
    102900,
    801000,
    983900,
    729900,
    650200,
    702700,
    328600,
    358700,
    570700,
    483800,
    726400,
    116300,
    852300,
    129700,
    192300,
    94800,
    729600,
    271200,
    747800,
    436400,
    861700,
    574700,
    99300,
    760700,
    569300,
    549800,
    382400,
    674900,
    440500,
    438000,
    520700,
    993300,
    8300,
    960800,
    899000,
    509300,
    417900,
    562400,
    956400,
    84300,
    616800,
    305500,
    866700,
    318700,
    620500,
    354400,
    109200,
    802600,
    398300,
    574400,
    911700,
    20500,
    591400,
    54700,
    555700,
    480100,
    913100,
    2500,
    429000,
    528700,
    922800,
    230100,
    580800,
    191300,
    212700,
    180100,
    680700,
    891500,
    352200,
    832300,
    38600,
    740500,
    792300,
    795000,
    503600,
    907400,
    571200,
    469100,
    5900,
    434900,
    109900,
    912300,
    846300,
    614600,
    445800,
    663400,
    776000,
    82200,
    871200,
    520200,
    690600,
    726200,
    588300,
    987800,
    350400,
    938700,
    918300,
    891800,
    729200,
    933700,
    72000,
    834400,
    595400,
    618700,
    672500,
    191600,
    829100,
    297700,
    535600,
    531800,
    615500,
    696300,
    401300,
    601200,
    459900,
    7500,
    922000,
    852100,
    226000,
    153000,
    659800,
    672000,
    964900,
    325400,
    24600,
    129800,
    241600,
    909900,
    155100,
    382800,
    366300,
    38500,
    49100,
    147600,
    759300,
    102300,
    320900,
    385300,
    998100,
    944600,
    658300,
    977900,
    507900,
    330000,
    597700,
    761600,
    880900,
    302700,
    261200,
    556000,
    200200,
    733300,
    737900,
    684000,
    338700,
    91600,
    819500,
    479600,
    165000,
    169500,
    515500,
    829500,
    212100,
    898900,
    960300,
    234800,
    104500,
    211500,
    134400,
    156800,
    101200,
    561600,
    688700,
    391200,
    628100,
    338400,
    738500,
    635100,
    981800,
    479200,
    183500,
    315100,
    801900,
    985600,
    204700,
    383600,
    302600,
    795700,
    614400,
    579500,
    303100,
    771600,
    640700,
    716800,
    361300,
    304400,
    994900,
    953400,
    693100,
    205600,
    767900,
    763700,
    3900,
    807200,
    775500,
    389300,
    686300,
    134700,
    313200,
    577300,
    129100,
    580400,
    777600,
    434600,
    723800,
    141900,
    353300,
    783000,
    193000,
    486000,
    967300,
    584800,
    243400,
    369300,
    393400,
    344400,
    826100,
    157800,
)
QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN = {
    100: "f9dbb0d5a09fc60e64b76497561f3e56df94df3768de33caf0e996d99e29bc6a",
    128: "5a9d34fdc6b83b34e094ac9402bafe27c24c7b2bc38398ae7b4c23e10f22446f",
}
CATALOG_SIZE = 33148
CATALOG_SHA256 = "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"


def canonical_json_sha256(document: object) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _sample_key(user_id: int, seed: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{user_id}".encode(), digest_size=8).digest()


def _select_sequence(batch: dict[str, Any], sequence_index: int) -> dict[str, Any]:
    cumulative = batch["cumulative_lens"]
    start = int(cumulative[sequence_index])
    stop = int(cumulative[sequence_index + 1])
    events = torch.arange(start, stop, dtype=torch.long)
    return {
        "int_columns": {
            name: values.select(events) for name, values in batch["int_columns"].items()
        },
        "float_columns": {
            name: values.select(events)
            for name, values in batch["float_columns"].items()
        },
        "timestamp": batch["timestamp"][events],
        "cumulative_lens": torch.tensor([0, stop - start], dtype=torch.long),
    }


def _merge_features(rows: list[FeatureValues]) -> FeatureValues:
    counts = torch.cat([row.offsets.diff() for row in rows])
    return FeatureValues(
        values=torch.cat([row.values for row in rows]),
        offsets=torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))),
    )


def _merge_sequences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor(
        [int(row["cumulative_lens"][-1]) for row in rows], dtype=torch.long
    )
    return {
        "int_columns": {
            name: _merge_features([row["int_columns"][name] for row in rows])
            for name in rows[0]["int_columns"]
        },
        "float_columns": {
            name: _merge_features([row["float_columns"][name] for row in rows])
            for name in rows[0]["float_columns"]
        },
        "timestamp": torch.cat([row["timestamp"] for row in rows]),
        "cumulative_lens": torch.cat(
            (torch.zeros(1, dtype=torch.long), lengths.cumsum(0))
        ),
    }


def select_query_batch_by_user_hash(
    batches: list[dict[str, Any]],
    *,
    user_column: str,
    query_batch_size: int = QUERY_BATCH_SIZE,
    seed: int = QUERY_SELECTION_SEED,
    eligible_user_ids: set[int] | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not batches:
        raise ValueError("query selection requires at least one packed batch")
    candidates = []
    seen_users: set[int] = set()
    for batch_index, batch in enumerate(batches):
        cumulative = batch["cumulative_lens"]
        last_events = cumulative[1:] - 1
        user_ids = batch["int_columns"][user_column].dense()[last_events]
        for sequence_index, raw_user_id in enumerate(user_ids.tolist()):
            user_id = int(raw_user_id)
            if eligible_user_ids is not None and user_id not in eligible_user_ids:
                continue
            if user_id in seen_users:
                raise ValueError("query population contains a duplicate user ID")
            seen_users.add(user_id)
            candidates.append(
                (_sample_key(user_id, seed), user_id, batch_index, sequence_index)
            )
    if len(candidates) < query_batch_size:
        raise ValueError(
            f"query population has {len(candidates)} users, needs {query_batch_size}"
        )
    selected = sorted(candidates)[:query_batch_size]
    rows = [
        _select_sequence(batches[batch_index], sequence_index)
        for _, _, batch_index, sequence_index in selected
    ]
    user_ids = torch.tensor([row[1] for row in selected], dtype=torch.long)
    return _merge_sequences(rows), user_ids


def _tensor_sha256(values: torch.Tensor) -> str:
    array = np.asarray(values.detach().cpu().tolist(), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _query_payload_sha256(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()

    def update(name: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    for kind in ("int_columns", "float_columns"):
        for name, feature in sorted(batch[kind].items()):
            update(f"{kind}.{name}.values", feature.values)
            update(f"{kind}.{name}.offsets", feature.offsets)
    update("timestamp", batch["timestamp"])
    update("cumulative_lens", batch["cumulative_lens"])
    return digest.hexdigest()


def benchmark_selected_model(
    score_full_catalog: Callable[[], object],
    *,
    query_batch_size: int,
    device: torch.device,
    warmup_iterations: int = WARMUP_ITERATIONS,
    timed_iterations: int = TIMED_ITERATIONS,
    require_a100: bool = True,
) -> dict[str, float | int | str]:
    if query_batch_size < 1 or warmup_iterations < 0 or timed_iterations < 1:
        raise ValueError("benchmark sizes and iteration counts must be positive")
    if require_a100:
        if device.type != "cuda" or "A100" not in torch.cuda.get_device_name(device):
            raise RuntimeError("selected-model benchmark requires one A100")
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    for _ in range(warmup_iterations):
        score_full_catalog()
    synchronize()
    latencies = []
    for _ in range(timed_iterations):
        started = perf_counter()
        score_full_catalog()
        synchronize()
        latencies.append(perf_counter() - started)
    latencies.sort()
    median = _percentile(latencies, 0.5)
    return {
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "query_batch_size": query_batch_size,
        "latency_p50_seconds": median,
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "queries_per_second": query_batch_size / median,
    }


def run_production_benchmark(
    model: torch.nn.Module,
    *,
    query_batches: list[dict[str, Any]],
    item_batch: dict[str, Any],
    user_column: str,
    item_id_column: str,
    device: torch.device,
    initialization_seed: int,
    initializer_std: float | None,
    eligible_user_ids: set[int],
    max_seq_len: int,
    weight_source: str = "fresh_initialized_no_checkpoint",
    optimizer_steps: int = 0,
    best_epoch: int | None = None,
    warmup_iterations: int = WARMUP_ITERATIONS,
    timed_iterations: int = TIMED_ITERATIONS,
    require_a100: bool = True,
) -> dict[str, Any]:
    query_batch, query_user_ids = select_query_batch_by_user_hash(
        query_batches,
        user_column=user_column,
        eligible_user_ids=eligible_user_ids,
    )
    population_user_ids = sorted(
        int(user_id)
        for batch in query_batches
        for user_id in batch["int_columns"][user_column]
        .dense()[batch["cumulative_lens"][1:] - 1]
        .tolist()
        if int(user_id) in eligible_user_ids
    )
    population_sha256 = _tensor_sha256(torch.tensor(population_user_ids))
    query_payload_sha256 = _query_payload_sha256(query_batch)
    query_user_ids_sha256 = _tensor_sha256(query_user_ids)
    query_user_ids_tuple = tuple(query_user_ids.tolist())
    observed_workload = {
        "population_size": len(population_user_ids),
        "population_user_ids_sha256": population_sha256,
        "user_ids_sha256": query_user_ids_sha256,
        "user_ids": query_user_ids_tuple,
        "packed_query_payload_sha256": query_payload_sha256,
    }
    expected_workload = {
        "population_size": QUERY_POPULATION_SIZE,
        "population_user_ids_sha256": QUERY_POPULATION_SHA256,
        "user_ids_sha256": QUERY_USER_IDS_SHA256,
        "user_ids": QUERY_USER_IDS,
        "packed_query_payload_sha256": QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN.get(
            max_seq_len
        ),
    }
    if observed_workload != expected_workload:
        raise ValueError(
            "selected-model benchmark query workload changed: "
            f"observed={observed_workload!r}"
        )
    query_batch = move_to_device(query_batch, device)
    item_batch = move_to_device(item_batch, device)
    item_ids = item_batch["int_columns"][item_id_column].dense()
    if item_ids.numel() != CATALOG_SIZE:
        raise ValueError("selected-model benchmark requires all 33,148 items")
    catalog_sha256 = _tensor_sha256(item_ids)
    if catalog_sha256 != CATALOG_SHA256:
        raise ValueError("selected-model benchmark catalog identity changed")
    state_sha256 = _model_state_sha256(model)
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=True),
    ):
        item_repr = model.encode_items(item_batch)

    observed_query_dtypes: set[str] = set()

    def score_full_catalog() -> object:
        with (
            torch.inference_mode(),
            torch.autocast(device.type, dtype=torch.bfloat16, enabled=True),
        ):
            query_repr = model.encode_cutoff_queries(query_batch)
        observed_query_dtypes.add(str(query_repr.dtype))
        scores = query_repr.float() @ item_repr.float().t()
        return torch.topk(scores, 100, dim=1).indices

    result = benchmark_selected_model(
        score_full_catalog,
        query_batch_size=QUERY_BATCH_SIZE,
        device=device,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        require_a100=require_a100,
    )
    return {
        **result,
        "catalog_sha256": catalog_sha256,
        "catalog_source": "full mapped pre-split catalog",
        "catalog_encoding_timed": False,
        "autocast_dtype": "torch.bfloat16",
        "item_representation_dtype": str(item_repr.dtype),
        "query_representation_dtypes": sorted(observed_query_dtypes),
        "ranking_dtype": "torch.float32",
        "top_k": 100,
        "query_selection": {
            "algorithm": "blake2b-64(seed:user_id)",
            "seed": QUERY_SELECTION_SEED,
            "population_size": len(population_user_ids),
            "population_user_ids_sha256": population_sha256,
            "user_ids": query_user_ids.tolist(),
            "user_ids_sha256": _tensor_sha256(query_user_ids),
            "packed_query_payload_sha256": query_payload_sha256,
        },
        "weights": {
            "source": weight_source,
            "seed": initialization_seed,
            "initializer_std": initializer_std,
            "optimizer_steps": optimizer_steps,
            **({} if best_epoch is None else {"best_epoch": best_epoch}),
            "state_sha256": state_sha256,
        },
    }


def write_benchmark(
    result: dict[str, Any],
    *,
    run_name: str,
    catalog_size: int,
    destination: Path,
) -> None:
    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")
    document = {
        "run_name": run_name,
        "catalog_size": catalog_size,
        "protocol": "one A100, fixed full catalog and query batch",
        **result,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"benchmark evidence already exists: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _positive_finite(document: dict[str, Any], name: str) -> float:
    value = document.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"selected-model benchmark {name} is missing")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"selected-model benchmark {name} is invalid")
    return result


def load_selected_benchmark(
    path: Path,
    *,
    run_name: str,
    expected_compiled: CompiledJob | None = None,
    logs_root: Path = Path("generated/logs"),
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("selected-model benchmark evidence is missing") from error
    if not isinstance(document, dict):
        raise ValueError("selected-model benchmark evidence must be an object")
    if document.get("run_name") != run_name:
        raise ValueError("benchmark does not belong to the selected run")
    expected = {
        "catalog_size": CATALOG_SIZE,
        "protocol": "one A100, fixed full catalog and query batch",
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise ValueError(f"selected-model benchmark {name} changed")
    device_name = document.get("device_name")
    if not isinstance(device_name, str) or "A100" not in device_name:
        raise ValueError("selected-model benchmark was not measured on one A100")
    query_batch_size = document.get("query_batch_size")
    if query_batch_size != QUERY_BATCH_SIZE:
        raise ValueError("selected-model benchmark query_batch_size is invalid")
    p50 = _positive_finite(document, "latency_p50_seconds")
    p95 = _positive_finite(document, "latency_p95_seconds")
    throughput = _positive_finite(document, "queries_per_second")
    if p95 < p50:
        raise ValueError("selected-model benchmark latency percentiles are invalid")
    expected_throughput = query_batch_size / p50
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-6):
        raise ValueError("selected-model benchmark throughput is inconsistent")
    evidence = {
        "catalog_sha256": CATALOG_SHA256,
        "catalog_source": "full mapped pre-split catalog",
        "catalog_encoding_timed": False,
        "autocast_dtype": "torch.bfloat16",
        "ranking_dtype": "torch.float32",
        "top_k": 100,
    }
    for name, value in evidence.items():
        if document.get(name) != value:
            raise ValueError(f"selected-model benchmark {name} changed")
    item_dtype = document.get("item_representation_dtype")
    query_dtypes = document.get("query_representation_dtypes")
    valid_representation_dtypes = {"torch.bfloat16", "torch.float32"}
    if item_dtype not in valid_representation_dtypes or not (
        isinstance(query_dtypes, list)
        and len(query_dtypes) == 1
        and query_dtypes[0] in valid_representation_dtypes
    ):
        raise ValueError("selected-model benchmark representation dtypes are invalid")
    selection = document.get("query_selection")
    if not isinstance(selection, dict) or selection.get("algorithm") != (
        "blake2b-64(seed:user_id)"
    ):
        raise ValueError("selected-model benchmark query selection changed")
    if selection.get("seed") != QUERY_SELECTION_SEED:
        raise ValueError("selected-model benchmark query seed changed")
    if (
        selection.get("population_size") != QUERY_POPULATION_SIZE
        or selection.get("population_user_ids_sha256") != QUERY_POPULATION_SHA256
    ):
        raise ValueError("selected-model benchmark query population is invalid")
    user_ids = selection.get("user_ids")
    if (
        not isinstance(user_ids, list)
        or len(user_ids) != QUERY_BATCH_SIZE
        or any(
            not isinstance(user_id, int) or isinstance(user_id, bool)
            for user_id in user_ids
        )
        or len(set(user_ids)) != QUERY_BATCH_SIZE
    ):
        raise ValueError("selected-model benchmark query user IDs are incomplete")
    if selection.get("user_ids_sha256") != _tensor_sha256(torch.tensor(user_ids)):
        raise ValueError("selected-model benchmark query user IDs changed")
    if selection.get("user_ids_sha256") != QUERY_USER_IDS_SHA256:
        raise ValueError("selected-model benchmark selected users changed")
    if tuple(user_ids) != QUERY_USER_IDS:
        raise ValueError("selected-model benchmark selected user IDs changed")
    weights = document.get("weights")
    if not isinstance(weights, dict) or weights.get("source") != (
        "validation_selected_recipe_reproduction_restored_weights"
    ):
        raise ValueError("selected-model benchmark did not use restored weights")
    optimizer_steps = weights.get("optimizer_steps")
    best_epoch = weights.get("best_epoch")
    if (
        not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps < 1
        or not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch < 1
    ):
        raise ValueError("selected-model benchmark training evidence is invalid")
    for container, name in (
        (selection, "population_user_ids_sha256"),
        (selection, "user_ids_sha256"),
        (selection, "packed_query_payload_sha256"),
        (weights, "state_sha256"),
    ):
        value = container.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"selected-model benchmark {name} is invalid")
    basis = document.get("basis")
    if not isinstance(basis, dict) or basis.get("kind") != (
        "deterministic_selected_recipe_reproduction"
    ):
        raise ValueError("selected-model benchmark basis changed")
    if basis.get("selected_run_name") != run_name:
        raise ValueError("selected-model benchmark architecture run changed")
    raw_contract = basis.get("selected_job_contract")
    if not isinstance(raw_contract, dict):
        raise ValueError("selected-model benchmark compiled contract is missing")
    from experiments.g2_esasrec.launchers.compiled import persisted_job_contract
    from experiments.g2_esasrec.protocol.manifest import CompiledJob, approved_manifest

    raw_job = raw_contract.get("job")
    parameters = raw_contract.get("parameters")
    matches = [
        job
        for job in approved_manifest().jobs
        if job.run_name == run_name and job.to_dict() == raw_job
    ]
    if len(matches) != 1 or not isinstance(parameters, dict):
        raise ValueError("selected-model benchmark compiled job changed")
    observed_compiled = CompiledJob(matches[0], parameters)
    if expected_compiled is not None and observed_compiled != expected_compiled:
        raise ValueError("selected-model benchmark selected artifact changed")
    expected_contract = persisted_job_contract(observed_compiled)
    if raw_contract != expected_contract:
        raise ValueError("selected-model benchmark architecture contract changed")
    selected_metrics_sha256 = basis.get("selected_metrics_sha256")
    diagnostic_metrics_sha256 = basis.get("diagnostic_metrics_sha256")
    if (
        not isinstance(selected_metrics_sha256, str)
        or len(selected_metrics_sha256) != 64
        or selected_metrics_sha256 != diagnostic_metrics_sha256
    ):
        raise ValueError("selected-model benchmark restored metrics changed")
    selected_metadata_sha256 = basis.get("selected_training_metadata_sha256")
    diagnostic_metadata_sha256 = basis.get("diagnostic_training_metadata_sha256")
    if (
        not isinstance(selected_metadata_sha256, str)
        or len(selected_metadata_sha256) != 64
        or selected_metadata_sha256 != diagnostic_metadata_sha256
    ):
        raise ValueError("selected-model benchmark training metadata changed")
    max_seq_len = expected_contract["local_implementation"]["transfer_invariants"][
        "max_seq_len"
    ]
    expected_query_payload_sha256 = QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN.get(max_seq_len)
    if (
        expected_query_payload_sha256 is None
        or selection.get("packed_query_payload_sha256") != expected_query_payload_sha256
    ):
        raise ValueError("selected-model benchmark query histories changed")
    if weights.get("seed") != observed_compiled.approved.seed:
        raise ValueError("selected-model benchmark initialization seed changed")
    expected_initializer_std = expected_contract["local_implementation"]["training"][
        "initializer_std"
    ]
    if weights.get("initializer_std") != expected_initializer_std:
        raise ValueError("selected-model benchmark initializer changed")
    diagnostic_run_name = document.get("diagnostic_run_name")
    expected_diagnostic_run_name = (
        f"g2_selected_benchmark_{run_name}_deterministic_reproduction_offline"
    )
    if diagnostic_run_name != expected_diagnostic_run_name:
        raise ValueError("selected-model benchmark diagnostic identity changed")
    selected_directory = logs_root / run_name
    diagnostic_directory = logs_root / diagnostic_run_name
    try:
        selected_metrics = json.loads(
            (selected_directory / "final_metrics.json").read_text()
        )
        diagnostic_metrics = json.loads(
            (diagnostic_directory / "final_metrics.json").read_text()
        )
        selected_metadata = json.loads(
            (selected_directory / "training_metadata.json").read_text()
        )
        diagnostic_metadata = json.loads(
            (diagnostic_directory / "training_metadata.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("selected-model benchmark source artifacts are missing") from error
    if canonical_json_sha256(selected_metrics) != selected_metrics_sha256:
        raise ValueError("selected-model benchmark selected metrics changed")
    if canonical_json_sha256(diagnostic_metrics) != diagnostic_metrics_sha256:
        raise ValueError("selected-model benchmark diagnostic metrics changed")
    if canonical_json_sha256(selected_metadata) != selected_metadata_sha256:
        raise ValueError("selected-model benchmark selected training metadata changed")
    if canonical_json_sha256(diagnostic_metadata) != diagnostic_metadata_sha256:
        raise ValueError("selected-model benchmark diagnostic training metadata changed")
    if selected_metrics != diagnostic_metrics or selected_metadata != diagnostic_metadata:
        raise ValueError("selected-model benchmark reproduction changed")
    if (
        selected_metadata.get("best_epoch") != best_epoch
        or selected_metadata.get("optimizer_steps") != optimizer_steps
        or selected_metadata.get("selection_resolved") is not True
    ):
        raise ValueError("selected-model benchmark training evidence is stale")
    return document
