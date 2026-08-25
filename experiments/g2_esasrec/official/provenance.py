from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

RECTOOLS_VERSION = "0.19.0"
OFFICIAL_PYTHON_VERSION = "3.12.13"
OFFICIAL_PACKAGE_VERSIONS = {
    "RecTools": RECTOOLS_VERSION,
    "torch": "2.7.1",
    "pytorch-lightning": "2.5.2",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "polars": "1.43.2",
}
RECTOOLS_SOURCE_MODULES = {
    "rectools_init": "rectools",
    "rectools_columns": "rectools.columns",
    "rectools_dataset": "rectools.dataset.dataset",
    "rectools_identifiers": "rectools.dataset.identifiers",
    "rectools_interactions": "rectools.dataset.interactions",
    "rectools_utils_array_set_ops": "rectools.utils.array_set_ops",
    "rectools_utils_indexing": "rectools.utils.indexing",
    "rectools_utils_misc": "rectools.utils.misc",
    "rectools_model_base": "rectools.models.base",
    "rectools_item_net": "rectools.models.nn.item_net",
    "rectools_transformer_base": "rectools.models.nn.transformers.base",
    "rectools_data_preparator": "rectools.models.nn.transformers.data_preparator",
    "rectools_transformer_constants": "rectools.models.nn.transformers.constants",
    "rectools_negative_sampler": "rectools.models.nn.transformers.negative_sampler",
    "rectools_net_blocks": "rectools.models.nn.transformers.net_blocks",
    "rectools_ligr": "rectools.models.nn.transformers.ligr",
    "rectools_lightning": "rectools.models.nn.transformers.lightning",
    "rectools_torch_backbone": "rectools.models.nn.transformers.torch_backbone",
    "rectools_similarity": "rectools.models.nn.transformers.similarity",
    "rectools_sasrec": "rectools.models.nn.transformers.sasrec",
    "rectools_rank_init": "rectools.models.rank",
    "rectools_rank_base": "rectools.models.rank.rank",
    "rectools_rank_torch": "rectools.models.rank.rank_torch",
}
RECTOOLS_SOURCE_SHA256 = {
    "rectools_init": "6d56ec90f6e2c2a60a6439e4a382eeafd4e49b809ab04b39914ebf8dae105384",
    "rectools_columns": "bedcbb829944662bbca5b50be704cbf03b30302e217ba413d818bd82b29bd918",
    "rectools_dataset": "0fbdf07058bb68d5d0016dfcc9b1830122a544ac9a98cf42bb5780de55a9e757",
    "rectools_identifiers": (
        "e2775e0eaa51da6ed28b22ef12a7c4a119cbc5f3edf6b5429a12db7fe98c970a"
    ),
    "rectools_interactions": (
        "f292cac6305ed3559fccedb92be2133a953e1c4e785f6eae16f463914af26552"
    ),
    "rectools_utils_array_set_ops": (
        "d5967ae165b917bc57f5e49f0f8c53771ebb28a8afacaf8bc20ad303da7a388e"
    ),
    "rectools_utils_indexing": (
        "d751763f58c3d15000a7479c850dbe89f98112c89c42d5cd6ce86ee7acfab27f"
    ),
    "rectools_utils_misc": (
        "52ad17d6e55ff9556f4a073e5c0528c096c0a9024c06069e796b2a22c6165b36"
    ),
    "rectools_model_base": (
        "33306dba9fd6b3fcea2ba08b2087eef94b2d3f8b0ffd94d1b7211393552366c3"
    ),
    "rectools_item_net": (
        "8b0abd6d64f291df337fccd98370a1028240af38103345905412aa352cec87fa"
    ),
    "rectools_transformer_base": (
        "688183fad14ba2a7802922802637350cca461972d6d02103c7b3240238d12186"
    ),
    "rectools_data_preparator": (
        "c8c9a6bb16accc563407b91c78e89740faa16050dac60ff2f11047ef90f57c0c"
    ),
    "rectools_transformer_constants": (
        "5a099f16865bd9b04b6346f1ec74405d5fa5b15fb092afc041aca7c3dd88b1c5"
    ),
    "rectools_negative_sampler": (
        "0e3831dc06c171965f21686770f686db6b9d57363767588bc48578aac7fefa91"
    ),
    "rectools_net_blocks": (
        "2a41aa8f3d0d7f50c6620002d90bd9be3a8820f23d03b76e25bb7bc7e9b76483"
    ),
    "rectools_ligr": (
        "1970236e381b1361680903e7327cc219a47fec6a1d5a8cb51840bdb5b3fccb60"
    ),
    "rectools_lightning": (
        "fa7fa54fd8db2b888e75a32e105c2bb068f8b0046932030129d1bde94d8e1db9"
    ),
    "rectools_torch_backbone": (
        "b246a1a9c89e56ee466920b7f86468791c7c7b930157d52191d3be2c4e542d47"
    ),
    "rectools_similarity": (
        "b7820f3ba807b2d14aeb64449a04a10bf0502173defc8fc748215f964c867032"
    ),
    "rectools_sasrec": (
        "464d2cb24552eeeb194c76620573e72a48ab90732ae930f10c45ce57dd822c25"
    ),
    "rectools_rank_init": (
        "cf5064ef6152e900800f35d17bec0453e20707d5d4693355b49b822086c3bdd7"
    ),
    "rectools_rank_base": (
        "efa1b88bdc14fa0bb57f9ac33c4bc5d45b67a77d5faad5f3344774f9677f4956"
    ),
    "rectools_rank_torch": (
        "b34b7fae790c05589b14649cf319e353775472223d726bd47a9ca08c03642747"
    ),
}
OFFICIAL_HYPERPARAMETERS = {
    "n_blocks": 2,
    "n_heads": 4,
    "n_factors": 256,
    "dropout_rate": 0.2,
    "session_max_len": 100,
    "n_negatives": 256,
    "lr": 0.001,
    "batch_size": 128,
}
OFFICIAL_PROTOCOL = {
    "cutoff": 25394930,
    "catalog_size": 33148,
    "model_candidate_catalog_size": 33148,
    "train_catalog_size": 33112,
    "mapped_items_absent_from_training": 36,
    "candidate_catalog_sha256": (
        "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"
    ),
    "candidate_catalog_source": "full mapped pre-split catalog",
    "train_events": 614244,
    "model_train_events_after_session_truncation": 362723,
    "validation_events": 20398,
    "evaluable_users": 3414,
    "eval_ks": [10, 50, 100],
    "selection_metric": "recall@100",
    "exclude_seen": False,
}


def source_manifest(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in sorted(paths.items())
    }


def rectools_source_paths() -> dict[str, Path]:
    paths = {}
    for name, module_name in RECTOOLS_SOURCE_MODULES.items():
        module_path = getattr(importlib.import_module(module_name), "__file__", None)
        if not isinstance(module_path, str):
            raise RuntimeError(f"RecTools source module has no file: {module_name}")
        paths[name] = Path(module_path)
    return paths


def rectools_source_contract_sha256() -> str:
    payload = {
        "environment": {
            "packages": OFFICIAL_PACKAGE_VERSIONS,
            "python": OFFICIAL_PYTHON_VERSION,
        },
        "modules": RECTOOLS_SOURCE_MODULES,
        "sha256": RECTOOLS_SOURCE_SHA256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
