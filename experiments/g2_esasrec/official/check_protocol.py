from pathlib import Path

from experiments.g2_esasrec.official import protocol, provenance

EXPECTED = {
    "cutoff": 25394930,
    "catalog_size": 33148,
    "train_catalog_size": 33112,
    "train_events": 614244,
    "validation_events": 20398,
    "evaluable_users": 3414,
    "candidate_catalog_sha256": (
        "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"
    ),
    "rectools_source_contract_sha256": (
        "837dee3b2e6026d1618dc3d6e5aef762de1b8d07306ea024688b7ca506039eae"
    ),
}


def observed(generated: Path) -> dict[str, int | str]:
    split = protocol.load_split(generated)
    users = protocol.evaluable_users(
        protocol.query_histories(split, max_seq_len=100), protocol.relevance(split)
    )
    evidence = protocol.candidate_catalog_evidence(split, split.catalog)
    return {
        "cutoff": split.cutoff,
        "catalog_size": split.catalog_size,
        "train_catalog_size": split.train.get_column(protocol.ITEM_COLUMN).n_unique(),
        "train_events": split.train.height,
        "validation_events": split.validation.height,
        "evaluable_users": len(users),
        "candidate_catalog_sha256": evidence["candidate_catalog_sha256"],
        "rectools_source_contract_sha256": (
            provenance.rectools_source_contract_sha256()
        ),
    }


def main() -> int:
    generated = Path(__file__).resolve().parents[3] / "generated"
    actual = observed(generated)
    print(actual)
    return actual != EXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
