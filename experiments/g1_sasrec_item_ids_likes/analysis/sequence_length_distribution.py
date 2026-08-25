from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from dcn.datasets.yambda import UserSample
from experiments.generation_protocol import TEST_INTERVAL_SECONDS


EXPERIMENT_DIR = Path(__file__).parents[1]
DEFAULT_EVENTS = Path(
    "generated/datasets/yambda/500m_like_core5_knownitems/events.parquet"
)


@dataclass(frozen=True)
class SequenceLengthAnalysis:
    lengths: tuple[int, ...]
    source: Path
    validation_interval_seconds: int
    sample_name: str | None

    @property
    def user_count(self) -> int:
        return len(self.lengths)

    @property
    def median(self) -> float:
        return float(np.median(self.lengths))

    @classmethod
    def from_parquet(
        cls,
        source: Path,
        *,
        validation_interval_seconds: int = TEST_INTERVAL_SECONDS,
        max_users: int | None = None,
        sample_seed: int = 42,
    ) -> SequenceLengthAnalysis:
        source = Path(source)
        sample = (
            UserSample(max_users=max_users, seed=sample_seed)
            if max_users is not None
            else None
        )
        selected_users = (
            "SELECT DISTINCT uid FROM events"
            if sample is None
            else sample.duckdb_query("events")
        )
        connection = duckdb.connect()
        try:
            connection.execute(
                f"CREATE VIEW events AS SELECT * FROM '{source.resolve()}'"
            )
            lengths = connection.execute(
                f"""
                WITH selected_users AS MATERIALIZED ({selected_users}),
                     cutoff AS (
                         SELECT max(timestamp) - ? AS timestamp FROM events
                     )
                SELECT count(*) AS sequence_length
                FROM events, cutoff
                WHERE uid IN (SELECT uid FROM selected_users)
                  AND events.timestamp < cutoff.timestamp
                GROUP BY uid
                HAVING count(*) >= 2
                ORDER BY uid
                """,
                [validation_interval_seconds],
            ).fetchnumpy()["sequence_length"]
        finally:
            connection.close()
        if not len(lengths):
            raise ValueError("no training histories remain before the validation cutoff")
        return cls(
            lengths=tuple(int(length) for length in lengths),
            source=source,
            validation_interval_seconds=validation_interval_seconds,
            sample_name=sample.name if sample is not None else None,
        )

    def write(self, evidence_dir: Path) -> tuple[Path, Path]:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        summary_path = evidence_dir / "sequence_length_distribution.md"
        plot_path = evidence_dir / "sequence_length_distribution.png"
        percentiles = np.percentile(self.lengths, [25, 50, 75, 90, 95, 99])
        sample = self.sample_name or "all training-eligible users"
        summary_path.write_text(
            "\n".join(
                [
                    "# G1 training-history length distribution",
                    "",
                    f"Source: `{self.source}` ({sample}); the final "
                    f"{self.validation_interval_seconds / 86_400:g} days are excluded.",
                    "",
                    f"Median training-history length: **{self.median:g} events**.",
                    "",
                    f"Users: {self.user_count:,}.",
                    "",
                    "| percentile | p25 | p50 | p75 | p90 | p95 | p99 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    "| events | "
                    + " | ".join(f"{value:g}" for value in percentiles)
                    + " |",
                    "",
                    "Reproduce from the repository root:",
                    "",
                    "```bash",
                    "python experiments/g1_sasrec_item_ids_likes/analysis/sequence_length_distribution.py",
                    "```",
                    "",
                    "![Training-history length distribution](sequence_length_distribution.png)",
                    "",
                ]
            )
        )

        upper = max(2, int(np.percentile(self.lengths, 99)))
        bins = np.unique(
            np.rint(np.geomspace(1, upper + 1, num=40)).astype(int)
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(np.minimum(self.lengths, upper), bins=bins, color="#4472C4")
        axis.axvline(self.median, color="#C00000", linestyle="--", label=f"median={self.median:g}")
        axis.set_xscale("log")
        axis.set_xlabel("Training-history length (events; top 1% clipped)")
        axis.set_ylabel("Users")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        return summary_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--evidence-dir", type=Path, default=EXPERIMENT_DIR / "evidence")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--sample-seed", type=int, default=42)
    arguments = parser.parse_args()
    analysis = SequenceLengthAnalysis.from_parquet(
        arguments.events,
        max_users=arguments.max_users,
        sample_seed=arguments.sample_seed,
    )
    summary, plot = analysis.write(arguments.evidence_dir)
    print(f"median={analysis.median:g} users={analysis.user_count} summary={summary} plot={plot}")


if __name__ == "__main__":
    main()
