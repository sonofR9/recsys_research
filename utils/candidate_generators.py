import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from data.counters.config import DecayConfig, FieldConfig, get_base_counter_columns
from utils.global_config import config
from data.counters.counter import EmaCounter

from data.utils import log_memory

DEFAULT_N_CANDIDATES = 100

UID = "uid"
ITEM_ID = "item_id"
COL_SOURCE = "source"


@dataclass(frozen=True)
class CounterStateSpec:
    key_columns: tuple[str, ...]
    # None means "count all rows" (no filtering by event_type)
    interaction_type: str | None = "like"
    half_life_days: int = 7

    @property
    def field_config(self) -> FieldConfig:
        return FieldConfig.matching(
            "event_type",
            self.interaction_type,
            [DecayConfig(half_life_days=self.half_life_days)],
        )

    @property
    def out_column(self) -> str:
        cols = get_base_counter_columns(list(self.key_columns), [self.field_config])
        return cols[0]


def _build_counter_state(
    source_path: str | Path,
    counters_dir: str | Path,
    days: list[int],
    spec: CounterStateSpec,
    force: bool = False,
) -> Path:
    """Walk the days in order so the counter accumulates, and return its states.

    Candidate generators read the per-day *state* — every key the counter knows
    about — rather than the enriched events, so nothing here writes the packed
    training columns ``preprocess_counters`` produces.
    """
    days = sorted(days)
    log_memory(f"Building counter state for {spec.key_columns} up to day {days[-1]}")

    counter = EmaCounter(
        keys=list(spec.key_columns),
        fields=[spec.field_config],
        cache_dir=Path(counters_dir),
    )

    for day in days:
        day_df = pl.read_parquet(Path(source_path) / f"day_{day:04d}.parquet")
        counter.process_day(day, day_df, invalidate_cache=force)

    return counter.state_path()


class BaseCandidateGenerator(ABC):
    def __init__(self):
        self._is_fitted = False

        self.cache_dir = config.candgen_path / self.generator_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def generator_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self, data_path: str | Path, force: bool = False
    ) -> "BaseCandidateGenerator":
        log_memory(f"{self.generator_name}.fit start force={force}")

        if force and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=False)

        self._fit_impl(str(data_path), force)
        self._is_fitted = True

        log_memory(f"{self.generator_name} fitted", logging.INFO)
        return self

    @abstractmethod
    def _fit_impl(self, data_path: str, force: bool) -> None:
        pass

    @abstractmethod
    def _generate_batch_impl(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> pl.DataFrame:
        pass

    def generate_batch(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        filter_seen: bool = False,
        deduplicate: bool = True,
    ) -> pl.DataFrame:
        log_memory(f"{self.generator_name}.generate_batch start")
        self._check_fitted()

        result = self._generate_batch_impl(users_df, day, n_candidates)

        if deduplicate:
            # Generators return their candidates best-first and the head() below
            # reads that order, so dedup has to keep it.
            result = result.unique(
                subset=[UID, ITEM_ID], keep="first", maintain_order=True
            )
        if filter_seen:
            result = self._filter_seen(result, users_df, day)
        result = result.group_by(UID, maintain_order=True).head(n_candidates)

        log_memory(f"{self.generator_name}.generate_batch end shape={result.shape}")
        return result

    def _filter_seen(
        self, candidates: pl.DataFrame, users_df: pl.DataFrame, day: int
    ) -> pl.DataFrame:
        # FIXME: implement        # FIXME: extremelly scatchy and why here and not in the base?
        # if "user_seen" not in self.state_specs_by_counter:
        #     return candidates

        # seen_col = self.state_specs_by_counter["user_seen"].out_column
        # user_seen = self._get_cached_state("user_seen", day).filter(
        #     pl.col(seen_col) > 0
        # )
        # return candidates.join(user_seen, on=[COL_UID, COL_ITEM_ID], how="anti")

        return candidates

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

    def _get_cache_path(self, spec_key: str, day: int) -> Path:
        _dir = self.cache_dir / spec_key / f"day_{day:04d}.parquet"
        _dir.parent.mkdir(parents=True, exist_ok=True)
        return _dir


class CounterBasedCandidateGeneratorBase(BaseCandidateGenerator):
    def __init__(self, decay_days: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.counters_dir = config.counters_path

        self.decay_days = decay_days
        self.state_specs_by_counter: dict[str, CounterStateSpec] = {}

        self.states_by_counter: dict[str, Path] = {}

    def _ensure_counter_state(
        self,
        data_path: str,
        spec: CounterStateSpec,
        spec_key: str,
        force: bool = False,
    ) -> None:
        days = []
        for parquet in Path(data_path).iterdir():
            if parquet.name.startswith("day_") and parquet.name.endswith(".parquet"):
                days.append(int(parquet.name[4:-8]))

        self.states_by_counter[spec_key] = _build_counter_state(
            data_path,
            self.counters_dir,
            days,
            spec,
            force,
        )

    def _get_state_path(self, spec_key: str, day: int) -> Path:
        return self.states_by_counter[spec_key] / f"day_{day:04d}.parquet"

    def _get_cached_state(
        self,
        spec_key: str,
        day: int,
        preprocess_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        cache_path = self._get_cache_path(spec_key, day)

        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # FIXME: recheck everything on data leaks! It used day + 1 here...
        state_path = self._get_state_path(spec_key, day)
        df = pl.read_parquet(state_path)

        if preprocess_fn is not None:
            df = preprocess_fn(df)
            df.write_parquet(cache_path)
        else:
            os.symlink(state_path.absolute(), cache_path.absolute())

        return df


class DecayedPopularityGenerator(CounterBasedCandidateGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spec_key = "item_popularity"

    def _fit_impl(self, data_path: str, force: bool) -> None:
        spec = CounterStateSpec(
            key_columns=(ITEM_ID,),
            interaction_type="like",
            half_life_days=self.decay_days,
        )
        self.state_specs_by_counter["decay"] = spec
        self._ensure_counter_state(data_path, spec, self.spec_key, force)

    def _generate_batch_impl(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> pl.DataFrame:
        ema_column = self.state_specs_by_counter["decay"].out_column

        def preprocess(df: pl.DataFrame) -> pl.DataFrame:
            return df.sort(ema_column, descending=True).select(ITEM_ID)

        top_items = self._get_cached_state(
            self.spec_key, day, preprocess_fn=preprocess
        ).head(n_candidates)

        return (
            users_df.select(UID)
            .join(top_items, how="cross")
            .with_columns(
                pl.lit(f"popularity_decay_{self.decay_days}d").alias(COL_SOURCE)
            )
        )


class EntityBasedGeneratorBase(CounterBasedCandidateGeneratorBase):
    def __init__(self, entity_col: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_col = entity_col

    def _fit_impl(self, data_path: str, force: bool) -> None:
        for spec_key, key_columns in [
            ("entity_items", (self.entity_col, ITEM_ID)),
            ("user_entities", (UID, self.entity_col)),
        ]:
            spec = CounterStateSpec(
                key_columns=key_columns,
                interaction_type="like",
                half_life_days=self.decay_days,
            )
            self.state_specs_by_counter[spec_key] = spec
            self._ensure_counter_state(data_path, spec, spec_key, force)

    def _generate_batch_impl(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> pl.DataFrame:
        entity_items_col = self.state_specs_by_counter["entity_items"].out_column
        user_entities_col = self.state_specs_by_counter["user_entities"].out_column

        entity_items = self._get_cached_state("entity_items", day)
        user_entities = self._get_cached_state("user_entities", day)

        candidates = (
            user_entities.join(users_df.select(UID), on=UID)
            .join(entity_items, on=self.entity_col)
            .with_columns(
                (pl.col(user_entities_col) * pl.col(entity_items_col)).alias("score")
            )
            .group_by([UID, ITEM_ID])
            .agg(pl.col("score").sum())
        )

        return (
            candidates.sort([UID, "score"], descending=[False, True])
            .group_by(UID)
            .head(n_candidates)
            .select([UID, ITEM_ID])
            .with_columns(pl.lit(self.entity_col).alias(COL_SOURCE))
        )


class ArtistBasedGenerator(EntityBasedGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(entity_col="artist", *args, **kwargs)


class AlbumBasedGenerator(EntityBasedGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(entity_col="album", *args, **kwargs)


# FIXME: add new popular items generator


class UserSeenItemsGenerator(CounterBasedCandidateGeneratorBase):
    def __init__(self, interaction_type: str | None = "like", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interaction_type = interaction_type
        self.state_key = "user_seen"

    def _fit_impl(self, data_path: str, force: bool) -> None:
        user_seen_spec = CounterStateSpec(
            key_columns=(UID, ITEM_ID),
            interaction_type=self.interaction_type,
            half_life_days=self.decay_days,
        )
        self.state_specs_by_counter[self.state_key] = user_seen_spec
        self._ensure_counter_state(data_path, user_seen_spec, self.state_key, force)

    def _generate_batch_impl(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> pl.DataFrame:
        users = users_df.select(UID)

        seen_col = self.state_specs_by_counter[self.state_key].out_column
        return (
            self._get_cached_state(self.state_key, day)
            .filter(pl.col(seen_col) > 0)
            .join(users, on=UID, how="inner")
            .sample(fraction=1.0, shuffle=True)
            .group_by(UID)
            .head(n_candidates)
            .select([UID, ITEM_ID])
            .with_columns(pl.lit("user_seen").alias(COL_SOURCE))
        )


class EnsembleCandidateGenerator(BaseCandidateGenerator):
    def __init__(self, generators: list[tuple], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generators: list[tuple[BaseCandidateGenerator, float]] = []

        for gen_tuple in generators:
            if len(gen_tuple) == 2:
                gen_class, weight = gen_tuple
                gen_kwargs = {}
            else:
                gen_class, weight, gen_kwargs = gen_tuple
            generator = gen_class(**gen_kwargs)
            self.generators.append((generator, weight))

    @property
    def is_fitted(self) -> bool:
        return all(gen.is_fitted for gen, _ in self.generators)

    @property
    def total_weight(self) -> float:
        return sum(weight for _, weight in self.generators)

    def _fit_impl(self, data_path: str, force: bool) -> None:
        for generator, _ in self.generators:
            generator.fit(data_path, force=force)

    def _compute_n_candidates_per_generator(self, n_candidates: int) -> list[int]:
        allocations = [
            int(n_candidates * w / self.total_weight) for _, w in self.generators
        ]
        allocations[-1] = n_candidates - sum(allocations) + allocations[-1]
        return allocations

    def _generate_batch_impl(
        self,
        users_df: pl.DataFrame,
        day: int,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> pl.DataFrame:
        allocations = self._compute_n_candidates_per_generator(n_candidates)
        all_results = []

        for (generator, _), n_alloc in zip(self.generators, allocations):
            if n_alloc <= 0:
                continue

            gen_name = generator.__class__.__name__
            try:
                candidates_df = generator.generate_batch(
                    users_df,
                    day,
                    n_alloc * 1.2,
                )
                if candidates_df.height > 0:
                    all_results.append(candidates_df)
                # FIXME: perform deduplication gradually (before adding new deduplicate and preserve n_alloc candidates only)
            except Exception as e:
                log_memory(f"Generator {gen_name} failed: {e}", logging.WARNING)
                raise

        log_memory("EnsembleCandidateGenerator.generate_batch combining")
        return pl.concat(all_results)


# FIXME: days should be passed outside
def create_ensemble(
    generators_config: list[tuple],
    full_data_path: str | Path,
    force: bool = False,
) -> EnsembleCandidateGenerator:
    log_memory(f"create_ensemble start force={force}")

    ensemble = EnsembleCandidateGenerator(generators_config)
    ensemble.fit(full_data_path, force=force)

    log_memory("create_ensemble end")
    return ensemble
