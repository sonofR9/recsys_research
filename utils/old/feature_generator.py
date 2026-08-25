import gc
import logging
import os
import shutil
import tempfile
from pathlib import Path

import duckdb
import polars as pl

from utils.global_config import config as global_config
from .utils import TO_DAY, log_memory, setup_duckdb_connection


def validate_parquet_path(path: str | Path, source_name: str) -> str:
    path_str = str(path)
    log_memory(f"read {source_name}")
    return f"read_parquet('{path_str}')"


def validate_dataframe(
    data: pl.DataFrame,
    source_name: str,
    temp_dir_path: str,
    max_size_gb: float,
) -> str:
    estimated_size_gb = data.estimated_size() / (1024**3)

    if estimated_size_gb > max_size_gb:
        raise AssertionError(
            f"In-memory DataFrame for {source_name} is too large "
            f"({estimated_size_gb:.2f} GB > {max_size_gb} GB). "
            f"Please pass large datasets as parquet file paths."
        )

    parquet_path = os.path.join(temp_dir_path, f"{source_name}.parquet")
    log_memory(f"Writing small {source_name} DataFrame to parquet")
    data.write_parquet(parquet_path)
    log_memory(f"Written {source_name} to {parquet_path}")

    return f"read_parquet('{parquet_path}')"


def validate_lazyframe(
    data: pl.LazyFrame,
    source_name: str,
    temp_dir_path: str,
) -> str:
    parquet_path = os.path.join(temp_dir_path, f"{source_name}.parquet")
    log_memory(f"Sinking {source_name} LazyFrame to parquet")
    data.sink_parquet(parquet_path)
    log_memory(f"Written {source_name} LazyFrame to {parquet_path}")

    return f"read_parquet('{parquet_path}')"


def validate_and_get_source(
    data: pl.DataFrame | pl.LazyFrame | str | Path,
    source_name: str,
    temp_dir_path: str,
    max_size_gb: float,
) -> str:
    """
    Validate data source and return DuckDB-compatible source string.

    For parquet files: returns direct path for DuckDB streaming
    For in-memory data: asserts it's small, writes to temp parquet
    """
    if isinstance(data, (str, Path)):
        return validate_parquet_path(data, source_name)

    if isinstance(data, pl.DataFrame):
        return validate_dataframe(data, source_name, temp_dir_path, max_size_gb)

    if isinstance(data, pl.LazyFrame):
        return validate_lazyframe(data, source_name, temp_dir_path)

    raise ValueError(f"Unsupported data type for {source_name}: {type(data)}")


def get_table_schema(con: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    """Get column name to type mapping for a table."""
    schema_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    return {row[0]: row[1].upper() for row in schema_info}


def create_target_base_table(
    con: duckdb.DuckDBPyConnection,
    target_source: str,
) -> dict[str, str]:
    log_memory("Starting target_base materialization")

    schema_check = con.execute(f"DESCRIBE SELECT * FROM {target_source}").fetchall()
    source_cols = [row[0] for row in schema_check if row[0] != "day"]
    cols_str = ", ".join(source_cols)

    con.execute(
        f"""
        CREATE TABLE target_base AS
        SELECT {cols_str},
               {TO_DAY} AS day
        FROM {target_source};
    """
    )
    con.execute("CHECKPOINT;")
    log_memory("target_base created")

    return get_table_schema(con, "target_base")


def create_events_base(
    con: duckdb.DuckDBPyConnection, events_source: str, same_source: bool
) -> None:
    if same_source:
        log_memory("Same source detected - using VIEW for events")
    else:
        log_memory("Different sources - creating VIEW for events")

    con.execute(
        f"""
        CREATE {"VIEW" if same_source else "TABLE"} events_base AS
        SELECT *,
            {TO_DAY} AS day,
            (event_type = 'like')::TINYINT AS is_like,
            (event_type = 'dislike')::TINYINT AS is_dislike,
            (event_type = 'listen')::TINYINT AS is_listen,
            ((is_organic) AND (event_type = 'like'))::TINYINT AS is_org_like,
            ((is_organic) AND (event_type = 'dislike'))::TINYINT AS is_org_dislike,
            ((is_organic) AND (event_type = 'listen'))::TINYINT AS is_org_listen,
            ((is_organic) AND (event_type IN ['like', 'dislike', 'listen']))::TINYINT AS is_org_any
        FROM {events_source}
        WHERE event_type IN ['like', 'dislike', 'listen'];
    """
    )
    # FIXME(sashanovak): listen should go to total count?
    if not same_source:
        con.execute("CHECKPOINT;")

    log_memory("events_base created")


def materialize_base_tables(
    con: duckdb.DuckDBPyConnection,
    events_source: str,
    target_source: str,
    same_source: bool,
) -> dict[str, str]:
    tgt_schema = create_target_base_table(con, target_source)

    log_memory("Starting events processing")

    create_events_base(con, events_source, same_source)

    return tgt_schema


def build_initial_selects(
    tgt_schema: dict[str, str],
    target_for_fake: float,
) -> list[str]:
    """Build initial SELECT clauses for target columns."""
    t_cols = [c for c in tgt_schema.keys() if c != "target" and c != "day"]
    selects = ["t.day"] + [f"t.{c}" for c in t_cols]

    if "event_type" in tgt_schema.keys():
        selects.append(
            f"CASE WHEN t.event_type = 'like' THEN 1.0 "
            f"WHEN t.event_type = 'dislike' THEN 0.0 ELSE {target_for_fake} END::FLOAT AS target"
        )
    else:
        selects.append(f"{target_for_fake}::FLOAT AS target")

    return selects


def build_column_expressions(
    entity_cols: list[str],
    tgt_schema: dict[str, str],
    fake_int: int,
    fake_str: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    agg_cols = []
    key_cols = []
    restored_cols = []
    id_filters = []

    for col in entity_cols:
        is_str = "VARCHAR" in tgt_schema.get(col, "VARCHAR")

        if is_str:
            fill = f"'{fake_str}'"
            nullif_val = f"'{fake_str}'"
            val_0 = "'0'"
        else:
            fill = f"{fake_int}"
            nullif_val = str(fake_int)
            val_0 = "0"

        agg_cols.append(f"COALESCE({col}, {fill}) AS {col}")
        key_cols.append(f"COALESCE({col}, {fill}) AS {col}")
        restored_cols.append(f"NULLIF({col}, {nullif_val}) AS {col}")

        if "id" in col:
            id_filters.append(f"{col} != {val_0} AND {col} IS NOT NULL")

    return agg_cols, key_cols, restored_cols, id_filters


def build_window_definitions(
    entity_cols: list[str],
    e_plan: dict,
) -> list[str]:
    window_defs = [
        f"w_life AS (PARTITION BY {', '.join(entity_cols)} ORDER BY day "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
    ]

    for window in e_plan["windows"].keys():
        win_val = int(window.replace("d", ""))
        w_name = f"w_{window}"
        window_defs.append(
            f"{w_name} AS (PARTITION BY {', '.join(entity_cols)} ORDER BY day "
            f"RANGE BETWEEN INTERVAL {win_val} DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING)"
        )

    return window_defs


def build_base_expressions(
    prefix: str,
    e_plan: dict,
    base_agg_to_metric: dict[str, str],
) -> list[str]:
    base_exprs = []

    for agg in e_plan["life"]["base_aggs"]:
        metr = base_agg_to_metric[agg]
        base_exprs.append(
            f"COALESCE(SUM({agg}) OVER w_life, 0) AS {prefix}_{metr}_life"
        )

    for window, w_plan in e_plan["windows"].items():
        w_name = f"w_{window}"
        for agg in w_plan["base_aggs"]:
            metric = base_agg_to_metric[agg]
            base_exprs.append(
                f"COALESCE(SUM({agg}) OVER {w_name}, 0) AS {prefix}_{metric}_{window}"
            )

    return base_exprs


def build_ratio_expressions(
    prefix: str,
    e_plan: dict,
    alpha: int,
    beta: int,
) -> list[str]:
    ratio_exprs = []

    for window, w_plan in e_plan["windows"].items():
        ratios = w_plan["ratios"]

        if "like_ratio" in ratios:
            ratio_exprs.append(
                f"({prefix}_likes_{window} + {alpha})::FLOAT / "
                f"({prefix}_cnt_{window} + {beta})::FLOAT AS {prefix}_like_ratio_{window}"
            )

        if "like_all_ratio" in ratios:
            ratio_exprs.append(
                f"({prefix}_likes_{window} + {alpha})::FLOAT / "
                f"({prefix}_all_{window} + {beta})::FLOAT AS {prefix}_like_all_ratio_{window}"
            )

        if "org_like_ratio" in ratios:
            ratio_exprs.append(
                f"({prefix}_org_likes_{window} + {alpha})::FLOAT / "
                f"({prefix}_org_cnt_{window} + {beta})::FLOAT AS {prefix}_org_like_ratio_{window}"
            )

    return ratio_exprs


def get_final_required_columns(e_plan: dict) -> list[str]:
    final_req = list(e_plan["life"]["final_cols"])

    for w_plan in e_plan["windows"].values():
        final_req.extend(list(w_plan["final_cols"]))

    return final_req


def build_join_clause(
    prefix: str,
    entity_cols: list[str],
    table_name: str,
) -> str:
    f_alias = f"f_{prefix}"
    join_conds = [f"t.day = {f_alias}.day"]
    join_conds.extend(
        [f"t.{col} IS NOT DISTINCT FROM {f_alias}.{col}" for col in entity_cols]
    )

    return f"LEFT JOIN {table_name} AS {f_alias} ON {' AND '.join(join_conds)}"


def build_select_clauses(
    prefix: str,
    final_req: list[str],
) -> list[str]:
    f_alias = f"f_{prefix}"
    select_clauses = []

    for c in final_req:
        if "_ratio_" in c:
            select_clauses.append(f"COALESCE({f_alias}.{c}, 0.0)::FLOAT AS {c}")
        else:
            select_clauses.append(f"COALESCE({f_alias}.{c}, 0)::UINTEGER AS {c}")

    return select_clauses


def create_aggregation_table(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    agg_cols: list[str],
    where_clause: str,
) -> None:
    log_memory(f"Creating aggregation table for {prefix}")

    con.execute(
        f"""
        CREATE TABLE tbl_{prefix}_agg AS
        SELECT day, {", ".join(agg_cols)},
            SUM(is_like) AS daily_likes,
            SUM(is_dislike) AS daily_dislikes,
            SUM(is_listen) AS daily_listens,
            SUM(is_like + is_dislike) AS daily_count,
            COUNT(*) AS daily_all,
            SUM(is_org_like) AS daily_org_likes,
            SUM(is_org_dislike) AS daily_org_dislikes,
            SUM(is_org_like + is_org_dislike) AS daily_org_count,
            SUM(is_org_listen) AS daily_org_listens,
            SUM(is_org_any) AS daily_org_all
        FROM events_base {where_clause}
        GROUP BY ALL;
    """
    )
    con.execute("CHECKPOINT;")
    log_memory(f"Aggregation table created for {prefix}")


def create_spine_table(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    entity_cols: list[str],
    key_cols: list[str],
) -> None:
    tbl_prefix = f"tbl_{prefix}"
    cols_str = ", ".join(entity_cols)

    con.execute(
        f"""
        CREATE TABLE {tbl_prefix}_spine AS
        SELECT day, {cols_str},
               SUM(daily_likes) as daily_likes,
               SUM(daily_dislikes) as daily_dislikes,
               SUM(daily_listens) as daily_listens,
               SUM(daily_count) as daily_count,
               SUM(daily_all) as daily_all,
               SUM(daily_org_likes) as daily_org_likes,
               SUM(daily_org_dislikes) as daily_org_dislikes,
               SUM(daily_org_listens) as daily_org_listens,
               SUM(daily_org_count) as daily_org_count,
               SUM(daily_org_all) as daily_org_all
        FROM (
            SELECT day, {cols_str},
                   daily_likes, daily_dislikes, daily_listens,
                   daily_count, daily_all,
                   daily_org_likes, daily_org_dislikes, daily_org_listens,
                   daily_org_count, daily_org_all
            FROM {tbl_prefix}_agg
            UNION ALL
            SELECT DISTINCT day, {", ".join(key_cols)}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            FROM target_base
            WHERE day IS NOT NULL
        ) GROUP BY ALL;
    """
    )

    con.execute(f"DROP TABLE {tbl_prefix}_agg;")
    con.execute("CHECKPOINT;")
    log_memory(f"Spine table created for {prefix}")


def create_window_table(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    restored_cols: list[str],
    base_exprs: list[str],
    window_defs: list[str],
) -> None:
    tbl_prefix = f"tbl_{prefix}"

    base_select_str = f", {', '.join(base_exprs)}" if base_exprs else ""
    window_clause = f" WINDOW {', '.join(window_defs)}" if base_exprs else ""

    con.execute(
        f"""
        CREATE TABLE {tbl_prefix}_win AS
        SELECT day, {", ".join(restored_cols)} {base_select_str}
        FROM {tbl_prefix}_spine {window_clause};
    """
    )
    # FIXME(sashanovak): QUALIFY day BETWEEN 100 AND 120 if I want only between some days

    con.execute(f"DROP TABLE {tbl_prefix}_spine;")
    con.execute("CHECKPOINT;")
    log_memory(f"Window table created for {prefix}")


def create_final_entity_table(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    ratio_exprs: list[str],
) -> None:
    tbl_prefix = f"tbl_{prefix}"
    ratio_select_str = f", {', '.join(ratio_exprs)}" if ratio_exprs else ""

    con.execute(
        f"""
        CREATE TABLE {tbl_prefix}_final AS
        SELECT * {ratio_select_str} FROM {tbl_prefix}_win;
    """
    )

    con.execute(f"DROP TABLE {tbl_prefix}_win;")
    con.execute("CHECKPOINT;")
    log_memory(f"Final table created for {prefix}")


def process_entity_prefix(
    con: duckdb.DuckDBPyConnection,
    prefix: str,
    entity_cols: list[str],
    tgt_schema: dict[str, str],
    plan: dict,
    base_agg_to_metric: dict[str, str],
    alpha: int,
    beta: int,
    fake_int: int,
    fake_str: str,
) -> tuple[str, list[str]]:
    if prefix not in plan:
        return "", []

    log_memory(f"Processing entity prefix: {prefix}")

    e_plan = plan[prefix]

    agg_cols, key_cols, restored_cols, id_filters = build_column_expressions(
        entity_cols, tgt_schema, fake_int, fake_str
    )

    where_clause = "WHERE " + " AND ".join(id_filters) if id_filters else ""

    window_defs = build_window_definitions(entity_cols, e_plan)
    base_exprs = build_base_expressions(prefix, e_plan, base_agg_to_metric)
    ratio_exprs = build_ratio_expressions(prefix, e_plan, alpha, beta)
    final_req = get_final_required_columns(e_plan)

    if not final_req:
        return "", []

    create_aggregation_table(con, prefix, agg_cols, where_clause)
    create_spine_table(con, prefix, entity_cols, key_cols)
    create_window_table(con, prefix, restored_cols, base_exprs, window_defs)
    create_final_entity_table(con, prefix, ratio_exprs)

    table_name = f"tbl_{prefix}_final"
    join_clause = build_join_clause(prefix, entity_cols, table_name)
    select_clauses = build_select_clauses(prefix, final_req)

    return join_clause, select_clauses


def create_base_result_table(
    con: duckdb.DuckDBPyConnection,
    final_selects: list[str],
) -> None:
    base_selects = [s for s in final_selects if s.startswith("t.") or "AS target" in s]

    con.execute(
        f"""
        CREATE TABLE result_current AS
        SELECT {", ".join(base_selects)}
        FROM target_base t;
    """
    )
    # FIXME(sashanovak): optional filtering by day here

    con.execute("DROP TABLE target_base;")
    # con.execute("DROP VIEW IF EXISTS events_base;")
    # con.execute("DROP TABLE IF EXISTS events_base;")
    con.execute("CHECKPOINT;")
    log_memory("Base result created")


def join_entity_to_result(
    con: duckdb.DuckDBPyConnection,
    join_clause: str,
    prefix: str,
    final_selects: list[str],
) -> None:
    """Join a single entity table to the current result."""
    prefix_selects = [
        s
        for s in final_selects
        if f"f_{prefix}." in s and "AS target" not in s and not s.startswith("t.")
    ]

    if not prefix_selects:
        con.execute(f"DROP TABLE IF EXISTS tbl_{prefix}_final;")
        return

    adjusted_join = join_clause.replace(" t.", " result_current.")
    adjusted_join = adjusted_join.replace("(t.", "(result_current.")

    con.execute(
        f"""
        CREATE TABLE result_next AS
        SELECT result_current.*, {", ".join(prefix_selects)}
        FROM result_current
        {adjusted_join};
    """
    )

    con.execute("DROP TABLE result_current;")
    con.execute(f"DROP TABLE IF EXISTS tbl_{prefix}_final;")
    con.execute("ALTER TABLE result_next RENAME TO result_current;")
    con.execute("CHECKPOINT;")
    log_memory(f"Joined {prefix}")


def export_result_to_parquet(
    con: duckdb.DuckDBPyConnection,
    output_path: str,
) -> None:
    con.execute(
        f"""
        COPY result_current TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """
    )
    con.execute("DROP TABLE result_current;")
    con.execute("CHECKPOINT;")
    log_memory("Exported to parquet")


def assemble_final_result(
    con: duckdb.DuckDBPyConnection,
    joins: list[str],
    join_metadata: list[str],
    final_selects: list[str],
    output_path: str,
) -> None:
    log_memory("Starting incremental final assembly")

    create_base_result_table(con, final_selects)

    for join_clause, prefix in zip(joins, join_metadata):
        join_entity_to_result(con, join_clause, prefix, final_selects)

    export_result_to_parquet(con, output_path)


def load_result(
    result_parquet: str,
    temp_dir_name: str,
    return_lazy: bool,
) -> pl.DataFrame | pl.LazyFrame:
    """Load result from parquet file."""
    if return_lazy:
        persistent_path = f".tmp/features_result_{os.getpid()}.parquet"
        shutil.copy(result_parquet, persistent_path)
        result_df = pl.scan_parquet(persistent_path)
        log_memory("Returning lazy frame")
    else:
        result_df = pl.read_parquet(result_parquet)
        log_memory("Loaded result into memory")

    return result_df


def generate_all_possible_features(
    entity_mappings: dict[str, list[str]],
    windows: list[str],
) -> list[str]:
    features = []

    for prefix in entity_mappings.keys():
        for metric in [
            "likes_life",
            "cnt_life",
            "all_life",
            "listens_life",
            "org_likes_life",
            "org_cnt_life",
            "org_all_life",
            "org_listens_life",
        ]:
            features.append(f"{prefix}_{metric}")

        for w in windows:
            for metric in [
                "likes",
                "dislikes",
                "listens",
                "cnt",
                "all",
                "org_likes",
                "org_listens",
                "org_cnt",
                "org_all",
                "like_ratio",
                "like_all_ratio",
                "org_like_ratio",
            ]:
                features.append(f"{prefix}_{metric}_{w}")

    return features


def parse_feature_name(
    feat: str,
    prefixes: list[str],
) -> tuple[str | None, str]:
    matched_prefix = next((p for p in prefixes if feat.startswith(p + "_")), None)

    if matched_prefix is None:
        return None, ""

    remainder = feat[len(matched_prefix) + 1 :]
    return matched_prefix, remainder


def add_lifetime_feature_to_plan(
    plan: dict,
    prefix: str,
    feat: str,
    remainder: str,
) -> None:
    plan[prefix]["life"]["final_cols"].add(feat)

    if "org_likes" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_org_likes")
    elif "org_listens" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_org_listens")
    elif "org_cnt" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_org_count")
    elif "org_all" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_org_all")
    elif "listens" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_listens")
    elif "all" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_all")
    elif "likes" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_likes")
    elif "cnt" in remainder:
        plan[prefix]["life"]["base_aggs"].add("daily_count")


def add_window_feature_to_plan(
    plan: dict,
    prefix: str,
    feat: str,
    remainder: str,
) -> None:
    parts = remainder.rsplit("_", 1)

    if len(parts) != 2 or not parts[1].endswith("d"):
        return

    metric, window = parts

    w_plan = plan[prefix]["windows"].setdefault(
        window,
        {"base_aggs": set(), "ratios": set(), "final_cols": set()},
    )
    w_plan["final_cols"].add(feat)

    if metric == "like_ratio":
        w_plan["base_aggs"].update(["daily_likes", "daily_count"])
        w_plan["ratios"].add("like_ratio")
    elif metric == "like_all_ratio":
        w_plan["base_aggs"].update(["daily_likes", "daily_all"])
        w_plan["ratios"].add("like_all_ratio")
    elif metric == "org_like_ratio":
        w_plan["base_aggs"].update(["daily_org_likes", "daily_org_count"])
        w_plan["ratios"].add("org_like_ratio")
    elif metric in [
        "likes",
        "dislikes",
        "listens",
        "cnt",
        "all",
        "org_likes",
        "org_listens",
        "org_cnt",
        "org_all",
    ]:
        agg_map = {
            "cnt": "daily_count",
            "all": "daily_all",
            "likes": "daily_likes",
            "dislikes": "daily_dislikes",
            "listens": "daily_listens",
            "org_likes": "daily_org_likes",
            "org_listens": "daily_org_listens",
            "org_cnt": "daily_org_count",
            "org_all": "daily_org_all",
        }
        w_plan["base_aggs"].add(agg_map.get(metric, f"daily_{metric}"))


def build_execution_plan(
    importances: dict[str, float] | None,
    entity_mappings: dict[str, list[str]],
    windows: list[str],
    threshold: float,
) -> dict:
    plan = {}

    if importances is None:
        req_features = generate_all_possible_features(entity_mappings, windows)
    else:
        req_features = [f for f, imp in importances.items() if imp > threshold]

    prefixes = sorted(entity_mappings.keys(), key=len, reverse=True)

    for feat in req_features:
        matched_prefix, remainder = parse_feature_name(feat, prefixes)

        if matched_prefix is None:
            continue

        if matched_prefix not in plan:
            plan[matched_prefix] = {
                "windows": {},
                "life": {"final_cols": set(), "base_aggs": set()},
            }

        if remainder.endswith("life"):
            add_lifetime_feature_to_plan(plan, matched_prefix, feat, remainder)
        else:
            add_window_feature_to_plan(plan, matched_prefix, feat, remainder)

    return plan


class FeatureGenerator:
    MAX_INMEMORY_PARQUET_GB = 2.0

    def __init__(
        self,
        feature_importances: dict[str, float] | None = None,
        feature_importance_threshold: float = 0.0,
        windows: list[str] | None = None,
        smooth_alpha: int = 10,
        smooth_beta: int = 20,
        target_for_fake: float = 0.05,
    ):
        self.alpha = smooth_alpha
        self.beta = smooth_beta
        self.target_for_fake = target_for_fake
        self.threshold = feature_importance_threshold
        self.windows = windows or ["1d", "7d", "30d", "180d"]

        self._fake_int = -999999
        self._fake_str = "__NULL_PLACEHOLDER__"

        self.entity_mappings = {
            "item": ["item_id"],
            "uid": ["uid"],
            "artist": ["artist_id"],
            "album": ["album_id"],
            "uid_artist": ["uid", "artist_id"],
            "uid_item": ["uid", "item_id"],
            "uid_album": ["uid", "album_id"],
        }

        self.base_agg_to_metric = {
            "daily_likes": "likes",
            "daily_dislikes": "dislikes",
            "daily_listens": "listens",
            "daily_count": "cnt",
            "daily_all": "all",
            "daily_org_likes": "org_likes",
            "daily_org_listens": "org_listens",
            "daily_org_count": "org_cnt",
            "daily_org_all": "org_all",
        }

        self.plan = build_execution_plan(
            feature_importances,
            self.entity_mappings,
            self.windows,
            self.threshold,
        )

    def _process_all_entities(
        self,
        con: duckdb.DuckDBPyConnection,
        tgt_schema: dict[str, str],
        final_selects: list[str],
    ) -> tuple[list[str], list[str]]:
        """Process all entity prefixes and collect joins/selects."""
        joins = []
        join_metadata = []

        for entity_cols in self.entity_mappings.values():
            prefix = "_".join([c.replace("_id", "") for c in entity_cols])
            print(f"Processing features for prefix: {prefix}...")

            join_clause, cols_selects = process_entity_prefix(
                con=con,
                prefix=prefix,
                entity_cols=entity_cols,
                tgt_schema=tgt_schema,
                plan=self.plan,
                base_agg_to_metric=self.base_agg_to_metric,
                alpha=self.alpha,
                beta=self.beta,
                fake_int=self._fake_int,
                fake_str=self._fake_str,
            )

            if join_clause:
                joins.append(join_clause)
                join_metadata.append(prefix)
                final_selects.extend(cols_selects)

            con.execute("CHECKPOINT;")
            log_memory(f"Completed processing for {prefix}")

        return joins, join_metadata

    def create_features(
        self,
        event_history_df: (pl.DataFrame | pl.LazyFrame | str | Path | None) = None,
        target_df: pl.DataFrame | pl.LazyFrame | str | Path | None = None,
        return_lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Create features from event history and target data."""
        log_memory("Starting create_features")

        if event_history_df is None:
            event_history_df = target_df
        if target_df is None:
            target_df = event_history_df

        if event_history_df is None and target_df is None:
            raise ValueError(
                "You must provide at least one of `target_df` or `event_history_df`!"
            )

        same_source = event_history_df is target_df
        log_memory(f"Same source: {same_source}")

        tmp_dir = global_config.tmp_path
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(dir=str(tmp_dir))
        con = None

        try:
            events_source = validate_and_get_source(
                event_history_df,
                "events",
                temp_dir.name,
                self.MAX_INMEMORY_PARQUET_GB,
            )

            if isinstance(event_history_df, pl.DataFrame):
                del event_history_df
                gc.collect()
                log_memory("Freed events DataFrame memory", logging.DEBUG)

            if same_source:
                target_source = events_source
            else:
                target_source = validate_and_get_source(
                    target_df,
                    "target",
                    temp_dir.name,
                    self.MAX_INMEMORY_PARQUET_GB,
                )
                if isinstance(target_df, pl.DataFrame):
                    del target_df
                    gc.collect()
                    log_memory("Freed target DataFrame memory")

            con = setup_duckdb_connection(temp_dir.name, "features.db")
            log_memory("DuckDB environment setup complete")

            tgt_schema = materialize_base_tables(
                con, events_source, target_source, same_source
            )

            log_memory("Building initial selects")
            final_selects = build_initial_selects(tgt_schema, self.target_for_fake)

            joins, join_metadata = self._process_all_entities(
                con, tgt_schema, final_selects
            )

            result_parquet = os.path.join(temp_dir.name, "final_result.parquet")
            assemble_final_result(
                con, joins, join_metadata, final_selects, result_parquet
            )

            con.close()
            con = None
            gc.collect()
            log_memory("DuckDB closed")

            return load_result(result_parquet, temp_dir.name, return_lazy)

        finally:
            log_memory("Cleaning up")
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
            temp_dir.cleanup()
            gc.collect()
            log_memory("Cleanup complete")
