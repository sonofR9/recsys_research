import polars as pl
import duckdb
import tempfile
import os


class FeatureGenerator:
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
            "uid_album": ["uid", "album_id"],
        }

        self.plan = self._build_execution_plan(feature_importances)

    def _generate_all_possible_features(self) -> list[str]:
        features = []
        for prefix in self.entity_mappings.keys():
            for metric in [
                "likes_life",
                "cnt_life",
                "org_likes_life",
                "org_cnt_life",
            ]:
                features.append(f"{prefix}_{metric}")
            for w in self.windows:
                for metric in [
                    "likes",
                    "dislikes",
                    "cnt",
                    "org_likes",
                    "org_cnt",
                    "like_ratio",
                    "org_like_ratio",
                ]:
                    features.append(f"{prefix}_{metric}_{w}")
        return features

    def _build_execution_plan(
        self, importances: dict[str, float] | None
    ) -> dict:
        plan = {}
        if importances is None:
            req_features = self._generate_all_possible_features()
        else:
            req_features = [
                f for f, imp in importances.items() if imp > self.threshold
            ]

        prefixes = sorted(self.entity_mappings.keys(), key=len, reverse=True)

        for feat in req_features:
            matched_prefix = next(
                (p for p in prefixes if feat.startswith(p + "_")), None
            )
            if not matched_prefix:
                continue

            remainder = feat[len(matched_prefix) + 1 :]

            if matched_prefix not in plan:
                plan[matched_prefix] = {
                    "windows": {},
                    "life": {"final_cols": set(), "base_aggs": set()},
                }

            if remainder.endswith("life"):
                plan[matched_prefix]["life"]["final_cols"].add(feat)
                if "org_likes" in remainder:
                    plan[matched_prefix]["life"]["base_aggs"].add(
                        "daily_org_likes"
                    )
                elif "org_cnt" in remainder:
                    plan[matched_prefix]["life"]["base_aggs"].add(
                        "daily_org_count"
                    )
                elif "likes" in remainder:
                    plan[matched_prefix]["life"]["base_aggs"].add(
                        "daily_likes"
                    )
                elif "cnt" in remainder:
                    plan[matched_prefix]["life"]["base_aggs"].add(
                        "daily_count"
                    )
            else:
                parts = remainder.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].endswith("d"):
                    continue
                metric, window = parts

                w_plan = plan[matched_prefix]["windows"].setdefault(
                    window,
                    {"base_aggs": set(), "ratios": set(), "final_cols": set()},
                )
                w_plan["final_cols"].add(feat)

                if metric == "like_ratio":
                    w_plan["base_aggs"].update(["daily_likes", "daily_count"])
                    w_plan["ratios"].add("like_ratio")
                elif metric == "org_like_ratio":
                    w_plan["base_aggs"].update(
                        ["daily_org_likes", "daily_org_count"]
                    )
                    w_plan["ratios"].add("org_like_ratio")
                elif metric == "likes":
                    w_plan["base_aggs"].add("daily_likes")
                elif metric == "dislikes":
                    w_plan["base_aggs"].add("daily_dislikes")
                elif metric == "cnt":
                    w_plan["base_aggs"].add("daily_count")
                elif metric == "org_likes":
                    w_plan["base_aggs"].add("daily_org_likes")
                elif metric == "org_cnt":
                    w_plan["base_aggs"].add("daily_org_count")

        return plan

    @staticmethod
    def _add_day_column(df: pl.LazyFrame) -> pl.LazyFrame:
        if "day" not in df.collect_schema().names():
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="s")
                .dt.truncate("1d")
                .alias("day")
            )
        return df

    def create_features(
        self,
        event_history_df: pl.DataFrame | pl.LazyFrame | None = None,
        target_df: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:

        if event_history_df is None:
            event_history_df = target_df.lazy()
        elif isinstance(event_history_df, pl.DataFrame):
            event_history_df = event_history_df.lazy()

        if target_df is None:
            target_df = event_history_df
        elif isinstance(target_df, pl.DataFrame):
            target_df = target_df.lazy()

        # Step 1: Pre-Process & Column Limit Lazily via Polars
        target_df_lazy = self._add_day_column(target_df)
        history_lazy = self._add_day_column(event_history_df)

        base_lazy = history_lazy.filter(
            pl.col("event_type") != "random_negative"
        ).with_columns(
            [
                (pl.col("event_type") == "like")
                .cast(pl.UInt32)
                .alias("is_like"),
                (pl.col("event_type") == "dislike")
                .cast(pl.UInt32)
                .alias("is_dislike"),
                (pl.col("is_organic") & (pl.col("event_type") == "like"))
                .cast(pl.UInt32)
                .alias("is_org_like"),
                (pl.col("is_organic") & (pl.col("event_type") == "dislike"))
                .cast(pl.UInt32)
                .alias("is_org_dislike"),
                pl.col("is_organic").cast(pl.UInt32).alias("is_org_any"),
            ]
        )

        needed_cols = {
            "day",
            "is_like",
            "is_dislike",
            "is_org_like",
            "is_org_dislike",
            "is_org_any",
        }
        for prefix in self.plan.keys():
            needed_cols.update(self.entity_mappings[prefix])

        avail_cols = base_lazy.collect_schema().names()
        keep = [c for c in needed_cols if c in avail_cols]

        # Step 2: Push dataset state to DuckDB streams over Zero-Copy maps
        base_events_df = base_lazy.select(keep).collect(engine="streaming")
        target_df_eager = target_df_lazy.collect(engine="streaming")
        tgt_schema = target_df_eager.schema

        con = duckdb.connect()
        con.execute("SET enable_progress_bar=false;")
        con.execute("SET memory_limit = '20GB';")
        con.execute("SET preserve_insertion_order = false;")
        con.execute("SET force_compression = 'auto';")

        con.register("base_events_df", base_events_df)
        con.register("target_df_eager", target_df_eager)

        base_agg_to_metric = {
            "daily_likes": "likes",
            "daily_dislikes": "dislikes",
            "daily_count": "cnt",
            "daily_org_likes": "org_likes",
            "daily_org_dislikes": "org_dislikes",
            "daily_org_count": "org_cnt",
        }

        ctes = []
        joins = []

        # Construct standard fields (excluding `target` if it already accidentally happens to exist, preventing duplicates)
        t_cols = [c for c in tgt_schema.names() if c != "target"]
        final_selects = [f"t.{c}" for c in t_cols]

        if "event_type" in tgt_schema.names():
            final_selects.append(
                f"CASE WHEN t.event_type = 'like' THEN 1.0 "
                f"WHEN t.event_type = 'dislike' THEN 0.0 ELSE {self.target_for_fake} END::FLOAT AS target"
            )
        else:
            final_selects.append(f"{self.target_for_fake}::FLOAT AS target")

        for entity_idx_tuple, entity_cols in self.entity_mappings.items():
            prefix = "_".join([c.replace("_id", "") for c in entity_cols])
            if prefix not in self.plan:
                continue
            e_plan = self.plan[prefix]

            agg_cols, key_cols, restored_cols, id_filters = [], [], [], []

            for col in entity_cols:
                is_str = tgt_schema[col] in (
                    getattr(pl, "String", pl.Utf8),
                    pl.Utf8,
                    getattr(pl, "Categorical", None),
                )
                fill, nullif_val, val_0 = (
                    (
                        f"'{self._fake_str}'::VARCHAR",
                        f"'{self._fake_str}'",
                        "'0'",
                    )
                    if is_str
                    else (
                        f"{self._fake_int}::BIGINT",
                        str(self._fake_int),
                        "0",
                    )
                )

                agg_cols.append(f"COALESCE({col}, {fill}) AS {col}")
                key_cols.append(f"COALESCE({col}, {fill}) AS {col}")
                restored_cols.append(f"NULLIF({col}, {nullif_val}) AS {col}")

                if "id" in col:
                    id_filters.append(
                        f"{col} != {val_0} AND {col} IS NOT NULL"
                    )

            where_clause = (
                "WHERE " + " AND ".join(id_filters) if id_filters else ""
            )

            base_exprs = []
            life_aggs = e_plan["life"]["base_aggs"]
            for agg in [
                "daily_likes",
                "daily_count",
                "daily_org_likes",
                "daily_org_count",
            ]:
                if agg in life_aggs:
                    metr = base_agg_to_metric[agg]
                    base_exprs.append(
                        f"COALESCE(SUM({agg}) OVER w_life, 0) AS {prefix}_{metr}_life"
                    )

            window_defs = [
                f"w_life AS (PARTITION BY {', '.join(entity_cols)} ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
            ]
            ratio_exprs = []

            for window, w_plan in e_plan["windows"].items():
                win_val = int(window.replace("d", ""))
                w_name = f"w_{window}"
                window_defs.append(
                    f"{w_name} AS (PARTITION BY {', '.join(entity_cols)} ORDER BY day "
                    f"RANGE BETWEEN INTERVAL {win_val} DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING)"
                )

                for agg in w_plan["base_aggs"]:
                    metric = base_agg_to_metric[agg]
                    base_exprs.append(
                        f"COALESCE(SUM({agg}) OVER {w_name}, 0) AS {prefix}_{metric}_{window}"
                    )

                ratios = w_plan["ratios"]
                if "like_ratio" in ratios:
                    ratio_exprs.append(
                        f"({prefix}_likes_{window} + {self.alpha})::FLOAT / ({prefix}_cnt_{window} + {self.beta})::FLOAT AS {prefix}_like_ratio_{window}"
                    )
                if "org_like_ratio" in ratios:
                    ratio_exprs.append(
                        f"({prefix}_org_likes_{window} + {self.alpha})::FLOAT / ({prefix}_org_cnt_{window} + {self.beta})::FLOAT AS {prefix}_org_like_ratio_{window}"
                    )

            final_req = list(e_plan["life"]["final_cols"])
            for window, w_plan in e_plan["windows"].items():
                final_req.extend(list(w_plan["final_cols"]))

            if not final_req:
                continue

            base_select_str = (
                f", {', '.join(base_exprs)}" if base_exprs else ""
            )
            ratio_select_str = (
                f", {', '.join(ratio_exprs)}" if ratio_exprs else ""
            )
            window_clause = (
                f" WINDOW {', '.join(window_defs)}" if base_exprs else ""
            )

            c_pfx = f"cte_{prefix}"
            ctes.append(
                f"""
                {c_pfx}_agg AS (
                    SELECT day, {', '.join(agg_cols)},
                        SUM(is_like) AS daily_likes, SUM(is_dislike) AS daily_dislikes, COUNT(*) AS daily_count,
                        SUM(is_org_like) AS daily_org_likes, SUM(is_org_dislike) AS daily_org_dislikes, SUM(is_org_any) AS daily_org_count
                    FROM base_events_df {where_clause} GROUP BY ALL),
                {c_pfx}_keys AS (
                    SELECT DISTINCT day, {', '.join(key_cols)}
                    FROM target_df_eager WHERE day IS NOT NULL),
                {c_pfx}_spine AS (
                    SELECT day, {', '.join(entity_cols)}, SUM(daily_likes) as daily_likes, SUM(daily_dislikes) as daily_dislikes,
                           SUM(daily_count) as daily_count, SUM(daily_org_likes) as daily_org_likes, SUM(daily_org_dislikes) as daily_org_dislikes, SUM(daily_org_count) as daily_org_count
                    FROM (SELECT day, {', '.join(entity_cols)}, daily_likes, daily_dislikes, daily_count, daily_org_likes, daily_org_dislikes, daily_org_count FROM {c_pfx}_agg
                          UNION ALL 
                          SELECT day, {', '.join(entity_cols)}, 0, 0, 0, 0, 0, 0 FROM {c_pfx}_keys)
                    GROUP BY ALL),
                {c_pfx}_win AS (
                    SELECT day, {', '.join(restored_cols)} {base_select_str}
                    FROM {c_pfx}_spine {window_clause}),
                {c_pfx}_final AS (
                    SELECT * {ratio_select_str} FROM {c_pfx}_win)
            """
            )

            # --- FIX APPLIED HERE: Using dynamic f_alias correctly for explicit joins ---
            f_alias = f"f_{prefix}"
            join_conds = [f"t.day = {f_alias}.day"]
            for col in entity_cols:
                join_conds.append(
                    f"t.{col} IS NOT DISTINCT FROM {f_alias}.{col}"
                )

            joins.append(
                f"LEFT JOIN {c_pfx}_final AS {f_alias} ON {' AND '.join(join_conds)}"
            )

            for c in final_req:
                if "_ratio_" in c:
                    final_selects.append(
                        f"COALESCE({f_alias}.{c}, 0.0)::FLOAT AS {c}"
                    )
                else:
                    final_selects.append(
                        f"COALESCE({f_alias}.{c}, 0)::UINTEGER AS {c}"
                    )

        if not ctes:
            return target_df_eager

        complete_query = (
            "WITH "
            + ",\n".join(ctes)
            + "\nSELECT "
            + ", ".join(final_selects)
            + "\nFROM target_df_eager t "
            + "\n".join(joins)
        )

        result_df = con.query(complete_query).pl()
        con.close()

        return result_df
