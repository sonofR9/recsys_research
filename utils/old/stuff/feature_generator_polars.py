import polars as pl


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
        """
        Dynamically builds the execution plan based on feature_importances.
        If feature_importances is None, computes ALL normally available features.
        Only features with importance > feature_importance_threshold are computed.
        """
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
        """Generates all feature names if no importance dict is provided."""
        features = []
        for prefix in self.entity_mappings.keys():
            # Lifetime metrics
            for metric in [
                "likes_life",
                "cnt_life",
                "org_likes_life",
                "org_cnt_life",
            ]:
                features.append(f"{prefix}_{metric}")
            # Windowed metrics
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
        """Parses requested feature strings to build an optimized computing plan."""
        plan = {}

        # Decide the target feature list based on provided importance or fallback
        if importances is None:
            req_features = self._generate_all_possible_features()
        else:
            req_features = [
                f for f, imp in importances.items() if imp > self.threshold
            ]

        # Sort prefixes by length (longest first e.g. 'uid_artist' before 'uid')
        prefixes = sorted(self.entity_mappings.keys(), key=len, reverse=True)

        for feat in req_features:
            matched_prefix = None
            for p in prefixes:
                if feat.startswith(p + "_"):
                    matched_prefix = p
                    break

            if not matched_prefix:
                continue

            # Remainder (e.g., 'like_ratio_7d' or 'org_likes_life')
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
                if len(parts) != 2:
                    continue
                metric, window = parts

                # Verify suffix is actually a window
                if not window.endswith("d"):
                    continue

                w_plan = plan[matched_prefix]["windows"].setdefault(
                    window,
                    {"base_aggs": set(), "ratios": set(), "final_cols": set()},
                )
                w_plan["final_cols"].add(feat)

                # Flag dependencies so .rolling() only calculates needed base items
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

    def _get_base_events(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(
            pl.col("event_type") != "random_negative"
        ).with_columns(
            [
                (pl.col("event_type") == "like").alias("is_like"),
                (pl.col("event_type") == "dislike").alias("is_dislike"),
                (
                    pl.col("is_organic") & (pl.col("event_type") == "like")
                ).alias("is_org_like"),
                (
                    pl.col("is_organic") & (pl.col("event_type") == "dislike")
                ).alias("is_org_dislike"),
                pl.col("is_organic").alias("is_org_any"),
            ]
        )

    def _aggregate_daily_stats(
        self, df: pl.LazyFrame, entity_cols: list[str]
    ) -> pl.LazyFrame:
        agg_cols = [
            pl.col("is_like").sum().cast(pl.UInt32).alias("daily_likes"),
            pl.col("is_dislike").sum().cast(pl.UInt32).alias("daily_dislikes"),
            pl.len().cast(pl.UInt32).alias("daily_count"),
            pl.col("is_org_like")
            .sum()
            .cast(pl.UInt32)
            .alias("daily_org_likes"),
            pl.col("is_org_dislike")
            .sum()
            .cast(pl.UInt32)
            .alias("daily_org_dislikes"),
            pl.col("is_org_any")
            .sum()
            .cast(pl.UInt32)
            .alias("daily_org_count"),
        ]
        return df.group_by(["day"] + entity_cols).agg(agg_cols).sort("day")

    def _inject_target_spine(
        self,
        history_agg: pl.LazyFrame,
        target_df: pl.LazyFrame,
        entity_cols: list[str],
    ) -> pl.LazyFrame:
        target_spine = target_df.select(["day"] + entity_cols).unique()
        target_spine = self._fill_nulls_for_groupby(target_spine, entity_cols)

        zero_cols_exprs = [
            pl.lit(0, dtype=pl.UInt32).alias(c)
            for c in [
                "daily_likes",
                "daily_dislikes",
                "daily_count",
                "daily_org_likes",
                "daily_org_dislikes",
                "daily_org_count",
            ]
        ]
        target_spine = target_spine.with_columns(zero_cols_exprs)

        return (
            pl.concat([history_agg, target_spine], how="vertical_relaxed")
            .group_by(["day"] + entity_cols)
            .agg(
                [
                    pl.col("daily_likes").sum(),
                    pl.col("daily_dislikes").sum(),
                    pl.col("daily_count").sum(),
                    pl.col("daily_org_likes").sum(),
                    pl.col("daily_org_dislikes").sum(),
                    pl.col("daily_org_count").sum(),
                ]
            )
        )

    def _fill_nulls_for_groupby(
        self, df: pl.LazyFrame, entity_cols: list[str]
    ) -> pl.LazyFrame:
        schema = pl.LazyFrame.collect_schema(df)
        fill_exprs = []

        for col in entity_cols:
            dtype = schema[col]
            if dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                fill_exprs.append(pl.col(col).fill_null(self._fake_int))
            elif dtype in [pl.String, pl.Utf8]:
                fill_exprs.append(pl.col(col).fill_null(self._fake_str))
            else:
                fill_exprs.append(pl.col(col).fill_null(self._fake_int))

        if fill_exprs:
            df = df.with_columns(fill_exprs)

        return df.filter(pl.col("day").is_not_null())

    def _restore_nulls_from_fake(
        self, df: pl.LazyFrame, entity_cols: list[str]
    ) -> pl.LazyFrame:
        schema = pl.LazyFrame.collect_schema(df)
        restore_exprs = []

        for col in entity_cols:
            dtype = schema[col]
            if dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                restore_exprs.append(
                    pl.when(pl.col(col) == self._fake_int)
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            elif dtype in [pl.String, pl.Utf8]:
                restore_exprs.append(
                    pl.when(pl.col(col) == self._fake_str)
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        if restore_exprs:
            df = df.with_columns(restore_exprs)

        return df

    def _calc_rolling_features(
        self,
        daily_agg_df: pl.LazyFrame,
        entity_cols: list[str],
        window: str,
        w_plan: dict,
    ) -> pl.LazyFrame:
        prefix = "_".join([c.replace("_id", "") for c in entity_cols])

        agg_exprs = []
        if "daily_likes" in w_plan["base_aggs"]:
            agg_exprs.append(
                pl.col("daily_likes")
                .sum()
                .cast(pl.UInt32)
                .alias(f"{prefix}_likes_{window}")
            )
        if "daily_dislikes" in w_plan["base_aggs"]:
            agg_exprs.append(
                pl.col("daily_dislikes")
                .sum()
                .cast(pl.UInt32)
                .alias(f"{prefix}_dislikes_{window}")
            )
        if "daily_count" in w_plan["base_aggs"]:
            agg_exprs.append(
                pl.col("daily_count")
                .sum()
                .cast(pl.UInt32)
                .alias(f"{prefix}_cnt_{window}")
            )
        if "daily_org_likes" in w_plan["base_aggs"]:
            agg_exprs.append(
                pl.col("daily_org_likes")
                .sum()
                .cast(pl.UInt32)
                .alias(f"{prefix}_org_likes_{window}")
            )
        if "daily_org_count" in w_plan["base_aggs"]:
            agg_exprs.append(
                pl.col("daily_org_count")
                .sum()
                .cast(pl.UInt32)
                .alias(f"{prefix}_org_cnt_{window}")
            )

        result = (
            daily_agg_df.sort(entity_cols + ["day"])
            .rolling(
                index_column="day",
                by=entity_cols,
                period=window,
                closed="left",
            )
            .agg(agg_exprs)
        )

        ratio_exprs = []
        if "like_ratio" in w_plan["ratios"]:
            ratio_exprs.append(
                (
                    (pl.col(f"{prefix}_likes_{window}") + self.alpha)
                    / (pl.col(f"{prefix}_cnt_{window}") + self.beta)
                )
                .cast(pl.Float32)
                .alias(f"{prefix}_like_ratio_{window}")
            )
        if "org_like_ratio" in w_plan["ratios"]:
            ratio_exprs.append(
                (
                    (pl.col(f"{prefix}_org_likes_{window}") + self.alpha)
                    / (pl.col(f"{prefix}_org_cnt_{window}") + self.beta)
                )
                .cast(pl.Float32)
                .alias(f"{prefix}_org_like_ratio_{window}")
            )

        if ratio_exprs:
            result = result.with_columns(ratio_exprs)

        # Clean up memory immediately to keep joins lean
        cols_to_keep = ["day"] + entity_cols + list(w_plan["final_cols"])
        result = result.select(cols_to_keep)

        return self._restore_nulls_from_fake(result, entity_cols)

    def _calc_lifetime_features(
        self, daily_agg_df: pl.LazyFrame, entity_cols: list[str], l_plan: dict
    ) -> pl.LazyFrame:
        prefix = "_".join([c.replace("_id", "") for c in entity_cols])

        exprs = []
        if "daily_likes" in l_plan["base_aggs"]:
            exprs.append(
                pl.col("daily_likes")
                .shift(1)
                .fill_null(0)
                .cum_sum()
                .over(entity_cols)
                .cast(pl.UInt32)
                .alias(f"{prefix}_likes_life")
            )
        if "daily_count" in l_plan["base_aggs"]:
            exprs.append(
                pl.col("daily_count")
                .shift(1)
                .fill_null(0)
                .cum_sum()
                .over(entity_cols)
                .cast(pl.UInt32)
                .alias(f"{prefix}_cnt_life")
            )
        if "daily_org_likes" in l_plan["base_aggs"]:
            exprs.append(
                pl.col("daily_org_likes")
                .shift(1)
                .fill_null(0)
                .cum_sum()
                .over(entity_cols)
                .cast(pl.UInt32)
                .alias(f"{prefix}_org_likes_life")
            )
        if "daily_org_count" in l_plan["base_aggs"]:
            exprs.append(
                pl.col("daily_org_count")
                .shift(1)
                .fill_null(0)
                .cum_sum()
                .over(entity_cols)
                .cast(pl.UInt32)
                .alias(f"{prefix}_org_cnt_life")
            )

        result = daily_agg_df.sort(entity_cols + ["day"]).with_columns(exprs)

        cols_to_keep = ["day"] + entity_cols + list(l_plan["final_cols"])
        result = result.select(cols_to_keep)

        return self._restore_nulls_from_fake(result, entity_cols)

    @staticmethod
    def _add_day_column(df: pl.LazyFrame) -> pl.LazyFrame:
        if "day" not in pl.LazyFrame.collect_schema(df).names():
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
        else:
            event_history_df = event_history_df.lazy()

        if target_df is None:
            target_df = event_history_df
        elif isinstance(target_df, pl.DataFrame):
            target_df = target_df.lazy()

        target_df = self._add_day_column(target_df)
        history_df = self._add_day_column(event_history_df)
        base_events = self._get_base_events(history_df)

        result_df = target_df

        # Skip logic dynamically filters unused tables
        for entity_idx_tuple, entity_cols in self.entity_mappings.items():
            prefix = "_".join([c.replace("_id", "") for c in entity_cols])

            # IMPORTANT: Compute logic completely skips unused entity subsets
            if prefix not in self.plan:
                continue

            e_plan = self.plan[prefix]

            curr_base = base_events
            for col in entity_cols:
                if "id" in col:
                    curr_base = curr_base.filter(pl.col(col) != 0)

            history_agg = self._aggregate_daily_stats(curr_base, entity_cols)
            history_agg = self._fill_nulls_for_groupby(
                history_agg, entity_cols
            )
            daily_agg = self._inject_target_spine(
                history_agg, target_df, entity_cols
            )

            if e_plan["life"]["final_cols"]:
                lifetime_stats = self._calc_lifetime_features(
                    daily_agg, entity_cols, e_plan["life"]
                )
                result_df = result_df.join(
                    lifetime_stats, on=["day"] + entity_cols, how="left"
                )

            for window, w_plan in e_plan["windows"].items():
                if w_plan["final_cols"]:
                    rolling_stats = self._calc_rolling_features(
                        daily_agg, entity_cols, window, w_plan
                    )
                    result_df = result_df.join(
                        rolling_stats, on=["day"] + entity_cols, how="left"
                    )

        # Compile expected feature generation schema and enforce formatting bounds
        generated_features = []
        for e in self.plan.values():
            generated_features.extend(e["life"]["final_cols"])
            for w in e["windows"].values():
                generated_features.extend(w["final_cols"])

        available_cols = pl.LazyFrame.collect_schema(result_df).names()
        valid_feat_cols = [
            c for c in generated_features if c in available_cols
        ]

        result_df = result_df.with_columns(
            [
                (
                    pl.col(c).fill_null(0).cast(pl.Float32)
                    if "_ratio_" in c
                    else pl.col(c).fill_null(0)
                )
                for c in valid_feat_cols
            ]
        )

        if "event_type" in available_cols:
            result_df = result_df.with_columns(
                pl.when(pl.col("event_type") == "like")
                .then(1)
                .when(pl.col("event_type") == "dislike")
                .then(0)
                .otherwise(self.target_for_fake)
                .alias("target")
            )
        else:
            result_df = result_df.with_columns(
                pl.lit(self.target_for_fake).alias("target")
            )

        return result_df.collect(engine="streaming")
