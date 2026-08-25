import polars as pl

MODEL_PATH = "model.cbm"
FINAL_MODEL_PATH = "final_model.cbm"


def recall_at_k(
    positive_interactions: pl.DataFrame,
    candidates: pl.DataFrame,
    k: int = 100,
) -> float:
    user_col = "uid"
    item_col = "item_id"
    rank_col = "rank"
    top_k_candidates = candidates.filter(pl.col(rank_col) <= k).select(
        [user_col, item_col]
    )

    total_positives = positive_interactions.group_by(user_col).agg(
        pl.col(item_col).n_unique().alias("total_positives")
    )

    hits = (
        positive_interactions.select([user_col, item_col])
        .join(top_k_candidates, on=[user_col, item_col], how="inner")
        .group_by(user_col)
        .agg(pl.col(item_col).n_unique().alias("hits"))
    )

    recall = (
        total_positives.join(hits, on=user_col, how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .with_columns((pl.col("hits") / pl.col("total_positives")).alias("recall"))
        .select(pl.col("recall").mean())
        .item()
    )

    return recall


def prepare_submission(df: pl.DataFrame, output_path: str, top_k: int = 100):
    """
    Filters the top K items per user, joins them into a space-separated string,
    and writes the result to a CSV file.
    """
    (
        df.filter(pl.col("rank") <= top_k)
        .sort(["uid", "rank"])
        .with_columns(pl.col("item_id").cast(pl.String))
        .group_by("uid")
        .agg(pl.col("item_id").str.concat(" ").alias("item_ids"))
        .write_csv(output_path)
    )

    print(f"Submission saved to {output_path}")
