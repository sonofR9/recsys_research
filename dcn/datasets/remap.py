from pathlib import Path

import polars as pl


def build_id_remap_and_remapped_embeddings(
    main_parquet: Path,
    embeddings_parquet: Path,
    remap_parquet: Path,
    remapped_embeddings_parquet: Path,
    raw_id_column: str = "item_id",
    embedding_column: str = "normalized_embed",
) -> None:
    """Assign compact ids 1..N to ids in BOTH parquets; 0 stays the unknown item."""
    main_ids = pl.scan_parquet(main_parquet).select(pl.col(raw_id_column)).unique()

    # Streamed: the table covers the whole catalog, far more than a run needs.
    joined = (
        main_ids.join(
            pl.scan_parquet(embeddings_parquet)
            .select(raw_id_column, embedding_column)
            .unique(subset=raw_id_column),
            on=raw_id_column,
            how="inner",
        )
        .sort(raw_id_column)
        .with_row_index(name="compact_id", offset=1)
        .collect(engine="streaming")
    )

    remap = joined.select(
        pl.col(raw_id_column),
        pl.col("compact_id").cast(pl.Int64),
    )
    remap_parquet.parent.mkdir(parents=True, exist_ok=True)
    remap.write_parquet(remap_parquet)

    joined.select(
        pl.col("compact_id").cast(pl.Int64),
        pl.col(embedding_column),
    ).write_parquet(remapped_embeddings_parquet)


def apply_id_remap_to_parquet(
    main_parquet: Path,
    remap_parquet: Path,
    output_parquet: Path,
    id_column: str = "item_id",
    compact_column: str | None = None,
    drop_unmapped: bool = False,
) -> None:
    """Add a compact id column to ``main_parquet``; unmapped ids become 0."""
    if compact_column is None:
        compact_column = f"compact_{id_column}"

    # Streamed: the eager path holds the input and the joined copy at once.
    main = pl.scan_parquet(main_parquet).join(
        pl.scan_parquet(remap_parquet),
        on=id_column,
        how="inner" if drop_unmapped else "left",
        maintain_order="left",
    )
    compact_id = pl.col("compact_id")
    if not drop_unmapped:
        compact_id = compact_id.fill_null(0)
    main = main.with_columns(compact_id.alias(compact_column))
    if compact_column != "compact_id":
        main = main.drop("compact_id")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    main.sink_parquet(output_parquet)
