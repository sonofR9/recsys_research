from pathlib import Path

import polars as pl

from dcn.datasets.remap import (
    apply_id_remap_to_parquet,
    build_id_remap_and_remapped_embeddings,
)


def test_build_remap_assigns_contiguous_compact_ids(tmp_path: Path) -> None:
    embeddings_parquet = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "item_id": [42, 7, 99],
            "normalized_embed": [[1.0], [2.0], [3.0]],
        }
    ).write_parquet(embeddings_parquet)

    main_parquet = tmp_path / "main.parquet"
    pl.DataFrame({"item_id": [42, 7, 99]}).write_parquet(main_parquet)

    remap_parquet = tmp_path / "remap.parquet"
    compact_embeddings_parquet = tmp_path / "compact.parquet"
    build_id_remap_and_remapped_embeddings(
        main_parquet=main_parquet,
        embeddings_parquet=embeddings_parquet,
        remap_parquet=remap_parquet,
        remapped_embeddings_parquet=compact_embeddings_parquet,
    )

    remap = pl.read_parquet(remap_parquet)
    assert remap["item_id"].to_list() == [7, 42, 99]
    assert remap["compact_id"].to_list() == [1, 2, 3]

    compact = pl.read_parquet(compact_embeddings_parquet)
    assert compact["compact_id"].to_list() == [1, 2, 3]
    assert compact["normalized_embed"].to_list() == [[2.0], [1.0], [3.0]]


def test_build_remap_ignores_repeated_embedding_rows(tmp_path: Path) -> None:
    embeddings_parquet = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "item_id": [7, 42, 7],
            "normalized_embed": [[2.0], [1.0], [2.0]],
        }
    ).write_parquet(embeddings_parquet)

    main_parquet = tmp_path / "main.parquet"
    pl.DataFrame({"item_id": [42, 7]}).write_parquet(main_parquet)

    remap_parquet = tmp_path / "remap.parquet"
    compact_embeddings_parquet = tmp_path / "compact.parquet"
    build_id_remap_and_remapped_embeddings(
        main_parquet=main_parquet,
        embeddings_parquet=embeddings_parquet,
        remap_parquet=remap_parquet,
        remapped_embeddings_parquet=compact_embeddings_parquet,
    )

    remap = pl.read_parquet(remap_parquet)
    assert remap["item_id"].to_list() == [7, 42]
    assert remap["compact_id"].to_list() == [1, 2]
    assert pl.read_parquet(compact_embeddings_parquet)["compact_id"].to_list() == [1, 2]


def test_apply_remap_keeps_the_row_order_of_the_main_table(tmp_path: Path) -> None:
    main_parquet = tmp_path / "main.parquet"
    pl.DataFrame({"item_id": [42, 7, 42, 1234], "x": [1, 2, 3, 4]}).write_parquet(
        main_parquet
    )

    remap_parquet = tmp_path / "remap.parquet"
    pl.DataFrame({"item_id": [42, 7], "compact_id": [1, 2]}).write_parquet(
        remap_parquet
    )

    output_parquet = tmp_path / "remapped.parquet"
    apply_id_remap_to_parquet(main_parquet, remap_parquet, output_parquet)

    output = pl.read_parquet(output_parquet)
    assert output["x"].to_list() == [1, 2, 3, 4]
    assert output["compact_item_id"].to_list() == [1, 2, 1, 0]
    assert output["item_id"].to_list() == [42, 7, 42, 1234]


def test_apply_remap_unknown_ids_become_zero(tmp_path: Path) -> None:
    main_parquet = tmp_path / "main.parquet"
    pl.DataFrame({"item_id": [42, 7, 1234], "x": [1, 2, 3]}).write_parquet(main_parquet)

    remap_parquet = tmp_path / "remap.parquet"
    pl.DataFrame({"item_id": [42, 7], "compact_id": [1, 2]}).write_parquet(
        remap_parquet
    )

    output_parquet = tmp_path / "remapped.parquet"
    apply_id_remap_to_parquet(
        main_parquet, remap_parquet, output_parquet, compact_column="item_id"
    )

    output = pl.read_parquet(output_parquet)
    assert output.sort("x")["item_id"].to_list() == [1, 2, 0]


def test_apply_remap_can_drop_items_without_embeddings(tmp_path: Path) -> None:
    main_parquet = tmp_path / "main.parquet"
    pl.DataFrame({"item_id": [42, 1234, 7], "x": [1, 2, 3]}).write_parquet(
        main_parquet
    )
    remap_parquet = tmp_path / "remap.parquet"
    pl.DataFrame({"item_id": [42, 7], "compact_id": [1, 2]}).write_parquet(
        remap_parquet
    )
    output_parquet = tmp_path / "remapped.parquet"

    apply_id_remap_to_parquet(
        main_parquet,
        remap_parquet,
        output_parquet,
        drop_unmapped=True,
    )

    output = pl.read_parquet(output_parquet)
    assert output["x"].to_list() == [1, 3]
    assert output["compact_item_id"].to_list() == [1, 2]
