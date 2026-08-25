"""Tests for embedding binary converter."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch


# FIXME: it is no longer a class
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_embeddings_parquet(temp_dir):
    """Create a sample embeddings.parquet file for testing."""
    # Create sample data
    n_items = 100
    embedding_dim = 128

    item_ids = np.arange(n_items)
    embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)

    df = pd.DataFrame(
        {"item_id": item_ids, "item_embedding": [emb.tolist() for emb in embeddings]}
    )

    parquet_path = os.path.join(temp_dir, "embeddings.parquet")
    df.to_parquet(parquet_path, index=False)

    return parquet_path, item_ids, embeddings


class TestEmbeddingBinaryConverter:
    """Test suite for EmbeddingBinaryConverter."""

    def test_convert_creates_files(self, temp_dir, sample_embeddings_parquet):
        """Test that convert() creates all expected files."""
        parquet_path, _, _ = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        converter.convert()

        # Check that all files are created
        assert os.path.exists(os.path.join(output_dir, "embeddings.bin"))
        assert os.path.exists(os.path.join(output_dir, "id_to_offset.pkl"))
        assert os.path.exists(os.path.join(output_dir, "embeddings_metadata.pkl"))

    def test_convert_correctness(self, temp_dir, sample_embeddings_parquet):
        """Test that converted embeddings match original data."""
        parquet_path, item_ids, original_embeddings = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        converter.convert()

        # Load the binary file
        bin_path = os.path.join(output_dir, "embeddings.bin")
        loaded_embeddings = np.memmap(
            bin_path,
            dtype=np.float32,
            mode="r",
            shape=(len(item_ids), original_embeddings.shape[1]),
        )

        # Check that embeddings match
        np.testing.assert_allclose(
            loaded_embeddings, original_embeddings, rtol=1e-5, atol=1e-7
        )

    def test_metadata_correctness(self, temp_dir, sample_embeddings_parquet):
        """Test that metadata is correctly saved."""
        parquet_path, item_ids, original_embeddings = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        converter.convert()

        # Load metadata
        import pickle

        metadata_path = os.path.join(output_dir, "embeddings_metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Check metadata fields
        assert metadata["num_items"] == len(item_ids)
        assert metadata["embedding_dim"] == original_embeddings.shape[1]
        assert metadata["dtype"] == "float32"
        assert "source_file" in metadata

    def test_id_to_offset_mapping(self, temp_dir, sample_embeddings_parquet):
        """Test that id_to_offset mapping is correct."""
        parquet_path, item_ids, _ = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        converter.convert()

        # Load id_to_offset
        import pickle

        id_to_offset_path = os.path.join(output_dir, "id_to_offset.pkl")
        with open(id_to_offset_path, "rb") as f:
            id_to_offset = pickle.load(f)

        # Check that all item_ids are present
        assert len(id_to_offset) == len(item_ids)
        for item_id in item_ids:
            assert item_id in id_to_offset

        # Check that offsets are sequential
        offsets = sorted(id_to_offset.values())
        assert offsets == list(range(len(item_ids)))

    def test_skip_if_exists(self, temp_dir, sample_embeddings_parquet):
        """Test that conversion is skipped if files already exist."""
        parquet_path, _, _ = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        # First conversion
        converter.convert()

        # Get modification time of binary file
        bin_path = os.path.join(output_dir, "embeddings.bin")
        mtime_before = os.path.getmtime(bin_path)

        # Second conversion (should skip)
        import time

        time.sleep(0.1)  # Ensure time difference
        converter.convert()

        # Check that file was not modified
        mtime_after = os.path.getmtime(bin_path)
        assert mtime_before == mtime_after

    def test_force_reconvert(self, temp_dir, sample_embeddings_parquet):
        """Test that force=True forces reconversion."""
        parquet_path, _, _ = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        # First conversion
        converter.convert()

        # Get modification time of binary file
        bin_path = os.path.join(output_dir, "embeddings.bin")
        mtime_before = os.path.getmtime(bin_path)

        # Second conversion with force=True
        import time

        time.sleep(0.1)  # Ensure time difference
        converter.convert(force=True)

        # Check that file was modified
        mtime_after = os.path.getmtime(bin_path)
        assert mtime_after > mtime_before

    def test_custom_item_id_column(self, temp_dir):
        """Test conversion with custom item_id column name."""
        # Create sample data with custom column name
        n_items = 50
        embedding_dim = 64

        item_ids = np.arange(n_items)
        embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)

        df = pd.DataFrame(
            {
                "track_id": item_ids,  # Custom column name
                "item_embedding": [emb.tolist() for emb in embeddings],
            }
        )

        parquet_path = os.path.join(temp_dir, "embeddings_custom.parquet")
        df.to_parquet(parquet_path, index=False)

        output_dir = os.path.join(temp_dir, "binary_custom")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path,
            output_dir=output_dir,
            item_id_column="track_id",
        )

        converter.convert()

        # Load id_to_offset and check
        import pickle

        id_to_offset_path = os.path.join(output_dir, "id_to_offset.pkl")
        with open(id_to_offset_path, "rb") as f:
            id_to_offset = pickle.load(f)

        assert len(id_to_offset) == n_items

    def test_custom_embedding_column(self, temp_dir):
        """Test conversion with custom embedding column name."""
        # Create sample data with custom column name
        n_items = 50
        embedding_dim = 64

        item_ids = np.arange(n_items)
        embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)

        df = pd.DataFrame(
            {
                "item_id": item_ids,
                "track_embedding": [
                    emb.tolist() for emb in embeddings
                ],  # Custom column name
            }
        )

        parquet_path = os.path.join(temp_dir, "embeddings_custom.parquet")
        df.to_parquet(parquet_path, index=False)

        output_dir = os.path.join(temp_dir, "binary_custom")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path,
            output_dir=output_dir,
            embedding_column="track_embedding",
        )

        converter.convert()

        # Load binary file and check shape
        bin_path = os.path.join(output_dir, "embeddings.bin")
        loaded_embeddings = np.memmap(
            bin_path, dtype=np.float32, mode="r", shape=(n_items, embedding_dim)
        )

        assert loaded_embeddings.shape == (n_items, embedding_dim)

    def test_empty_embeddings(self, temp_dir):
        """Test handling of empty embeddings file."""
        # Create empty parquet file
        df = pd.DataFrame({"item_id": [], "item_embedding": []})

        parquet_path = os.path.join(temp_dir, "embeddings_empty.parquet")
        df.to_parquet(parquet_path, index=False)

        output_dir = os.path.join(temp_dir, "binary_empty")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        # Should handle empty file gracefully
        converter.convert()

        # Check that files are created but empty
        bin_path = os.path.join(output_dir, "embeddings.bin")
        assert os.path.exists(bin_path)
        assert os.path.getsize(bin_path) == 0

    def test_duplicate_item_ids(self, temp_dir):
        """Test handling of duplicate item_ids."""
        # Create data with duplicate item_ids
        n_items = 50
        embedding_dim = 64

        item_ids = np.concatenate([np.arange(n_items), [0, 1, 2]])  # Duplicates
        embeddings = np.random.randn(len(item_ids), embedding_dim).astype(np.float32)

        df = pd.DataFrame(
            {
                "item_id": item_ids,
                "item_embedding": [emb.tolist() for emb in embeddings],
            }
        )

        parquet_path = os.path.join(temp_dir, "embeddings_dup.parquet")
        df.to_parquet(parquet_path, index=False)

        output_dir = os.path.join(temp_dir, "binary_dup")

        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )

        # Should handle duplicates (keep last occurrence)
        converter.convert()

        # Load id_to_offset
        import pickle

        id_to_offset_path = os.path.join(output_dir, "id_to_offset.pkl")
        with open(id_to_offset_path, "rb") as f:
            id_to_offset = pickle.load(f)

        # Should have unique item_ids only
        assert len(id_to_offset) == n_items


class TestEmbeddingBinaryConverterIntegration:
    """Integration tests for EmbeddingBinaryConverter with EmbeddingCache."""

    def test_converter_cache_integration(self, temp_dir, sample_embeddings_parquet):
        """Test that converted embeddings can be loaded by EmbeddingCache."""
        parquet_path, item_ids, original_embeddings = sample_embeddings_parquet
        output_dir = os.path.join(temp_dir, "binary")

        # Convert
        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )
        converter.convert()

        # Load with EmbeddingCache
        from data.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(binary_dir=output_dir)

        # Test random item_ids
        test_ids = np.random.choice(item_ids, size=10, replace=False)
        test_ids_tensor = torch.tensor(test_ids, dtype=torch.long)

        embeddings = cache.get_embeddings(test_ids_tensor)

        # Check shape
        assert embeddings.shape == (len(test_ids), original_embeddings.shape[1])

        # Check values match
        for i, item_id in enumerate(test_ids):
            expected = original_embeddings[item_id]
            actual = embeddings[i].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow: create parquet -> convert -> load -> use."""
        # 1. Create embeddings parquet
        n_items = 200
        embedding_dim = 256

        item_ids = np.arange(n_items)
        embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)

        df = pd.DataFrame(
            {
                "item_id": item_ids,
                "item_embedding": [emb.tolist() for emb in embeddings],
            }
        )

        parquet_path = os.path.join(temp_dir, "embeddings.parquet")
        df.to_parquet(parquet_path, index=False)

        # 2. Convert to binary
        output_dir = os.path.join(temp_dir, "binary")
        converter = EmbeddingBinaryConverter(
            embeddings_parquet=parquet_path, output_dir=output_dir
        )
        converter.convert()

        # 3. Load with cache
        from data.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(binary_dir=output_dir)

        # 4. Use in batch
        batch_size = 32
        batch_ids = torch.randint(0, n_items, (batch_size,), dtype=torch.long)
        batch_embeddings = cache.get_embeddings(batch_ids)

        # 5. Verify
        assert batch_embeddings.shape == (batch_size, embedding_dim)
        assert batch_embeddings.dtype == torch.float32

        # Check that embeddings are correct
        for i, item_id in enumerate(batch_ids):
            expected = embeddings[item_id.item()]
            actual = batch_embeddings[i].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
