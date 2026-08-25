# Data Module Tests

This directory contains tests for the data module components.

## Running Tests

### Run all tests
```bash
pytest data/tests/
```

### Run specific test file
```bash
pytest data/tests/test_embedding_binary_converter.py
```

### Run with verbose output
```bash
pytest data/tests/ -v
```

### Run with coverage
```bash
pytest data/tests/ --cov=data --cov-report=html
```

## Test Files

### `test_embedding_binary_converter.py`
Tests for the embedding binary converter that converts embeddings from parquet to binary format.

**Test Coverage:**
- File creation and structure
- Conversion correctness (embeddings match original)
- Metadata correctness
- ID to offset mapping
- Caching behavior (skip if exists, force reconvert)
- Custom column names
- Edge cases (empty files, duplicates)
- Integration with EmbeddingCache
- End-to-end workflow

## Test Structure

Tests use pytest fixtures for:
- `temp_dir`: Temporary directory for test files (auto-cleanup)
- `sample_embeddings_parquet`: Sample embeddings data for testing

## Requirements

```bash
pip install pytest pytest-cov
```

## Adding New Tests

When adding new test files:
1. Name them `test_*.py`
2. Use pytest fixtures for setup/teardown
3. Group related tests in classes
4. Add docstrings explaining what is being tested
5. Update this README with test coverage information