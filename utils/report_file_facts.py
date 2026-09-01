from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import TracebackType
from typing import TypeVar, cast


Value = TypeVar("Value")
_CURRENT: ContextVar[ReportFileFacts | None] = ContextVar(
    "report_file_facts", default=None
)


class ReportFileFacts:
    def __init__(self, database: Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database, timeout=30)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS file_facts (
                namespace TEXT NOT NULL,
                identities TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, identities)
            )
            """
        )
        self._connection.commit()
        self._memory: dict[tuple[str, str], object] = {}
        self._canonical_paths: dict[Path, Path] = {}

    def load_or_compute(
        self,
        namespace: str,
        paths: Iterable[Path],
        compute: Callable[[], Value],
    ) -> Value:
        sources = tuple(Path(path) for path in paths)
        for _ in range(3):
            identities = self._identities(sources)
            key = namespace, identities
            if key in self._memory:
                return cast(Value, self._memory[key])
            row = self._connection.execute(
                "SELECT value FROM file_facts WHERE namespace = ? AND identities = ?",
                key,
            ).fetchone()
            if row is not None:
                value = json.loads(row[0])
                self._memory[key] = value
                return cast(Value, value)
            value = compute()
            if identities != self._identities(sources):
                continue
            serialized = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO file_facts(namespace, identities, value) "
                "VALUES (?, ?, ?)",
                (namespace, identities, serialized),
            )
            self._connection.commit()
            self._memory[key] = value
            return value
        raise RuntimeError("source files changed repeatedly while computing report facts")

    def sha256(self, path: Path) -> str:
        source = Path(path)

        def compute() -> str:
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            return digest.hexdigest()

        return self.load_or_compute("sha256", (source,), compute)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ReportFileFacts:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _identities(self, paths: tuple[Path, ...]) -> str:
        identities = []
        for path in paths:
            canonical = self._canonical_paths.get(path)
            if canonical is None:
                canonical = path.resolve(strict=True)
                if canonical == path.absolute():
                    self._canonical_paths[path] = canonical
            status = canonical.stat()
            identities.append(
                [os.fspath(canonical), status.st_size, status.st_mtime_ns]
            )
        return json.dumps(identities, separators=(",", ":"))


def report_file_facts(base_path: Path) -> ReportFileFacts:
    override = os.environ.get("DCN_REPORT_FILE_FACTS")
    database = (
        Path(override)
        if override is not None
        else Path(base_path) / "report_file_facts.sqlite3"
    )
    return _report_file_facts(os.fspath(database.resolve()))


@cache
def _report_file_facts(database: str) -> ReportFileFacts:
    return ReportFileFacts(Path(database))


def current_report_file_facts() -> ReportFileFacts | None:
    return _CURRENT.get()


@contextmanager
def report_file_fact_scope(base_path: Path) -> Iterator[None]:
    token = _CURRENT.set(report_file_facts(base_path))
    try:
        yield
    finally:
        _CURRENT.reset(token)
