import fcntl
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def hold(path: Path | None, what: str, *, shared: bool = False) -> Iterator[None]:
    """Hold an advisory lock while a section runs.

    ``None`` skips the lock, for a caller with nothing to serialize against.
    Shared holders coexist but still block an exclusive holder.

    An advisory lock on an open file: the kernel drops it when the holder dies,
    so a killed run cannot leave the queue stuck. The pid inside is for whoever
    is looking at the box wondering who has it; nothing reads it back.
    """
    if path is None:
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # Not "w": that truncates before the lock is taken, so a waiting process
    # erases the pid of the run it is waiting for.
    with open(path, "a+") as handle:
        start = perf_counter()
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        logger.info("Took the %s after %.1fs", what, perf_counter() - start)
        if not shared:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
