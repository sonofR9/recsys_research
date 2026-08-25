from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_process_group(
    backend: str | None = None,
) -> int:
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
        )

    return rank


def is_chief() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
