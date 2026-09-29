"""Handing the GPU back between stages.

The GPU this runs on is shared, and `songtrace serve` outlives every job it
runs, so a stage that caches its weights holds VRAM until the process exits —
including while a later stage is waiting on the Whisper server, which can be
minutes of a GPU nothing is using. Two things have to be undone: the caches
themselves, and PyTorch's allocator, which keeps freed blocks for reuse rather
than returning them to the driver.

A stage that caches a model registers how to drop it with @on_release, so this
module needs to know nothing about the pipeline. Registration happens at import
time, which run_pipeline has already done by the time it calls release().
"""
import gc
import sys
from typing import Callable

_callbacks: list[Callable[[], None]] = []


def on_release(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a callback that drops a cached model."""
    _callbacks.append(fn)
    return fn


def release() -> int:
    """Drop every cached model and return its VRAM to the driver.

    Answers the bytes handed back, which is the only part of this a caller can
    show someone. Safe whenever: a stage that never loaded anything has nothing
    to drop, and the allocator is left alone unless this process actually
    initialised CUDA.
    """
    for callback in _callbacks:
        callback()
    # weights reach the allocator's free list only once the last reference
    # goes, and a module graph is cyclic enough to outlive refcounting
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return 0
    if not torch.cuda.is_initialized():
        return 0
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return max(before - torch.cuda.memory_reserved(), 0)
