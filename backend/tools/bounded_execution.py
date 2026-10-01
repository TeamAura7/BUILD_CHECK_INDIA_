"""
Bounded-processing helper (phase3.1 FIX #10).

The extraction pipeline must never hang indefinitely on a pathological
PDF (huge page count, dense raster geometry, degenerate OCR/OpenCV
input). This module provides a small, dependency-free wall-clock
timeout wrapper: run a callable in a worker thread, and if it doesn't
finish within the configured budget, raise `ExtractionTimeoutError`
instead of blocking forever.

Note: because CPython (and native extensions like PyMuPDF/OpenCV that
release the GIL during heavy C calls) is used here via a thread rather
than a process, a truly stuck C-level call may keep running in the
background after we give up waiting on it. That's an acceptable
trade-off for this project's scope (no subprocess sandboxing
infrastructure) — the important behavioral contract is upheld: the
*caller* gets an explicit, bounded failure instead of hanging forever.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, TypeVar

T = TypeVar("T")


class ExtractionTimeoutError(Exception):
    """Raised when a bounded operation exceeds its configured time budget."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{operation} did not complete within the configured "
            f"{timeout_seconds:.1f}s timeout."
        )


def run_with_timeout(fn: Callable[[], T], timeout_seconds: float, operation: str = "operation") -> T:
    """
    Run `fn()` with a hard wall-clock budget.

    Raises `ExtractionTimeoutError` (never hangs) if `fn` does not
    complete in time. Any exception raised by `fn` itself propagates
    unchanged.
    """
    if timeout_seconds <= 0:
        return fn()
    # Deliberately NOT `with ThreadPoolExecutor(...) as pool:` -- the
    # executor's __exit__ calls shutdown(wait=True), which blocks until the
    # worker thread finishes naturally even after future.result(timeout=...)
    # has already raised. That defeats the whole point of this wrapper: a
    # genuinely hung callable would make the *caller* block indefinitely too,
    # the exact failure mode this function exists to prevent. Instead, shut
    # down with wait=False so a still-running worker is abandoned rather than
    # waited on -- the caller gets control back at the documented timeout
    # regardless of whether the underlying call ever returns.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        pool.shutdown(wait=False)
        raise ExtractionTimeoutError(operation, timeout_seconds) from exc
    else:
        pool.shutdown(wait=False)
        return result


__all__ = ["ExtractionTimeoutError", "run_with_timeout"]
