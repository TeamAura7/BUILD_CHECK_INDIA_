"""
Tests for `backend.tools.bounded_execution.run_with_timeout`.

The regression this pins: a call whose underlying work keeps running well
past the configured timeout must still return control to the caller at
(approximately) the timeout, not block until the runaway work finishes.
The original implementation used
`with ThreadPoolExecutor(...) as pool:`, whose `__exit__` calls
`shutdown(wait=True)` -- that blocks past `future.result(timeout=...)`
raising, until the still-running worker thread finishes naturally. For a
genuinely pathological/hung callable, that meant the documented timeout
did not actually bound wall-clock time for the caller.
"""

from __future__ import annotations

import threading
import time

import pytest

from backend.tools.bounded_execution import ExtractionTimeoutError, run_with_timeout


def test_fast_callable_returns_its_result():
    assert run_with_timeout(lambda: 42, timeout_seconds=5.0, operation="test") == 42


def test_slow_callable_raises_timeout_error():
    with pytest.raises(ExtractionTimeoutError):
        run_with_timeout(lambda: time.sleep(1.0), timeout_seconds=0.05, operation="test")


def test_exception_from_callable_propagates_unchanged():
    def boom():
        raise ValueError("real failure")

    with pytest.raises(ValueError, match="real failure"):
        run_with_timeout(boom, timeout_seconds=5.0, operation="test")


def test_caller_regains_control_at_timeout_even_if_work_keeps_running():
    """The core regression test: a callable that runs far longer than the
    timeout must not make the caller wait for it to finish."""
    still_running = threading.Event()

    def runs_way_past_the_timeout():
        time.sleep(2.0)
        still_running.set()  # only reached if something (wrongly) waited for us

    start = time.monotonic()
    with pytest.raises(ExtractionTimeoutError):
        run_with_timeout(runs_way_past_the_timeout, timeout_seconds=0.1, operation="test")
    elapsed = time.monotonic() - start

    # Generous margin over the 0.1s budget for scheduling jitter, but nowhere
    # near the callable's own 2s runtime -- if the old shutdown(wait=True)
    # bug were still present, this would take ~2s instead.
    assert elapsed < 1.0
    assert not still_running.is_set()
