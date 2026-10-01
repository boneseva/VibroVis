"""
Lightweight performance and cache diagnostics utility for VibroVis.
Provides timing instrumentation for the main data/plot pipeline and
cache state inspection.

Controlled via the VIBROVIS_DIAGNOSTICS environment variable or programmatically
via set_diagnostics_enabled(True/False). Default is False (disabled).
"""
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Single global flag for enabling diagnostics
_DIAGNOSTICS_ENABLED = os.environ.get("VIBROVIS_DIAGNOSTICS", "").lower() in ("1", "true", "yes", "on")


def is_diagnostics_enabled() -> bool:
    """Return whether diagnostics and performance profiling are enabled."""
    return _DIAGNOSTICS_ENABLED


def set_diagnostics_enabled(enabled: bool = True) -> None:
    """Enable or disable diagnostics and performance profiling."""
    global _DIAGNOSTICS_ENABLED
    _DIAGNOSTICS_ENABLED = bool(enabled)


def get_cache_diagnostics(current_sample_df: Optional[Any] = None) -> Dict[str, Any]:
    """
    Measure cache diagnostic metrics:
    - number of entries in server_cache
    - approximate number of rows stored in server_cache
    - size (rows) of the current filtered/sample result
    - number of manual labels in MANUAL_LABELS_CACHE
    """
    try:
        from callbacks.callbacks_constants import server_cache, MANUAL_LABELS_CACHE, MANUAL_LABELS_LOCK
    except ImportError:
        server_cache = {}
        MANUAL_LABELS_CACHE = {}
        MANUAL_LABELS_LOCK = None

    server_cache_entries = len(server_cache)
    server_cache_rows = 0
    for v in server_cache.values():
        if hasattr(v, '__len__'):
            try:
                server_cache_rows += len(v)
            except Exception:
                pass

    sample_rows = 0
    if current_sample_df is not None and hasattr(current_sample_df, '__len__'):
        try:
            sample_rows = len(current_sample_df)
        except Exception:
            pass

    manual_labels_count = 0
    if MANUAL_LABELS_LOCK is not None:
        try:
            with MANUAL_LABELS_LOCK:
                manual_labels_count = len(MANUAL_LABELS_CACHE)
        except Exception:
            manual_labels_count = len(MANUAL_LABELS_CACHE)
    else:
        manual_labels_count = len(MANUAL_LABELS_CACHE)

    return {
        'server_cache_entries': server_cache_entries,
        'server_cache_rows': server_cache_rows,
        'sample_rows': sample_rows,
        'manual_labels_count': manual_labels_count,
    }


def format_cache_diagnostics(stats: Dict[str, Any]) -> str:
    """Format cache diagnostics dictionary into a concise one-line summary."""
    entries = stats.get('server_cache_entries', 0)
    rows = stats.get('server_cache_rows', 0)
    sample = stats.get('sample_rows', 0)
    labels = stats.get('manual_labels_count', 0)
    if sample > 0:
        return f"[cache] server_cache: {entries} entries (~{rows:,} rows) | sample: {sample:,} rows | manual labels: {labels:,}"
    return f"[cache] server_cache: {entries} entries (~{rows:,} rows) | manual labels: {labels:,}"


class PipelineTimer:
    """
    Lightweight timing utility using time.perf_counter() for pipeline diagnostics.
    Can be used to record named stages and format them cleanly.
    """
    def __init__(self, name: str = "pipeline", enabled: Optional[bool] = None, output_stream=None):
        self.name = name
        self.enabled = is_diagnostics_enabled() if enabled is None else enabled
        self.output_stream = output_stream or sys.stdout
        self.timings: Dict[str, float] = {}
        self._t0: Optional[float] = None
        self._cur_step: Optional[str] = None
        self._cur_start: Optional[float] = None

    def start(self) -> "PipelineTimer":
        """Start total timer."""
        if not self.enabled:
            return self
        self._t0 = time.perf_counter()
        return self

    def step(self, step_name: str) -> "PipelineTimer":
        """Switch to a new named step, recording elapsed time of the previous step."""
        if not self.enabled:
            return self
        now = time.perf_counter()
        if self._cur_step is not None and self._cur_start is not None:
            elapsed = now - self._cur_start
            self.timings[self._cur_step] = self.timings.get(self._cur_step, 0.0) + elapsed
        self._cur_step = step_name
        self._cur_start = now
        return self

    def stop_step(self) -> None:
        """Finish the current step."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._cur_step is not None and self._cur_start is not None:
            elapsed = now - self._cur_start
            self.timings[self._cur_step] = self.timings.get(self._cur_step, 0.0) + elapsed
            self._cur_step = None
            self._cur_start = None

    def stop(self, cache_df: Optional[Any] = None, extra_info: Optional[str] = None) -> Dict[str, float]:
        """
        Stop total timer, format output if enabled, and return timing dictionary.
        """
        if not self.enabled:
            return {}
        self.stop_step()
        total = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        self.timings['total'] = total

        lines = [f"{self.name} total: {total:.2f} s" + (f" ({extra_info})" if extra_info else "")]
        for k, v in self.timings.items():
            if k == 'total':
                continue
            lines.append(f"  {k:<12}: {v:.2f} s")

        cache_stats = get_cache_diagnostics(cache_df)
        lines.append(f"  {format_cache_diagnostics(cache_stats)}")

        print("\n".join(lines), file=self.output_stream)
        return self.timings


@contextmanager
def time_block(block_name: str, extra_info: Optional[str] = None, show_cache: bool = True):
    """
    Context manager for timing a standalone operation (e.g. data loading).
    """
    if not is_diagnostics_enabled():
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        extra_str = f" ({extra_info})" if extra_info else ""
        lines = [f"{block_name} total: {elapsed:.2f} s{extra_str}"]
        if show_cache:
            stats = get_cache_diagnostics()
            lines.append(f"  {format_cache_diagnostics(stats)}")
        print("\n".join(lines))
