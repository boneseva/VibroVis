"""
Bounded in-memory caches for multi-user VibroVis (one gunicorn worker, many browser sessions).

``SessionFrameCache``  -> ``callbacks_constants.server_cache``
    Per-browser sampled/filtered frames. Every entry is addressed by a random key that only the owning browser
    holds (in its ``filtered-data`` Store), so entries never mix between users. It is bounded (entries and bytes,
    least-recently-used first) and a key that a browser has replaced is kept for a short grace period so callbacks
    that are still running against it finish.

``DatasetCache`` (added in a later step)
    Shared, read-only model DataFrames keyed by (project, model, sorted locations).

Eviction only drops the cache's reference. A callback that already holds a frame keeps it alive until it returns,
so an active callback can never lose its DataFrame halfway through. A miss is always reported as a miss (None /
KeyError); nothing here ever answers a lookup with another key's data.
"""
import os
import sys
import threading
import time
from collections import OrderedDict

DEFAULT_SESSION_MAX_ENTRIES = 256
DEFAULT_SESSION_MAX_BYTES = 1 * 1024 ** 3       # 1 GB
DEFAULT_REPLACED_GRACE_SECONDS = 120.0


def _env_int(name, default):
    try:
        value = int(float(os.environ[name]))
        return value if value > 0 else default
    except (KeyError, ValueError):
        return default


def frame_nbytes(value):
    """Real memory of a DataFrame (``memory_usage(deep=True)``); a rough size for anything else."""
    try:
        return int(value.memory_usage(deep=True).sum())
    except Exception:
        return sys.getsizeof(value)


def cache_key_of(payload):
    """The cache key inside a ``filtered-data`` Store value (``{'cache_key': k, ...}`` or the bare key)."""
    if isinstance(payload, dict):
        return payload.get('cache_key')
    return payload


class _Entry:
    __slots__ = ('value', 'nbytes', 'expires_at')

    def __init__(self, value, nbytes):
        self.value = value
        self.nbytes = nbytes
        self.expires_at = None      # set when the owning browser replaced this key


class SessionFrameCache:
    """Thread-safe, bounded, dict-like cache of per-browser frames (see module docstring)."""

    def __init__(self, max_entries=None, max_bytes=None, grace_seconds=None, clock=time.monotonic):
        self.max_entries = max_entries or _env_int('VIBROVIS_SERVER_CACHE_MAX_ENTRIES', DEFAULT_SESSION_MAX_ENTRIES)
        self.max_bytes = max_bytes or _env_int('VIBROVIS_SERVER_CACHE_MAX_BYTES', DEFAULT_SESSION_MAX_BYTES)
        self.grace_seconds = (DEFAULT_REPLACED_GRACE_SECONDS if grace_seconds is None else float(grace_seconds))
        self._clock = clock
        self._entries = OrderedDict()           # key -> _Entry, least recently used first
        self._bytes = 0
        self._lock = threading.RLock()
        self.evictions = 0

    # -- dict-style API (used by the callbacks and the existing tests) --------------------------------------------
    def get(self, key, default=None):
        with self._lock:
            self._purge_expired()
            entry = self._entries.get(key)
            if entry is None:
                return default
            self._entries.move_to_end(key)
            return entry.value

    def __getitem__(self, key):
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.pop(key)

    def __contains__(self, key):
        with self._lock:
            self._purge_expired()
            return key in self._entries

    def __len__(self):
        with self._lock:
            self._purge_expired()
            return len(self._entries)

    def pop(self, key, *default):
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                if default:
                    return default[0]
                raise KeyError(key)
            self._bytes -= entry.nbytes
            return entry.value

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def keys(self):
        with self._lock:
            self._purge_expired()
            return list(self._entries.keys())

    def values(self):
        """Snapshot of the cached values (does not refresh recency)."""
        with self._lock:
            self._purge_expired()
            return [e.value for e in self._entries.values()]

    def items(self):
        with self._lock:
            self._purge_expired()
            return [(k, e.value) for k, e in self._entries.items()]

    # -- bounded storage ------------------------------------------------------------------------------------------
    def put(self, key, value):
        nbytes = frame_nbytes(value)                # measured outside the lock: can take a few ms for big frames
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[key] = _Entry(value, nbytes)
            self._bytes += nbytes
            self._purge_expired()
            self._evict(protect=key)

    def replace(self, key, value, replaces=None):
        """
        Store *value* under *key* and mark the browser's previous key (and the entries derived from it, named
        ``"<previous>::..."``) for removal after the grace period, so a filter change does not leave one orphan
        frame behind per click while callbacks still running on the previous frame can finish.
        """
        self.put(key, value)
        if replaces is None or replaces == key:
            return
        with self._lock:
            deadline = self._clock() + self.grace_seconds
            prefix = f"{replaces}::" if isinstance(replaces, str) else None
            for existing, entry in self._entries.items():
                if existing == replaces or (prefix and isinstance(existing, str) and existing.startswith(prefix)):
                    entry.expires_at = deadline

    def stats(self):
        with self._lock:
            self._purge_expired()
            return {'entries': len(self._entries), 'bytes': self._bytes,
                    'max_entries': self.max_entries, 'max_bytes': self.max_bytes, 'evictions': self.evictions}

    # -- internals (lock held) ------------------------------------------------------------------------------------
    def _drop(self, key):
        entry = self._entries.pop(key)
        self._bytes -= entry.nbytes

    def _purge_expired(self):
        now = self._clock()
        for key in [k for k, e in self._entries.items() if e.expires_at is not None and e.expires_at <= now]:
            self._drop(key)

    def _evict(self, protect):
        """Enforce both bounds: replaced (grace) entries go first, then least recently used. Never the new entry."""
        while len(self._entries) > 1 and (len(self._entries) > self.max_entries or self._bytes > self.max_bytes):
            victim = next((k for k, e in self._entries.items() if e.expires_at is not None and k != protect), None)
            if victim is None:
                victim = next((k for k in self._entries if k != protect), None)
            if victim is None:
                break
            self._drop(victim)
            self.evictions += 1


# ======================================================================================================================
# Shared, read-only model datasets
# ======================================================================================================================
DEFAULT_DATASET_CACHE_BYTES = 4 * 1024 ** 3     # 4 GB
DEFAULT_MAX_CONCURRENT_LOADS = 1


def dataset_key(project, model, locations):
    """
    Full identity of a loaded dataset: ``(project, model, sorted distinct locations)``.
    ``project`` is None for the legacy flat cache. Returns None when the selection is incomplete.
    """
    if isinstance(locations, (list, tuple, set, frozenset)):
        locs = sorted({str(v) for v in locations if v is not None and str(v) != ''})
    elif locations is None or str(locations) == '':
        locs = []
    else:
        locs = [str(locations)]
    if not model or not locs:
        return None
    return (None if project in (None, '') else str(project), str(model), tuple(locs))


class Dataset:
    """A loaded model DataFrame plus its derived metadata. Treat as IMMUTABLE: it is shared by every user."""
    __slots__ = ('key', 'df', 'metadata', 'nbytes')

    def __init__(self, key, df, metadata, nbytes):
        self.key = key
        self.df = df
        self.metadata = metadata
        self.nbytes = nbytes


class _Flight:
    __slots__ = ('event', 'result', 'error')

    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None


class DatasetCache:
    """
    Bounded, thread-safe cache of shared read-only datasets.

    * keyed by ``dataset_key`` (project, model, sorted locations): different selections never overwrite each other,
      and a late load only ever fills its OWN key, so a stale load cannot affect anyone else's view;
    * single-flight: concurrent requests for the same key share one load;
    * a load semaphore (default 1) bounds the transient memory of loading, which is several times the frame size;
    * LRU eviction by measured bytes (``memory_usage(deep=True)``), budget ``VIBROVIS_DATASET_CACHE_BYTES``
      (default 4 GB). A frame bigger than the whole budget is admitted alone. Eviction only drops the cache's
      reference, so a callback that already holds a frame keeps it until it returns;
    * the loader's result is never mutated after insertion. Empty/None results are not cached.
    """

    def __init__(self, max_bytes=None, max_concurrent_loads=None):
        self.max_bytes = max_bytes or _env_int('VIBROVIS_DATASET_CACHE_BYTES', DEFAULT_DATASET_CACHE_BYTES)
        slots = max_concurrent_loads or _env_int('VIBROVIS_MAX_CONCURRENT_LOADS', DEFAULT_MAX_CONCURRENT_LOADS)
        self.max_concurrent_loads = slots
        self._slots = threading.BoundedSemaphore(slots)
        self._entries = OrderedDict()           # key -> Dataset, least recently used first
        self._bytes = 0
        self._inflight = {}                     # key -> _Flight
        self._lock = threading.RLock()
        self.loads = 0                          # completed loader calls (tests / diagnostics)
        self.evictions = 0

    def peek(self, key):
        """The cached Dataset for *key* (refreshing its recency), or None. Never loads."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def get(self, key, loader):
        """
        Return the Dataset for *key*, calling ``loader()`` -> ``(DataFrame, metadata)`` at most once across
        concurrent callers. Returns None when the loader yields no rows. A loader error propagates to every waiter
        and is not cached.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            flight = self._inflight.get(key)
            leader = flight is None
            if leader:
                flight = self._inflight[key] = _Flight()

        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.result

        try:
            with self._slots:
                loaded = loader()
            result = None
            if loaded is not None:
                df, metadata = loaded
                if df is not None and len(df) > 0:
                    result = Dataset(key, df, metadata, frame_nbytes(df))
                    self._insert(result)
            self.loads += 1
            flight.result = result
            return result
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            flight.event.set()

    def put(self, key, df, metadata):
        """Seed or replace an entry directly (tests / warm-up). Returns the Dataset."""
        dataset = Dataset(key, df, metadata, frame_nbytes(df))
        self._insert(dataset)
        return dataset

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def keys(self):
        with self._lock:
            return list(self._entries.keys())

    def stats(self):
        with self._lock:
            return {'entries': len(self._entries), 'bytes': self._bytes, 'max_bytes': self.max_bytes,
                    'loads': self.loads, 'evictions': self.evictions,
                    'per_key': {k: e.nbytes for k, e in self._entries.items()}}

    def _insert(self, dataset):
        with self._lock:
            old = self._entries.pop(dataset.key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[dataset.key] = dataset
            self._bytes += dataset.nbytes
            while self._bytes > self.max_bytes and len(self._entries) > 1:
                victim_key = next(k for k in self._entries if k != dataset.key)
                victim = self._entries.pop(victim_key)
                self._bytes -= victim.nbytes
                self.evictions += 1
