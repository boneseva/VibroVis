"""
Concurrency helpers for the shared, persistent manual-label stores.

Manual labels (``MANUAL_LABELS_CACHE``) and their clip durations (``CLIP_DURATION_CACHE``) are global, shared by
every user and persisted on disk. Storage locations, key tuples ``(location, microlocation, file, channel,
round(clip_time*10))`` and value formats are NOT changed here; this module only makes access to them safe:

* every writer holds ``MANUAL_LABELS_LOCK`` and applies a group of changes in ONE diskcache transaction, so other
  threads, store handles and processes see all of a group or none of it;
* every reader that iterates takes a consistent snapshot (lock + transaction) instead of walking keys while
  another user writes;
* rename = scan + apply in one atomic section, so a concurrent save cannot be overwritten by a stale scan.

Conflict policy (documented limitation): two users writing the SAME clip is last-write-wins. No author or version is
stored, so there is no conflict warning.

The stores are looked up on ``callbacks.callbacks_constants`` at call time, so a test that swaps them is honoured.
"""
import contextlib
import time

UNLABELED = 'Unlabeled'


def _constants():
    from callbacks import callbacks_constants
    return callbacks_constants


@contextlib.contextmanager
def _transaction(cache):
    """One write transaction for diskcache stores; a no-op for plain dicts (used by some tests)."""
    transact = getattr(cache, 'transact', None)
    if transact is None:
        yield
    else:
        with transact():
            yield


def log_write(op, key, value):
    """Append one line to the label-write audit log (best effort, never raises)."""
    try:
        with open(_constants().LABEL_WRITE_LOG, 'a', encoding='utf-8') as log:
            log.write(f"{time.time()}\t{op}\t{key}\t{value}\n")
    except Exception:
        pass


def snapshot_store(cache, lock=None):
    """A consistent ``dict`` copy of *cache*: taken under *lock* and inside one transaction."""
    snapshot = {}
    with (lock if lock is not None else contextlib.nullcontext()):
        with _transaction(cache):
            iter_keys = cache.iterkeys() if hasattr(cache, 'iterkeys') else iter(list(cache))
            for key in list(iter_keys):
                try:
                    snapshot[key] = cache[key]
                except KeyError:
                    continue
    return snapshot


def snapshot_labels():
    """Consistent copy of every shared manual label: ``{(loc, micro, file, channel, t10): label}``."""
    c = _constants()
    return snapshot_store(c.MANUAL_LABELS_CACHE, c.MANUAL_LABELS_LOCK)


def unique_label_values():
    """Sorted distinct label texts currently stored."""
    return sorted(set(snapshot_labels().values()))


def set_clip_labels(clips, label, model_name, op):
    """
    Save (or, for 'Unlabeled', delete) *label* for every ``(instance_key, duration)`` in *clips* as one atomic group.

    Durations are written before labels and deleted after them, so a reader never sees a label without the
    duration it was saved with. Returns the number of clips written or removed.
    """
    c = _constants()
    labels, durations = c.MANUAL_LABELS_CACHE, c.CLIP_DURATION_CACHE
    with c.MANUAL_LABELS_LOCK:
        if label == UNLABELED:
            with _transaction(labels):
                for key, _dur in clips:
                    labels.pop(key, None)
            with _transaction(durations):
                for key, _dur in clips:
                    durations.pop(key, None)
            for key, _dur in clips:
                log_write(f"{op}_delete", key, 'DELETED')
            return len(clips)

        with _transaction(durations):
            for key, dur in clips:
                durations[key] = (float(dur), model_name)
        with _transaction(labels):
            for key, _dur in clips:
                labels[key] = label
        for key, _dur in clips:
            log_write(f"{op}_save", key, label)
        return len(clips)


def bulk_set_labels(items, op=None):
    """Upsert ``(key, label)`` pairs (import, server-file load) as one atomic group. Returns the count."""
    c = _constants()
    items = list(items)
    with c.MANUAL_LABELS_LOCK:
        with _transaction(c.MANUAL_LABELS_CACHE):
            for key, label in items:
                c.MANUAL_LABELS_CACHE[key] = label
        if op:
            for key, label in items:
                log_write(op, key, label)
    return len(items)


def rename_label(old_label, new_label, op='clusters_rename'):
    """
    Rename/merge every clip labelled *old_label* to *new_label* (or delete them when *new_label* is 'Unlabeled').

    The scan and the update happen in one atomic section: a label saved by another user cannot be overwritten with
    a stale value, and other users never see a half-renamed set. Returns True when anything changed.
    """
    c = _constants()
    labels = c.MANUAL_LABELS_CACHE
    with c.MANUAL_LABELS_LOCK:
        with _transaction(labels):
            matches = [key for key, value in snapshot_store(labels).items() if value == old_label]
            if not matches:
                return False
            for key in matches:
                if new_label == UNLABELED:
                    labels.pop(key, None)
                else:
                    labels[key] = new_label
        for key in matches:
            if new_label == UNLABELED:
                log_write(f"{op}_delete", key, 'DELETED')
            else:
                log_write(f"{op}_update", key, new_label)
    return True


def clear_labels():
    """Remove every manual label (the 'Reset All Labels' action)."""
    c = _constants()
    with c.MANUAL_LABELS_LOCK:
        c.MANUAL_LABELS_CACHE.clear()
