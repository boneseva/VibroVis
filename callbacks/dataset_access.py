"""
How callbacks obtain the DataFrame that belongs to THEIR browser.

The load callback (``load_data_into_server_cache``) stores the loaded dataset in the shared read-only
``DATASETS`` cache and returns the dataset's key inside ``derived-model-metadata``, a Store that only the
requesting browser holds. Every other callback reads that key (as a State/Input) and resolves its frame here, so a
callback can only ever see the dataset its own browser loaded. There is no global "current dataset".

Rules for callers:
* the returned frame is shared by every user and is READ-ONLY: filter with ``df[mask]`` (which copies) or call
  ``.copy()`` before adding columns, never assign into it;
* a ``None`` frame means "nothing loaded for this browser (yet)".
"""
import os

import read_data
from callbacks import callbacks_constants as C
from dataset_cache import dataset_key


def key_to_payload(key):
    """JSON-safe form of a dataset key for the browser's metadata Store."""
    return [key[0], key[1], list(key[2])]


def key_from_metadata(metadata):
    """The dataset key a browser's ``derived-model-metadata`` points at, or None."""
    raw = metadata.get('key') if isinstance(metadata, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3 or not isinstance(raw[2], (list, tuple)):
        return None
    return (raw[0], raw[1], tuple(raw[2]))


def is_resident(key):
    """True if the shared cache still holds *key* (a lookup only: it never loads, it refreshes the key's recency)."""
    return key is not None and C.DATASETS.peek(key) is not None


def init_marker(url_init):
    """``{'epoch', 'attempt'}`` of a TERMINAL (ready/failed) URL initialisation, else None. Used for idempotency."""
    if not isinstance(url_init, dict) or url_init.get('phase') not in ('ready', 'failed'):
        return None
    attempt = url_init.get('attempt')
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        return None
    return {'epoch': url_init.get('epoch'), 'attempt': attempt}


def recorded_attempt(metadata):
    """The initialisation attempt this browser's last load was made under (0 if none was recorded)."""
    init = metadata.get('init') if isinstance(metadata, dict) else None
    a = init.get('attempt') if isinstance(init, dict) else None
    return a if isinstance(a, int) and not isinstance(a, bool) else 0


def _read_partition(key):
    """Read one dataset from disk (project partitions, or the legacy flat cache when no project)."""
    import pandas as pd
    project, model, locations = key
    if project:
        return read_data.load_project_model_data(project_id=project, selected_model=model,
                                                 selected_location=list(locations))
    data_path = read_data.SAVE_PATH
    if not os.path.exists(data_path):
        vibro_path = os.path.join("data", "vibro", "cache", "final_data.parquet")
        if os.path.exists(vibro_path):
            data_path = vibro_path
    if not os.path.exists(data_path):
        return None
    cols = ['x', 'y', 'location', 'microlocation', 'model_name', 'channel', 'cluster_num', 'cluster_id', 'day_dt',
            'start_hour_float', 'file_name', 'clip_time', 'clip_duration', 'mp3_file', 'row_idx', 'start_dt',
            'time_of_day', 'recorder_type', 'wav_file']
    df = pd.read_parquet(data_path, filters=[[('model_name', '==', model), ('location', 'in', list(locations))]],
                         columns=cols)
    return read_data.process_loaded_model_dataframe(df)


def _build(key):
    """Loader for ``DatasetCache.get``: ``(frame, metadata)`` ready to be shared, or None."""
    import pandas as pd
    from callbacks.callbacks_data import extract_model_metadata
    df = _read_partition(key)
    if df is None or df.empty:
        return None
    # Sanity guard: the reader falls back to the WHOLE partition when no row matches the locations. Never cache
    # (or serve) a frame that holds locations the key does not name.
    if 'location' in df.columns and not {str(v) for v in pd.unique(df['location'])} <= set(key[2]):
        print(f"[dataset] refusing to cache {key}: loaded rows outside the requested locations")
        return None
    if 'clip_duration' not in df.columns:
        df['clip_duration'] = 5.0           # filled BEFORE the frame is shared; callbacks must not add it later
    return df, extract_model_metadata(df)


def load_dataset(key):
    """The shared Dataset for *key* (loading it once, single-flight), or None for an invalid/empty selection."""
    if key is None:
        return None
    project, _model, locations = key
    if project and not read_data.is_valid_project_location(project, list(locations)):
        return None                         # unknown location/project: never load (could read a whole partition)
    return C.DATASETS.get(key, lambda: _build(key))


def resolve_dataset(metadata, project=None, model=None, locations=None):
    """
    ``(frame, state)`` for the browser whose Store content is *metadata*.

    state: 'ok'    -> frame is this browser's loaded dataset
           'stale' -> the browser has since selected something else; its load has not finished (frame is None)
           'none'  -> nothing loaded (frame is None)

    Pass the browser's current project/model/locations to detect 'stale'.
    """
    key = key_from_metadata(metadata)
    if key is None:
        return None, 'none'                 # this browser has no dataset loaded (yet)
    expected = dataset_key(project, model, locations) if (model and locations) else None
    if expected is not None and expected != key:
        return None, 'stale'
    dataset = C.DATASETS.peek(key)
    if dataset is None:                     # evicted since it was loaded: reload (single-flight), same key
        try:
            dataset = load_dataset(key)
        except Exception as exc:            # noqa: BLE001
            print(f"[dataset] reload failed for {key}: {exc}")
            dataset = None
    return (dataset.df, 'ok') if dataset is not None else (None, 'none')


def dataset_object(metadata):
    """The shared ``Dataset`` (frame + server-side metadata) the browser's *metadata* Store points at, or None."""
    key = key_from_metadata(metadata)
    if key is None:
        return None
    dataset = C.DATASETS.peek(key)
    if dataset is None:
        try:
            dataset = load_dataset(key)
        except Exception as exc:            # noqa: BLE001
            print(f"[dataset] reload failed for {key}: {exc}")
            dataset = None
    return dataset


def dataset_for(metadata, project=None, model=None, locations=None):
    """Just the frame of :func:`resolve_dataset` (None unless state is 'ok')."""
    return resolve_dataset(metadata, project, model, locations)[0]
