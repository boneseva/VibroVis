"""
URL view state for VibroVis (schema v1): parsing, validation, serialisation and the staged-restore bookkeeping.

PURE MODULE: standard library only, no Dash, no callbacks, no I/O. Nothing here touches the layout or any callback;
Phase 2 wires it in. Everything that has to travel through a ``dcc.Store`` is plain JSON (dict/list/str/number/bool).

Two validation layers, deliberately separate
--------------------------------------------
1. SYNTAX  (``parse``)      Needs no data. Types, enums, ranges, encodings, limits, duplicates. Produces a ``UrlState``
                            plus warnings. A value that is syntactically wrong is dropped with a warning.
2. APPLICATION (``validate_stage``)  Needs the real option lists (``Catalog``), which only exist once the previous
                            stage has been applied: project -> location -> model -> k -> filters. A value that is not
                            offered (any more) is dropped with a reason and the rest still applies.

URL semantics
-------------
* Absent key = "not specified" (the app keeps its normal default; for filters that means "everything").
* ``key=none`` (only where meaningful) = an EXPLICIT empty selection, e.g. ``micro=none``. ``key=`` (empty value) is
  malformed and ignored with a warning; it is never read as "none".
* Multi-value keys are comma separated. Items are split on the RAW comma and then percent-decoded, so an item that
  contains a comma is written ``%2C``.
* Duplicate owned keys: the LAST one wins (deterministic), with a ``duplicate_param`` warning.
* VibroVis only ever reads/writes the keys in ``OWNED_KEYS``. Every other query parameter is preserved verbatim, in its
  original order, by ``rewrite_search``.
* Never in a URL (by design): free-text manual-label names, Raven annotation content or category names, cluster
  colours/names, spectrogram settings, animation or playback state. ``labels`` only takes the predefined values
  ``unlabeled`` / ``labeled``.

Manual-label selection (``labels``)
-----------------------------------
``labels=unlabeled`` / ``labels=labeled`` describe the legend selection AT RESTORATION TIME: ``unlabeled`` ticks only
the ``Unlabeled`` row, ``labeled`` ticks every other row (see ``labels_to_tick``). That is exactly what the corresponding
manual ticking means in the UI. They are not a standing rule: a label that appears later (for example one saved by
another user through the shared label store) gets a new legend row, which keeps the existing default of being ticked.
A value that would leave NO row ticked is dropped, because the UI can not represent it faithfully.

Consistent snapshots (``consistent_state``)
-------------------------------------------
Whatever is serialised (live URL or share link) never contains a value that restoration would have to drop or that
contradicts another value: ``mth``/``mcc`` only with ``merge=1``; ``clusters``/``hide`` only with cluster colouring (and
never an empty selection); ``labels`` only with manual colouring; ``annot``/``cmp`` only with Raven colouring.

Precedence between a pending URL restore and the existing preset reset mechanism (``reset_decision``)
------------------------------------------------------------------------------------------------------
For a control that some reset callback is about to overwrite:
  1. the restore still has an unresolved (``pending``) value for that control  -> ``'restore'``  (the URL value wins;
     the reset callback must not overwrite it, and must not consume the rest of the restore);
  2. otherwise, a preset was just loaded (the 6-second window)                 -> ``'keep'``     (today's behaviour);
  3. otherwise                                                                  -> ``'default'``  (today's behaviour).
A restore that is cancelled (``cancel_restore``; e.g. the user pressed the preset "Load" button, a later explicit
action) owns nothing, so rules 2 and 3 apply again. With no restore at all, the answer is exactly today's behaviour.
"""
import dataclasses
import math
import re
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.parse import quote, unquote_plus

import limits as _limits       # stdlib-only leaf module: the single authority for the point limit/default/hour step

SCHEMA_VERSION = 1

MAX_QUERY_LENGTH = 8192           # longer queries are ignored entirely (all parameters are still preserved)
MAX_ITEMS = 500                   # per multi-value key
MAX_TEXT_LENGTH = 200             # per text item
NONE_TOKEN = "none"
MAX_CLUSTER_COUNT = 1_000        # sanity bound for ``k`` (real values are 10, 20, ...); unrelated to the point limit

VIEWS = ("scatter", "table")
COLORS = ("cluster", "manual", "raven")
COMPARES = ("all", "agree", "disagree")
LABEL_FILTERS = ("unlabeled", "labeled")

# Values that need no catalog because the app default is known statically (used for minimal "live" URLs and to pin
# defaults in an explicit "share" snapshot).
DEFAULTS = {"view": "scatter", "color": "cluster", "cmp": "all", "merge": False, "prio": True,
            "max": _limits.DEFAULT_MAX_POINTS}

# url key -> UrlState field. The order is the canonical serialisation order.
KEY_TO_FIELD = {
    "v": "version", "project": "project", "loc": "loc", "model": "model", "k": "k", "view": "view", "color": "color",
    "annot": "annot", "cmp": "cmp", "micro": "micro", "rec": "rec", "ch": "ch", "dates": "dates", "hours": "hours",
    "clusters": "clusters", "hide": "hide", "labels": "labels", "merge": "merge", "mth": "mth", "mcc": "mcc",
    "max": "max_points", "prio": "prio", "sort": "sort",
}
FIELD_TO_KEY = {f: k for k, f in KEY_TO_FIELD.items()}
OWNED_KEYS = frozenset(KEY_TO_FIELD)
STATE_KEYS = tuple(k for k in KEY_TO_FIELD if k != "v")

# Restore stages, in order. Nothing in a later stage is applied before the earlier stage is resolved.
STAGES = ("project", "location", "model", "k", "filters")
STAGE_KEYS = {
    "project": ("project",),
    "location": ("loc",),
    "model": ("model",),
    "k": ("k",),
    # Order matters inside the last stage (dependencies): recorder after micro; annotators/compare after colour;
    # merge threshold and clip count after merge; clusters after k and colour; labels after colour.
    "filters": ("view", "color", "annot", "cmp", "micro", "rec", "ch", "dates", "hours", "clusters", "hide",
                "labels", "merge", "mth", "mcc", "max", "prio", "sort"),
}
KEY_STAGE = {k: s for s, keys in STAGE_KEYS.items() for k in keys}
assert set(KEY_STAGE) == set(STATE_KEYS)

_CONTROL = re.compile(r"[\x00-\x1f\x7f�]")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_YEAR = re.compile(r"^\d{4}$")
_NUM = r"\d+(?:\.\d+)?"
_HOURS = re.compile(rf"^({_NUM})-({_NUM})$")
_COLNAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# ======================================================================================================================
# State
# ======================================================================================================================
@dataclass
class UrlState:
    """Everything a URL can describe. ``None`` = not specified; an empty list = explicit empty selection."""
    version: int = SCHEMA_VERSION
    project: str = None
    loc: list = None
    model: str = None
    k: int = None
    view: str = None
    color: str = None
    annot: list = None
    cmp: str = None
    micro: list = None
    rec: list = None
    ch: list = None
    dates: list = None          # list of [start_iso, end_iso] inclusive ranges, sorted and non-overlapping
    hours: tuple = None         # (lo, hi)
    clusters: list = None       # visible cluster ids
    hide: list = None           # hidden cluster ids (the shorter way to say "all but these")
    labels: str = None
    merge: bool = None
    mth: float = None
    mcc: int = None
    max_points: int = None
    prio: bool = None
    sort: tuple = None          # (column, ascending)

    def requested_keys(self):
        """URL keys that carry a value (in canonical order, excluding the version)."""
        return [k for k in STATE_KEYS if getattr(self, KEY_TO_FIELD[k]) is not None]

    def is_empty(self):
        return not self.requested_keys()

    def to_dict(self):
        """JSON-safe dict of the specified keys only (URL-key names), suitable for a dcc.Store."""
        out = {}
        for key in self.requested_keys():
            value = getattr(self, KEY_TO_FIELD[key])
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data):
        state = cls()
        for key, value in (data or {}).items():
            if key not in KEY_TO_FIELD or key == "v":
                continue
            if key in ("hours", "sort") and value is not None:
                value = tuple(value)
            setattr(state, KEY_TO_FIELD[key], value)
        return state


def _warn(code, key=None, detail=None):
    return {"code": code, "key": key, "detail": detail}


# ======================================================================================================================
# Syntax layer: query string -> UrlState
# ======================================================================================================================
def split_query(search):
    """
    Split a query string into ``[(raw_segment, key, raw_value)]`` (key percent-decoded, value left raw).
    A leading ``?`` and any ``#fragment`` are dropped; empty segments (``&&``) are skipped.
    A segment without ``=`` has an empty value.
    """
    s = search or ""
    if "#" in s:
        s = s.split("#", 1)[0]
    if s.startswith("?"):
        s = s[1:]
    out = []
    for seg in s.split("&"):
        if seg == "":
            continue
        raw_key, _, raw_value = seg.partition("=")
        out.append((seg, unquote_plus(raw_key), raw_value))
    return out


def _decode_item(raw):
    return unquote_plus(raw)


def _clean_text(text):
    return bool(text) and len(text) <= MAX_TEXT_LENGTH and not _CONTROL.search(text)


def _parse_list(raw_value, key, warnings, allow_none):
    """Raw value -> list of decoded items, ``[]`` for the explicit ``none``, or None when invalid."""
    parts = raw_value.split(",")
    items = [_decode_item(p) for p in parts]
    if any(i == "" for i in items):
        warnings.append(_warn("empty_item", key, raw_value))
        return None
    if len(items) > MAX_ITEMS:
        warnings.append(_warn("too_many_items", key, str(len(items))))
        return None
    lowered = [i.lower() for i in items]
    if NONE_TOKEN in lowered:
        if not allow_none:
            warnings.append(_warn("none_not_allowed", key, raw_value))
            return None
        if len(items) != 1:
            warnings.append(_warn("none_mixed_with_values", key, raw_value))
            return None
        return []
    return items


def _parse_text_list(raw_value, key, warnings, allow_none):
    items = _parse_list(raw_value, key, warnings, allow_none)
    if items is None or items == []:
        return items
    good = [i for i in items if _clean_text(i)]
    if len(good) != len(items):
        warnings.append(_warn("invalid_text", key, raw_value))
        return None
    seen, uniq = set(), []
    for i in good:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _parse_int(text, lo, hi):
    if not re.fullmatch(r"\d{1,12}", text or ""):       # bounded length: never feed a huge digit string to int()
        return None
    n = int(text)
    return n if lo <= n <= hi else None


def _parse_int_list(raw_value, key, warnings, allow_none, lo=0, hi=1_000_000):
    items = _parse_list(raw_value, key, warnings, allow_none)
    if items is None or items == []:
        return items
    nums = []
    for i in items:
        n = _parse_int(i, lo, hi)
        if n is None:
            warnings.append(_warn("invalid_number", key, i))
            return None
        nums.append(n)
    return sorted(set(nums))


def _parse_bool(text):
    t = (text or "").lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    return None


def _month_range(year, month):
    first = date(year, month, 1)
    last = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
    return first, last


def _parse_date_item(item):
    """One ``dates`` item -> (start, end) as ``date`` objects, or None. Accepts day, day..day, YYYY-MM, YYYY."""
    try:
        if ".." in item:
            a, _, b = item.partition("..")
            if not (_DATE.match(a) and _DATE.match(b)):
                return None
            start, end = date.fromisoformat(a), date.fromisoformat(b)
            return (start, end) if start <= end else None
        if _DATE.match(item):
            d = date.fromisoformat(item)
            return d, d
        if _MONTH.match(item):
            return _month_range(int(item[:4]), int(item[5:7]))
        if _YEAR.match(item):
            y = int(item)
            return date(y, 1, 1), date(y, 12, 31)
    except ValueError:
        return None
    return None


def merge_date_ranges(ranges):
    """Sort and merge overlapping or adjacent inclusive ``(start, end)`` date ranges."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _parse_dates(raw_value, key, warnings):
    items = _parse_list(raw_value, key, warnings, allow_none=True)
    if items is None or items == []:
        return items
    ranges = []
    for i in items:
        r = _parse_date_item(i)
        if r is None:
            warnings.append(_warn("invalid_date", key, i))
            continue
        ranges.append(r)
    if not ranges:
        return None
    return [[a.isoformat(), b.isoformat()] for a, b in merge_date_ranges(ranges)]


def _fmt_num(x):
    return str(int(x)) if float(x) == int(x) else repr(float(x))


def _parse_hours(text, key, warnings):
    m = _HOURS.match(text or "")
    if not m:
        warnings.append(_warn("invalid_hours", key, text))
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if not (0 <= lo <= hi <= 24):
        warnings.append(_warn("hours_out_of_range", key, text))
        return None
    return (lo, hi)


def _parse_sort(text, key, warnings):
    col, sep, direction = (text or "").partition(":")
    if not sep or direction.lower() not in ("asc", "desc") or not _COLNAME.match(col):
        warnings.append(_warn("invalid_sort", key, text))
        return None
    return (col, direction.lower() == "asc")


def parse(search, max_points_limit=None):
    """
    Syntax-validate the VibroVis parameters of a query string. Returns ``(UrlState, warnings)``.

    ``max_points_limit`` defaults to ``limits.MAX_POINTS_LIMIT`` (read at call time); pass the app's actual limit to
    override it.

    Never raises. Unrelated parameters are ignored here (``rewrite_search`` preserves them). An empty or
    parameter-free query gives an empty state and no warnings (= today's behaviour).
    """
    state, warnings = UrlState(), []
    if max_points_limit is None:
        max_points_limit = _limits.MAX_POINTS_LIMIT
    if len(search or "") > MAX_QUERY_LENGTH:
        warnings.append(_warn("query_too_long", None, str(len(search))))
        return state, warnings
    segments = split_query(search)

    owned, duplicated = {}, []
    for _seg, key, raw_value in segments:
        if key in OWNED_KEYS:
            if key in owned and key not in duplicated:
                duplicated.append(key)      # one warning per key, however often it repeats
            owned[key] = raw_value          # last wins
    warnings.extend(_warn("duplicate_param", key, "last value wins") for key in duplicated)

    # -- version -------------------------------------------------------------------------------------------------------
    if "v" in owned:
        v = _parse_int(owned["v"], 1, 10 ** 6)
        if v is None:
            warnings.append(_warn("invalid_version", "v", owned["v"]))     # treated as the current version
        elif v > SCHEMA_VERSION:
            warnings.append(_warn("unsupported_version", "v", f"{v} is newer than {SCHEMA_VERSION}; nothing restored"))
            state.version = v
            return state, warnings
        else:
            state.version = v

    for key in STATE_KEYS:
        if key not in owned:
            continue
        raw = owned[key]
        if raw == "":
            warnings.append(_warn("empty_value", key))
            continue
        field_name = KEY_TO_FIELD[key]
        value = None
        if key in ("project", "model"):
            text = _decode_item(raw)
            if _clean_text(text) and text.lower() != NONE_TOKEN:
                value = text
            else:
                warnings.append(_warn("invalid_text", key, raw))
        elif key in ("loc", "annot"):
            value = _parse_text_list(raw, key, warnings, allow_none=False)
        elif key in ("micro", "rec"):
            value = _parse_text_list(raw, key, warnings, allow_none=True)
        elif key in ("ch", "clusters"):
            value = _parse_int_list(raw, key, warnings, allow_none=True)
        elif key == "hide":
            value = _parse_int_list(raw, key, warnings, allow_none=False)
        elif key == "k":
            value = _parse_int(_decode_item(raw), 1, MAX_CLUSTER_COUNT)
            if value is None:
                warnings.append(_warn("invalid_number", key, raw))
        elif key == "mcc":
            value = _parse_int(_decode_item(raw), 1, 1_000_000)
            if value is None:
                warnings.append(_warn("invalid_number", key, raw))
        elif key == "mth":
            text = _decode_item(raw)
            if re.fullmatch(_NUM, text) and float(text) <= 1_000_000:
                value = float(text)
            else:
                warnings.append(_warn("invalid_number", key, raw))
        elif key == "max":
            n = _parse_int(_decode_item(raw), 1, 10 ** 9)
            if n is None:
                warnings.append(_warn("invalid_number", key, raw))
            elif n > max_points_limit:
                warnings.append(_warn("clamped", key, f"{n} -> {max_points_limit}"))
                value = max_points_limit
            else:
                value = n
        elif key in ("merge", "prio"):
            value = _parse_bool(_decode_item(raw))
            if value is None:
                warnings.append(_warn("invalid_boolean", key, raw))
        elif key == "view":
            value = _decode_item(raw).lower()
            if value not in VIEWS:
                warnings.append(_warn("unknown_value", key, raw))
                value = None
        elif key == "color":
            value = _decode_item(raw).lower()
            if value not in COLORS:
                warnings.append(_warn("unknown_value", key, raw))
                value = None
        elif key == "cmp":
            value = _decode_item(raw).lower()
            if value not in COMPARES:
                warnings.append(_warn("unknown_value", key, raw))
                value = None
        elif key == "labels":
            value = _decode_item(raw).lower()
            if value not in LABEL_FILTERS:
                warnings.append(_warn("unknown_value", key, raw))     # free-text label names are never accepted
                value = None
        elif key == "dates":
            value = _parse_dates(raw, key, warnings)
        elif key == "hours":
            value = _parse_hours(_decode_item(raw), key, warnings)
        elif key == "sort":
            value = _parse_sort(_decode_item(raw), key, warnings)
        if value is not None:
            setattr(state, field_name, value)

    if state.clusters is not None and state.hide is not None:
        warnings.append(_warn("conflicting_params", "hide", "'clusters' and 'hide' both given; 'clusters' wins"))
        state.hide = None
    return state, warnings


# ======================================================================================================================
# Initialisation (Phase 2.1): what the server answers when the page asks "is there a view to restore?"
# ======================================================================================================================
INIT_PHASES = ("pending", "running", "ready", "failed")      # public phases of the initialisation state machine


def has_owned_params(search):
    """True when the query string contains ANY VibroVis-owned key (even with an invalid or empty value)."""
    return any(key in OWNED_KEYS for _seg, key, _v in split_query(search))


def make_init_payload(search, max_points_limit=None, token=None):
    """
    The answer to the page's initialisation request. Phase 2.1 only PARSES the URL: nothing here (or anywhere in
    Phase 2.1) restores a selected value.

    ``status`` is ``'none'`` when there is nothing restorable (no query, only unrelated parameters, only invalid
    parameters, or an unsupported schema version) and ``'active'`` when at least one valid VibroVis value was parsed
    (it then carries a fresh ``token`` and the syntax-validated ``state``). ``hasOwnedParams`` records whether the URL
    contained any VibroVis-owned key at all, so a later notice can tell "a link that failed" from "no link".
    """
    state, warnings = parse(search, max_points_limit)
    payload = {"status": "none", "token": None, "state": {}, "warnings": warnings,
               "hasOwnedParams": has_owned_params(search), "schema": SCHEMA_VERSION}
    if not state.is_empty():
        payload.update(status="active", token=token or uuid.uuid4().hex, state=state.to_dict())
    return payload


def init_should_hold(public_state):
    """
    True while initialisation is unresolved (``pending`` or ``running``): consumers that would load data must wait.
    ``ready``, ``failed`` and anything unrecognised (including ``None``) do not hold, so a missing or broken state can
    never keep the app from loading data.
    """
    return isinstance(public_state, dict) and public_state.get("phase") in ("pending", "running")


# ======================================================================================================================
# Serialisation: UrlState -> query string, and merging into an existing query string
# ======================================================================================================================
def restore_should_hold(url_init, identity):
    """
    Should the data load keep holding after the initialisation is ``ready``?

    True while the URL carries a restorable view (``active`` payload) and the browser has not yet reported, for THAT
    payload's token, that the identity stages (project, location, model, k) are settled. ``identity`` is the
    ``url-identity`` Store content ``{"token", "done"}``. Anything else (no URL, nothing restorable, failed
    initialisation, a missing or foreign token's record) does not hold, with one exception: a foreign (older) token's
    record never releases a newer payload.
    """
    if not isinstance(url_init, dict) or url_init.get("phase") != "ready":
        return False
    payload = url_init.get("payload")
    if not isinstance(payload, dict) or payload.get("status") != "active" or not payload.get("token"):
        return False
    return not (isinstance(identity, dict) and identity.get("token") == payload["token"] and identity.get("done") is True)


def _enc(text):
    return quote(str(text), safe="")


def _join(items):
    return ",".join(_enc(i) for i in items)


def format_dates(ranges):
    """Canonical ``dates`` value: whole months as ``YYYY-MM``, whole years as ``YYYY``, else ``a..b`` or a day."""
    out = []
    for start_iso, end_iso in ranges:
        start, end = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
        if start == end:
            out.append(start_iso)
        elif start.day == 1 and (start, end) == _month_range(start.year, start.month):
            out.append(f"{start.year:04d}-{start.month:02d}")
        elif (start, end) == (date(start.year, 1, 1), date(start.year, 12, 31)):
            out.append(f"{start.year:04d}")
        else:
            out.append(f"{start_iso}..{end_iso}")
    return ",".join(out)


def _format_value(key, value):
    if key in ("loc", "annot", "micro", "rec"):
        return NONE_TOKEN if value == [] else _join(value)
    if key in ("ch", "clusters"):
        return NONE_TOKEN if value == [] else ",".join(str(int(x)) for x in sorted(set(value)))
    if key == "hide":
        return ",".join(str(int(x)) for x in sorted(set(value)))
    if key == "dates":
        return NONE_TOKEN if value == [] else format_dates(value)
    if key == "hours":
        return f"{_fmt_num(value[0])}-{_fmt_num(value[1])}"
    if key == "sort":
        return f"{value[0]}:{'asc' if value[1] else 'desc'}"
    if key in ("merge", "prio"):
        return "1" if value else "0"
    if key == "mth":
        return _fmt_num(value)
    if key in ("k", "mcc", "max"):
        return str(int(value))
    return _enc(value)          # project, model, view, color, cmp, labels


def serialize_pairs(state, mode="live"):
    """
    Canonical ``[(key, value)]`` for *state*.

    mode ``'live'``  minimal: omits values equal to the static defaults (``DEFAULTS``) and the version.
    mode ``'share'`` fully explicit snapshot: starts with ``v=1`` and pins the defaulted keys (view, color, merge, prio,
                     max, and cmp for Raven colouring) even when unset, so the link means the same thing if a default
                     ever changes.
    Both modes first apply ``consistent_state``: nothing is written that restoration would have to drop.
    """
    if mode not in ("live", "share"):
        raise ValueError("mode must be 'live' or 'share'")
    state = consistent_state(state)
    pairs = []
    if mode == "share":
        pairs.append(("v", str(SCHEMA_VERSION)))
    for key in STATE_KEYS:
        value = getattr(state, KEY_TO_FIELD[key])
        if value is None:
            if mode == "share" and key in DEFAULTS and (key != "cmp" or state.color == "raven"):
                value = DEFAULTS[key]
            else:
                continue
        if mode == "live" and key in DEFAULTS and value == DEFAULTS[key]:
            continue
        if key == "hide" and state.clusters is not None:
            continue                                    # 'clusters' wins; never write both
        pairs.append((key, _format_value(key, value)))
    return pairs


def serialize(state, mode="live"):
    """Query string (without the leading ``?``) for *state*; ``''`` when there is nothing to say."""
    return "&".join(f"{k}={v}" for k, v in serialize_pairs(state, mode))


def rewrite_search(existing_search, state, mode="live"):
    """
    New ``search`` string: every parameter VibroVis does not own is kept VERBATIM and in its original order;
    every owned parameter (all duplicates, however malformed) is replaced by the canonical rendering of *state*.
    Unrelated parameters come first, then the owned ones in canonical order. Returns ``''`` or ``'?...'``.
    """
    kept = [seg for seg, key, _v in split_query(existing_search) if key not in OWNED_KEYS]
    owned = [f"{k}={v}" for k, v in serialize_pairs(state, mode)]
    query = "&".join(kept + owned)
    return f"?{query}" if query else ""


def unrelated_params(search):
    """The raw ``key=value`` segments of *search* that VibroVis does not own."""
    return [seg for seg, key, _v in split_query(search) if key not in OWNED_KEYS]


def compress_cluster_selection(selected, universe):
    """
    ``(clusters, hide)`` for a visible-cluster selection: ``(None, None)`` when everything is visible (the default),
    otherwise whichever of ``clusters=<visible>`` / ``hide=<hidden>`` is shorter.
    """
    universe = sorted(set(int(c) for c in universe))
    selected = sorted(set(int(c) for c in selected) & set(universe))
    if selected == universe:
        return None, None
    hidden = [c for c in universe if c not in set(selected)]
    if not selected or len(selected) <= len(hidden):
        return selected, None
    return None, hidden


def compress_dates(selected_iso_dates):
    """Selected ISO dates -> merged inclusive ranges in the ``UrlState.dates`` form (``[]`` for none)."""
    ranges = [(date.fromisoformat(d), date.fromisoformat(d)) for d in selected_iso_dates]
    return [[a.isoformat(), b.isoformat()] for a, b in merge_date_ranges(ranges)]


def snap_hours(lo, hi, bounds, step=None):
    """
    Snap a requested hour range OUTWARD to the slider step (lower bound down, upper bound up, so no requested data is
    silently excluded), then clamp it to the dataset's slider bounds.

    Returns ``(lo, hi, reasons)`` where ``reasons`` is a subset of ``['snapped', 'clamped']``, or ``None`` when the
    requested range lies entirely outside the bounds (or nothing is left after clamping).
    """
    step = float(step if step is not None else _limits.HOUR_SLIDER_STEP)
    b_lo, b_hi = float(bounds[0]), float(bounds[1])
    eps = 1e-9
    s_lo = math.floor(lo / step + eps) * step
    s_hi = math.ceil(hi / step - eps) * step
    if s_hi < b_lo or s_lo > b_hi:
        return None
    c_lo, c_hi = max(s_lo, b_lo), min(s_hi, b_hi)
    if c_lo > c_hi:
        return None
    reasons = []
    if abs(s_lo - lo) > eps or abs(s_hi - hi) > eps:
        reasons.append("snapped")
    if abs(c_lo - s_lo) > eps or abs(c_hi - s_hi) > eps:
        reasons.append("clamped")
    return round(c_lo, 6), round(c_hi, 6), reasons


def labels_to_tick(labels, rows):
    """
    Map ``labels=unlabeled|labeled`` onto the manual-label legend rows that exist at restoration time.

    *rows*: the legend row ids in legend order (including ``'Unlabeled'``). Returns
    ``{'action': 'apply'|'noop'|'drop', 'ticked': [...], 'reason': str|None}``. Rows not in ``ticked`` are unticked.
    Never returns an ``apply`` with zero ticked rows.
    """
    rows = list(rows)
    others = [r for r in rows if r != "Unlabeled"]
    if labels == "unlabeled":
        if "Unlabeled" not in rows:
            return {"action": "drop", "ticked": [], "reason": "no_unlabeled_row"}
        if not others:
            return {"action": "noop", "ticked": ["Unlabeled"], "reason": None}      # already the only (ticked) row
        return {"action": "apply", "ticked": ["Unlabeled"], "reason": None}
    if labels == "labeled":
        if not others:
            return {"action": "drop", "ticked": [], "reason": "no_labeled_rows"}
        return {"action": "apply", "ticked": others, "reason": None}
    return {"action": "drop", "ticked": [], "reason": "unknown_value"}


def cluster_ticks(clusters, hide, cluster_ids):
    """
    Map ``clusters=`` / ``hide=`` onto the cluster legend rows (cluster colouring only).

    Returns ``{'action': 'apply'|'noop'|'drop', 'ticked': [...], 'reason': str|None}``. An empty selection (nothing
    visible) is dropped: the UI turns filtering off again on the next unrelated recompute.
    """
    ids = sorted(set(int(c) for c in cluster_ids))
    if clusters is not None:
        ticked = [c for c in ids if c in set(clusters)]
    elif hide is not None:
        ticked = [c for c in ids if c not in set(hide)]
    else:
        return {"action": "noop", "ticked": ids, "reason": None}
    if not ticked:
        return {"action": "drop", "ticked": [], "reason": "empty_cluster_selection"}
    if ticked == ids:
        return {"action": "noop", "ticked": ids, "reason": None}
    return {"action": "apply", "ticked": ticked, "reason": None}


def consistent_state(state):
    """
    A copy of *state* without any value that restoration would have to drop or that contradicts another value
    (see the module docstring). Used by every serialiser, so live URLs and share links are always restorable.
    """
    c = dataclasses.replace(state)
    color = c.color or "cluster"
    if c.merge is not True:
        c.mth = None
        c.mcc = None
    if color != "cluster" or c.clusters == []:
        c.clusters = None
        if color != "cluster":
            c.hide = None
    if c.clusters is not None:
        c.hide = None                                   # never both
    if color != "manual":
        c.labels = None
    if color != "raven":
        c.annot = None
        c.cmp = None
    return c


# ======================================================================================================================
# Application layer: validation against the real option lists, one restore stage at a time
# ======================================================================================================================
@dataclass
class Catalog:
    """Option lists the app offers right now. ``None`` = not available yet (the stage is not ready)."""
    projects: list = None
    locations: list = None
    models: list = None
    k_values: list = None
    microlocations: list = None
    recorders: list = None
    channels: list = None
    dates: list = None                 # ISO strings available in the loaded dataset
    hour_bounds: tuple = None          # (min, max) of the loaded dataset
    cluster_ids: list = None
    annotators: list = None
    columns: list = None               # sortable table columns
    raven_available: bool = None
    clip_count_max: int = None
    merge_max: float = None
    max_points_limit: int = None       # None -> limits.MAX_POINTS_LIMIT
    hour_step: float = None            # None -> limits.HOUR_SLIDER_STEP (the real slider step)


# Option lists a requested key is validated against. A stage is "ready" when every list needed by the keys the URL
# actually requested in that stage is available; a stage with nothing requested is trivially ready.
_KEY_REQUIRES = {
    "project": ("projects",), "loc": ("locations",), "model": ("models",), "k": ("k_values",),
    "color": ("raven_available",), "annot": ("annotators", "raven_available"),
    "cmp": ("annotators", "raven_available"), "micro": ("microlocations",), "rec": ("recorders",),
    "ch": ("channels",), "dates": ("dates",), "hours": ("hour_bounds",), "clusters": ("cluster_ids",),
    "hide": ("cluster_ids",), "sort": ("columns",),
}


def stage_ready(stage, state, catalog):
    """True when every option list needed by the keys *state* requests in *stage* is available."""
    for key in STAGE_KEYS[stage]:
        if getattr(state, KEY_TO_FIELD[key]) is None:
            continue
        if any(getattr(catalog, name) is None for name in _KEY_REQUIRES.get(key, ())):
            return False
    return True


def _drop(key, value, reason):
    return {"key": key, "value": value, "reason": reason}


def _adjust(key, value, applied, reason):
    """A value that WAS applied, but changed to fit (snapped to a step or clamped to a bound). Not a drop."""
    return {"key": key, "value": value, "applied": applied, "reason": reason}


def _intersect(requested, available):
    avail = set(available)
    kept = [r for r in requested if r in avail]
    return kept, [r for r in requested if r not in avail]


def resolve_dates(ranges, available_iso_dates):
    """Expand ``[[start, end], ...]`` against the dates that exist; returns the sorted ISO dates inside the ranges."""
    spans = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in ranges]
    out = []
    for d in sorted(available_iso_dates):
        dd = date.fromisoformat(d)
        if any(a <= dd <= b for a, b in spans):
            out.append(d)
    return out


def validate_stage(stage, state, catalog):
    """
    Check the keys of *stage* that *state* requests against *catalog*.

    Returns ``{'stage', 'ready', 'applied': {key: value}, 'dropped': [{key, value, reason}], 'adjusted': [...]}``
    (``adjusted`` lists values that were applied after being snapped/clamped; they are not drops). When the catalog is
    not ready for this stage, ``ready`` is False and nothing is applied or dropped (the stage stays pending).
    Every requested key of the stage ends up in exactly one of ``applied`` / ``dropped``.
    """
    result = {"stage": stage, "ready": stage_ready(stage, state, catalog), "applied": {}, "dropped": [], "adjusted": []}
    if not result["ready"]:
        return result
    ap, dr = result["applied"], result["dropped"]

    def requested(key):
        return getattr(state, KEY_TO_FIELD[key])

    if stage == "project":
        v = requested("project")
        if v is not None:
            if v in catalog.projects:
                ap["project"] = v
            else:
                dr.append(_drop("project", v, "unknown_project"))
    elif stage == "location":
        v = requested("loc")
        if v is not None:
            kept, gone = _intersect(v, catalog.locations)
            if kept:
                ap["loc"] = kept
            else:
                dr.append(_drop("loc", v, "unknown_location"))
            dr.extend(_drop("loc", g, "unknown_location") for g in gone if kept)
    elif stage == "model":
        v = requested("model")
        if v is not None:
            if v in catalog.models:
                ap["model"] = v
            else:
                dr.append(_drop("model", v, "unknown_model"))
    elif stage == "k":
        v = requested("k")
        if v is not None:
            if v in catalog.k_values:
                ap["k"] = v
            else:
                dr.append(_drop("k", v, "unknown_k"))
    elif stage == "filters":
        _validate_filters(state, catalog, ap, dr, result["adjusted"])
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return result


def _validate_filters(state, catalog, ap, dr, adj):
    def requested(key):
        return getattr(state, KEY_TO_FIELD[key])

    def list_filter(key, available, reason):
        v = requested(key)
        if v is None:
            return
        if v == []:
            ap[key] = []                                  # explicit empty selection stays explicit
            return
        kept, gone = _intersect(v, available)
        if kept:
            ap[key] = kept
            dr.extend(_drop(key, g, reason) for g in gone)
        else:
            dr.append(_drop(key, v, reason))              # nothing valid: keep the app default (everything)

    if requested("view") is not None:
        ap["view"] = requested("view")

    color = requested("color")
    if color is not None:
        if color == "raven" and not catalog.raven_available:
            dr.append(_drop("color", color, "raven_not_available"))
            color = None
        else:
            ap["color"] = color
    # The colour mode the browser will be in once colour is resolved: a URL that does not mention colour (or whose
    # colour was dropped) leaves the app on its default, cluster colouring.
    effective_color = ap.get("color") or "cluster"

    v = requested("annot")
    if v is not None:
        if not catalog.raven_available:
            dr.append(_drop("annot", v, "raven_not_available"))
        else:
            list_filter("annot", catalog.annotators, "unknown_annotator")
    cmp_v = requested("cmp")
    if cmp_v is not None:
        n_annot = len(ap["annot"]) if ap.get("annot") else len(catalog.annotators or [])
        if not catalog.raven_available:
            dr.append(_drop("cmp", cmp_v, "raven_not_available"))
        elif cmp_v != "all" and n_annot < 2:
            dr.append(_drop("cmp", cmp_v, "needs_two_annotators"))
        else:
            ap["cmp"] = cmp_v

    list_filter("micro", catalog.microlocations, "unknown_microlocation")
    list_filter("rec", catalog.recorders, "unknown_recorder")
    list_filter("ch", catalog.channels, "unknown_channel")

    v = requested("dates")
    if v is not None:
        if v == []:
            ap["dates"] = []
        else:
            days = resolve_dates(v, catalog.dates)
            if days:
                ap["dates"] = days
            else:
                dr.append(_drop("dates", v, "no_matching_dates"))

    v = requested("hours")
    if v is not None:
        snapped = snap_hours(v[0], v[1], catalog.hour_bounds, catalog.hour_step)
        if snapped is None:
            dr.append(_drop("hours", list(v), "outside_available_hours"))
        else:
            lo, hi, reasons = snapped
            ap["hours"] = [lo, hi]
            adj.extend(_adjust("hours", list(v), [lo, hi], r) for r in reasons)

    for key in ("clusters", "hide"):
        v = requested(key)
        if v is None:
            continue
        if effective_color != "cluster":
            dr.append(_drop(key, v, "requires_cluster_color"))     # the cluster rows only exist in cluster colouring
            continue
        if v == []:
            # "No cluster visible" is not stable in the UI (filtering switches itself off on the next unrelated
            # recompute), so it can not be restored faithfully.
            dr.append(_drop(key, v, "empty_cluster_selection"))
            continue
        kept, gone = _intersect(v, catalog.cluster_ids)
        if not kept:
            dr.append(_drop(key, v, "unknown_cluster"))
        elif key == "hide" and set(kept) >= set(catalog.cluster_ids):
            dr.append(_drop(key, v, "empty_cluster_selection"))    # hiding every cluster = nothing visible
        else:
            ap[key] = kept
            dr.extend(_drop(key, g, "unknown_cluster") for g in gone)

    v = requested("labels")
    if v is not None:
        if effective_color == "manual":
            ap["labels"] = v
        else:
            dr.append(_drop("labels", v, "requires_manual_color"))

    merge = requested("merge")
    if merge is not None:
        ap["merge"] = merge
    for key, cap, reason in (("mth", catalog.merge_max, "merge_threshold"), ("mcc", catalog.clip_count_max, "clip_count")):
        v = requested(key)
        if v is None:
            continue
        if not ap.get("merge") and not merge:
            dr.append(_drop(key, v, "requires_merge"))
        elif cap is not None and v > cap:
            ap[key] = cap
            adj.append(_adjust(key, v, cap, "clamped"))
        else:
            ap[key] = v

    v = requested("max")
    if v is not None:
        limit = catalog.max_points_limit if catalog.max_points_limit is not None else _limits.MAX_POINTS_LIMIT
        ap["max"] = min(v, limit)
        if v > limit:
            adj.append(_adjust("max", v, limit, "clamped"))
    if requested("prio") is not None:
        ap["prio"] = requested("prio")

    v = requested("sort")
    if v is not None:
        if v[0] in catalog.columns:
            ap["sort"] = list(v)
        else:
            dr.append(_drop("sort", list(v), "unknown_column"))


# ======================================================================================================================
# Restore bookkeeping (a plain-JSON "token" that lives in a dcc.Store)
# ======================================================================================================================
def start_restore(state, warnings=None, token=None):
    """
    New restore token for *state*: every requested key starts ``pending``. The WHOLE token stays alive until each
    key is ``applied`` or ``dropped``; applying one widget never consumes the rest.
    """
    keys = state.requested_keys()
    return {
        "v": SCHEMA_VERSION,
        "token": token or uuid.uuid4().hex,
        "state": state.to_dict(),
        "status": {k: "pending" for k in keys},
        "applied": {},
        "dropped": [],
        "adjusted": [],
        "warnings": list(warnings or []),
        "cancelled": None,
    }


def is_active(restore):
    return bool(restore) and restore.get("cancelled") is None and any(
        s == "pending" for s in restore.get("status", {}).values())


def is_complete(restore):
    return not restore or all(s != "pending" for s in restore.get("status", {}).values())


def pending_keys(restore, stage=None):
    if not restore or restore.get("cancelled") is not None:
        return []
    keys = [k for k, s in restore.get("status", {}).items() if s == "pending"]
    return [k for k in keys if stage is None or KEY_STAGE[k] == stage]


def current_stage(restore):
    """The earliest stage that still has pending keys, or None. Later stages must wait for it."""
    for stage in STAGES:
        if pending_keys(restore, stage):
            return stage
    return None


def resolve_stage(restore, result):
    """
    Fold a ``validate_stage`` result into *restore* (returns a NEW token). A stage that is not ``ready`` changes
    nothing. Only keys of that stage are touched; keys of other stages stay pending.
    """
    if not result.get("ready"):
        return restore
    new = {**restore, "status": dict(restore["status"]), "applied": dict(restore["applied"]),
           "dropped": list(restore["dropped"]), "adjusted": list(restore.get("adjusted", []))}
    stage = result["stage"]
    for key in STAGE_KEYS[stage]:
        if new["status"].get(key) != "pending":
            continue
        if key in result["applied"]:
            new["status"][key] = "applied"
        elif any(d["key"] == key for d in result["dropped"]):
            new["status"][key] = "dropped"
        else:
            # requested but not reported: resolve it as dropped rather than leave the token pending forever
            new["status"][key] = "dropped"
            new["dropped"].append(_drop(key, new["state"].get(key), "not_reported"))
    new["applied"].update(result["applied"])
    new["dropped"].extend(result["dropped"])
    new["adjusted"].extend(result.get("adjusted", []))
    return new


def cancel_restore(restore, reason):
    """Abandon whatever is still pending (e.g. the user loaded a preset). Returns a NEW token."""
    if not restore:
        return restore
    return {**restore, "cancelled": reason}


def owns(restore, key):
    """True while the restore still has an unresolved value for *key*."""
    return key in pending_keys(restore)


def reset_decision(restore, key, preset_active):
    """
    What a reset callback must do for control *key*: ``'restore'`` (leave/apply the URL value), ``'keep'`` (a
    preset was just loaded; today's behaviour) or ``'default'`` (today's reset). See the module docstring.
    """
    if owns(restore, key):
        return "restore"
    return "keep" if preset_active else "default"


def notice_text(restore):
    """Short, non-blocking message for what could not be applied, or ''."""
    drops = list((restore or {}).get("dropped", []))             # adjustments (snapped/clamped) are not reported
    warns = [w for w in (restore or {}).get("warnings", []) if w.get("code") != "clamped"]
    n = len(drops) + len(warns)
    if not n:
        return ""
    names = sorted({d["key"] for d in drops if d.get("key")} | {w["key"] for w in warns if w.get("key")})
    plural = "setting" if n == 1 else "settings"
    return f"{n} {plural} in this link could not be applied" + (f" ({', '.join(names)})." if names else ".")
