"""
Deep link from a VibroVis clip to the same recording in the Citimon Explorer.

Pure (standard library only). The Explorer identifies a recording by a numeric database id that VibroVis does not
know, but it also stores each recording's ``path`` (``<folder>/<stem>.WAV``, e.g. ``2026_03-C20/20260228_211000.WAV``).
A VibroVis clip knows the same thing: ``mp3_file`` is ``<folder>/<stem>_ch<N>.mp3`` and ``clip_time`` is the clip's
start inside the recording file, which is the Explorer's time axis too.  So the link carries ``path`` (without
extension) and ``t`` in a descriptor URL ``<base>/go?dataset=citimon&path=<folder>/<stem>&t=<seconds>``.  The
Explorer has no ``/go`` route and none is needed: ``assets/explorer_open.js`` intercepts the click, finds the numeric id
through the Explorer's existing API with the user's own Explorer login, and opens ``#/recording/<id>?t=<seconds>``.

Configuration (environment, read on every call):
  VIBROVIS_EXPLORER_URL   base URL of the Explorer. Unset = ``/explorer`` (the Explorer next to VibroVis on the ranavis VM),
                          so the link is on by default; set it to another URL to move it, or to an empty string to switch
                          the link off (a local run without an Explorer then shows no dead link).
Only Citimon recordings are linked (the Explorer dataset ``citimon``); a folder that does not look like a Citimon
deployment (``<year>_<month>-<recorder>``, the Explorer's own rule) yields no link.
"""
import os
import re
from urllib.parse import quote

EXPLORER_URL_ENV = "VIBROVIS_EXPLORER_URL"
DEFAULT_BASE = "/explorer"
DATASET = "citimon"
# Same rule as explorer/datasets/citimon.py FOLDER (year_month-recorder, optional -N suffix of renamed folders).
_FOLDER = re.compile(r"^\d{4}[-_]\d{2}-[A-Za-z]\d+(?:-\d)?$")
_CHANNEL_MP3 = re.compile(r"_ch\d+\.mp3$", re.IGNORECASE)


def explorer_base(environ=None):
    """Explorer base URL without a trailing slash (default ``/explorer``), or ``""`` when switched off (set empty)."""
    value = (environ if environ is not None else os.environ).get(EXPLORER_URL_ENV, DEFAULT_BASE)
    return str(value).strip().rstrip("/")


def recording_path(mp3_file):
    """``<folder>/<stem>`` of a Citimon clip's ``mp3_file`` (no channel suffix, no extension), else ``None``."""
    if not isinstance(mp3_file, str) or not mp3_file:
        return None
    parts = [p for p in mp3_file.replace("\\", "/").split("/") if p]
    if len(parts) < 2 or not _FOLDER.match(parts[-2]):
        return None
    stem = _CHANNEL_MP3.sub("", parts[-1])
    if not stem or stem == parts[-1]:          # not a ``..._chN.mp3`` name
        return None
    return f"{parts[-2]}/{stem}"


def explorer_url(mp3_file, clip_time, environ=None):
    """Explorer link for the clip, or ``None`` (feature off, not a Citimon clip, or no usable start time)."""
    base = explorer_base(environ)
    path = recording_path(mp3_file)
    if not base or path is None:
        return None
    try:
        t = float(clip_time)
    except (TypeError, ValueError):
        return None
    if t != t or t < 0 or t in (float("inf"),):
        return None
    return f"{base}/go?dataset={DATASET}&path={quote(path, safe='/')}&t={t:.2f}"
