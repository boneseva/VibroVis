"""
Server-side planners for URL restoration (Phase 2.2).

PURE: no Dash and no I/O. Given what a browser asked for (``UrlState``) and what the browser's OWN loaded dataset
offers, they return a JSON plan that says which values may be applied and which are dropped (with a reason). They never
write a widget; the browser-side reconciler (``assets/url_reconcile.js``) applies the plan.

Identity (project -> location -> model -> k) is validated in the browser against the option lists the app actually
offers at that moment (see ``IDENTITY_REASONS``), because those lists only exist after the previous stage has taken
effect. Everything else (the ``filters`` stage) is planned here from the loaded dataset, once the browser says which
dataset it holds.
"""
import url_state as u
import limits

IDENTITY_KEYS = ("project", "loc", "model", "k")
# Same reason codes url_state.validate_stage uses for the identity stages; the browser reports them as dropped.
IDENTITY_REASONS = {"project": "unknown_project", "loc": "unknown_location", "model": "unknown_model", "k": "unknown_k"}
# Order in which filter values are written (dependencies: recorder after microlocation, clusters after colour, ...).
FILTER_ORDER = u.STAGE_KEYS["filters"]


def _micro_values(metadata):
    return [m for m in (metadata or {}).get("microlocations", []) if str(m).strip() and str(m).strip().lower() != "default"]


def narrowed_recorders(dff, metadata, micros):
    """Recorder options the app offers for a microlocation selection (``None`` = no microlocation filter)."""
    all_micros = _micro_values(metadata)
    if micros is None or set(micros) == set(all_micros):
        return list((metadata or {}).get("recorders", []))
    if not micros or dff is None or dff.empty:
        return []
    mask = dff["microlocation"].isin(micros)
    return sorted(dff.loc[mask, "recorder_type"].dropna().unique().tolist()) if mask.any() else []


def cluster_ids_for(dff, k):
    if dff is None or dff.empty or k is None:
        return []
    if "cluster_num" in dff.columns:
        dff = dff[dff["cluster_num"] == k]
    return sorted(int(c) for c in dff["cluster_id"].dropna().unique())


def build_filters_catalog(dff, metadata, state, *, k, raven_available, annotators, columns, merge_max=None,
                          clip_max=None):
    """``url_state.Catalog`` for the filters stage of the dataset described by *metadata* / *dff*."""
    metadata = metadata or {}
    micros = _micro_values(metadata)
    requested_micro = state.micro
    micro_eff = None if requested_micro is None else [m for m in requested_micro if m in micros]
    if requested_micro == []:
        micro_eff = []
    hour_bounds = (metadata.get("hour_min", 0), metadata.get("hour_max", 24))
    return u.Catalog(
        microlocations=micros,
        recorders=narrowed_recorders(dff, metadata, micro_eff),
        channels=list(metadata.get("channels", [])),
        dates=list(metadata.get("dates", [])),
        hour_bounds=hour_bounds,
        cluster_ids=cluster_ids_for(dff, k),
        annotators=list(annotators or []),
        columns=list(columns or []),
        raven_available=bool(raven_available),
        clip_count_max=clip_max,
        merge_max=merge_max,
        max_points_limit=limits.MAX_POINTS_LIMIT,
        hour_step=limits.HOUR_SLIDER_STEP,
    )


def _number(value, integer=False):
    """*value* as a finite number, else None (the request comes from the browser: never trust its types)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
        return None
    return int(value) if integer else value


def build_filters_plan(init_payload, request, *, dff, metadata, loaded_key, raven_available, annotators, columns):
    """
    Plan for the filters stage.

    *init_payload*: the ``url-init`` payload (token + parsed state). *request*: what the browser sent
    (``token``, ``key``, ``k``, optional ``merge_max`` / ``clip_max``). *loaded_key*: the dataset key of the browser's own
    ``derived-model-metadata`` (already resolved to a frame by the caller as *dff*).

    Returns ``{"status": "ok"|"stale"|"dataset_mismatch"|"none", "token", "key", "applied", "dropped", "adjusted",
    "cluster_ids"}``. Only ``"ok"`` carries values.
    """
    base = {"token": (request or {}).get("token"), "key": (request or {}).get("key"), "applied": {}, "dropped": [],
            "adjusted": [], "cluster_ids": []}
    if not isinstance(init_payload, dict) or init_payload.get("status") != "active" \
            or init_payload.get("token") != base["token"]:
        return {**base, "status": "stale"}
    if loaded_key is None or [loaded_key[0], loaded_key[1], list(loaded_key[2])] != list(base["key"] or []) or dff is None:
        return {**base, "status": "dataset_mismatch"}
    state = u.UrlState.from_dict(init_payload.get("state"))
    catalog = build_filters_catalog(dff, metadata, state, k=_number(request.get("k"), integer=True),
                                    raven_available=raven_available, annotators=annotators, columns=columns,
                                    merge_max=_number(request.get("merge_max")),
                                    # The clip-count slider only reaches its real maximum once merging is ON, so a cap
                                    # observed before that (always 1) would wrongly clamp every request. The browser clamps
                                    # `mcc` against the live slider at apply time instead (assets/url_reconcile.js).
                                    clip_max=None)
    result = u.validate_stage("filters", state, catalog)
    if not result["ready"]:
        # The catalog is complete by construction; a not-ready stage means a malformed request. Fail closed.
        return {**base, "status": "dataset_mismatch"}
    return {**base, "status": "ok", "applied": result["applied"], "dropped": result["dropped"],
            "adjusted": result["adjusted"], "cluster_ids": catalog.cluster_ids}


def table_columns(all_columns, raven_columns, raven_available):
    """Sortable table columns the Table View offers (Raven columns only with Raven data)."""
    return [c for c in all_columns if raven_available or c not in raven_columns]


# ======================================================================================================================
# Live URL / share link: widget snapshot -> UrlState -> the EXISTING serialiser (url_state.rewrite_search)
# ======================================================================================================================
def _as_list(value):
    return [] if value is None else (list(value) if isinstance(value, (list, tuple)) else [value])


def _same_set(a, b):
    return {str(x) for x in a} == {str(x) for x in b}


def state_from_snapshot(snap, *, meta, dff, explicit=False):
    """
    ``UrlState`` describing the view in *snap* (raw widget values collected by the browser, see
    ``assets/url_reconcile.js`` ``collectSnapshot``), relative to the dataset the browser holds (*meta*, *dff*).

    A value equal to the app default (everything selected, full hour range, ...) is left unspecified so live URLs stay
    short; with ``explicit=True`` (share links) the lists that define the view are always written. Transient state (hover,
    playback, loading, free text) is not part of the snapshot at all.
    """
    meta = meta or {}
    st = u.UrlState()
    st.project = snap.get("project")
    st.loc = sorted(str(x) for x in _as_list(snap.get("loc"))) or None
    st.model = snap.get("model")
    k = snap.get("k")
    st.k = int(k) if isinstance(k, (int, float)) and not isinstance(k, bool) else None
    st.view = "table" if snap.get("view") == "table-tab" else "scatter"
    color = snap.get("color") if snap.get("color") in u.COLORS else "cluster"
    st.color = color

    if color == "raven":
        annot = [str(a) for a in _as_list(snap.get("annot"))]
        st.annot = annot or None
        st.cmp = snap.get("cmp") if snap.get("cmp") in u.COMPARES else None

    micros = [m for m in meta.get("microlocations", []) if str(m).strip() and str(m).strip().lower() != "default"]
    micro_sel = [str(m) for m in _as_list(snap.get("micro")) if str(m) in set(micros)]   # only what the dataset offers
    if micro_sel == [] and micros:
        st.micro = []
    elif explicit or not _same_set(micro_sel, micros):
        st.micro = sorted(micro_sel) or None

    rec_avail = narrowed_recorders(dff, meta, micro_sel)
    rec_sel = [str(r) for r in _as_list(snap.get("rec")) if str(r) in set(map(str, rec_avail))]
    if (explicit or not _same_set(rec_sel, rec_avail)) and (rec_sel or rec_avail):
        st.rec = sorted(rec_sel)

    ch_avail = list(meta.get("channels", []))
    ch_sel = [c for c in _as_list(snap.get("ch")) if isinstance(c, (int, float)) and c in set(ch_avail)]
    if explicit or not _same_set(ch_sel, ch_avail):
        st.ch = sorted(int(c) for c in ch_sel)

    dates_avail = list(meta.get("dates", []))
    dates_sel = sorted({str(d) for d in _as_list(snap.get("dates")) if str(d) in set(dates_avail)})
    if dates_sel != sorted(dates_avail) or explicit:
        st.dates = u.compress_dates(dates_sel)

    hours = snap.get("hours")
    bounds = (meta.get("hour_min", 0), meta.get("hour_max", 24))
    if isinstance(hours, (list, tuple)) and len(hours) == 2 and all(isinstance(h, (int, float)) for h in hours):
        if explicit or (abs(hours[0] - bounds[0]) > 1e-9 or abs(hours[1] - bounds[1]) > 1e-9):
            st.hours = (float(hours[0]), float(hours[1]))

    cl = snap.get("clusters") or {}
    if color == "cluster" and cl.get("ids"):
        ids = [int(c) for c in cl["ids"]]
        ticked = [int(c) for c in cl.get("ticked", [])]
        if explicit and ticked:
            st.clusters = sorted(set(ticked) & set(ids))
        else:
            st.clusters, st.hide = u.compress_cluster_selection(ticked, ids)

    lb = snap.get("labels") or {}
    if color == "manual" and lb.get("rows"):
        rows = [str(r) for r in lb["rows"]]
        ticked = [str(r) for r in lb.get("ticked", [])]
        others = [r for r in rows if r != "Unlabeled"]
        if ticked == ["Unlabeled"] and others:
            st.labels = "unlabeled"
        elif others and _same_set(ticked, others):
            st.labels = "labeled"

    st.merge = bool(snap.get("merge"))
    if st.merge:
        mth, mcc = snap.get("mth"), snap.get("mcc")
        st.mth = float(mth) if isinstance(mth, (int, float)) and not isinstance(mth, bool) else None
        st.mcc = int(mcc) if isinstance(mcc, (int, float)) and not isinstance(mcc, bool) else None
    mx = snap.get("max")
    st.max_points = int(mx) if isinstance(mx, (int, float)) and not isinstance(mx, bool) and mx >= 1 else None
    st.prio = "on" in _as_list(snap.get("prio"))
    sort = snap.get("sort") or {}
    if isinstance(sort, dict) and sort.get("col"):
        st.sort = (str(sort["col"]), bool(sort.get("asc", True)))
    return u.consistent_state(st)


def search_for_snapshot(snap, *, meta, dff, mode="live"):
    """New ``location.search`` for *snap*: unrelated parameters kept verbatim, owned ones from the existing serialiser."""
    state = state_from_snapshot(snap, meta=meta, dff=dff, explicit=(mode == "share"))
    return u.rewrite_search(snap.get("search") or "", state, mode)
