"""
Aggregate imported Raven annotations onto plotted VibroVis points (read-only; pure pandas, no Dash).

Mapping (see viewer.map_links_to_points): a selection lands only on the clips stored as its links for the active
model - the plotted single clip with the same recording, channel and start, or (merge mode) the one merged point
whose span contains the clip. Links matching several merged points are "ambiguous" and land nowhere.

Per annotation, a comparison key is derived from the interpretation; the original label text is kept for display:
  recognised species  -> the species name            ("Bufo bufo?" -> "Bufo bufo", uncertain)
  lone "?"            -> "Unidentified (?)"          (uncertain)
  other text          -> "Other: <text>"             (trailing "?" stripped, case-folded; "?" -> uncertain)
  empty label         -> "(no label)"                (shown, but not counted as labelling the clip)

Per clip, among the selected annotators (each with the set of keys from all of their selections on that clip):
  none     - no selected annotator labelled it        (missing annotations are never disagreements)
  single   - exactly one annotator labelled it        (one label alone is never agreement)
  agree    - >= 2 annotators with identical key sets
  disagree - >= 2 annotators with different key sets
A point (merged points contain several clips) is disagree if any clip disagrees, else agree if any clip agrees,
else single if any clip is annotated, else none.

Point category (used for colour): "Annotators disagree" for disagree; otherwise the single key present, or
"Multiple labels" when several keys apply; points without annotations are "No Raven annotation".
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from . import viewer

NO_ANNOTATION = "No Raven annotation"
DISAGREE = "Annotators disagree"
MULTIPLE = "Multiple labels"
UNIDENTIFIED = "Unidentified (?)"
NO_LABEL = "(no label)"
OTHER_PREFIX = "Other: "

STATUS_TEXT = {"none": "", "single": "single annotator", "agree": "agree", "disagree": "disagree"}
POINT_COLUMNS = ["raven_category", "raven_status", "raven_labels", "raven_annotators", "raven_uncertain",
                 "raven_n_annotations", "raven_clips_annotated"]


def label_key(label: Optional[str], species: Optional[str], rule: Optional[str]) -> str:
    if isinstance(species, str) and species:
        return species
    text = (label if isinstance(label, str) else "").strip()
    core = text.rstrip("?").strip()
    if not text or rule == "empty":
        return NO_LABEL
    if not core or rule == "uncertain_only":
        return UNIDENTIFIED
    return OTHER_PREFIX + core.casefold()


# Minimum overlap for a link to colour a clip: the selection must cover at least MIN_OVERLAP_S of the clip,
# or at least MIN_FRAC_OF_SELECTION of the selection must lie inside the clip (so short calls still count).
# Links below this stay in the database and are listed in the point details, marked as not used for colouring.
MIN_OVERLAP_S = 1.0
MIN_FRAC_OF_SELECTION = 0.5


def counts_for_colouring(ann: pd.DataFrame) -> pd.Series:
    """True for links whose overlap is large enough to colour the clip (see MIN_OVERLAP_S)."""
    if ann.empty:
        return pd.Series(dtype=bool, index=ann.index)
    overlap = pd.to_numeric(ann["overlap_s"], errors="coerce").fillna(0.0)
    frac_sel = pd.to_numeric(ann["frac_of_selection"], errors="coerce").fillna(0.0)
    return (overlap >= MIN_OVERLAP_S - 1e-9) | (frac_sel >= MIN_FRAC_OF_SELECTION - 1e-9)


def clip_annotations(selections: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """One row per (selection, linked clip) with the selection's annotator, label, key and original interval."""
    keep = ["selection_pk", "annotator", "label", "species", "uncertain", "interpretation_rule", "quick_pass",
            "t0", "t1", "t1_original", "source_stem", "table_file", "raven_selection"]
    if links.empty or selections.empty:
        return pd.DataFrame(columns=list(links.columns) + keep[1:] + ["key"])
    ann = links.merge(selections[keep], on="selection_pk", how="inner")
    ann["key"] = [label_key(l, s, r) for l, s, r in zip(ann["label"], ann["species"], ann["interpretation_rule"])]
    ann["uncertain"] = ann["uncertain"].fillna(False).astype(bool)
    return ann.reset_index(drop=True)


def map_to_frame(ann: pd.DataFrame, dff: pd.DataFrame, id_col: str = "row_idx") -> pd.DataFrame:
    """Map clip annotations onto rows of a (possibly large, possibly merged) plotted frame.

    Only rows whose recording has links are examined. Returns ``ann`` restricted to links that land on exactly
    one point, with that point's ``id_col`` value in ``point_id``. Ambiguous/not-plotted links are dropped
    (``mapping_counts`` reports them).
    """
    cols = list(ann.columns) + ["point_id", "match"]
    if ann.empty or dff is None or dff.empty or "mp3_file" not in dff.columns:
        return pd.DataFrame(columns=cols)
    audio = set(ann["audio_path"].dropna())
    variants = audio | {a.replace("/", "\\") for a in audio}
    cand = dff[dff["mp3_file"].astype(str).isin(variants)]
    if cand.empty:
        return pd.DataFrame(columns=cols)
    cand = cand.assign(plot_id=cand[id_col].values if id_col in cand.columns else cand.index)
    mapped = viewer.map_links_to_points(ann, cand)
    mapped = mapped.rename(columns={"plot_id": "point_id"})
    return mapped


def mapping_counts(mapped: pd.DataFrame) -> dict:
    c = mapped["match"].value_counts() if not mapped.empty else pd.Series(dtype=int)
    return {k: int(c.get(k, 0)) for k in ("exact", "merged", "ambiguous", "not_in_view")}


def _clip_status(per_annotator: dict) -> str:
    """per_annotator: annotator -> set of comparable keys (empty-label annotations excluded)."""
    labelled = {a: keys for a, keys in per_annotator.items() if keys}
    if not labelled:
        return "none"
    if len(labelled) == 1:
        return "single"
    sets = list(labelled.values())
    return "agree" if all(s == sets[0] for s in sets[1:]) else "disagree"


def _roll_up(statuses: Iterable[str]) -> str:
    statuses = set(statuses)
    for s in ("disagree", "agree", "single"):
        if s in statuses:
            return s
    return "none"


def _display(row) -> str:
    return f"{row['annotator']}: {row['label'] if isinstance(row['label'], str) and row['label'] else NO_LABEL}"


def aggregate_points(mapped: pd.DataFrame, annotators: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-point Raven summary (index = point_id) for the selected annotators (None/empty -> all)."""
    landed = mapped[mapped["match"].isin(["exact", "merged"])] if not mapped.empty else mapped
    if annotators:
        landed = landed[landed["annotator"].isin(set(annotators))]
    if landed is None or landed.empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    out = {}
    for pid, g in landed.groupby("point_id", sort=False):
        statuses = []
        for _, clip in g.groupby(["audio_path", "channel", "clip_time_s"], sort=False):
            per = {}
            for a, ag in clip.groupby("annotator"):
                per[a] = {k for k in ag["key"] if k != NO_LABEL}
            statuses.append(_clip_status(per))
        status = _roll_up(statuses)
        keys = sorted(set(g["key"]))
        if status == "disagree":
            category = DISAGREE
        elif len(keys) == 1:
            category = keys[0]
        else:
            category = MULTIPLE
        per_sel = g.drop_duplicates("selection_pk")
        out[pid] = {
            "raven_category": category,
            "raven_status": status,
            "raven_labels": "; ".join(sorted({_display(r) for _, r in per_sel.iterrows()})),
            "raven_annotators": ", ".join(sorted(per_sel["annotator"].unique())),
            "raven_uncertain": bool(per_sel["uncertain"].any()),
            "raven_n_annotations": int(len(per_sel)),
            "raven_clips_annotated": int(g.groupby(["audio_path", "channel", "clip_time_s"]).ngroups),
        }
    return pd.DataFrame.from_dict(out, orient="index", columns=POINT_COLUMNS)


def attach(dff: pd.DataFrame, agg: pd.DataFrame, id_col: str = "row_idx") -> pd.DataFrame:
    """Copy of dff with the Raven point columns; unannotated rows get NO_ANNOTATION / 'none'."""
    out = dff.copy()
    ids = out[id_col] if id_col in out.columns else pd.Series(out.index, index=out.index)
    for col in POINT_COLUMNS:
        mapped = ids.map(agg[col]) if not agg.empty else pd.Series(index=out.index, dtype=object)
        out[col] = mapped.values
    out["raven_category"] = out["raven_category"].fillna(NO_ANNOTATION)
    out["raven_status"] = out["raven_status"].fillna("none")
    out["raven_labels"] = out["raven_labels"].fillna("")
    out["raven_annotators"] = out["raven_annotators"].fillna("")
    out["raven_uncertain"] = out["raven_uncertain"].fillna(False).astype(bool)
    out["raven_n_annotations"] = out["raven_n_annotations"].fillna(0).astype(int)
    out["raven_clips_annotated"] = out["raven_clips_annotated"].fillna(0).astype(int)
    return out


def compare_filter(dff: pd.DataFrame, compare: Optional[str]) -> pd.DataFrame:
    """'agree' / 'disagree' keep matching annotated points; unannotated points stay as context."""
    if compare not in ("agree", "disagree") or "raven_status" not in dff.columns:
        return dff
    return dff[(dff["raven_status"] == compare) | (dff["raven_status"] == "none")]


def category_order(categories: Iterable[str]) -> list:
    """Species first, then other labels, then the special categories; NO_ANNOTATION last."""
    def rank(c):
        if c == NO_ANNOTATION:
            return (6, c)
        if c == DISAGREE:
            return (5, c)
        if c == MULTIPLE:
            return (4, c)
        if c in (UNIDENTIFIED, NO_LABEL):
            return (3, c)
        if c.startswith(OTHER_PREFIX):
            return (2, c)
        return (1, c)
    return sorted(set(categories), key=rank)


def point_annotations(plot_id, dff: pd.DataFrame, ann: pd.DataFrame,
                      annotators: Optional[Sequence[str]] = None, id_col: str = "plot_id"):
    """(annotations landing on one plotted point, number from annotators not selected)."""
    if dff is None or plot_id not in dff.index or ann.empty:
        return ann.iloc[0:0], 0
    audio = str(dff.at[plot_id, "mp3_file"]).replace("\\", "/")
    same = dff[dff["mp3_file"].astype(str).str.replace("\\", "/", regex=False) == audio]
    mapped = map_to_frame(ann[ann["audio_path"] == audio], same, id_col=id_col)
    if mapped.empty:
        return mapped, 0
    hits = mapped[(mapped["point_id"].astype(str) == str(dff.at[plot_id, id_col]))
                  & mapped["match"].isin(["exact", "merged"])]
    if annotators:
        hidden = int(hits.loc[~hits["annotator"].isin(set(annotators)), "selection_pk"].nunique())
        hits = hits[hits["annotator"].isin(set(annotators))]
    else:
        hidden = 0
    return hits.sort_values(["clip_time_s", "annotator", "t0"]).reset_index(drop=True), hidden
