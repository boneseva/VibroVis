"""
Raven annotations as the "Imported labels (Raven)" colour mode: aggregation rules, scatter colouring, legend,
Table View, point details, histogram, and separation from manual labels / cluster colouring.

All annotation data is SYNTHETIC: rule tests use hand-built frames, the integration tests import synthetic Raven
tables into a temporary database with the real import pipeline. The real annotation DB and the real manual-label
store are never opened (the manual-label diskcache is redirected to a temporary directory before import).
"""
import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from dash import Dash, Patch
import dash._callback_context as dash_ctx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
for p in (ROOT, TESTS, os.path.join(TESTS, "fixtures", "raven_synthetic")):
    if p not in sys.path:
        sys.path.insert(0, p)

_DISKCACHE_TMP = tempfile.mkdtemp(prefix="raven_labels_diskcache_")
os.environ.setdefault("VIBROVIS_MANUAL_LABELS_DIR", os.path.join(_DISKCACHE_TMP, "manual_labels"))
os.environ.setdefault("VIBROVIS_CLIP_DURATIONS_DIR", os.path.join(_DISKCACHE_TMP, "clip_durations"))

from generate_fixtures import LOC_A  # noqa: E402
from test_raven_citimon_delivery import frame_with_locations, row, table  # noqa: E402
import callbacks.callbacks_raven_labels as rl  # noqa: E402
from callbacks.callbacks_clusters import register_cluster_callbacks  # noqa: E402
from callbacks.callbacks_constants import MANUAL_LABELS_CACHE, MANUAL_LABELS_LOCK, server_cache  # noqa: E402
from tests.dataset_fixtures import install_dataset, clear_datasets  # noqa: E402
from callbacks.callbacks_plots import register_plot_callbacks  # noqa: E402
from callbacks.callbacks_table import register_table_callbacks  # noqa: E402
from raven_import import aggregate as agg  # noqa: E402
from raven_import.clip_index import DataFrameClipSource  # noqa: E402
from raven_import.config import ImportOptions  # noqa: E402
from raven_import.service import commit_import, preview_import  # noqa: E402
from raven_import.store import AnnotationStore  # noqa: E402

REC = "20260320_185000"
AUDIO = f"Citimon/{LOC_A}/{REC}_ch1.mp3"
REAL_MANUAL_DIR = os.path.join(ROOT, "cache", "manual_labels")


@contextlib.contextmanager
def triggered(prop_id, value=1):
    token = dash_ctx.context_value.set(types.SimpleNamespace(triggered_inputs=[{'prop_id': prop_id, 'value': value}]))
    try:
        yield
    finally:
        dash_ctx.context_value.reset(token)


# ---------------------------------------------------------------------------
# Rule tests on hand-built frames
# ---------------------------------------------------------------------------
def ann_frame(rows):
    """rows: (selection_pk, annotator, label, species, rule, uncertain, clip_time)."""
    out = []
    for pk, who, label, species, rule, unc, t in rows:
        out.append({"selection_pk": pk, "annotator": who, "label": label, "species": species,
                    "interpretation_rule": rule, "uncertain": unc, "audio_path": AUDIO, "channel": 1,
                    "clip_time_s": t, "clip_duration_s": 5.0, "overlap_s": 1.0, "frac_of_clip": 0.2,
                    "frac_of_selection": 1.0, "relation": "inside", "quick_pass": False, "t0": t + 1, "t1": t + 2,
                    "t1_original": None, "source_stem": REC, "table_file": f"{who}.txt", "raven_selection": str(pk),
                    "recording_id": f"Citimon::{AUDIO}::ch1", "model_name": "perchv2"})
    df = pd.DataFrame(out)
    df["key"] = [agg.label_key(l, s, r) for l, s, r in zip(df["label"], df["species"], df["interpretation_rule"])]
    return df


def points(times=(0.0, 2.5, 5.0, 7.5, 10.0), merged=()):
    """Plotted frame (row_idx ids); merged: (start, duration, clip_count, row_idx) extra merged points."""
    rows = [{"row_idx": i, "mp3_file": AUDIO.replace("/", "\\"), "channel": 1, "clip_time": t, "clip_duration": 5.0,
             "clip_count": 1} for i, t in enumerate(times)]
    for start, dur, n, rid in merged:
        rows.append({"row_idx": rid, "mp3_file": AUDIO, "channel": 1, "clip_time": start, "clip_duration": dur,
                     "clip_count": n})
    return pd.DataFrame(rows)


def summarize(ann, dff, annotators=None):
    mapped = agg.map_to_frame(ann, dff)
    return agg.aggregate_points(mapped, annotators), mapped


class TestAggregationRules(unittest.TestCase):
    def test_minimum_overlap_rule(self):
        # >= 1 s of the clip covered, or >= 50 % of the selection inside the clip (short calls still count)
        ann = pd.DataFrame({"overlap_s": [0.32, 1.0, 0.4, 0.9, 3.8],
                            "frac_of_selection": [0.08, 0.2, 1.0, 0.45, 0.9]})
        self.assertEqual(agg.counts_for_colouring(ann).tolist(), [False, True, True, False, True])

    def test_label_keys(self):
        self.assertEqual(agg.label_key("Bufo bufo?", "Bufo bufo", "exact"), "Bufo bufo")
        self.assertEqual(agg.label_key("?", None, "uncertain_only"), agg.UNIDENTIFIED)
        self.assertEqual(agg.label_key("Govor ", None, "unrecognized"), "Other: govor")
        self.assertEqual(agg.label_key("govor?", None, "unrecognized"), "Other: govor")
        self.assertEqual(agg.label_key("", None, "empty"), agg.NO_LABEL)
        self.assertEqual(agg.label_key(float("nan"), float("nan"), None), agg.NO_LABEL)

    def test_agreement_needs_two_annotators_with_the_same_species(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "bufo bufo?", "Bufo bufo", "case_normalized", True, 5.0)])
        res, _ = summarize(ann, points())
        self.assertEqual(res.loc[2, "raven_status"], "agree")
        self.assertEqual(res.loc[2, "raven_category"], "Bufo bufo")
        self.assertTrue(res.loc[2, "raven_uncertain"])                     # certainty differs -> marked, still agreement
        self.assertEqual(res.loc[2, "raven_labels"], "A: Bufo bufo; B: bufo bufo?")   # original text preserved

    def test_single_annotator_is_never_agreement(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0)])
        res, _ = summarize(ann, points())
        self.assertEqual(res.loc[2, "raven_status"], "single")
        self.assertEqual(res.loc[2, "raven_category"], "Bufo bufo")

    def test_different_species_and_unidentified_are_disagreements(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "Rana dalmatina", "Rana dalmatina", "exact", False, 5.0),
                         (3, "A", "Hyla arborea", "Hyla arborea", "exact", False, 7.5),
                         (4, "B", "?", None, "uncertain_only", True, 7.5)])
        res, _ = summarize(ann, points())
        self.assertEqual(res.loc[2, "raven_category"], agg.DISAGREE)
        self.assertEqual(res.loc[3, "raven_status"], "disagree")
        self.assertTrue(res.loc[3, "raven_uncertain"])

    def test_missing_annotation_is_not_disagreement(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "Rana dalmatina", "Rana dalmatina", "exact", False, 7.5)])
        res, _ = summarize(ann, points())
        self.assertEqual(res["raven_status"].to_dict(), {2: "single", 3: "single"})

    def test_empty_label_does_not_count_as_labelling(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "", None, "empty", False, 5.0)])
        res, _ = summarize(ann, points())
        self.assertEqual(res.loc[2, "raven_status"], "single")
        self.assertEqual(res.loc[2, "raven_category"], agg.MULTIPLE)        # both labels still shown
        self.assertIn("B: (no label)", res.loc[2, "raven_labels"])

    def test_non_species_labels_compare_by_text(self):
        ann = ann_frame([(1, "A", "govor", None, "unrecognized", False, 5.0),
                         (2, "B", "Govor", None, "unrecognized", False, 5.0)])
        res, _ = summarize(ann, points())
        self.assertEqual((res.loc[2, "raven_status"], res.loc[2, "raven_category"]), ("agree", "Other: govor"))

    def test_multiple_selections_per_clip(self):
        both = [(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                (2, "A", "Hyla arborea", "Hyla arborea", "exact", False, 5.0)]
        same = ann_frame(both + [(3, "B", "Hyla arborea", "Hyla arborea", "exact", False, 5.0),
                                 (4, "B", "Bufo bufo", "Bufo bufo", "exact", False, 5.0)])
        res, _ = summarize(same, points())
        self.assertEqual((res.loc[2, "raven_status"], res.loc[2, "raven_category"]), ("agree", agg.MULTIPLE))
        self.assertEqual(res.loc[2, "raven_n_annotations"], 4)
        subset = ann_frame(both + [(3, "B", "Bufo bufo", "Bufo bufo", "exact", False, 5.0)])
        res, _ = summarize(subset, points())
        self.assertEqual(res.loc[2, "raven_status"], "disagree")            # {Bufo, Hyla} vs {Bufo}
        alone = ann_frame(both)
        res, _ = summarize(alone, points())
        self.assertEqual((res.loc[2, "raven_status"], res.loc[2, "raven_category"]), ("single", agg.MULTIPLE))

    def test_multiple_clips_per_selection_and_no_neighbours(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, t) for t in (2.5, 5.0, 7.5)])
        res, _ = summarize(ann, points())
        self.assertEqual(sorted(res.index), [1, 2, 3])                      # exactly the linked clips
        attached = agg.attach(points(), res)
        self.assertEqual(attached.set_index("row_idx").loc[[0, 4], "raven_category"].tolist(),
                         [agg.NO_ANNOTATION, agg.NO_ANNOTATION])

    def test_annotator_filter(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "Rana dalmatina", "Rana dalmatina", "exact", False, 5.0),
                         (3, "B", "Hyla arborea", "Hyla arborea", "exact", False, 10.0)])
        res, _ = summarize(ann, points(), annotators=["A"])
        self.assertEqual(list(res.index), [2])
        self.assertEqual((res.loc[2, "raven_status"], res.loc[2, "raven_category"]), ("single", "Bufo bufo"))

    def test_merged_points_roll_up_per_clip(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 20.0),
                         (2, "B", "Bufo bufo", "Bufo bufo", "exact", False, 20.0),
                         (3, "A", "Bufo bufo", "Bufo bufo", "exact", False, 22.5),
                         (4, "B", "Rana dalmatina", "Rana dalmatina", "exact", False, 22.5),
                         (5, "A", "Hyla arborea", "Hyla arborea", "exact", False, 30.0),
                         (6, "B", "Hyla arborea", "Hyla arborea", "exact", False, 32.5)])
        dff = points(times=(), merged=[(20.0, 7.5, 2, 50), (30.0, 7.5, 2, 60)])
        res, _ = summarize(ann, dff)
        self.assertEqual(res.loc[50, "raven_status"], "disagree")          # one clip agrees, one disagrees
        self.assertEqual(res.loc[50, "raven_clips_annotated"], 2)
        # A and B labelled different clips of one merged point: each clip is single -> not agreement
        self.assertEqual((res.loc[60, "raven_status"], res.loc[60, "raven_category"]), ("single", "Hyla arborea"))

    def test_ambiguous_merged_points_get_nothing(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 10.0)])
        dff = points(times=(), merged=[(7.5, 10.0, 3, 70), (10.0, 10.0, 3, 71)])
        res, mapped = summarize(ann, dff)
        self.assertTrue(res.empty)
        self.assertEqual(agg.mapping_counts(mapped)["ambiguous"], 1)

    def test_compare_filter_keeps_unannotated_context(self):
        ann = ann_frame([(1, "A", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (2, "B", "Bufo bufo", "Bufo bufo", "exact", False, 5.0),
                         (3, "A", "Hyla arborea", "Hyla arborea", "exact", False, 7.5)])
        res, _ = summarize(ann, points())
        attached = agg.attach(points(), res)
        agree = agg.compare_filter(attached, "agree").set_index("row_idx")
        self.assertEqual(sorted(agree.index), [0, 1, 2, 4])                # 7.5 (single) hidden
        disagree = agg.compare_filter(attached, "disagree").set_index("row_idx")
        self.assertEqual(sorted(disagree.index), [0, 1, 4])


# ---------------------------------------------------------------------------
# Integration: synthetic import -> colour mode callbacks
# ---------------------------------------------------------------------------
# Expected per clip start (s) of the synthetic recording, given the tables below.
EXPECTED = {
    7.5: ("agree", "Pelophylax sp."), 10.0: ("agree", "Pelophylax sp."), 12.5: ("agree", "Pelophylax sp."),
    17.5: ("disagree", agg.DISAGREE), 20.0: ("disagree", agg.DISAGREE),
    27.5: ("single", "Rana temporaria"), 30.0: ("single", "Rana temporaria"),
    37.5: ("single", "Other: govor"), 40.0: ("single", "Other: govor"),
    42.5: ("single", agg.UNIDENTIFIED), 45.0: ("single", agg.UNIDENTIFIED),
    47.5: ("agree", agg.MULTIPLE), 50.0: ("agree", agg.MULTIPLE),
}


def plotted_model_frame():
    times = [t / 2 for t in range(0, 111, 5)]
    n = len(times)
    day = pd.Timestamp("2026-03-20")
    return pd.DataFrame({
        "row_idx": np.arange(n), "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float) * 0.5,
        "cluster_num": 10, "cluster_id": [int(t) % 3 for t in times], "location": LOC_A, "microlocation": LOC_A,
        "recorder_type": "AudioMoth", "file_name": f"{REC}.WAV", "mp3_file": f"Citimon\\{LOC_A}\\{REC}_ch1.mp3",
        "channel": 1, "clip_time": times, "clip_duration": 5.0, "clip_count": 1, "day_dt": day,
        "day_dt_str": "2026-03-20", "start_hour_float": [18.833 + t / 3600 for t in times],
        "time_of_day": [day + pd.Timedelta(hours=18, minutes=50, seconds=t) for t in times],
        "start_dt": [day + pd.Timedelta(hours=18, minutes=50, seconds=t) for t in times],
        "date_time_str": "20-03-2026 18:50",
    })


def callback(app, name):
    for v in app.callback_map.values():
        fn = v.get("callback")
        if fn is None:
            continue
        inner = getattr(fn, "__wrapped__", fn)
        if inner.__name__ == name:
            return inner
    raise KeyError(name)


def texts(component):
    out = []

    def walk(c):
        if isinstance(c, (list, tuple)):
            for x in c:
                walk(x)
        elif isinstance(c, str):
            out.append(c)
        elif c is not None and hasattr(c, "children"):
            if hasattr(c, "value") and isinstance(getattr(c, "value", None), str):
                out.append(c.value)
            walk(c.children)
    walk(component)
    return out


def manual_snapshot():
    with MANUAL_LABELS_LOCK:
        return {k: MANUAL_LABELS_CACHE[k] for k in MANUAL_LABELS_CACHE.iterkeys()}


class _IntegrationBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="raven_labels_")
        cls.db = os.path.join(cls.tmp, "ann.sqlite3")
        source = DataFrameClipSource(frame_with_locations(), "synthetic-labels")
        store = AnnotationStore(cls.db)
        f = f"{REC}.WAV"
        jp = table(row(1, 11.0, 14.0, f, 11.0, "2026/3/20", "18:50:11", "Pelophylax"),
                   row(2, 21.0, 22.0, f, 21.0, "2026/3/20", "18:50:21", "Bufo bufo"),
                   row(3, 31.0, 32.0, f, 31.0, "2026/3/20", "18:50:31", "Rana temporaria"),
                   row(4, 51.0, 52.0, f, 51.0, "2026/3/20", "18:50:51", "Bufo bufo"),
                   row(5, 51.5, 52.5, f, 51.5, "2026/3/20", "18:50:51", "Hyla arborea"))
        ss = table(row(1, 11.0, 14.0, f, 11.0, "2026/3/20", "18:50:11", "Pelophylax?"),
                   row(2, 21.0, 22.0, f, 21.0, "2026/3/20", "18:50:21", "Rana dalmatina"),
                   row(3, 41.0, 42.0, f, 41.0, "2026/3/20", "18:50:41", "govor"),
                   row(4, 51.0, 52.0, f, 51.0, "2026/3/20", "18:50:51", "bufo bufo"),
                   row(5, 51.5, 52.5, f, 51.5, "2026/3/20", "18:50:51", "Hyla arborea"),
                   row(6, 46.0, 46.5, f, 46.0, "2026/3/20", "18:50:46", "?"))
        files = [(f"{LOC_A}-JP-hitro.txt", jp), (f"{LOC_A}-SS.txt", ss)]
        opts = ImportOptions(location_mode="auto", annotator_from_table_name=True, primary_label_auto=True,
                             truncate_at_recording_end=True)
        prev = preview_import(files, opts, "", source, store)
        commit_import(files, opts, "", source, store, prev.token)
        del store
        cls.db_hash = hashlib.sha256(open(cls.db, "rb").read()).hexdigest()
        cls._old_db, cls._old_root = rl.DB_PATH, rl.RECORDINGS_ROOT
        rl.DB_PATH, rl.RECORDINGS_ROOT = cls.db, cls.tmp
        with rl._cache_lock:
            rl._cache.clear()
        cls.app = Dash(__name__)
        register_plot_callbacks(cls.app)
        register_cluster_callbacks(cls.app)
        register_table_callbacks(cls.app)
        rl.register_raven_label_callbacks(cls.app)
        cls.update_figure = staticmethod(callback(cls.app, "update_figure"))

    @classmethod
    def tearDownClass(cls):
        rl.DB_PATH, rl.RECORDINGS_ROOT = cls._old_db, cls._old_root
        with rl._cache_lock:
            rl._cache.clear()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clear_datasets()
        self.meta = install_dataset(plotted_model_frame(), project="Citimon", model="perchv2", locations=[LOC_A])

    def tearDown(self):
        clear_datasets()

    def figure(self, mode, trigger='color-mode-radio.value', annotators=None, compare='all', raven_values=None,
               raven_ids=None, raven_colors=None, figure_state=None, label_colors=None):
        args = (None, None, 10, [], {}, {}, None, None, 50000, 0, ['on'], False, 15, 1, None, None, mode, None,
                label_colors or {}, {}, [], figure_state, None, LOC_A, None, [], [], None)
        with triggered(trigger):
            return self.update_figure(*args, selected_project="Citimon", selected_model="perchv2",
                                      raven_annotators=annotators, raven_compare=compare,
                                      raven_checkbox_values=raven_values, raven_colors_data=raven_colors,
                                      raven_checkbox_ids=raven_ids, dataset_metadata=self.meta)

    @staticmethod
    def by_clip(fig):
        """clip_time -> (trace name, symbol) for every plotted point."""
        if isinstance(fig, tuple):
            fig = fig[0]
        out = {}
        for tr in fig.data:
            for cd in tr.customdata:
                out[float(cd[6])] = (tr.name, (tr.marker.symbol or "circle"))
        return out


class TestRavenColorMode(_IntegrationBase):
    def test_database_is_only_read(self):
        files_before = sorted(os.listdir(self.tmp))
        os.chmod(self.db, stat.S_IREAD)                    # any write attempt would now fail
        try:
            with rl._cache_lock:
                rl._cache.clear()
            ann = rl.get_annotations("Citimon", "perchv2")
        finally:
            os.chmod(self.db, stat.S_IREAD | stat.S_IWRITE)
        self.assertEqual(ann["selection_pk"].nunique(), 11)
        self.assertEqual(hashlib.sha256(open(self.db, "rb").read()).hexdigest(), self.db_hash)
        self.assertEqual(sorted(os.listdir(self.tmp)), files_before)

    def test_points_coloured_by_aggregated_annotation(self):
        out = self.figure('raven')
        fig, key, stats = out[0], out[1], out[7]
        clips = self.by_clip(fig)
        self.assertEqual(len(clips), len(plotted_model_frame()))          # same points, no duplicates or removals
        for t, (status, category) in EXPECTED.items():
            self.assertEqual(clips[t][0], category, t)
            self.assertEqual(server_cache[key].set_index("clip_time").loc[t, "raven_status"], status, t)
        for t in (0.0, 2.5, 5.0, 15.0, 22.5, 25.0, 32.5, 35.0, 52.5, 55.0):
            self.assertEqual(clips[t][0], agg.NO_ANNOTATION, t)              # neighbours stay unannotated
        self.assertEqual(clips[10.0][1], "circle-open")                     # "Pelophylax?" -> hollow
        self.assertEqual(clips[42.5][1], "circle-open")                     # lone "?"
        self.assertEqual(clips[30.0][1], "circle")
        self.assertEqual(fig.data[0].name, agg.NO_ANNOTATION)               # grey context drawn underneath
        # Annotated traces win clicks over the grey points they overlap (assets/scatter_click_hittest.js)
        for tr in fig.data:
            prio = (tr.meta or {}).get('click_priority') if isinstance(tr.meta, dict) else None
            self.assertEqual(prio, None if tr.name == agg.NO_ANNOTATION else 1, tr.name)
        self.assertEqual(stats[rl.STATS_PREFIX + agg.DISAGREE], 2)
        self.assertEqual(stats['__raven_links_landed__'], 24)              # JP 3+2+2+2+2, SS 3+2+2+2+2+2
        hover = {float(cd[6]): txt for tr in fig.data for cd, txt in zip(tr.customdata, tr.text or [])}
        self.assertEqual(hover[20.0], "JP: Bufo bufo; SS: Rana dalmatina (disagree)")

    def test_compare_modes_and_single_annotator(self):
        agree = self.by_clip(self.figure('raven', compare='agree'))
        self.assertEqual({c for c, _ in agree.values()}, {"Pelophylax sp.", agg.MULTIPLE, agg.NO_ANNOTATION})
        self.assertNotIn(30.0, agree)                                       # single-annotator clip hidden
        self.assertIn(0.0, agree)                                           # unannotated context kept
        disagree = self.by_clip(self.figure('raven', compare='disagree'))
        self.assertEqual({c for c, _ in disagree.values()}, {agg.DISAGREE, agg.NO_ANNOTATION})
        jp = self.by_clip(self.figure('raven', annotators=["JP"]))
        self.assertEqual(jp[20.0][0], "Bufo bufo")                          # only JP's label, no conflict
        self.assertEqual(jp[47.5][0], agg.MULTIPLE)
        self.assertEqual(jp[40.0][0], agg.NO_ANNOTATION)                    # SS-only clip
        self.assertEqual(jp[10.0][1], "circle")                              # JP's label was certain
        compare_options = callback(self.app, "compare_options")
        opts, value, hint = compare_options(["JP"], "Citimon", "agree")
        self.assertEqual(value, "all")
        self.assertTrue(all(o.get("disabled") for o in opts if o["value"] != "all"))
        self.assertIn("at least two annotators", hint)
        opts, value, hint = compare_options([], "Citimon", "agree")         # empty = all (JP, SS)
        self.assertEqual((value, hint), ("agree", ""))

    def test_legend_lists_categories_and_checkboxes_filter(self):
        out = self.figure('raven')
        render = callback(self.app, "render_cluster_controls")
        with triggered('color-mode-radio.value', 'raven'):
            rows = render(10, None, None, out[7], 'raven', None, {}, {}, {}, {}, None, [], [], [], [], {}, [], [], dataset_metadata=self.meta)
        ids = [r.children[0].id for r in rows]
        names = [r.children[2].children for r in rows]
        self.assertEqual(names[-1], agg.NO_ANNOTATION)
        self.assertEqual(names[:2], ["Pelophylax sp.", "Rana temporaria"])  # species first
        self.assertIn(agg.DISAGREE, names)
        self.assertTrue(all(i["type"] == "raven-checkbox" for i in ids))
        values = [[] if i["index"] == "Rana temporaria" else ["on"] for i in ids]
        filtered = self.by_clip(self.figure('raven', trigger='{"index":"Rana temporaria","type":"raven-checkbox"}.value',
                                            raven_values=values, raven_ids=ids))
        self.assertNotIn(30.0, filtered)
        self.assertIn(20.0, filtered)

    def test_colour_change_patches_raven_traces(self):
        fig = self.figure('raven')[0]
        with triggered('raven-color-store.data'):
            out = self.figure('raven', trigger='raven-color-store.data', figure_state=fig.to_dict(),
                              raven_colors={"Rana temporaria": "#000000"})
        self.assertIsInstance(out[0], Patch)

    def test_switching_modes_preserves_cluster_and_manual_colouring(self):
        def signature(fig):
            return [(t.name, t.marker.color, list(t.x), [list(map(str, c)) for c in t.customdata]) for t in fig.data]
        before_cluster = signature(self.figure('cluster')[0])
        before_manual = signature(self.figure('manual')[0])
        manual_before = manual_snapshot()
        out = self.figure('raven')
        self.assertTrue((server_cache[out[1]]["manual_label"] == "Unlabeled").all())   # manual data untouched
        self.assertEqual(signature(self.figure('cluster')[0]), before_cluster)
        self.assertEqual(signature(self.figure('manual')[0]), before_manual)
        self.assertEqual(manual_snapshot(), manual_before)
        cluster_df = server_cache[self.figure('cluster')[1]]
        self.assertFalse(any(c.startswith("raven_") for c in cluster_df.columns))  # no Raven data outside Raven mode

    def test_manual_label_colours_survive_raven_mode(self):
        self.assertNotEqual(os.path.abspath(MANUAL_LABELS_CACHE.directory), os.path.abspath(REAL_MANUAL_DIR))
        key = (LOC_A, LOC_A, f"{REC}_ch1.mp3", 1, 300)                          # clip at 30.0 s, temp store only
        with MANUAL_LABELS_LOCK:
            MANUAL_LABELS_CACHE[key] = "TestFrog"
        try:
            colors = {"TestFrog": "#123456", "Unlabeled": "#D9D9D9"}
            manual = self.figure('manual', label_colors=colors)[0]
            self.assertIn(("TestFrog", "#123456"), [(t.name, t.marker.color) for t in manual.data])
            raven = self.figure('raven', label_colors=colors)
            self.assertEqual(self.by_clip(raven[0])[30.0][0], "Rana temporaria")
            self.assertEqual(server_cache[raven[1]].set_index("clip_time").loc[30.0, "manual_label"], "TestFrog")
            again = self.figure('manual', label_colors=colors)[0]
            self.assertEqual([(t.name, t.marker.color, list(t.x)) for t in again.data],
                             [(t.name, t.marker.color, list(t.x)) for t in manual.data])
        finally:
            with MANUAL_LABELS_LOCK:
                del MANUAL_LABELS_CACHE[key]


class TestRavenTwoUsers(_IntegrationBase):
    """Two browsers with different datasets and different annotator filters on one worker."""

    def _other_dataset_meta(self):
        other = plotted_model_frame()
        other["location"] = "LocOther"
        other["microlocation"] = "LocOther"
        other["mp3_file"] = f"Citimon\LocOther\{REC}_ch1.mp3"      # a recording the Raven import knows nothing about
        return install_dataset(other, project="Citimon", model="perchv2", locations=["LocOther"])

    def _figure_for(self, meta, location, annotators, mode="raven", compare="all"):
        args = (None, None, 10, [], {}, {}, None, None, 50000, 0, ['on'], False, 15, 1, None, None, mode, None,
                {}, {}, [], None, None, location, None, [], [], None)
        with triggered('color-mode-radio.value'):
            return self.update_figure(*args, selected_project="Citimon", selected_model="perchv2",
                                      raven_annotators=annotators, raven_compare=compare, raven_checkbox_values=None,
                                      raven_colors_data=None, raven_checkbox_ids=None, dataset_metadata=meta)

    def test_each_user_gets_annotations_joined_against_their_own_dataset_and_filters(self):
        meta_a = self.meta                                              # user A: LOC_A, annotator JP
        meta_b = self._other_dataset_meta()                             # user B: another location, annotator SS
        a1 = self.by_clip(self._figure_for(meta_a, LOC_A, ["JP"]))
        b1 = self.by_clip(self._figure_for(meta_b, "LocOther", ["SS"]))
        a2 = self.by_clip(self._figure_for(meta_a, LOC_A, ["JP"]))      # interleaved: A again after B
        self.assertEqual(a1, a2)
        self.assertEqual(a1[20.0][0], "Bufo bufo")                      # JP's own label on A's recording
        self.assertEqual(a1[40.0][0], agg.NO_ANNOTATION)                # SS-only clip stays unannotated for JP
        self.assertEqual({name for name, _sym in b1.values()}, {agg.NO_ANNOTATION},
                         "B's recordings have no Raven import; nothing may be joined from A's dataset")
        ss_on_a = self.by_clip(self._figure_for(meta_a, LOC_A, ["SS"]))
        self.assertEqual(ss_on_a[40.0][0], "Other: govor")              # a different filter on the SAME dataset
        self.assertNotEqual(ss_on_a, a1)
        self.assertEqual(self.by_clip(self._figure_for(meta_a, LOC_A, ["JP"])), a1)


class TestRavenViews(_IntegrationBase):
    def test_table_view_shows_raven_columns_only_in_raven_mode(self):
        update_table = callback(self.app, "update_table")
        key = self.figure('raven')[1]
        with triggered('color-mode-radio.value'):
            div, info, _ = update_table("table-tab", key, None, {}, None, 0, None, "raven", False)
        t = texts(div)
        for header in ("Raven labels", "Raven annotators", "Raven agreement"):
            self.assertIn(header, t)
        self.assertIn("JP: Bufo bufo; SS: Rana dalmatina", t)
        self.assertIn("JP, SS", t)
        self.assertIn("disagree", t)
        self.assertIn("single annotator", t)
        key = self.figure('cluster')[1]
        with triggered('color-mode-radio.value'):
            div, _, _ = update_table("table-tab", key, None, {}, None, 0, None, "cluster", False)
        self.assertNotIn("Raven labels", texts(div))

    def test_point_details_show_all_annotations_with_original_intervals(self):
        key = self.figure('raven')[1]
        dff = server_cache[key]
        show = callback(self.app, "show_point_details")

        def details(t, annotators=None, mode='raven'):
            pid = int(dff[dff.clip_time == t]["plot_id"].iloc[0])
            click = {"points": [{"customdata": ["x", pid]}]}
            with triggered('scatter.clickData'):
                return " ".join(texts(show(click, None, mode, annotators, {"cache_key": key}, "Citimon", "perchv2")))

        d = details(20.0)
        self.assertIn("annotators disagree", d)
        self.assertIn("Bufo bufo", d)
        self.assertIn("Rana dalmatina", d)
        self.assertIn("21.00–22.00 s", d)                                    # original interval, unchanged
        self.assertIn("quick pass", d)                                       # JP table name has -hitro
        d = details(10.0)
        self.assertIn("Pelophylax?", d)                                      # original text kept
        self.assertIn("(uncertain)", d)
        self.assertIn("annotators agree", d)
        self.assertIn("not a species", details(40.0))
        self.assertIn("No Raven annotation on this point", details(0.0))
        self.assertIn("1 more from annotators not selected", details(20.0, annotators=["JP"]))
        self.assertEqual(details(20.0, mode='cluster'), "")                 # other modes unchanged

    def test_small_overlaps_are_listed_but_do_not_colour(self):
        show = callback(self.app, "show_point_details")
        with mock.patch.object(agg, "MIN_OVERLAP_S", 99.0), mock.patch.object(agg, "MIN_FRAC_OF_SELECTION", 2.0):
            out = self.figure('raven')
            key = out[1]
            self.assertEqual({c for c, _ in self.by_clip(out[0]).values()}, {agg.NO_ANNOTATION})
            dff = server_cache[key]
            pid = int(dff[dff.clip_time == 20.0]["plot_id"].iloc[0])
            with triggered('scatter.clickData'):
                d = " ".join(texts(show({"points": [{"customdata": ["x", pid]}]}, None, 'raven', None,
                                        {"cache_key": key}, "Citimon", "perchv2")))
        self.assertIn("Bufo bufo", d)                                         # still listed in the details
        self.assertIn("not used for colouring", d)
        self.assertIn("so this point is not coloured", d)

    def test_table_row_click_feeds_details_and_histogram(self):
        key = self.figure('raven')[1]
        dff = server_cache[key]
        pid = int(dff[dff.clip_time == 30.0]["plot_id"].iloc[0])
        hist = callback(self.app, "show_histogram_for_clicked_cluster")
        table_click = {"points": [{"customdata": ["Rana temporaria", pid, None]}]}
        with triggered('table-click-data-store.data'):
            fig = hist(None, 'count', 'daily', {}, {}, {}, 'raven', table_click, {"cache_key": key})
        self.assertEqual(int(sum(len(t.x) for t in fig.data if t.x is not None)), 2)   # two Rana temporaria clips
        show = callback(self.app, "show_point_details")
        with triggered('table-click-data-store.data'):
            d = " ".join(texts(show(None, table_click, 'raven', None, {"cache_key": key}, "Citimon", "perchv2")))
        self.assertIn("Rana temporaria", d)

    def test_select_all_drives_raven_checkboxes(self):
        select_all = callback(self.app, "raven_select_all")
        opts = [[{"label": "", "value": "on"}]] * 3
        self.assertEqual(select_all([], opts), [[], [], []])
        self.assertEqual(select_all(["All"], opts), [["on"]] * 3)

    def test_colour_option_only_for_projects_with_annotations(self):
        option = callback(self.app, "raven_color_option")
        opts, value, annotators = option("Citimon", "cluster")
        self.assertIn("raven", [o["value"] for o in opts])
        self.assertEqual([a["value"] for a in annotators], ["JP", "SS"])
        opts, value, annotators = option("OtherProject", "raven")
        self.assertNotIn("raven", [o["value"] for o in opts])
        self.assertEqual(value, "cluster")

    def test_play_interval_uses_original_selection_interval(self):
        play = callback(self.app, "play_interval")
        ann = rl.get_annotations("Citimon", "perchv2")
        pk = int(ann[(ann.annotator == "JP") & (ann.label == "Pelophylax")]["selection_pk"].iloc[0])
        prop = json.dumps({"index": pk, "type": "raven-play-interval"}, separators=(",", ":")) + ".n_clicks"
        tone = (np.sin(np.linspace(0, 4000 * np.pi, 16000 * 4)).astype(np.float32), 16000)   # no real audio read
        settings = ("linear", 1024, 0.5, "hann", 50, 5000, 128, "Viridis", -100)
        with mock.patch.object(rl.utils, "load_audio_segment", return_value=tone) as load, triggered(prop, 1):
            store, fig, msg = play([1], "Citimon", *settings)
        self.assertEqual(load.call_args.kwargs["clip_time"], 11.0)
        self.assertEqual(load.call_args.kwargs["clip_duration"], 3.0)        # 11-14 s, not a 5 s clip
        self.assertTrue(store["audio_path"].endswith(f"{AUDIO}/1/11.000/14.000"))
        # The time axis shows file time (audio loaded from 10.5 s), so the outline is the selection itself
        self.assertEqual((fig.layout.shapes[0].x0, fig.layout.shapes[0].x1), (11.0, 14.0))
        self.assertAlmostEqual(float(fig.data[0].x[0]), 10.5, delta=0.1)
        self.assertEqual(fig.layout.xaxis.title.text, "Time in file (s)")
        self.assertIn("Pelophylax", store["info"])
        self.assertEqual(msg, "")
        with mock.patch.object(rl.utils, "load_audio_segment", return_value=(None, None)), triggered(prop, 1):
            self.assertEqual(play([1], "Citimon", *settings)[2], "The audio file could not be read.")

    def test_raven_code_never_touches_manual_labels(self):
        for path in ("callbacks/callbacks_raven_labels.py", "raven_import/aggregate.py", "raven_import/viewer.py"):
            src = open(os.path.join(ROOT, path), encoding="utf-8").read()
            self.assertNotIn("MANUAL_LABELS_CACHE", src, path)
            self.assertNotIn("AnnotationStore(", src, path)
        self.assertEqual(hashlib.sha256(open(self.db, "rb").read()).hexdigest(), self.db_hash)


if __name__ == "__main__":
    unittest.main()
