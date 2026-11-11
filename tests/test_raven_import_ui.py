"""
Raven import wizard (Citimon): upload -> automatic analysis -> review exceptions -> import.

Uses SYNTHETIC tables and a synthetic clip source with temporary annotation and diskcache stores; the real
annotation database and manual-label store are never opened.
"""
import base64
import contextlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

from dash import Dash
import dash._callback_context as dash_ctx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
FIXTURES = os.path.join(TESTS, "fixtures", "raven_synthetic")
for p in (ROOT, TESTS, FIXTURES):
    if p not in sys.path:
        sys.path.insert(0, p)

# Importing the callbacks package opens diskcache stores at import time, which rewrites
# their SQLite files. Point them at throwaway dirs unless another test already imported it.
_DISKCACHE_TMP = tempfile.mkdtemp(prefix="raven_ui_diskcache_")
os.environ.setdefault("VIBROVIS_MANUAL_LABELS_DIR", os.path.join(_DISKCACHE_TMP, "manual_labels"))
os.environ.setdefault("VIBROVIS_CLIP_DURATIONS_DIR", os.path.join(_DISKCACHE_TMP, "clip_durations"))

from generate_fixtures import LOC_A  # noqa: E402
from test_raven_citimon_delivery import HEADER, frame_with_locations, row, table  # noqa: E402
import callbacks.callbacks_raven as cr  # noqa: E402
from callbacks.callbacks_constants import MANUAL_LABELS_CACHE  # noqa: E402
from layout_raven import (  # noqa: E402
    MODAL_HIDDEN, MODAL_VISIBLE, configured_aliases_text, raven_import_launcher, raven_import_modal,
)
from raven_import.clip_index import DataFrameClipSource  # noqa: E402
from raven_import.config import ROLES  # noqa: E402
from raven_import.store import AnnotationStore  # noqa: E402

DYNAMIC_IDS = {"raven-fix-annotator", "raven-fix-location", "raven-fix-alias", "raven-apply-fixes",
               "raven-conflict", "raven-ack", "raven-col"}
RID_185 = f"zabe::Citimon/{LOC_A}/20260320_185000_ch1.mp3::ch1"


@contextlib.contextmanager
def triggered(prop_id):
    token = dash_ctx.context_value.set(types.SimpleNamespace(triggered_inputs=[{'prop_id': prop_id, 'value': 1}]))
    try:
        yield
    finally:
        dash_ctx.context_value.reset(token)


def data_url(data: bytes) -> str:
    return "data:text/plain;base64," + base64.b64encode(data).decode()


def jp_table():
    return table(row(1, 900.0, 902.0, "2026320_185000.WAV", 1.0, "2026/3/20", "18:50:01", "Pelophylax"),
                 row(2, 950.0, 951.0, "2026320_185000.WAV", 30.0, "2026/3/20", "18:50:30", "Rana dalmatina?"))


def ss_table_cp1250():
    text = HEADER + row(1, 10.0, 12.0, "20260320_185000.WAV", 1.0, "2026/3/20", "18:50:01", "Bufo bufo") \
        + row(2, 20.0, 21.0, "20260320_185000.WAV", 20.0, "2026/3/20", "18:50:20", "šakal ali žaba?")
    return text.encode("cp1250")


def component_ids(component, acc=None):
    acc = {} if acc is None else acc
    cid = getattr(component, "id", None)
    if cid is not None:
        acc[str(cid) if not isinstance(cid, dict) else cid.get("type")] = component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, "to_plotly_json"):
            component_ids(child, acc)
    return acc


class TestRavenImportWizard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Dash(__name__, suppress_callback_exceptions=True)
        cr.register_raven_callbacks(cls.app)
        cls.fns = {}
        for entry in cls.app.callback_map.values():
            fn = getattr(entry["callback"], "__wrapped__", entry["callback"])
            cls.fns[fn.__name__] = fn

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="raven_ui_")
        self._orig = (cr.CLIP_SOURCE, cr.STORE)
        cr.CLIP_SOURCE = DataFrameClipSource(frame_with_locations("2026_03-R26", "2026_03-R99"), "synthetic-ui")
        cr.STORE = AnnotationStore(os.path.join(self.tmp, "ann.sqlite3"))

    def tearDown(self):
        cr.CLIP_SOURCE, cr.STORE = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers mirroring the browser --------------------------------------------------------------------

    @staticmethod
    def form(**kw):
        values = dict(model=None, annotator_source="auto", location_mode="auto", fixed_location=None,
                      time_mode="auto", file_from_name=[], encoding="auto", decimal=".", default_channel=1,
                      duration=60, role_values=[None] * len(ROLES), annotation_cols=[], primary_col=None,
                      import_flags=["repair_dates", "truncate"], aliases=configured_aliases_text())
        values.update(kw)
        return list(values.values())

    def upload(self, files, form=None):
        """files: list of (name, bytes). Returns (token, analysis_state, review_component, outputs)."""
        with triggered("raven-upload.contents"):
            out = self.fns["on_analyse"]([data_url(d) for _, d in files], 0, [], [n for n, _ in files], None, None,
                                         [], [], [], [], [], None, None, *(form or self.form()))
        return out[0], out[4], out[2], out

    def apply_fixes(self, token, analysis, annotators=None, locations=None, aliases=None, form=None):
        annotators, locations = annotators or {}, locations or {}
        with triggered('{"index":0,"type":"raven-apply-fixes"}.n_clicks'):
            out = self.fns["on_analyse"](
                None, 0, [1], None, token, analysis,
                list(annotators.values()), [{'type': 'raven-fix-annotator', 'index': k} for k in annotators],
                list(locations.values()), [{'type': 'raven-fix-location', 'index': k} for k in locations],
                [[json.dumps(a) for a in (aliases or [])]], None, None, *(form or self.form()))
        return out[4], out[2], out

    def ready(self, analysis, conflicts=None, ack=False, annotators=None, locations=None, aliases=None, form=None,
              fallback=None):
        """Import button state as the browser would compute it: (enabled, reason)."""
        conflicts, annotators, locations = conflicts or {}, annotators or {}, locations or {}
        disabled, style, reason = self.fns["update_import_state"](
            analysis, list(conflicts.values()), [{'type': 'raven-conflict', 'index': k} for k in conflicts],
            [["ack"]] if ack else [[]],
            list(annotators.values()), [{'type': 'raven-fix-annotator', 'index': k} for k in annotators],
            list(locations.values()), [{'type': 'raven-fix-location', 'index': k} for k in locations],
            [[json.dumps(a) for a in (aliases or [])]], None, fallback, *(form or self.form()))
        self.assertEqual(style["cursor"], "not-allowed" if disabled else "pointer")
        return not disabled, reason

    def confirm(self, analysis, conflicts=None, ack=False, form=None, note=None):
        conflicts = conflicts or {}
        return self.fns["on_confirm"](
            1, analysis, list(conflicts.values()), [{'type': 'raven-conflict', 'index': k} for k in conflicts],
            [["ack"]] if ack else [[]], note, 0, None, None, *(form or self.form()))

    # ---- layout --------------------------------------------------------------------------------------------

    def test_layout_ids_defaults_and_collapsed_advanced(self):
        comps = component_ids(raven_import_modal())
        comps.update(component_ids(raven_import_launcher()))
        for key, entry in self.app.callback_map.items():
            for dep in entry["inputs"] + entry["state"]:
                dep_id = dep["id"]
                name = json.loads(dep_id)["type"] if dep_id.startswith("{") else dep_id
                if name in DYNAMIC_IDS or name == "project-dropdown":
                    continue
                self.assertIn(name, comps, key)
        self.assertEqual(comps["raven-location-mode"].value, "auto")
        self.assertEqual(comps["raven-annotator-source"].value, "auto")
        self.assertEqual(sorted(comps["raven-import-flags"].value), ["repair_dates", "truncate"])
        self.assertIn("2026_03-r26=2026_03-R26", comps["raven-aliases"].value)
        advanced = [c for c in raven_import_modal().children.children
                    if type(c).__name__ == "Details" and "Advanced" in str(c.children[0].children)]
        self.assertEqual(len(advanced), 1)
        self.assertFalse(getattr(advanced[0], "open", False))
        self.assertTrue(comps["raven-confirm-btn"].disabled)

    def test_launcher_only_for_citimon_and_modal_toggle(self):
        self.assertEqual(self.fns["toggle_launcher"]("Citimon")["display"], "block")
        self.assertEqual(self.fns["toggle_launcher"]("Multimon")["display"], "none")
        with triggered("raven-open-btn.n_clicks"):
            self.assertEqual(self.fns["toggle_modal"](1, 0), MODAL_VISIBLE)
        with triggered("raven-close-btn.n_clicks"):
            self.assertEqual(self.fns["toggle_modal"](1, 1), MODAL_HIDDEN)

    def test_import_readiness_rules(self):
        r = cr.import_readiness
        self.assertEqual(r(None, {}, False, False, True)[0], False)
        good = {"preview_token": "t", "fingerprint": "f", "conflicts": {}, "needs_ack": False}
        self.assertEqual(r(good, {}, False, False, True), (True, ""))
        self.assertFalse(r({**good, "preview_token": None}, {}, False, False, True)[0])
        self.assertFalse(r(good, {}, False, False, False)[0])
        self.assertFalse(r(good, {}, False, True, True)[0])
        self.assertFalse(r({**good, "conflicts": {"a.txt": ["b1"]}}, {"a.txt": None}, False, False, True)[0])
        self.assertTrue(r({**good, "conflicts": {"a.txt": ["b1"]}}, {"a.txt": "replace"}, False, False, True)[0])
        self.assertFalse(r({**good, "needs_ack": True}, {}, False, False, True)[0])
        self.assertTrue(r({**good, "needs_ack": True}, {}, True, False, True)[0])

    def test_bad_uploads_rejected(self):
        files, errors = cr.decode_uploads(["data:x;base64,AAAA", "bad"], ["x.wav", "y.txt"])
        self.assertEqual(files, [])
        self.assertEqual(len(errors), 2)

    # ---- default workflow ----------------------------------------------------------------------------------

    def test_happy_path_needs_no_choices(self):
        files = [(f"{LOC_A}-JP-hitro.txt", jp_table()), (f"{LOC_A}-SS.txt", ss_table_cp1250())]
        token, analysis, review, out = self.upload(files)
        self.assertEqual(self.ready(analysis), (True, ""))        # Import enabled, no decisions needed
        self.assertIn("Import 2 table(s), 4 selection(s)", out[5])
        self.assertIn("Nothing needs your attention", str(review))
        self.assertFalse(analysis["needs_ack"])
        self.assertEqual(analysis["conflicts"], {})

        before_manual = len(MANUAL_LABELS_CACHE)
        result, signal, after = self.confirm(analysis)
        self.assertIn("Imported 2 table(s): 4 selection(s) linked, 0 awaiting", str(result))
        self.assertEqual((signal, after["preview_token"]), (1, None))
        self.assertFalse(self.ready(after)[0])                     # disabled after importing
        self.assertEqual(len(MANUAL_LABELS_CACHE), before_manual)  # manual labels untouched

        batches = {b["source_filename"]: b for b in cr.STORE.list_batches("Citimon")}
        self.assertEqual({b["annotator"] for b in batches.values()}, {"JP", "SS"})
        jp = cr.STORE.selections(batches[f"{LOC_A}-JP-hitro.txt"]["batch_id"])
        self.assertEqual(jp[0]["annotations"]["Annotation"], "Pelophylax")          # original text kept
        self.assertEqual(jp[0]["interpretations"]["Annotation"]["species"], "Pelophylax sp.")
        self.assertEqual(jp[0]["location"], LOC_A)
        self.assertIn("begin_file_date_repaired", jp[0]["flags_json"])
        ss = cr.STORE.selections(batches[f"{LOC_A}-SS.txt"]["batch_id"])
        self.assertEqual(ss[1]["annotations"]["Annotation"], "šakal ali žaba?")      # CP1250 decoded
        self.assertEqual(ss[1]["interpretations"]["Annotation"]["uncertain"], 1)
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(cr.STORE.db_path)) as c:
            meta = dict(c.execute("SELECT source_filename, table_meta_json FROM import_batches").fetchall())
            opts = json.loads(c.execute("SELECT options_json FROM import_batches LIMIT 1").fetchone()[0])
            raw = dict(c.execute("SELECT source_filename, raw_content FROM import_batches").fetchall())
        self.assertIn('"quick_pass": true', meta[f"{LOC_A}-JP-hitro.txt"])
        self.assertIn('"quick_pass": false', meta[f"{LOC_A}-SS.txt"])
        self.assertEqual((opts["location_mode"], opts["primary_label_auto"]), ("auto", True))
        self.assertEqual(raw[f"{LOC_A}-SS.txt"], ss_table_cp1250())                 # original bytes kept

    def test_missing_data_is_importable_and_kept_unlinked(self):
        new_location = table(row(1, 5.0, 6.0, "20260410_120000.WAV", 5.0, "2026/4/10", "12:00:05", "Bufo bufo"))
        not_in_partition = table(row(1, 5.0, 6.0, "20260321_000000.WAV", 5.0, "2026/3/21", "0:00:05", "Bufo bufo"))
        files = [("2026_04-C77-JP.txt", new_location), (f"{LOC_A}-JP.txt", not_in_partition)]
        token, analysis, review, out = self.upload(files)
        self.assertEqual(self.ready(analysis), (True, ""))        # coverage gaps do not block import
        self.assertFalse(analysis["needs_ack"])
        self.assertIn("awaiting recordings or clips", str(review))
        self.assertIn("2026_04-C77 (1)", str(review))
        result = self.confirm(analysis)[0]
        self.assertIn("0 selection(s) linked, 2 awaiting", str(result))
        statuses = sorted(s["status"] for b in cr.STORE.list_batches() for s in cr.STORE.selections(b["batch_id"]))
        self.assertEqual(statuses, ["recording_not_found", "unknown_location"])

    def test_missing_metadata_needs_fix(self):
        files = [("annotations.txt", table(row(1, 900.0, 902.0, "20260320_185000.WAV", 1.0, "2026/3/20",
                                               "18:50:01", "Bufo bufo")))]
        token, analysis, review, out = self.upload(files)
        enabled, reason = self.ready(analysis)
        self.assertFalse(enabled)                                  # nothing importable yet
        self.assertIn("needs a fix", reason)
        enabled, reason = self.ready(analysis, annotators={"annotations.txt": "SS"})
        self.assertFalse(enabled)
        self.assertIn("Annotator not found", str(review))
        self.assertIsNone(analysis["preview_token"])

        self.assertIn("Location not detected", str(review))        # both fixes offered in the same round
        analysis, review, out = self.apply_fixes(token, analysis, annotators={"annotations.txt": "SS"})
        self.assertIn("Location not detected", str(review))
        self.assertTrue(analysis["needs_ack"])                     # missing_location is unresolved
        enabled, reason = self.ready(analysis, locations={"annotations.txt": LOC_A})
        self.assertEqual((enabled, reason), (False, "Press 'Apply and re-analyse' to use the values you entered."))
        enabled, reason = self.ready(analysis)
        self.assertIn("Tick 'Import anyway'", reason)
        self.assertTrue(self.ready(analysis, ack=True)[0])
        refused = self.confirm(analysis)[0]                        # backend safeguard still refuses
        self.assertIn("Confirm how unresolved selections", str(refused))
        self.assertEqual(cr.STORE.list_batches(), [])

        analysis, review, out = self.apply_fixes(token, analysis, locations={"annotations.txt": LOC_A})
        self.assertFalse(analysis["needs_ack"])
        self.assertIn("Nothing needs your attention", str(review))
        self.confirm(analysis)
        b = cr.STORE.list_batches()[0]
        self.assertEqual(b["annotator"], "SS")
        sel = cr.STORE.selections(b["batch_id"])[0]
        self.assertEqual((sel["status"], sel["location_source"]), ("matched_multiple", "override"))
        self.assertIn("location_override", sel["flags_json"])

    def test_aliases_configured_validated_and_suggested(self):
        data = table(row(1, 5.0, 6.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Bufo bufo"))
        files = [("2026_03-r26-JP.txt", data), ("2026_03-r99-JP.txt", data)]
        token, analysis, review, out = self.upload(files)
        text = str(review)
        self.assertIn("'2026_03-R28' is not a known location", text)      # configured alias validated, ignored
        self.assertIn("2026_03-r99 → 2026_03-R99", text)                  # case-only suggestion, not applied
        self.assertNotIn("2026_03-r26 → ", text)                          # configured alias already applies
        self.assertFalse(analysis["needs_ack"])                           # unknown location = awaiting data

        analysis, review, out = self.apply_fixes(token, analysis, aliases=[["2026_03-r99", "2026_03-R99"]])
        self.assertEqual(analysis["fixes"]["extra_aliases"], {"2026_03-r99": "2026_03-R99"})
        self.confirm(analysis)
        locs = {s["location"] for b in cr.STORE.list_batches() for s in cr.STORE.selections(b["batch_id"])}
        self.assertEqual(locs, {"2026_03-R26", "2026_03-R99"})

    def test_unresolved_rows_need_acknowledgement(self):
        files = [(f"{LOC_A}-JP.txt", table(
            row(1, 900.0, 902.0, "20260320_185000.WAV", 1.0, "2026/3/20", "18:50:01", "Bufo bufo"),
            row(2, 950.0, 949.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Bufo bufo")))]
        token, analysis, review, out = self.upload(files)
        self.assertTrue(analysis["needs_ack"])
        self.assertIn("begin/end time missing, invalid or reversed", str(review))
        self.assertFalse(self.ready(analysis)[0])
        self.assertTrue(self.ready(analysis, ack=True)[0])
        self.assertIn("Confirm how unresolved", str(self.confirm(analysis)[0]))
        result = self.confirm(analysis, ack=True)[0]
        self.assertIn("1 selection(s) linked, 0 awaiting recordings or clips, 1 unresolved", str(result))
        statuses = [s["status"] for s in cr.STORE.selections(cr.STORE.list_batches()[0]["batch_id"])]
        self.assertEqual(statuses, ["matched_multiple", "invalid_interval"])

    def test_duplicates_conflicts_and_unchanged_settings(self):
        files = [(f"{LOC_A}-JP-hitro.txt", jp_table())]
        token, analysis, _, _ = self.upload(files)
        enabled, reason = self.ready(analysis, form=self.form(time_mode="file_offset"))
        self.assertFalse(enabled)
        self.assertIn("Advanced settings changed", reason)
        self.assertTrue(self.ready(analysis)[0])
        refused = self.confirm(analysis, form=self.form(time_mode="file_offset"))[0]
        self.assertIn("Settings changed since the analysis", str(refused))
        self.assertEqual(cr.STORE.list_batches(), [])
        self.confirm(analysis)
        old_id = cr.STORE.list_batches()[0]["batch_id"]

        token, analysis, review, out = self.upload(files)          # identical table again
        enabled, reason = self.ready(analysis)
        self.assertFalse(enabled)                                  # already imported: cannot duplicate
        self.assertIn("already imported", reason)
        self.assertIn("Skipped automatically", str(review))

        edited = [(f"{LOC_A}-JP-hitro.txt", jp_table().replace(b"Pelophylax", b"Pelophylax sp."))]
        token, analysis, review, out = self.upload(edited)
        self.assertIn("different content", str(review))
        self.assertEqual(list(analysis["conflicts"]), [f"{LOC_A}-JP-hitro.txt"])
        enabled, reason = self.ready(analysis)
        self.assertFalse(enabled)
        self.assertIn("Choose how to handle", reason)
        self.assertTrue(self.ready(analysis, conflicts={f"{LOC_A}-JP-hitro.txt": "keep"})[0])
        self.assertIn("Choose how to handle", str(self.confirm(analysis)[0]))
        self.confirm(analysis, conflicts={f"{LOC_A}-JP-hitro.txt": "replace"})
        status = {b["batch_id"]: b["status"] for b in cr.STORE.list_batches()}
        self.assertEqual(status[old_id], "superseded")
        self.assertEqual(sorted(status.values()), ["active", "superseded"])

    def test_advanced_one_annotator_for_all(self):
        files = [("annotations.txt", table(row(1, 900.0, 902.0, "20260320_185000.WAV", 1.0, "2026/3/20",
                                               "18:50:01", "Bufo bufo")))]
        form = self.form(annotator_source="fixed", location_mode="fixed", fixed_location=LOC_A)
        with triggered("raven-upload.contents"):
            out = self.fns["on_analyse"]([data_url(d) for _, d in files], 0, [], [n for n, _ in files], None, None,
                                         [], [], [], [], [], None, "Ana", *form)
        analysis = out[4]
        self.assertTrue(self.ready(analysis, form=form, fallback="Ana")[0])
        self.assertIn("Advanced settings changed", self.ready(analysis, form=form, fallback="Bo")[1])
        res = self.fns["on_confirm"](1, analysis, [], [], [[]], None, 0, None, "Ana", *form)[0]
        self.assertIn("Imported 1 table(s)", str(res))
        self.assertEqual(cr.STORE.list_batches()[0]["annotator"], "Ana")

    def test_revert_and_relink_from_advanced(self):
        files = [(f"{LOC_A}-JP.txt", table(row(1, 5.0, 6.0, "20260321_000000.WAV", 5.0, "2026/3/21", "0:00:05",
                                                "Bufo bufo")))]
        token, analysis, _, _ = self.upload(files)
        self.confirm(analysis)
        bid = cr.STORE.list_batches()[0]["batch_id"]
        self.assertEqual(cr.STORE.selections(bid)[0]["status"], "recording_not_found")

        # Partition gains the recording: dry run reports the change without writing.
        frame = frame_with_locations()
        extra = frame[frame.wav_file.str.endswith("20260320_185000.WAV")].copy()
        for col in ("wav_file", "rel_audio_path", "recording_id", "raw_clip_id", "semantic_clip_key"):
            extra[col] = extra[col].str.replace("20260320_185000", "20260321_000000", regex=False)
        import pandas as pd
        cr.CLIP_SOURCE = DataFrameClipSource(pd.concat([frame, extra]), "synthetic-ui-v2")
        dry = self.fns["relink_dry_run"](1)
        self.assertIn("1 would change", str(dry))
        self.assertEqual(cr.STORE.selections(bid)[0]["status"], "recording_not_found")
        out, signal = self.fns["do_relink"](1, 0)
        self.assertIn("1 changed", str(out))
        self.assertEqual(cr.STORE.selections(bid)[0]["status"], "matched_multiple")

        out, _ = self.fns["do_revert"](1, bid, 1)
        self.assertIn("Reverted", str(out))
        self.assertEqual(cr.STORE.list_batches()[0]["status"], "reverted")


def tearDownModule():
    shutil.rmtree(_DISKCACHE_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
