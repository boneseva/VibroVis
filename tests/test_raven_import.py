"""
Raven annotation import (Citimon) — parser, matcher, store and service tests.

All Raven tables and clip rows used here are SYNTHETIC (tests/fixtures/raven_synthetic).
Passing these tests does not demonstrate compatibility with real Raven exports.
"""
import hashlib
from contextlib import closing
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "raven_synthetic")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, FIXTURES)

from generate_fixtures import LOC_A, LOC_B, LOC_C, synthetic_clip_frame  # noqa: E402
from raven_import.clip_index import ClipIndex, DataFrameClipSource, PartitionClipSource  # noqa: E402
from raven_import.config import ImportOptions  # noqa: E402
from raven_import.matcher import match_table  # noqa: E402
from raven_import.parser import parse_raven_table  # noqa: E402
from raven_import.service import PreviewMismatch, commit_import, preview_import  # noqa: E402
from raven_import.store import AnnotationStore, SchemaVersionError, default_db_path  # noqa: E402

ANA = "SYNTHETIC_ana_sequence_utf8.Table.1.selections.txt"
BOSTJAN = "SYNTHETIC_bostjan_sequence_cp1250.Table.1.selections.txt"
SINGLE = "20260320_185000.Table.1.selections.txt"
FILENAME_ONLY = "SYNTHETIC_filename_only.txt"
NO_OFFSET = "SYNTHETIC_sequence_without_offset.txt"


def fixture(name):
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


def statuses(results, table):
    return {s.key: r.status for s, r in zip(table.selections, results)}


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = synthetic_clip_frame()
        cls.source = DataFrameClipSource(cls.clips, signature="synthetic-v1")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="raven_test_")
        self.store = AnnotationStore(os.path.join(self.tmp, "ann.sqlite3"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def preview(self, names_or_files, annotator="Ana", **opts):
        files = [(n, fixture(n)) if isinstance(n, str) else n for n in names_or_files]
        return files, preview_import(files, ImportOptions(**opts), annotator, self.source, self.store)

    def commit(self, names_or_files, annotator="Ana", supersede=None, **opts):
        files, prev = self.preview(names_or_files, annotator, **opts)
        return commit_import(files, ImportOptions(**opts), annotator, self.source, self.store, prev.token,
                             supersede=supersede, actor="test")


class TestRavenParser(_Base):
    def test_utf8_bom_crlf_and_duplicate_views(self):
        t = parse_raven_table(fixture(ANA), ANA, ImportOptions())
        self.assertIsNone(t.fatal_error)
        self.assertEqual(t.encoding, "utf-8")
        self.assertEqual(t.headers[0], "Selection")  # BOM stripped
        self.assertEqual(t.data_row_count, 20)
        self.assertEqual(len(t.selections), 10)
        self.assertEqual(t.duplicate_view_rows, 10)
        s1 = t.selections[0]
        self.assertEqual(s1.views, ["Waveform 1", "Spectrogram 1"])
        self.assertEqual(len(s1.raw_rows), 2)          # both views preserved
        self.assertIn("duplicate_views", s1.flags)
        self.assertEqual(s1.annotations["Notes"], "glasen klic, žaba v grmovju")
        self.assertEqual(t.column_map["file_offset"], "File Offset (s)")
        self.assertNotIn("Delta Time (s)", t.annotation_columns)  # measurement column
        self.assertEqual(t.annotation_columns, ["Species", "Call type", "Notes", "Synthetic"])

    def test_cp1250_auto_fallback_warns(self):
        t = parse_raven_table(fixture(BOSTJAN), BOSTJAN, ImportOptions())
        self.assertEqual(t.encoding, "cp1250")
        self.assertTrue(any("not valid UTF-8" in w for w in t.warnings))
        self.assertEqual(t.selections[0].annotations["Opomba"], "Žabji klic, slišno čez cesto")
        self.assertEqual(t.selections[1].annotations["Tip klica"], "zbor")

    def test_explicit_utf8_on_cp1250_is_fatal_not_silent(self):
        t = parse_raven_table(fixture(BOSTJAN), BOSTJAN, ImportOptions(encoding="utf-8"))
        self.assertIsNotNone(t.fatal_error)
        self.assertEqual(t.selections, [])

    def test_cp1252_choice_changes_characters(self):
        # Same bytes, different Windows code page: č (cp1250) becomes è (cp1252). The user must choose.
        t = parse_raven_table(fixture(BOSTJAN), BOSTJAN, ImportOptions(encoding="cp1252"))
        self.assertEqual(t.selections[2].annotations["Opomba"], "èuk v ozadju")

    def test_explicit_mapping_and_annotation_selection(self):
        data = ("Sel\tStart\tStop\tBegin File\tLabel\tExtra\n"
                "1\t1.0\t2.0\t20260320_184000.WAV\tHyla\tx\n").encode()
        opts = ImportOptions(column_mapping={"selection": "Sel", "begin_time": "Start", "end_time": "Stop",
                                             "channel": "", "view": "Nope"},
                             annotation_columns=["Label", "Missing"], primary_label_column="Label")
        t = parse_raven_table(data, "x.txt", opts)
        self.assertIsNone(t.fatal_error)
        self.assertEqual(t.column_map["begin_time"], "Start")
        self.assertIsNone(t.column_map["channel"])
        self.assertEqual(t.annotation_columns, ["Label"])
        self.assertEqual(t.primary_label_column, "Label")
        self.assertTrue(any("'Nope'" in w for w in t.warnings))
        self.assertTrue(any("Missing" in w for w in t.warnings))

    def test_missing_time_columns_is_fatal(self):
        t = parse_raven_table(b"Selection\tLabel\n1\tHyla\n", "x.txt", ImportOptions())
        self.assertIn("Begin/End time columns not found", t.fatal_error)

    def test_decimal_comma_configurable(self):
        data = "Selection\tBegin Time (s)\tEnd Time (s)\n1\t1,5\t2,5\n".encode()
        dot = parse_raven_table(data, "x.txt", ImportOptions())
        self.assertIn("begin_time", dot.selections[0].number_errors)
        opts = ImportOptions()
        opts.format.decimal_separator = ","
        comma = parse_raven_table(data, "x.txt", opts)
        self.assertEqual(comma.selections[0].numbers["begin_time"], 1.5)

    def test_malformed_row_kept_and_blocked(self):
        t = parse_raven_table(fixture(FILENAME_ONLY), FILENAME_ONLY, ImportOptions())
        bad = t.selections[2]
        self.assertEqual(bad.status, "malformed_row")
        self.assertIn("unexpected extra cell", bad.raw_rows[0]["__raw_line__"])

    def test_trailing_delimiter_tolerated(self):
        data = b"Selection\tBegin Time (s)\tEnd Time (s)\t\n1\t1.0\t2.0\t\n"
        t = parse_raven_table(data, "x.txt", ImportOptions())
        self.assertEqual(t.headers, ["Selection", "Begin Time (s)", "End Time (s)"])
        self.assertIsNone(t.selections[0].status)

    def test_missing_label_flag_uses_primary_column(self):
        t = parse_raven_table(fixture(ANA), ANA, ImportOptions(primary_label_column="Species"))
        flagged = [s.key for s in t.selections if "missing_label" in s.flags]
        self.assertEqual(flagged, ["5"])


class TestRavenMatcher(_Base):
    def _match(self, name, **opts):
        o = ImportOptions(**opts)
        t = parse_raven_table(fixture(name), name, o)
        index = self.source.load_index("perchv2", {LOC_A, LOC_B, LOC_C})
        return t, match_table(t, o, index, self.source.known_locations("perchv2"))

    def test_ana_statuses(self):
        t, res = self._match(ANA)
        self.assertEqual(statuses(res, t), {
            "1": "matched_single", "2": "matched_multiple", "3": "no_clip_overlap", "4": "spans_file_boundary",
            "5": "matched_multiple", "6": "invalid_interval", "7": "unknown_location_folder", "8": "view_conflict",
            "9": "missing_location", "10": "ambiguous_recording",
        })
        # Sequence-relative Begin Time 70.0 with File Offset 10.0 -> 10..12 s in the second file.
        r5 = res[4]
        self.assertEqual((r5.file_t0_s, r5.file_t1_s), (10.0, 12.0))
        self.assertTrue(r5.recording_id.endswith("20260320_185000_ch1.mp3::ch1"))
        self.assertEqual(r5.time_mode_used, "file_offset")

    def test_overlap_seconds_fractions_relations(self):
        _, res = self._match(ANA)
        links = {l.clip_time_s: l for l in res[1].links}  # selection 2: 11..14 s
        self.assertEqual(sorted(links), [7.5, 10.0, 12.5])
        self.assertEqual(links[10.0].relation, "selection_within_clip")
        self.assertAlmostEqual(links[10.0].overlap_s, 3.0)
        self.assertAlmostEqual(links[10.0].frac_of_clip, 0.6)
        self.assertAlmostEqual(links[10.0].frac_of_selection, 1.0)
        self.assertEqual(links[7.5].relation, "partial")
        self.assertAlmostEqual(links[7.5].overlap_s, 1.5)
        self.assertAlmostEqual(links[7.5].frac_of_selection, 0.5)

    def test_clip_within_selection_relation(self):
        data = ("Selection\tBegin Time (s)\tEnd Time (s)\tBegin Path\tFile Offset (s)\n"
                f"1\t0.0\t12.0\tD:\\x\\{LOC_A}\\20260320_185000.WAV\t0.0\n").encode()
        o = ImportOptions()
        t = parse_raven_table(data, "x.txt", o)
        res = match_table(t, o, self.source.load_index("perchv2", {LOC_A}), [LOC_A])
        rel = {l.clip_time_s: l.relation for l in res[0].links}
        self.assertEqual(rel[0.0], "clip_within_selection")
        self.assertEqual(rel[10.0], "partial")

    def test_cluster_num_duplicates_removed_before_matching(self):
        index = ClipIndex(self.clips[self.clips.location == LOC_A], "perchv2", "s")
        self.assertEqual(index.duplicate_rows_removed, index.rows_before_dedup // 2)
        single = ClipIndex(self.clips[(self.clips.location == LOC_A) & (self.clips.cluster_num == 10)], "perchv2", "s")
        o = ImportOptions()
        t = parse_raven_table(fixture(ANA), ANA, o)
        a = match_table(t, o, index, [LOC_A, LOC_B, LOC_C])
        b = match_table(t, o, single, [LOC_A, LOC_B, LOC_C])
        for ra, rb in zip(a, b):
            keys = [l.semantic_clip_key for l in ra.links]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(keys, [l.semantic_clip_key for l in rb.links])

    def test_filename_alone_never_identifies_recording(self):
        t, res = self._match(FILENAME_ONLY)
        self.assertEqual(res[0].status, "missing_location")
        _, res_a = self._match(FILENAME_ONLY, location_mode="fixed", fixed_location=LOC_A)
        _, res_b = self._match(FILENAME_ONLY, location_mode="fixed", fixed_location=LOC_B)
        self.assertEqual(res_a[0].status, "matched_single")
        self.assertEqual(res_b[0].status, "matched_single")
        self.assertNotEqual(res_a[0].recording_id, res_b[0].recording_id)
        self.assertEqual(res_a[1].status, "invalid_interval")  # decimal comma
        self.assertEqual(res_a[2].status, "malformed_row")

    def test_path_and_fixed_location_conflict_blocks(self):
        t, res = self._match(ANA, location_mode="fixed", fixed_location=LOC_B)
        self.assertEqual(res[0].status, "location_conflict")
        # Fallback only used where Begin Path is missing (selection 9).
        t, res = self._match(ANA, fixed_location=LOC_A)
        st = statuses(res, t)
        self.assertEqual(st["9"], "matched_multiple")
        self.assertEqual(st["1"], "matched_single")

    def test_single_file_table_name_inference_is_opt_in(self):
        t, res = self._match(SINGLE, fixed_location=LOC_A)
        self.assertEqual(set(statuses(res, t).values()), {"missing_source_file"})
        t, res = self._match(SINGLE, fixed_location=LOC_A, file_from_table_name=True)
        self.assertEqual(statuses(res, t), {"1": "matched_multiple", "2": "matched_multiple",
                                            "3": "exceeds_recording_duration"})
        self.assertEqual(res[0].time_mode_used, "single_file")

    def test_sequence_without_offset_is_blocked(self):
        for mode in ("auto", "single_file"):
            t, res = self._match(NO_OFFSET, time_mode=mode)
            self.assertEqual(set(statuses(res, t).values()), {"ambiguous_time_origin"}, mode)

    def test_file_offset_mode_requires_offset(self):
        t, res = self._match(SINGLE, fixed_location=LOC_A, file_from_table_name=True, time_mode="file_offset")
        self.assertEqual(set(statuses(res, t).values()), {"missing_file_offset"})

    def test_time_origin_conflict_in_single_file_mode(self):
        data = ("Selection\tBegin Time (s)\tEnd Time (s)\tBegin Path\tFile Offset (s)\n"
                f"1\t70.0\t71.0\tD:\\x\\{LOC_A}\\20260320_185000.WAV\t10.0\n").encode()
        o = ImportOptions(time_mode="single_file")
        t = parse_raven_table(data, "x.txt", o)
        res = match_table(t, o, self.source.load_index("perchv2", {LOC_A}), [LOC_A])
        self.assertEqual(res[0].status, "time_origin_conflict")

    def test_channel_not_found_and_missing_channel(self):
        data = ("Selection\tChannel\tBegin Time (s)\tEnd Time (s)\tBegin Path\n"
                f"1\t2\t1.0\t2.0\tD:\\x\\{LOC_A}\\20260320_185000.WAV\n").encode()
        o = ImportOptions()
        t = parse_raven_table(data, "x.txt", o)
        idx = self.source.load_index("perchv2", {LOC_A})
        self.assertEqual(match_table(t, o, idx, [LOC_A])[0].status, "channel_not_found")
        data2 = f"Selection\tBegin Time (s)\tEnd Time (s)\tBegin Path\n1\t1.0\t2.0\tD:\\x\\{LOC_A}\\20260320_185000.WAV\n".encode()
        o2 = ImportOptions(default_channel=None)
        t2 = parse_raven_table(data2, "x.txt", o2)
        self.assertEqual(match_table(t2, o2, idx, [LOC_A])[0].status, "missing_channel")


class TestRavenStore(_Base):
    def test_commit_preserves_original_data(self):
        out = self.commit([ANA], primary_label_column="Species")
        self.assertEqual(out[0]["outcome"], "imported")
        bid = out[0]["batch_id"]
        sels = self.store.selections(bid)
        self.assertEqual(len(sels), 10)                       # blocked rows kept too
        by_key = {s["raven_selection"]: s for s in sels}
        self.assertEqual(by_key["6"]["status"], "invalid_interval")
        self.assertEqual(by_key["6"]["begin_time_raw"], "80.0")
        self.assertIn("Waveform 1", by_key["1"]["raw_rows_json"])
        self.assertEqual(by_key["1"]["annotations"]["Notes"], "glasen klic, žaba v grmovju")
        self.assertEqual(sum(len(s["links"]) for s in sels), 6)
        self.assertEqual(by_key["2"]["links"][1]["relation"], "selection_within_clip")
        with closing(sqlite3.connect(self.store.db_path)) as c, c:
            raw, enc, primary = c.execute("SELECT raw_content, encoding, primary_label_column FROM import_batches").fetchone()
        self.assertEqual(raw, fixture(ANA))
        self.assertEqual((enc, primary), ("utf-8", "Species"))
        self.assertEqual([e["event"] for e in self.store.batch_events(bid)], ["imported"])

    def test_annotator_disagreement_kept_separately(self):
        self.commit([ANA])
        self.commit([BOSTJAN], annotator="Boštjan")
        rows = self.store.active_clip_annotations("Citimon", "perchv2")
        key = f"zabe::Citimon/{LOC_A}/20260320_184000_ch1.mp3::ch1::0.0"
        per_annotator = {(r["annotator"], r["column_name"]): r["value"] for r in rows if r["semantic_clip_key"] == key}
        self.assertEqual(per_annotator[("Ana", "Species")], "Hyla arborea")
        self.assertEqual(per_annotator[("Boštjan", "Vrsta")], "Rana dalmatina")
        self.assertEqual(sorted(self.store.list_annotators()), ["Ana", "Boštjan"])

    def test_reimport_is_idempotent(self):
        first = self.commit([ANA])
        files, prev = self.preview([ANA])
        self.assertEqual(prev.files[0].duplicate_of, first[0]["batch_id"])
        second = self.commit([ANA])
        self.assertEqual(second[0]["outcome"], "duplicate")
        self.assertEqual(len(self.store.list_batches()), 1)
        # Same file, same upload twice in one commit: also one batch.
        third = self.commit([(ANA, fixture(ANA)), ("copy_" + ANA, fixture(ANA))], annotator="Cene")
        self.assertEqual([o["outcome"] for o in third], ["imported", "duplicate"])

    def test_same_file_other_annotator_is_new_batch(self):
        self.commit([ANA])
        out = self.commit([ANA], annotator="Boštjan")
        self.assertEqual(out[0]["outcome"], "imported")

    def test_explicit_supersede(self):
        old = self.commit([ANA])[0]["batch_id"]
        edited = fixture(ANA).replace("Bufo bufo".encode(), "Bufo viridis".encode())
        files, prev = self.preview([(ANA, edited)])
        cands = prev.files[0].supersede_candidates
        self.assertEqual([c["batch_id"] for c in cands], [old])
        # Without an explicit choice both stay active.
        kept = commit_import(files, ImportOptions(), "Ana", self.source, self.store, prev.token)
        self.assertEqual({b["status"] for b in self.store.list_batches()}, {"active"})
        self.store.revert_batch(kept[0]["batch_id"], actor="test")
        files, prev = self.preview([(ANA, edited)])
        new = commit_import(files, ImportOptions(), "Ana", self.source, self.store, prev.token,
                            supersede={ANA: [old]})[0]["batch_id"]
        status = {b["batch_id"]: (b["status"], b["superseded_by"]) for b in self.store.list_batches()}
        self.assertEqual(status[old], ("superseded", new))
        self.assertEqual(status[new], ("active", None))
        self.assertEqual(len(self.store.selections(old)), 10)       # old data retained
        self.assertEqual([e["event"] for e in self.store.batch_events(old)], ["imported", "superseded"])

    def test_supersede_rejects_other_annotator_and_rolls_back(self):
        ana = self.commit([ANA])[0]["batch_id"]
        files, prev = self.preview([BOSTJAN, SINGLE], annotator="Boštjan")
        with self.assertRaises(ValueError):
            commit_import(files, ImportOptions(), "Boštjan", self.source, self.store, prev.token,
                          supersede={BOSTJAN: [ana]})
        self.assertEqual([b["batch_id"] for b in self.store.list_batches()], [ana])

    def test_revert(self):
        bid = self.commit([ANA])[0]["batch_id"]
        self.store.revert_batch(bid, actor="test", reason="wrong annotator")
        self.assertEqual(self.store.list_batches()[0]["status"], "reverted")
        self.assertEqual(self.store.active_clip_annotations("Citimon"), [])
        self.assertEqual(len(self.store.selections(bid)), 10)
        with self.assertRaises(ValueError):
            self.store.revert_batch(bid)
        again = self.commit([ANA])
        self.assertEqual(again[0]["outcome"], "imported")

    def test_preview_token_mismatch_writes_nothing(self):
        files, prev = self.preview([ANA])
        with self.assertRaises(PreviewMismatch):
            commit_import(files, ImportOptions(time_mode="file_offset"), "Ana", self.source, self.store, prev.token)
        self.assertEqual(self.store.list_batches(), [])

    def test_unparseable_file_reported_not_imported(self):
        files, prev = self.preview([BOSTJAN, ANA], encoding="utf-8")
        out = commit_import(files, ImportOptions(encoding="utf-8"), "Ana", self.source, self.store, prev.token)
        self.assertEqual([o["outcome"] for o in out], ["not_imported", "imported"])

    def test_persistence_schema_version_and_backup(self):
        bid = self.commit([ANA])[0]["batch_id"]
        reopened = AnnotationStore(self.store.db_path)
        self.assertEqual([b["batch_id"] for b in reopened.list_batches()], [bid])
        self.assertEqual(reopened.schema_version(), 3)
        backup = self.store.backup_to(os.path.join(self.tmp, "backup", "copy.sqlite3"))
        self.assertFalse(os.path.exists(backup + "-wal") or os.path.exists(backup + "-shm"))
        with closing(sqlite3.connect(backup)) as c:
            self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        self.assertEqual(len(AnnotationStore(backup).selections(bid)), 10)
        with closing(sqlite3.connect(self.store.db_path)) as c, c:
            c.execute("UPDATE schema_meta SET value='99' WHERE key='schema_version'")
        with self.assertRaises(SchemaVersionError):
            AnnotationStore(self.store.db_path).list_batches()

    def test_default_db_location_is_separate_and_ignored(self):
        old = os.environ.pop("VIBROVIS_ANNOTATIONS_DIR", None)
        try:
            path = default_db_path()
            self.assertEqual(os.path.dirname(path), os.path.join(ROOT, "cache", "annotations"))
            os.environ["VIBROVIS_ANNOTATIONS_DIR"] = self.tmp
            self.assertEqual(os.path.dirname(default_db_path()), self.tmp)
        finally:
            os.environ.pop("VIBROVIS_ANNOTATIONS_DIR", None)
            if old is not None:
                os.environ["VIBROVIS_ANNOTATIONS_DIR"] = old
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as f:
            self.assertIn("/cache/annotations/", f.read().splitlines())
        with open(os.path.join(ROOT, ".dockerignore"), encoding="utf-8") as f:
            self.assertIn("cache/annotations/", f.read().splitlines())

    def test_manual_label_store_untouched(self):
        manual_dir = os.path.join(ROOT, "cache", "manual_labels")

        def snapshot():
            out = {}
            for name in sorted(os.listdir(manual_dir)):
                with open(os.path.join(manual_dir, name), "rb") as f:
                    out[name] = hashlib.sha256(f.read()).hexdigest()
            return out

        before = snapshot()
        self.commit([ANA])
        self.commit([BOSTJAN], annotator="Boštjan")
        self.assertEqual(before, snapshot())
        for mod in ("parser", "matcher", "store", "service", "clip_index", "config"):
            with open(os.path.join(ROOT, "raven_import", f"{mod}.py"), encoding="utf-8") as f:
                self.assertNotIn("MANUAL_LABELS", f.read(), mod)


class TestRavenRealPartition(unittest.TestCase):
    """Optional check against the local Citimon partition; skipped when data is absent."""

    def setUp(self):
        self.source = PartitionClipSource(base_dir=os.path.join(ROOT, "data"))
        if "perchv2" not in self.source.models() or not self.source.known_locations("perchv2"):
            self.skipTest("Citimon partition not available")

    def test_real_recording_identity_and_dedup(self):
        loc = "2026_05-C35"  # 2026-06-C35 (a duplicate encoding of this folder) is excluded by the registry
        self.assertIn(loc, self.source.known_locations("perchv2"))
        data = ("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tBegin File\tBegin Path\tFile Offset (s)\tSpecies\n"
                f"1\tSpectrogram 1\t1\t0.5\t2.0\t20260511_184000.WAV\tE:\\{loc}\\20260511_184000.WAV\t0.5\tSYNTHETIC\n").encode()
        prev = preview_import([("synthetic.txt", data)], ImportOptions(), "Synthetic", self.source)
        self.assertEqual(prev.errors, [])
        res = prev.files[0].results[0]
        self.assertEqual(res.status, "matched_single")
        self.assertEqual(res.links[0].semantic_clip_key, f"zabe::Citimon/{loc}/20260511_184000_ch1.mp3::ch1::0.0")
        self.assertGreater(prev.duplicate_clip_rows_removed, 0)


if __name__ == "__main__":
    unittest.main()
