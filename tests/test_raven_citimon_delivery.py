"""
Citimon Raven delivery format (2026-10): table-name metadata, verified file-date repair, location aliases,
truncation at the recording end, species interpretation, re-linking and the v1 -> v2 schema migration.

Tables built here are SYNTHETIC copies of the real delivery's layout. TestRealDelivery runs only when the
biologists' CITI-MON-anotacije folder is present locally and never writes to any database.
"""
import glob
import pathlib
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "raven_synthetic")
for p in (ROOT, FIXTURES):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
from generate_fixtures import LOC_A, synthetic_clip_frame  # noqa: E402
from raven_import.clip_index import DataFrameClipSource, PartitionClipSource  # noqa: E402
from raven_import.config import ImportOptions  # noqa: E402
from raven_import.interpret import interpret_label  # noqa: E402
from raven_import.matcher import clock_offset_discrepancy, match_table, resolve_source_stem  # noqa: E402
from raven_import.parser import parse_raven_table, parse_table_meta  # noqa: E402
from raven_import.service import commit_import, preview_import, relink_batches  # noqa: E402
from raven_import.store import SCHEMA_V1, AnnotationStore, _execute_statements  # noqa: E402

HEADER = ("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tBegin File\t"
          "File Offset (s)\tBegin Date\tBegin Clock Time\tAnnotation\r\n")
VOCAB = ["Bufo bufo", "Rana dalmatina", "Rana temporaria", "Hyla arborea", "Bombina variegata", "Pelophylax sp."]


def row(sel, begin, end, begin_file, offset, date, clock, label):
    return f"{sel}\tSpectrogram 1\t1\t{begin}\t{end}\t500.0\t3000.0\t{begin_file}\t{offset}\t{date}\t{clock}\t{label}\r\n"


def table(*rows):
    return (HEADER + "".join(rows)).encode("utf-8")


def frame_with_locations(*extra_locations):
    """Synthetic clips: LOC_A plus copies of its 20260320_185000 recording at other locations."""
    base = synthetic_clip_frame()
    frames = [base]
    rec = base[base.wav_file.str.endswith("20260320_185000.WAV")]
    for loc in extra_locations:
        copy = rec.copy()
        for col in ("wav_file", "rel_audio_path", "recording_id", "raw_clip_id", "semantic_clip_key"):
            copy[col] = copy[col].str.replace(LOC_A, loc, regex=False)
        copy["location"] = copy["microlocation"] = loc
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


class TestInterpretation(unittest.TestCase):
    def test_rules(self):
        cases = {
            "Bufo bufo": ("Bufo bufo", False, "exact"),
            "bufo bufo": ("Bufo bufo", False, "case_normalized"),
            "Pelophylax": ("Pelophylax sp.", False, "synonym"),
            "Pelophylax sp.? ": ("Pelophylax sp.", True, "exact"),
            "Rana temporaria?": ("Rana temporaria", True, "exact"),
            "?": (None, True, "uncertain_only"),
            "": (None, False, "empty"),
            "govor": (None, False, "unrecognized"),
            "žerjav?": (None, True, "unrecognized"),
            "Rana arvalis": (None, False, "unrecognized"),
        }
        for value, expected in cases.items():
            it = interpret_label(value, VOCAB, {"Pelophylax": "Pelophylax sp."})
            self.assertEqual((it.species, it.uncertain, it.rule), expected, value)


class TestTableMeta(unittest.TestCase):
    def test_pattern(self):
        fmt = ImportOptions().format
        self.assertEqual(parse_table_meta("2026_03-C02-JP-hitro.txt", fmt),
                         {"location": "2026_03-C02", "annotator": "JP", "quick_pass": True})
        self.assertEqual(parse_table_meta("2026_03-C04-SS.txt", fmt),
                         {"location": "2026_03-C04", "annotator": "SS", "quick_pass": False})
        self.assertEqual(parse_table_meta("2026_03-C19-1-JP.txt", fmt)["location"], "2026_03-C19-1")
        self.assertIsNone(parse_table_meta("annotations.csv", fmt))


class _DeliveryBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="raven_delivery_")
        self.store = AnnotationStore(os.path.join(self.tmp, "ann.sqlite3"))
        self.source = DataFrameClipSource(frame_with_locations("2026_03-R26", "2026_04-r26"), "synthetic-v2")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def opts(**kw):
        base = dict(location_mode="table_name", annotator_from_table_name=True, primary_label_column="Annotation")
        base.update(kw)
        return ImportOptions(**base)

    def match(self, name, data, **kw):
        o = self.opts(**kw)
        t = parse_raven_table(data, name, o)
        known = self.source.known_locations("perchv2")
        return t, match_table(t, o, self.source.load_index("perchv2", set(known)), known)


class TestDateRepair(_DeliveryBase):
    def test_unpadded_date_repaired_only_when_begin_date_confirms(self):
        data = table(
            row(1, 900.0, 901.0, "2026320_185000.WAV", 1.0, "2026/3/20", "18:50:01.0000", "Bufo bufo"),
            row(2, 960.0, 961.0, "2026320_185000.WAV", 1.0, "2026/3/21", "18:50:01.0000", "Bufo bufo"),
            row(3, 990.0, 991.0, "20260320_185000.WAV", 1.0, "2026/3/22", "18:50:01.0000", "Bufo bufo"),
        )
        t, res = self.match(f"{LOC_A}-JP-hitro.txt", data)
        self.assertEqual(res[0].status, "matched_single")
        self.assertEqual(res[0].source_stem, "20260320_185000")
        self.assertIn("begin_file_date_repaired", res[0].flags)
        self.assertEqual(res[1].status, "source_file_date_mismatch")
        self.assertEqual(res[2].status, "source_file_date_mismatch")  # 8-digit names are checked too
        self.assertEqual(t.selections[0].fields["begin_file"], "2026320_185000.WAV")  # original kept

        _, off = self.match(f"{LOC_A}-JP-hitro.txt", data, repair_unpadded_file_dates=False)
        self.assertEqual(off[0].status, "recording_not_found")

    def test_unverifiable_without_begin_date(self):
        data = ("Selection\tBegin Time (s)\tEnd Time (s)\tBegin File\tFile Offset (s)\tAnnotation\n"
                "1\t1.0\t2.0\t2026320_185000.WAV\t1.0\tBufo bufo\n").encode()
        _, res = self.match(f"{LOC_A}-JP.txt", data)
        self.assertEqual(res[0].status, "source_file_date_unverified")

    def test_midnight_file_and_clock_jitter(self):
        # Pattern seen in real exports: clock ~0.47 s behind file time + offset, so for a midnight file
        # clock - offset falls on the previous day. Values here are synthetic.
        data = table(row(10, 5012.5, 5020.0, "2026228_000000.WAV", 12.5, "2026/2/28", "0:00:12.0300",
                         "Rana temporaria"))
        o = self.opts()
        t = parse_raven_table(data, f"{LOC_A}-JP-hitro.txt", o)
        stem, flags, problem = resolve_source_stem(t.selections[0], t, o)
        self.assertIsNone(problem)
        self.assertEqual(stem, "20260228_000000")
        self.assertAlmostEqual(clock_offset_discrepancy(t.selections[0], stem), -0.47, places=3)


class TestAliasesAndTruncation(_DeliveryBase):
    def test_alias_is_explicit_and_location_specific(self):
        data = table(row(1, 5.0, 6.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Bufo bufo"))
        _, res = self.match("2026_03-r26-JP-hitro.txt", data)
        self.assertEqual(res[0].status, "unknown_location")  # never matched ignoring case
        aliases = {"2026_03-r26": "2026_03-R26"}
        _, res = self.match("2026_03-r26-JP-hitro.txt", data, location_aliases=aliases)
        self.assertEqual((res[0].status, res[0].location), ("matched_multiple", "2026_03-R26"))
        self.assertIn("location_alias:2026_03-r26->2026_03-R26", res[0].flags)
        self.assertEqual(res[0].location_source, "table_name+alias")
        _, res = self.match("2026_04-r26-JP-hitro.txt", data, location_aliases=aliases)
        self.assertEqual(res[0].location, "2026_04-r26")
        self.assertEqual(res[0].flags, [])

    def test_truncation_keeps_original_end(self):
        data = table(
            row(1, 100.0, 104.0, "20260320_185000.WAV", 58.0, "2026/3/20", "18:50:58", "Bufo bufo"),
            row(2, 200.0, 201.0, "20260320_185000.WAV", 61.0, "2026/3/20", "18:51:01", "Bufo bufo"),
        )
        _, res = self.match(f"{LOC_A}-JP.txt", data)
        self.assertEqual(res[0].status, "exceeds_recording_duration")
        t, res = self.match(f"{LOC_A}-JP.txt", data, truncate_at_recording_end=True)
        r = res[0]
        self.assertEqual((r.file_t0_s, r.file_t1_s, r.file_t1_original_s), (58.0, 60.0, 62.0))
        self.assertIn("truncated_at_recording_end", r.flags)
        self.assertTrue(all(l.clip_time_s + l.clip_duration_s > 58.0 for l in r.links))
        self.assertAlmostEqual(max(l.overlap_s for l in r.links), 2.0)
        self.assertEqual(res[1].status, "exceeds_recording_duration")  # starts after the end: still blocked
        self.assertEqual(t.selections[0].fields["end_time"], "104.0")

    def test_actual_audio_duration_used_when_known(self):
        rid = f"zabe::Citimon/{LOC_A}/20260320_185000_ch1.mp3::ch1"
        self.source.durations = {rid: 55.0}
        data = table(row(1, 100.0, 104.0, "20260320_185000.WAV", 53.0, "2026/3/20", "18:50:53", "Bufo bufo"))
        _, res = self.match(f"{LOC_A}-JP.txt", data)
        self.assertEqual(res[0].status, "exceeds_recording_duration")   # 57 s > 55 s although < 60 s
        self.assertIn("recording_duration:55s", res[0].flags)
        _, res = self.match(f"{LOC_A}-JP.txt", data, truncate_at_recording_end=True)
        self.assertEqual((res[0].file_t1_s, res[0].file_t1_original_s), (55.0, 57.0))
        _, res = self.match(f"{LOC_A}-JP.txt", data, recording_duration_from_audio=False)
        self.assertEqual(res[0].status, "matched_multiple")              # configured 60 s


class TestAutoLocationAndOverrides(_DeliveryBase):
    """location_mode='auto' composes the existing table-name / Begin Path / fallback rules."""

    def path_table(self, folder):
        return ("Selection\tBegin Time (s)\tEnd Time (s)\tBegin Path\tFile Offset (s)\tAnnotation\n"
                f"1\t5.0\t6.0\tD:\\x\\{folder}\\20260320_185000.WAV\t5.0\tBufo bufo\n").encode()

    def test_auto_mode_branches(self):
        data = table(row(1, 5.0, 6.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Bufo bufo"))
        cases = [
            (f"{LOC_A}-JP.txt", data, ("matched_multiple", LOC_A, "table_name")),
            ("2026_05-C77-JP.txt", data, ("unknown_location", None, None)),
            ("annotations.txt", data, ("missing_location", None, None)),
            ("annotations.txt", self.path_table(LOC_A), ("matched_multiple", LOC_A, "path")),
            ("notes-JP.txt", self.path_table(LOC_A), ("matched_multiple", LOC_A, "path")),  # name not a location
            ("2026_03-R26-JP.txt", self.path_table(LOC_A), ("location_conflict", None, None)),
        ]
        for name, raw, expected in cases:
            _, res = self.match(name, raw, location_mode="auto")
            self.assertEqual((res[0].status, res[0].location, res[0].location_source), expected, name)

    def test_override_is_explicit_and_checked(self):
        data = table(row(1, 5.0, 6.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Bufo bufo"))
        _, res = self.match("annotations.txt", data, location_mode="auto",
                            location_overrides={"annotations.txt": LOC_A})
        self.assertEqual((res[0].status, res[0].location_source), ("matched_multiple", "override"))
        self.assertIn("location_override", res[0].flags)
        _, res = self.match("2026_03-R26-JP.txt", data, location_mode="auto",
                            location_overrides={"2026_03-R26-JP.txt": LOC_A})
        self.assertEqual(res[0].status, "location_conflict")       # override never silently beats the name
        _, res = self.match("annotations.txt", data, location_mode="auto",
                            location_overrides={"other.txt": LOC_A})
        self.assertEqual(res[0].status, "missing_location")         # keyed by exact table name

    def test_annotator_override_and_primary_label_auto(self):
        data = table(row(1, 5.0, 6.0, "20260320_185000.WAV", 5.0, "2026/3/20", "18:50:05", "Pelophylax"))
        opts = self.opts(location_mode="auto", primary_label_column=None, primary_label_auto=True,
                         annotator_overrides={"annotations.txt": "SS"}, location_overrides={"annotations.txt": LOC_A})
        prev = preview_import([("annotations.txt", data)], opts, "", self.source, self.store)
        fp = prev.files[0]
        self.assertEqual((fp.annotator, fp.error, fp.table.primary_label_column), ("SS", None, "Annotation"))
        no_auto = parse_raven_table(data, "annotations.txt", ImportOptions())
        self.assertIsNone(no_auto.primary_label_column)            # default behaviour unchanged

    def test_auto_mode_identity_includes_table_location(self):
        empty = HEADER.encode("utf-8")
        pair = [(f"{LOC_A}-JP.txt", empty), ("2026_03-R26-JP.txt", empty)]
        opts = self.opts(location_mode="auto")
        prev = preview_import(pair, opts, "", self.source, self.store)
        out = commit_import(pair, opts, "", self.source, self.store, prev.token)
        self.assertEqual([o["outcome"] for o in out], ["imported", "imported"])
        # A table imported earlier in 'table_name' mode is recognised as the same import in 'auto' mode.
        prev = preview_import(pair[:1], self.opts(location_mode="table_name"), "", self.source, self.store)
        self.assertIsNotNone(prev.files[0].duplicate_of)


class TestDeliveryImportAndRelink(_DeliveryBase):
    def files(self):
        jp = table(row(1, 900.0, 902.0, "2026320_185000.WAV", 1.0, "2026/3/20", "18:50:01", "Pelophylax"),
                   row(2, 950.0, 951.0, "2026320_185000.WAV", 30.0, "2026/3/20", "18:50:30", "Rana dalmatina?"))
        ss = table(row(1, 10.0, 12.0, "2026320_185000.WAV", 1.0, "2026/3/20", "18:50:01", "Bufo bufo"))
        return [(f"{LOC_A}-JP-hitro.txt", jp), (f"{LOC_A}-SS.txt", ss)]

    def test_per_file_annotator_meta_and_interpretations(self):
        prev = preview_import(self.files(), self.opts(), "", self.source, self.store)
        self.assertEqual(prev.errors, [])
        self.assertEqual([fp.annotator for fp in prev.files], ["JP", "SS"])
        out = commit_import(self.files(), self.opts(), "", self.source, self.store, prev.token)
        self.assertEqual([o["outcome"] for o in out], ["imported", "imported"])
        self.assertEqual(sorted(self.store.list_annotators()), ["JP", "SS"])
        with closing(sqlite3.connect(self.store.db_path)) as c:
            meta = dict(c.execute("SELECT source_filename, table_meta_json FROM import_batches").fetchall())
        self.assertIn('"quick_pass": true', meta[f"{LOC_A}-JP-hitro.txt"])
        sels = self.store.selections(out[0]["batch_id"])
        self.assertEqual(sels[0]["annotations"]["Annotation"], "Pelophylax")  # original untouched
        self.assertEqual(sels[0]["interpretations"]["Annotation"]["species"], "Pelophylax sp.")
        self.assertEqual(sels[1]["interpretations"]["Annotation"]["uncertain"], 1)
        rows = self.store.active_clip_annotations("Citimon")
        self.assertTrue({r["interpreted_species"] for r in rows} >= {"Pelophylax sp.", "Bufo bufo"})

        # Byte-identical header-only tables for different locations are distinct imports (seen in a real
        # delivery); the same table again is still a duplicate.
        empty = HEADER.encode("utf-8")
        pair = [(f"{LOC_A}-JP-hitro.txt", empty), ("2026_03-R26-JP-hitro.txt", empty)]
        prev = preview_import(pair, self.opts(), "", self.source, self.store)
        out2 = commit_import(pair, self.opts(), "", self.source, self.store, prev.token)
        self.assertEqual([o["outcome"] for o in out2], ["imported", "imported"])
        prev = preview_import(pair[:1], self.opts(), "", self.source, self.store)
        self.assertEqual(commit_import(pair[:1], self.opts(), "", self.source, self.store, prev.token)[0]["outcome"],
                         "duplicate")

        # Unmatched table name: annotator cannot be taken from it -> file not imported, others unaffected.
        bad = [("annotations.txt", self.files()[0][1])]
        prev = preview_import(bad, self.opts(), "", self.source, self.store)
        self.assertFalse(prev.files[0].importable)

    def test_relink_after_partition_rebuild(self):
        empty = DataFrameClipSource(frame_with_locations().iloc[0:0].copy(), "empty")
        empty.known_locations = lambda model: [LOC_A]   # location known, but no clips yet
        prev = preview_import(self.files(), self.opts(), "", empty, self.store)
        out = commit_import(self.files(), self.opts(), "", empty, self.store, prev.token)
        bid = out[0]["batch_id"]
        self.assertEqual({s["status"] for s in self.store.selections(bid)}, {"recording_not_found"})

        dry = relink_batches(self.store, self.source, dry_run=True)
        self.assertEqual({r["outcome"] for r in dry}, {"dry_run"})
        self.assertEqual({s["status"] for s in self.store.selections(bid)}, {"recording_not_found"})

        self.store.revert_batch(out[1]["batch_id"])
        reports = relink_batches(self.store, self.source, batch_ids=[bid, out[1]["batch_id"]], actor="test")
        self.assertEqual([r["outcome"] for r in reports], ["relinked", "skipped"])
        sels = self.store.selections(bid)
        self.assertEqual([s["status"] for s in sels], ["matched_multiple", "matched_multiple"])
        self.assertTrue(all(l["partition_signature"] == "synthetic-v2" for s in sels for l in s["links"]))
        self.assertEqual(sels[0]["begin_file"], "2026320_185000.WAV")
        self.assertIn("begin_file_date_repaired", sels[0]["flags_json"])
        self.assertEqual([e["event"] for e in self.store.batch_events(bid)], ["imported", "relinked"])
        again = relink_batches(self.store, self.source, batch_ids=[bid])
        self.assertEqual(sum(len(s["links"]) for s in self.store.selections(bid)), again[0]["links"])


class TestSchemaMigration(unittest.TestCase):
    def test_v1_database_migrates_in_place(self):
        tmp = tempfile.mkdtemp(prefix="raven_migrate_")
        try:
            path = os.path.join(tmp, "v1.sqlite3")
            with closing(sqlite3.connect(path, isolation_level=None)) as c:
                _execute_statements(c, SCHEMA_V1)
                c.execute("INSERT INTO schema_meta VALUES ('schema_version', '1')")
                c.execute("INSERT INTO annotators(name, created_at) VALUES ('Ana', 'x')")
            store = AnnotationStore(path)
            self.assertEqual(store.schema_version(), 3)
            self.assertEqual(store.list_annotators(), ["Ana"])
            with closing(sqlite3.connect(path)) as c:
                cols = {r[1] for r in c.execute("PRAGMA table_info(selections)")}
                self.assertIn("file_t1_original_s", cols)
                self.assertTrue(c.execute("SELECT 1 FROM sqlite_master WHERE name='annotation_interpretations'").fetchone())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSchemaV3Backfill(unittest.TestCase):
    def test_v2_table_name_batches_get_location_scope(self):
        tmp = tempfile.mkdtemp(prefix="raven_v3_")
        try:
            path = os.path.join(tmp, "v2.sqlite3")
            from raven_import.store import MIGRATIONS
            with closing(sqlite3.connect(path, isolation_level=None)) as c:
                _execute_statements(c, SCHEMA_V1)
                _execute_statements(c, MIGRATIONS[1])
                c.execute("INSERT INTO schema_meta VALUES ('schema_version', '2')")
                c.execute("INSERT INTO annotators(name, created_at) VALUES ('JP', 'x')")
                for bid, loc, mode in (("a", "2026_02-C13", "table_name"), ("b", "2026_03-C01", "path")):
                    c.execute(
                        "INSERT INTO import_batches(batch_id, project, annotator_id, source_filename, file_sha256, "
                        "file_size, raw_content, headers_json, column_map_json, annotation_columns_json, options_json, "
                        "model_name, partition_signature, stats_json, warnings_json, created_at, status, table_meta_json) "
                        "VALUES (?, 'Citimon', 1, ?, ?, 0, x'', '[]', '{}', '[]', ?, 'perchv2', 's', '{}', '[]', 'x', "
                        "'active', ?)",
                        (bid, f"{loc}-JP.txt", f"hash-{bid}", f'{{"location_mode": "{mode}"}}', f'{{"location": "{loc}"}}'))
            store = AnnotationStore(path)
            self.assertEqual(store.schema_version(), 3)
            with closing(sqlite3.connect(path)) as c:
                scopes = dict(c.execute("SELECT batch_id, identity_scope FROM import_batches").fetchall())
            self.assertEqual(scopes, {"a": "location:2026_02-C13", "b": ""})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestRealDelivery(unittest.TestCase):
    """Read-only checks on the real delivery; skipped when the folder or partition is absent."""
    FOLDER = os.path.join(ROOT, "CITI-MON-anotacije")

    def test_real_delivery_dry_run(self):
        files = sorted(glob.glob(os.path.join(self.FOLDER, "*.txt")))
        source = PartitionClipSource(base_dir=os.path.join(ROOT, "data"))
        if not files or not source.known_locations("perchv2"):
            self.skipTest("real delivery or Citimon partition not available")
        data = [(os.path.basename(f), pathlib.Path(f).read_bytes()) for f in files]
        opts = ImportOptions(location_mode="table_name", annotator_from_table_name=True,
                             truncate_at_recording_end=True, primary_label_column="Annotation",
                             location_aliases={"2026_03-r26": "2026_03-R26", "2026_03-r28": "2026_03-R28"})
        prev = preview_import(data, opts, "", source, store=None)
        self.assertEqual(prev.errors, [])
        self.assertTrue(all(fp.importable for fp in prev.files))
        self.assertEqual({fp.annotator for fp in prev.files}, {"JP", "SS"})
        status, flags = {}, {}
        for fp in prev.files:
            for k, v in fp.stats["status_counts"].items():
                status[k] = status.get(k, 0) + v
            for k, v in fp.stats["flag_counts"].items():
                flags[k] = flags.get(k, 0) + v
        self.assertEqual(sum(status.values()), 1183)
        self.assertEqual(flags.get("begin_file_date_repaired"), 770)
        self.assertNotIn("source_file_date_mismatch", status)
        self.assertNotIn("source_file_date_unverified", status)
        matched = sum(status.get(k, 0) for k in ("matched_single", "matched_multiple", "no_clip_overlap"))
        self.assertGreaterEqual(matched, 86)  # grows when the partition gains positions for annotated files


if __name__ == "__main__":
    unittest.main()
