"""
Citimon hierarchy: registry (config/citimon_registry.json), cache builder, Raven clip index and map.

Builder tests run on an isolated temporary data tree with a SYNTHETIC registry and master TSV
(VIBROVIS_CITIMON_REGISTRY points at it); they never touch data/, cache/ or the real registry.
The TestRealRegistry checks read config/citimon_registry.json only.
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import build_cache  # noqa: E402
import citimon_map  # noqa: E402
import citimon_registry  # noqa: E402
from raven_import.clip_index import DataFrameClipSource, PartitionClipSource  # noqa: E402

TSV_HEADER = "file_name\tday\tstart_time\tchannel\tclip_time\tclip_duration\tembedding\tmodel_name\tcluster_id\tnum_clusters\n"


def synthetic_registry():
    """Two recorders: X01 (two positions, one folder without coordinates), X02 (one position).
    2026_03-X09 is an excluded duplicate; 2026_03-X01-1 is a former name of 2026_03-X01."""
    return {
        "schema": "vibrovis.citimon.registry", "schema_version": "1.0", "project": "Citimon",
        "unassigned_microlocation": "Unassigned",
        "recorders": {
            "X01": {"microlocations": ["X01-P1", "X01-P2"], "source_folders": ["2026_03-X01", "2026_04-X01", "2026_05-X01"]},
            "X02": {"microlocations": ["X02-P1"], "source_folders": ["2026_03-X02"]},
            "X09": {"microlocations": [], "source_folders": ["2026_03-X09"]},
        },
        "microlocations": {
            "X01-P1": {"recorder": "X01", "latitude": 46.05, "longitude": 14.5, "source_folders": ["2026_03-X01"]},
            "X01-P2": {"recorder": "X01", "latitude": 46.06, "longitude": 14.51, "source_folders": ["2026_04-X01"]},
            "X02-P1": {"recorder": "X02", "latitude": 45.6, "longitude": 13.8, "source_folders": ["2026_03-X02"]},
        },
        "source_folders": {
            "2026_03-X01": {"recorder": "X01", "microlocation": "X01-P1", "aliases": ["2026_03-X01-1"]},
            "2026_04-X01": {"recorder": "X01", "microlocation": "X01-P2", "aliases": []},
            "2026_05-X01": {"recorder": "X01", "microlocation": None, "aliases": []},
            "2026_03-X02": {"recorder": "X02", "microlocation": "X02-P1", "aliases": []},
            "2026_03-X09": {"recorder": "X09", "microlocation": None, "aliases": []},
        },
        "excluded_source_folders": {"2026_03-X09": "duplicate"},
        "name_index": {"2026_03-X01-1": "2026_03-X01"},
        "microlocation_id_ledger": {"X01": {"next_free": "X01-P3"}, "X02": {"next_free": "X02-P2"}},
    }


class TestRegistryModule(unittest.TestCase):
    def setUp(self):
        self.reg = citimon_registry.CitimonRegistry(synthetic_registry())

    def test_folder_resolution(self):
        i = self.reg.folder_info("2026_04-X01")
        self.assertEqual((i.recorder, i.microlocation, i.latitude), ("X01", "X01-P2", 46.06))
        # Former name and spelling variants resolve to the canonical folder.
        self.assertEqual(self.reg.folder_info("2026_03-X01-1").source_folder, "2026_03-X01")
        self.assertEqual(self.reg.folder_info("2026-03-x01").microlocation, "X01-P1")
        # No coordinates -> Unassigned, no coordinates.
        u = self.reg.folder_info("2026_05-X01")
        self.assertEqual((u.microlocation, u.latitude), ("Unassigned", None))
        self.assertIsNone(self.reg.folder_info("2026_03-X77"))

    def test_exclusion_is_literal_and_spelling_insensitive(self):
        self.assertTrue(self.reg.is_excluded("2026_03-X09"))
        self.assertTrue(self.reg.is_excluded("2026-03-x09"))
        self.assertFalse(self.reg.is_excluded("2026_03-X01"))
        self.assertFalse(self.reg.is_excluded("2026_03-X01-1"))

    def test_resolve_display(self):
        self.assertEqual(self.reg.resolve_display("X01"), ("X01", ["X01-P1", "X01-P2"]))
        self.assertEqual(self.reg.resolve_display("2026_04-X01"), ("X01", ["X01-P2"]))
        self.assertEqual(self.reg.resolve_display("2026_05-X01"), ("X01", []))
        self.assertEqual(self.reg.resolve_display("nope"), (None, []))

    def test_validate_clean_and_broken(self):
        self.assertEqual(self.reg.validate(), [])
        bad = synthetic_registry()
        bad["microlocations"]["X01-P3"] = dict(bad["microlocations"]["X01-P1"])          # same recorder+coords
        bad["source_folders"]["2026_03-X02"]["microlocation"] = "X01-P1"                  # other recorder's ID
        bad["name_index"]["ghost"] = "2026_09-X99"                                        # broken alias
        bad["microlocation_id_ledger"]["X01"]["next_free"] = "X01-P2"                     # would reuse an ID
        errs = " | ".join(citimon_registry.CitimonRegistry(bad).validate())
        for needle in ("duplicates X01-P1", "belongs to another recorder", "unknown folder", "reuse"):
            self.assertIn(needle, errs)

    def test_sun_times(self):
        # Master TSV value for 2026_03-C08 on 2026-03-17 is 06:11 / 18:10 (CET).
        rise, set_ = citimon_registry.sun_times(dt.date(2026, 3, 17), 46.24, 14.36)
        mins = lambda s: int(s[:2]) * 60 + int(s[3:])
        self.assertLessEqual(abs(mins(rise) - (6 * 60 + 11)), 3)
        self.assertLessEqual(abs(mins(set_) - (18 * 60 + 10)), 3)
        # Daylight saving: last Sunday of March 2026 is the 29th -> clocks jump one hour.
        r28, _ = citimon_registry.sun_times(dt.date(2026, 3, 28), 46.05, 14.5)
        r29, _ = citimon_registry.sun_times(dt.date(2026, 3, 29), 46.05, 14.5)
        self.assertGreater(mins(r29) - mins(r28), 55)
        self.assertEqual(citimon_registry._central_europe_utc_offset(dt.date(2026, 10, 24)), 2)
        self.assertEqual(citimon_registry._central_europe_utc_offset(dt.date(2026, 10, 25)), 1)


class TestRealRegistry(unittest.TestCase):
    """The committed registry: structure and the decisions it encodes."""

    @classmethod
    def setUpClass(cls):
        cls.reg = citimon_registry.load_registry(citimon_registry.DEFAULT_PATH)

    def test_present_and_consistent(self):
        self.assertIsNotNone(self.reg, "config/citimon_registry.json missing")
        self.assertEqual(self.reg.validate(), [])

    def test_decisions(self):
        r = self.reg
        # C19: both renamed folders and both former names -> one recorder, one microlocation.
        for name in ("2026_03-C19", "2026_04-C19", "2026_03-C19-1", "2026_03-C19-2"):
            self.assertEqual(r.folder_info(name).microlocation, "C19-P1", name)
        # R35 is a former name of C35 and its folder is excluded; the other duplicates too.
        self.assertEqual(r.folder_info("2026_03-R35").source_folder, "2026_03-C35")
        for name in ("2026_03-R35", "2026-06-C35", "2026_02-C49"):
            self.assertTrue(r.is_excluded(name), name)
        self.assertFalse(r.is_excluded("2026_03-C35"))
        self.assertFalse(r.is_excluded("2026_03-C49"))
        # C47 keeps two positions; C35 positions are distinct.
        self.assertEqual(r.recorder_microlocations("C47"), ["C47-P1", "C47-P2"])
        self.assertEqual(r.folder_info("2026_05-C35").microlocation, "C35-P2")
        # Raven table names without coordinates resolve to their recorder only.
        self.assertEqual((r.folder_info("2026_04-C08").recorder, r.folder_info("2026_04-C08").microlocation),
                         ("C08", "Unassigned"))
        # Lower-case Raven spellings resolve.
        self.assertEqual(r.folder_info("2026_03-r26").recorder, "R26")


class _TempTree(unittest.TestCase):
    """Isolated data tree: <tmp>/data/recordings/Citimon/<folder>/{audio, positions/}."""

    FOLDERS = {
        # folder -> (stem, day) ; cataloged in the master TSV?
        "2026_03-X01": [("20260317_223000", True)],
        "2026_04-X01": [("20260410_060000", False)],
        "2026_05-X01": [("20260512_120000", False)],
        "2026_03-X02": [("20260317_223000", False)],
        "2026_03-X09": [("20260317_223000", False)],   # excluded duplicate
    }

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vv_citimon_")
        self.data = os.path.join(self.tmp, "data")
        self.cit = os.path.join(self.data, "recordings", "Citimon")
        master = ["wav_file\tday\tstart_time\tlocation\tmicrolocation\tchannel\tchannel_name\trecorder_type\tsunrise\tsunset"]
        for folder, items in self.FOLDERS.items():
            for stem, cataloged in items:
                self._add_recording(folder, stem)
                if cataloged:
                    d, t = stem.split("_")
                    master.append(f"{folder}/{stem}.WAV\t{d[:4]}-{d[4:6]}-{d[6:]}\t{t[:2]}:{t[2:4]}:{t[4:]}\t{folder}\tunknown\t1\t1\tunknown\t06:11\t18:10")
        with open(os.path.join(self.data, "zabe+hyla+citimon.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(master) + "\n")
        self.reg_path = os.path.join(self.tmp, "citimon_registry.json")
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_registry(), f)
        self._old_env = os.environ.get(citimon_registry.ENV_VAR)
        os.environ[citimon_registry.ENV_VAR] = self.reg_path

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop(citimon_registry.ENV_VAR, None)
        else:
            os.environ[citimon_registry.ENV_VAR] = self._old_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add_recording(self, folder, stem):
        pos = os.path.join(self.cit, folder, "positions")
        os.makedirs(pos, exist_ok=True)
        open(os.path.join(self.cit, folder, f"{stem}_ch1.mp3"), "wb").close()
        d, t = stem.split("_")
        for k in (10, 20):
            with open(os.path.join(pos, f"{stem}_perchv2_{k}.tsv"), "w", encoding="utf-8") as f:
                f.write(TSV_HEADER)
                for i, ct in enumerate((2.5, 7.5)):
                    f.write(f"{stem}.WAV\t{d[2:]}\t{t}\t1\t{ct}\t5.0\t[{i}.5, {k}.25]\tperchv2\t{i}\t{k}\n")

    def builder(self, force=False, project="zabe"):
        # Same setup as the real data: cache project "zabe" (recording-ID prefix), data folder "data".
        return build_cache.CacheBuilder(project=project, base_dir=self.tmp, force=force, project_dir=self.data)

    def partition(self):
        return pd.read_parquet(os.path.join(self.data, "cache", "partitions", "Citimon", "perchv2.parquet"))


class TestBuilderHierarchy(_TempTree):
    def test_build_writes_registry_hierarchy(self):
        manifest = self.builder().build()
        self.assertTrue(manifest)
        df = self.partition()
        # Excluded folder is not ingested at all.
        self.assertNotIn("2026_03-X09", set(df["source_folder"]))
        self.assertFalse(df["recording_id"].str.contains("2026_03-X09").any())
        # location = recorder, microlocation = registry ID / Unassigned, source_folder = on-disk folder.
        per = df.drop_duplicates("source_folder").set_index("source_folder")
        self.assertEqual(per.loc["2026_03-X01", ["location", "microlocation"]].tolist(), ["X01", "X01-P1"])
        self.assertEqual(per.loc["2026_04-X01", ["location", "microlocation"]].tolist(), ["X01", "X01-P2"])
        self.assertEqual(per.loc["2026_05-X01", ["location", "microlocation"]].tolist(), ["X01", "Unassigned"])
        self.assertEqual(per.loc["2026_03-X02", ["location", "microlocation"]].tolist(), ["X02", "X02-P1"])
        # Recording IDs and audio paths still use the folder.
        self.assertTrue(df["recording_id"].str.match(r"^zabe::Citimon/2026_0\d-X0\d/\d{8}_\d{6}_ch1\.mp3::ch1$").all())
        self.assertTrue((df["rel_audio_path"].str.split("/").str[1] == df["source_folder"]).all())
        # Sunrise/sunset: master value kept; uncataloged rows computed from coordinates; Unassigned stays unknown.
        self.assertEqual(per.loc["2026_03-X01", ["sunrise", "sunset"]].tolist(), ["06:11", "18:10"])
        self.assertRegex(per.loc["2026_04-X01", "sunrise"], r"^0[5-7]:\d\d$")
        self.assertRegex(per.loc["2026_03-X02", "sunset"], r"^1[7-9]:\d\d$")
        self.assertEqual(per.loc["2026_05-X01", "sunrise"], "unknown")
        # Two k values x two clips x four folders.
        self.assertEqual(len(df), 2 * 2 * 4)
        self.assertEqual(manifest["locations"], ["Citimon"])

    def test_validate_reports_hierarchy(self):
        b = self.builder()
        self.assertTrue(b.validate())
        msgs = " ".join(m for _, m in b.diagnostics)
        self.assertIn("2026_03-X09", msgs)            # skipped (excluded)
        self.assertIn("2 recorders", msgs)
        self.assertIn("2026_05-X01", msgs)            # listed as Unassigned

    def test_unregistered_folder_blocks_citimon_and_keeps_previous_partition(self):
        self.assertTrue(self.builder().build())
        before = self.partition()
        self._add_recording("2026_06-X03", "20260601_100000")   # not in the registry
        b = self.builder(force=True)
        self.assertFalse(b.validate())
        manifest = b.build()
        self.assertIn("Citimon", manifest["locations"], "previous Citimon partition must stay in the manifest")
        pd.testing.assert_frame_equal(before, self.partition())
        self.assertTrue(any("not in the Citimon registry" in m for lvl, m in b.diagnostics if lvl == "ERROR"))

    def test_recording_id_prefix_guard(self):
        self.assertTrue(self.builder().build())
        before = self.partition()
        b = self.builder(force=True, project="data")          # would write "data::" IDs
        manifest = b.build()
        pd.testing.assert_frame_equal(before, self.partition())
        self.assertTrue(manifest["partitions"])               # previous partition kept in the manifest
        self.assertTrue(any("prefix 'zabe::'" in m for lvl, m in b.diagnostics if lvl == "ERROR"))

    def test_missing_registry_blocks_build(self):
        os.environ[citimon_registry.ENV_VAR] = os.path.join(self.tmp, "absent.json")
        b = self.builder()
        self.assertFalse(b.validate())
        b.build()
        self.assertFalse(os.path.exists(os.path.join(self.data, "cache", "partitions", "Citimon", "perchv2.parquet")))

    def test_registry_change_invalidates_partition_signature(self):
        m1 = self.builder().build()
        reg = synthetic_registry()
        reg["source_folders"]["2026_05-X01"]["note"] = "edited"
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        m2 = self.builder().build()
        self.assertNotEqual(m1["partitions"][0]["signature"], m2["partitions"][0]["signature"])


class TestRavenFolderColumn(_TempTree):
    def test_partition_clip_source_uses_source_folder(self):
        self.assertTrue(self.builder().build())
        src = PartitionClipSource(project="Citimon", base_dir=self.data)
        self.assertEqual(src.known_locations("perchv2"), ["2026_03-X01", "2026_03-X02", "2026_04-X01", "2026_05-X01"])
        idx = src.load_index("perchv2", ["2026_03-X01", "2026_03-X02"])
        x01 = idx.channels_for("2026_03-X01", "20260317_223000")
        x02 = idx.channels_for("2026_03-X02", "20260317_223000")
        self.assertEqual(x01, {1: {"zabe::Citimon/2026_03-X01/20260317_223000_ch1.mp3::ch1"}})
        self.assertEqual(x02, {1: {"zabe::Citimon/2026_03-X02/20260317_223000_ch1.mp3::ch1"}})
        self.assertEqual(idx.channels_for("X01", "20260317_223000"), {})   # recorder is not a Raven location

    def test_dataframe_clip_source_legacy_and_new(self):
        base = dict(wav_file="2026_03-C08/20260317_223000.WAV", channel=1,
                    recording_id="zabe::Citimon/2026_03-C08/20260317_223000_ch1.mp3::ch1",
                    semantic_clip_key="k1", clip_time=2.5, clip_duration=5.0, model_name="perchv2")
        legacy = DataFrameClipSource(pd.DataFrame([dict(base, location="2026_03-C08")]))
        new = DataFrameClipSource(pd.DataFrame([dict(base, location="C08", source_folder="2026_03-C08")]))
        for src in (legacy, new):
            self.assertEqual(src.known_locations("perchv2"), ["2026_03-C08"])
            self.assertTrue(src.load_index("perchv2", ["2026_03-C08"]).channels_for("2026_03-C08", "20260317_223000"))


class TestMapHierarchy(unittest.TestCase):
    """Map preview against the committed registry."""

    def test_recorder_highlights_all_its_positions(self):
        self.assertEqual(citimon_map.resolve_selection("C47"), ("C47", ["C47-P1", "C47-P2"]))
        svg = citimon_map._build_svg_string("C47")
        self.assertEqual(svg.count('r="4.5"'), 2)
        self.assertIn("C47 &#183; C47-P2", svg)

    def test_microlocation_filter_narrows_markers(self):
        self.assertEqual(citimon_map.resolve_selection("C47", ["C47-P2"]), ("C47", ["C47-P2"]))
        self.assertEqual(citimon_map._build_svg_string("C47", ["C47-P2"]).count('r="4.5"'), 1)
        unassigned_only = citimon_map._build_svg_string("C47", ["Unassigned"])
        self.assertIn("No coordinates", unassigned_only)

    def test_historical_folder_names_still_work(self):
        # Caches built before the registry have folder names in location and microlocation.
        self.assertEqual(citimon_map.resolve_selection("2026_03-C02", ["2026_03-C02"]), ("C02", ["C02-P1"]))
        self.assertEqual(citimon_map.resolve_selection("2026_03-C19-1"), ("C19", ["C19-P1"]))
        self.assertIn("No coordinates", citimon_map._build_svg_string("2026_04-C08"))

    def test_precomputed_per_recorder_and_callback(self):
        reg = citimon_registry.load_registry(citimon_registry.DEFAULT_PATH)
        self.assertEqual(set(citimon_map._PRECOMPUTED_DATA_URIS), set(reg.recorders))
        from dash import Dash
        from callbacks.callbacks_filters import register_filter_callbacks
        app = Dash(__name__)
        register_filter_callbacks(app)
        fn = next(v for k, v in app.callback_map.items() if 'location-map-container' in k)['callback'].__wrapped__
        all_children, style = fn("C47", "Citimon", ["C47-P1", "C47-P2"])
        one_children, _ = fn("C47", "Citimon", ["C47-P1"])
        self.assertEqual(style["display"], "flex")
        self.assertNotEqual(all_children.children[0].src, one_children.children[0].src)


class TestMultiLocation(unittest.TestCase):
    """The location dropdown is multi-select: loading, filtering and the map accept several recorders."""

    def test_location_list_normalisation(self):
        import read_data
        self.assertEqual(read_data.location_list(None), [])
        self.assertEqual(read_data.location_list(""), [])
        self.assertEqual(read_data.location_list("C08"), ["C08"])
        self.assertEqual(read_data.location_list(["C08", None, "", "C47"]), ["C08", "C47"])

    def test_reader_filters_to_all_selected_locations(self):
        from cache import reader
        df = pd.DataFrame({"location": ["C08", "C47", "C02", "C47"], "x": [1, 2, 3, 4]})
        self.assertEqual(reader._filter_by_locations(df, ["C08", "C47"])["x"].tolist(), [1, 2, 4])
        self.assertEqual(reader._filter_by_locations(df, "C02")["x"].tolist(), [3])
        self.assertEqual(len(reader._filter_by_locations(df, None)), 4)
        self.assertEqual(len(reader._filter_by_locations(df, ["Missing"])), 4)   # unchanged legacy behaviour

    def test_data_callback_loads_several_locations(self):
        from unittest import mock
        from dash import Dash
        import read_data
        from callbacks.callbacks_data import register_data_callbacks
        from callbacks.callbacks_constants import MODEL_DATA_CACHE
        app = Dash(__name__)
        register_data_callbacks(app)
        fn = next(v for k, v in app.callback_map.items() if 'model-data-ready-signal' in k)['callback'].__wrapped__
        frame = pd.DataFrame({"location": ["C08", "C47"], "microlocation": ["C08-P1", "C47-P2"], "channel": [1, 1]})
        saved = dict(MODEL_DATA_CACHE)
        try:
            with mock.patch.object(read_data, "is_valid_project_location", return_value=True) as valid, \
                    mock.patch.object(read_data, "load_project_model_data", return_value=frame) as load:
                _, meta = fn("perchv2", ["C08", "C47"], "Citimon")
            self.assertEqual(valid.call_args[0][1], ["C08", "C47"])
            self.assertEqual(load.call_args.kwargs["selected_location"], ["C08", "C47"])
            self.assertEqual(MODEL_DATA_CACHE["key"], ("Citimon", ("C08", "C47"), "perchv2"))
            self.assertEqual(meta["microlocations"], ["C08-P1", "C47-P2"])
        finally:
            MODEL_DATA_CACHE.clear()
            MODEL_DATA_CACHE.update(saved)

    def test_map_marks_positions_of_every_selected_recorder(self):
        uri_multi = citimon_map.get_map_data_uri(["C47", "C02"])
        import base64
        svg = base64.b64decode(uri_multi.split(",", 1)[1]).decode("utf-8")
        self.assertEqual(svg.count('r="4.5"'), 3)                   # C47-P1, C47-P2, C02-P1
        self.assertIn("C47 &#183; C47-P2", svg)
        self.assertIn("C02 &#183; C02-P1", svg)
        # The microlocation filter still narrows a multi-selection; one-item lists use the precomputed map
        narrowed = base64.b64decode(citimon_map.get_map_data_uri(["C47", "C02"], ["C02-P1"]).split(",", 1)[1]).decode()
        self.assertEqual(narrowed.count('r="4.5"'), 1)
        self.assertEqual(citimon_map.get_map_data_uri(["C47"]), citimon_map.get_map_data_uri("C47"))
        comp = citimon_map.render_citimon_map_component(["C47", "C02"])
        self.assertEqual(comp.title, "Location: C47, C02. Click to choose locations on the map")

    def test_map_picker_figure_and_toggle(self):
        fig = citimon_map.picker_figure(["C47"], ["C47", "C02"])
        by_name = {t.name: list(t.customdata) for t in fig.data if t.name}
        self.assertEqual(sorted(by_name["Selected"]), ["C47", "C47"])            # both C47 positions
        self.assertEqual(by_name["Not selected"], ["C02"])
        self.assertNotIn("C47", by_name["No data"])
        toggle = citimon_map.picker_toggle
        self.assertEqual(toggle(["C47"], ["C47", "C02"], click_data={"points": [{"customdata": "C02"}]}), ["C47", "C02"])
        self.assertEqual(toggle(["C47", "C02"], ["C47", "C02"], click_data={"points": [{"customdata": "C47"}]}), ["C02"])
        self.assertEqual(toggle(["C47"], ["C47"], click_data={"points": [{"customdata": "C99"}]}), ["C47"])  # no data
        box = {"points": [{"customdata": "C02"}, {"customdata": "C08"}, {"customdata": None}]}
        self.assertEqual(toggle(["C47"], ["C47", "C02", "C08"], selected_data=box), ["C47", "C02", "C08"])

    def test_multimon_map_positions(self):
        import base64
        import multimon_locations
        mm = multimon_locations.load()
        self.assertIsNotNone(mm, "config/multimon_locations.json must load")
        self.assertTrue(citimon_map.has_map("Multimon"))
        self.assertFalse(citimon_map.has_map("vibro"))
        # Locations with two positions (Renški glinokop is filed under Glinokop Vogrsko in the cache)
        self.assertEqual(citimon_map.resolve_selection("Glinokop Vogrsko", None, "Multimon"),
                         ("Glinokop Vogrsko", ["Glinokop Vogrsko · Vogrsko", "Glinokop Vogrsko · Renški glinokop"]))
        self.assertEqual(len(mm.resolve("Spodnje Gameljne")), 2)
        self.assertEqual(mm.resolve("mrtvica Vipave"), ["Mrtvica Vipave"])       # spelling-insensitive
        self.assertEqual(mm.resolve("Krakovski_gozd-2025"), ["Krakovski gozd"])  # folder not yet in the cache
        self.assertEqual(mm.resolve("Bioblitz"), [])                            # no coordinates
        self.assertEqual(mm.resolve("Prekmurje"), [])                           # VibroScape placeholder not used
        for key, (lat, lon) in mm.points.items():
            self.assertTrue(45.4 < lat < 46.9 and 13.3 < lon < 16.7, key)
        svg = base64.b64decode(citimon_map.get_map_data_uri(["Draga", "Banjšice"], None, "Multimon").split(",", 1)[1]).decode()
        self.assertEqual(svg.count('r="4.5"'), 2)
        self.assertIn("Banjšice (46.0641&#176; N, 13.6983&#176; E)", svg)
        glinokop = base64.b64decode(citimon_map.get_map_data_uri("Glinokop Vogrsko", None, "Multimon").split(",", 1)[1]).decode()
        self.assertIn("Glinokop Vogrsko &#183; Renški glinokop", glinokop)
        self.assertIn("No coordinates",
                      base64.b64decode(citimon_map.get_map_data_uri("Bioblitz", None, "Multimon").split(",", 1)[1]).decode())
        # Picker: location values as customdata, Multimon labels kept as is; Citimon unaffected
        fig = citimon_map.picker_figure(["Glinokop Vogrsko"], ["Glinokop Vogrsko", "Barka"], "Multimon")
        by_name = {t.name: list(t.customdata) for t in fig.data if t.name}
        self.assertEqual(by_name["Selected"], ["Glinokop Vogrsko", "Glinokop Vogrsko"])
        self.assertEqual(by_name["Not selected"], ["Barka"])
        self.assertIn("Krakovski_gozd", by_name["No data"])
        self.assertEqual(citimon_map.get_map_data_uri("C47"), citimon_map._PRECOMPUTED_DATA_URIS["C47"])

    def test_map_picker_callbacks(self):
        from unittest import mock
        from dash import Dash
        import dash
        from callbacks.callbacks_filters import register_filter_callbacks
        from layout_raven import MODAL_VISIBLE, MODAL_HIDDEN
        app = Dash(__name__)
        register_filter_callbacks(app)
        cb = lambda part: next(v for k, v in app.callback_map.items() if part in k)['callback'].__wrapped__

        def ctx(prop_id):
            fake = mock.MagicMock()
            fake.triggered_id = prop_id.split('.')[0]
            fake.triggered = [{'prop_id': prop_id}]
            return mock.patch.object(dash, 'callback_context', fake)

        toggle_modal = cb('location-map-modal.style')
        with ctx('location-map-container.n_clicks'):
            self.assertEqual(toggle_modal(1, 0, 'Citimon'), MODAL_VISIBLE)
            self.assertEqual(toggle_modal(1, 0, 'Multimon'), MODAL_VISIBLE)         # Multimon has a map too
            self.assertEqual(toggle_modal(1, 0, 'vibro'), MODAL_HIDDEN)             # projects without positions
        with ctx('location-map-close-btn.n_clicks'):
            self.assertEqual(toggle_modal(1, 1, 'Citimon'), MODAL_HIDDEN)

        render = cb('location-map-picker.figure')
        self.assertEqual(render(MODAL_HIDDEN, ['C47'], []), (dash.no_update, dash.no_update))
        fig, summary = render(MODAL_VISIBLE, ['C47', 'C02'], [{'label': 'C47', 'value': 'C47'}])
        self.assertEqual(summary, "Selected (2): C47, C02")

        choose = cb('location-map-picker.clickData')
        opts = [{'label': v, 'value': v} for v in ('C47', 'C02', 'C08')]
        with ctx('location-map-picker.clickData'):
            self.assertEqual(choose({'points': [{'customdata': 'C02'}]}, None, ['C47'], opts), (['C47', 'C02'], None, None))
        with ctx('location-map-picker.selectedData'):
            box = {'points': [{'customdata': 'C08'}, {'customdata': 'C47'}]}
            self.assertEqual(choose(None, box, ['C47'], opts), (['C47', 'C08'], None, None))


if __name__ == "__main__":
    unittest.main()
