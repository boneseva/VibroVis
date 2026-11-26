"""
Deep link from a clicked clip to the Citimon Explorer (explorer_link.py).

Covers the pure URL builder (folder/stem derivation, Windows separators, aliases of renamed folders, non-Citimon
folders, bad times, feature off by default), the layout contract (a hidden link in the info row) and the wiring in the
click callbacks (store carries ``explorer_url``; the clientside callback drives the link's href and visibility).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import explorer_link

ENV = {"VIBROVIS_EXPLORER_URL": "/explorer"}


class TestExplorerUrl(unittest.TestCase):

    def test_on_by_default_and_off_when_set_empty(self):
        mp3 = "2026_03-C20/20260228_211000_ch1.mp3"
        self.assertEqual(explorer_link.explorer_url(mp3, 12.0, environ={}),
                         "/explorer/go?dataset=citimon&path=2026_03-C20/20260228_211000&t=12.00")
        for off in ("", "  "):
            self.assertIsNone(explorer_link.explorer_url(mp3, 12.0, environ={"VIBROVIS_EXPLORER_URL": off}))

    def test_builds_go_link(self):
        url = explorer_link.explorer_url("2026_03-C20/20260228_211000_ch1.mp3", 12.345, environ=ENV)
        self.assertEqual(url, "/explorer/go?dataset=citimon&path=2026_03-C20/20260228_211000&t=12.35")

    def test_explorer_home_url(self):
        self.assertEqual(explorer_link.explorer_home_url(environ={}), "https://ranavis.musiclab.si/explorer")
        self.assertEqual(explorer_link.explorer_home_url(environ={"VIBROVIS_EXPLORER_URL": "https://example.org/explorer"}),
                         "https://example.org/explorer")
        self.assertEqual(explorer_link.explorer_home_url(environ={"VIBROVIS_EXPLORER_URL": ""}), "")

    def test_trailing_slash_and_absolute_base(self):
        url = explorer_link.explorer_url("2026_03-C20/20260228_211000_ch2.mp3", 0,
                                         environ={"VIBROVIS_EXPLORER_URL": "https://ranavis.musiclab.si/explorer/"})
        self.assertTrue(url.startswith("https://ranavis.musiclab.si/explorer/go?"))
        self.assertTrue(url.endswith("&t=0.00"))

    def test_windows_separators_and_prefix(self):
        for mp3 in ("2026_03-C20\\20260228_211000_ch1.mp3", "C:\\data\\Citimon\\2026_03-C20\\20260228_211000_ch1.mp3",
                    "/app/data/recordings/Citimon/2026_03-C20/20260228_211000_ch1.mp3"):
            self.assertEqual(explorer_link.recording_path(mp3), "2026_03-C20/20260228_211000", mp3)

    def test_renamed_folders_are_kept_for_the_explorer_alias_table(self):
        self.assertEqual(explorer_link.recording_path("2026_03-C19-1/20260301_131000_ch1.mp3"),
                         "2026_03-C19-1/20260301_131000")
        self.assertEqual(explorer_link.recording_path("2026-06-C35/20260511_184000_ch1.mp3"),
                         "2026-06-C35/20260511_184000")

    def test_non_citimon_or_malformed_gives_no_link(self):
        for mp3 in (None, "", float("nan"), 5, "20260228_211000_ch1.mp3", "Bioblitz/20260228_211000_ch1.mp3",
                    "Draga/2026-06-01/x_ch1.mp3", "2026_03-C20/20260228_211000.wav", "2026_03-C20/_ch1.mp3"):
            self.assertIsNone(explorer_link.explorer_url(mp3, 1.0, environ=ENV), repr(mp3))

    def test_bad_times_give_no_link(self):
        mp3 = "2026_03-C20/20260228_211000_ch1.mp3"
        for t in (None, "x", float("nan"), -1, float("inf")):
            self.assertIsNone(explorer_link.explorer_url(mp3, t, environ=ENV), repr(t))

    def test_reads_environment_on_every_call(self):
        mp3 = "2026_03-C20/20260228_211000_ch1.mp3"
        old = os.environ.pop("VIBROVIS_EXPLORER_URL", None)
        try:
            self.assertTrue(explorer_link.explorer_url(mp3, 1).startswith("/explorer/go?"))
            os.environ["VIBROVIS_EXPLORER_URL"] = ""
            self.assertIsNone(explorer_link.explorer_url(mp3, 1))
            os.environ["VIBROVIS_EXPLORER_URL"] = "https://example.org/x"
            self.assertTrue(explorer_link.explorer_url(mp3, 1).startswith("https://example.org/x/go?"))
        finally:
            os.environ.pop("VIBROVIS_EXPLORER_URL", None)
            if old is not None:
                os.environ["VIBROVIS_EXPLORER_URL"] = old


class TestExplorerLinkWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import layout
        import read_data
        cls.layout = layout.create_layout(read_data.get_initial_data_for_layout())

    def _find(self, comp, target):
        if getattr(comp, "id", None) == target:
            return comp
        kids = getattr(comp, "children", None)
        for k in (kids if isinstance(kids, list) else [kids]):
            if k is not None and not isinstance(k, (str, int, float)):
                found = self._find(k, target)
                if found is not None:
                    return found
        return None

    def test_link_is_hidden_in_the_info_row(self):
        link = self._find(self.layout, "explorer-link")
        self.assertIsNotNone(link, "explorer-link missing from the layout")
        self.assertEqual(link.style, {"display": "none"})
        self.assertEqual(link.target, "_blank")
        self.assertIn("noopener", link.rel)

    def test_title_bar_explorer_link(self):
        link = self._find(self.layout, "title-explorer-link")
        self.assertIsNotNone(link, "title-explorer-link missing from the layout")
        self.assertEqual(link.target, "_blank")
        self.assertIn("noopener", link.rel)
        self.assertEqual(link.href, "https://ranavis.musiclab.si/explorer")
        self.assertEqual(link.children, "Explorer ↗")

    def test_clientside_callback_drives_the_link(self):
        import dash
        import app as vibro_app  # noqa: F401  (registers the callbacks)
        specs = list(getattr(vibro_app.app, "_callback_list", [])) + list(dash._callback.GLOBAL_CALLBACK_LIST)
        found = [s for s in specs if "explorer-link.href" in str(s.get("output", ""))]
        self.assertEqual(len(found), 1, "exactly one callback writes explorer-link.href")
        out = str(found[0]["output"])
        self.assertIn("explorer-link.style", out)
        self.assertIn("spectrogram-raw-data-store", str(found[0]["inputs"]))

    def test_both_click_paths_store_the_link(self):
        for rel in (("callbacks", "callbacks_plots.py"), ("callbacks", "callbacks_table.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as fh:
                src = fh.read()
            self.assertIn("explorer_link.explorer_url(", src, rel[-1])
            self.assertIn("explorer_url", src.split("store_data = {")[1].split("}")[0], rel[-1])


if __name__ == "__main__":
    unittest.main()
