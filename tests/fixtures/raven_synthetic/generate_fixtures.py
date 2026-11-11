"""
SYNTHETIC TEST DATA GENERATOR — not real biologist annotations.

Writes fake Raven Pro selection tables into this directory and provides a synthetic
Citimon-like clip frame. Column names follow Raven Pro defaults as documented
publicly; they have NOT been checked against the biologists' real exports.

Every table carries a ``Synthetic`` column with the value SYNTHETIC-TEST-DATA.

Run:  python tests/fixtures/raven_synthetic/generate_fixtures.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "SYNTHETIC-TEST-DATA"

LOC_A = "2026_03-C02"
LOC_B = "2026_03-C03"
LOC_C = "2026-06-C35"


def _clip_times(missing=()):
    return [t / 2 for t in range(0, 111, 5) if t / 2 not in missing]  # 0.0 .. 55.0 step 2.5


def synthetic_clip_frame() -> pd.DataFrame:
    """Citimon-shaped partition rows (5 s clips, 2.5 s hop, duplicated per cluster_num)."""
    recordings = [
        # (location, relative wav, clip times)
        (LOC_A, f"Citimon/{LOC_A}/20260320_184000.WAV", _clip_times(missing={17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5})),
        (LOC_A, f"Citimon/{LOC_A}/20260320_185000.WAV", _clip_times()),
        # Same file name as LOC_A's first recording: all recorders share one schedule.
        (LOC_B, f"Citimon/{LOC_B}/20260320_184000.WAV", _clip_times()),
        (LOC_C, f"Citimon/{LOC_C}/20260511_184000.WAV", _clip_times()),
        # Two recordings with the same stem at one location -> ambiguous.
        (LOC_C, f"Citimon/{LOC_C}/20260511_190000.WAV", _clip_times()),
        (LOC_C, f"Citimon/{LOC_C}/copy/20260511_190000.WAV", _clip_times()),
    ]
    rows = []
    for loc, wav, times in recordings:
        stem_path = wav.rsplit(".", 1)[0]
        rel_audio = f"{stem_path}_ch1.mp3"
        recording_id = f"zabe::{rel_audio}::ch1"
        for cluster_num in (10, 20):
            for t in times:
                rows.append({
                    "file_name": wav.split("/")[-1], "channel": 1, "clip_time": t, "clip_duration": 5.0,
                    "model_name": "perchv2", "cluster_id": int(t) % cluster_num, "cluster_num": cluster_num,
                    "wav_file": wav, "location": loc, "microlocation": loc, "rel_audio_path": rel_audio,
                    "recording_id": recording_id,
                    "raw_clip_id": f"{recording_id}::perchv2::{round(t, 3)}",
                    "semantic_clip_key": f"{recording_id}::{round(t, 3)}",
                })
    return pd.DataFrame(rows)


def _write(name, header, rows, encoding, newline, bom=False):
    text = newline.join(["\t".join(header)] + ["\t".join(str(c) for c in r) for r in rows]) + newline
    data = text.encode(encoding)
    if bom:
        data = b"\xef\xbb\xbf" + data
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(data)


def _views(sel, rest_wave, rest_spec=None):
    return [[sel, "Waveform 1"] + rest_wave, [sel, "Spectrogram 1"] + (rest_spec or rest_wave)]


def write_all():
    pa = "D:\\Citimon\\" + LOC_A + "\\"
    # A: annotator "Ana", sequence of two files, UTF-8 with BOM, CRLF, two views per selection.
    header_a = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
                "Begin File", "End File", "Begin Path", "File Offset (s)", "Delta Time (s)",
                "Species", "Call type", "Notes", "Synthetic"]
    f1, f2 = "20260320_184000.WAV", "20260320_185000.WAV"

    def a(begin, end, bfile, efile, path, offset, species, call, notes):
        return [1, begin, end, 400, 3200, bfile, efile, path, offset, round(end - begin, 3), species, call, notes, MARK]

    rows_a = []
    rows_a += _views(1, a(0.5, 2.0, f1, f1, pa + f1, 0.5, "Hyla arborea", "advertisement", "glasen klic, žaba v grmovju"))
    rows_a += _views(2, a(11.0, 14.0, f1, f1, pa + f1, 11.0, "Pelophylax esculentus", "chorus", "zbor"))
    rows_a += _views(3, a(25.0, 30.0, f1, f1, pa + f1, 25.0, "Bufo bufo", "release call", "v vrzeli"))
    rows_a += _views(4, a(58.5, 61.5, f1, f2, pa + f1, 58.5, "Hyla arborea", "advertisement", "čez mejo datoteke"))
    rows_a += _views(5, a(70.0, 72.0, f2, f2, pa + f2, 10.0, "", "", ""))
    rows_a += _views(6, a(80.0, 79.0, f2, f2, pa + f2, 20.0, "Rana dalmatina", "advertisement", "napačen čas"))
    rows_a += _views(7, a(85.0, 86.0, f2, f2, "D:\\Citimon\\C99_unknown\\" + f2, 25.0, "Bufo bufo", "chorus", ""))
    rows_a += _views(8, a(90.0, 91.0, f2, f2, pa + f2, 30.0, "Hyla arborea", "chorus", "pogledi se ne ujemajo"),
                     a(90.4, 91.0, f2, f2, pa + f2, 30.4, "Hyla arborea", "chorus", "pogledi se ne ujemajo"))
    rows_a += _views(9, a(95.0, 96.0, f2, f2, "", 35.0, "Hyla arborea", "chorus", "brez poti"))
    rows_a += _views(10, a(100.0, 101.0, "20260511_190000.WAV", "20260511_190000.WAV",
                           "D:\\Citimon\\" + LOC_C + "\\20260511_190000.WAV", 40.0, "Bufo bufo", "chorus", "dvojnik"))
    _write("SYNTHETIC_ana_sequence_utf8.Table.1.selections.txt", header_a, rows_a, "utf-8", "\r\n", bom=True)

    # B: annotator "Boštjan", same intervals as Ana's 1-3 with different labels, CP1250, LF, one view.
    header_b = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
                "Begin File", "Begin Path", "File Offset (s)", "Vrsta", "Tip klica", "Opomba", "Synthetic"]
    rows_b = [
        [1, "Spectrogram 1", 1, 0.5, 2.0, 300, 3000, f1, pa + f1, 0.5, "Rana dalmatina", "oglašanje", "Žabji klic, slišno čez cesto", MARK],
        [2, "Spectrogram 1", 1, 11.0, 14.0, 300, 3000, f1, pa + f1, 11.0, "Pelophylax ridibundus", "zbor", "šum vetra", MARK],
        [3, "Spectrogram 1", 1, 25.0, 30.0, 300, 3000, f1, pa + f1, 25.0, "Bufo bufo", "", "čuk v ozadju", MARK],
    ]
    _write("SYNTHETIC_bostjan_sequence_cp1250.Table.1.selections.txt", header_b, rows_b, "cp1250", "\n")

    # C: single-file export (Raven default name), no file/path/offset columns, UTF-8 without BOM.
    header_c = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
                "Annotation", "Synthetic"]
    rows_c = [
        [1, "Spectrogram 1", 1, 3.0, 4.0, 500, 2500, "Hyla arborea", MARK],
        [2, "Spectrogram 1", 1, 41.0, 44.0, 500, 2500, "Bufo bufo", MARK],
        [3, "Spectrogram 1", 1, 59.0, 61.0, 500, 2500, "Hyla arborea", MARK],
    ]
    _write("20260320_185000.Table.1.selections.txt", header_c, rows_c, "utf-8", "\r\n")

    # D: Begin File only (no path) — the stem exists at two locations; plus malformed and decimal-comma rows.
    header_d = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Begin File", "File Offset (s)",
                "Species", "Synthetic"]
    rows_d = [
        [1, "Spectrogram 1", 1, 1.0, 2.0, f1, 1.0, "Hyla arborea", MARK],
        [2, "Spectrogram 1", 1, "1,5", "2,5", f1, "1,5", "Hyla arborea", MARK],
        [3, "Spectrogram 1", 1, 4.0, 5.0, f1, 4.0, "Hyla arborea", MARK, "unexpected extra cell"],
    ]
    _write("SYNTHETIC_filename_only.txt", header_d, rows_d, "utf-8", "\n")

    # E: sequence-relative times over two files without File Offset (time origin unknown).
    header_e = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Begin File", "Begin Path",
                "Species", "Synthetic"]
    rows_e = [
        [1, "Spectrogram 1", 1, 5.0, 6.0, f1, pa + f1, "Hyla arborea", MARK],
        [2, "Spectrogram 1", 1, 65.0, 66.0, f2, pa + f2, "Hyla arborea", MARK],
    ]
    _write("SYNTHETIC_sequence_without_offset.txt", header_e, rows_e, "utf-8", "\n")


if __name__ == "__main__":
    write_all()
    print("Synthetic Raven fixtures written to", HERE)
