"""Dashboard BI page: fixture rendering, empty states, lead-page navigation and import hygiene."""
import csv
import json
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from bi_fixture import NAMES, OLD_RUN_ID, SECRETS, patch_paths, write_fixture
from temajelajah import bi_dashboard
from temajelajah.bi_data import STARS_4_5

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "from temajelajah.bi_dashboard import render_dashboard\nrender_dashboard()"
KPI_LABELS = {"Ulasan pelanggan", "Rating rata-rata", "Ulasan berkeluhan", f"Keluhan di balik {STARS_4_5}", "Cakupan per target"}
TABS = ["Ringkasan", "Wawasan NLP", "Operasi scraping", "Metodologi"]


def _clear_caches():
    for cached in (bi_dashboard._bundle, bi_dashboard._terms, bi_dashboard._collector, bi_dashboard._audit_messages,
                   bi_dashboard._ops_tables, bi_dashboard._telemetry):
        cached.clear()


@pytest.fixture
def fixture_paths(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    patch_paths(monkeypatch, paths)
    _clear_caches()
    yield paths
    _clear_caches()


def _okay(at):
    assert not at.exception, [error.message for error in at.exception]


def _shown_text(at) -> str:
    values = [item.value for group in (at.markdown, at.caption, at.info, at.warning, at.error) for item in group]
    values += [frame.value.to_csv() for frame in at.dataframe]
    values += [metric.label + str(metric.value) + str(metric.help) for metric in at.metric]
    values += [chart.proto.spec + chart.proto.data.data.decode("latin-1") for chart in at.get("vega_lite_chart")]
    return "\n".join(map(str, values))


def test_renders_fixture_kpis_charts_badges_tabs(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert KPI_LABELS <= {metric.label for metric in at.metric}
    assert next(metric for metric in at.metric if metric.label == "Ulasan pelanggan").value == "11"
    assert len(at.get("vega_lite_chart")) >= 9
    assert any("Label asisten sementara" in item.value for item in at.markdown)
    assert any("Snapshot cocok dengan audit" in item.value for item in at.markdown)
    assert [tab.label for tab in at.tabs] == TABS
    assert not any(secret in _shown_text(at) for secret in SECRETS)
    at.text_input(key="bi_kwic_q").set_value("patient").run()
    _okay(at)
    context = "\n".join(frame.value.to_csv() for frame in at.dataframe)
    assert "[nama]" in context and "Wayan" not in context
    assert not any(secret in _shown_text(at) for secret in SECRETS)


def test_missing_corpus_waits_without_metrics(fixture_paths, monkeypatch):
    (fixture_paths["run"] / "corpus.csv").unlink()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert not at.metric and not at.get("vega_lite_chart") and not at.tabs
    assert any("Menunggu" in info.value for info in at.info)
    assert not any("-badge[" in item.value for item in at.markdown)
    at.button(key="bi_open_search").click().run()
    assert at.session_state["page"] == "Cari Tempat"
    monkeypatch.setattr(bi_dashboard.data, "EXPANSION_ROOT", fixture_paths["workspace"] / "absent")
    _okay(AppTest.from_string(SCRIPT, default_timeout=30).run())


def test_empty_filter_slice_warns_without_metrics(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert at.select_slider(key="bi_years").value == (2023, 2025)
    at.multiselect(key="bi_venues").set_value(["five_art_studio"])
    at.select_slider(key="bi_years").set_value((2024, 2024)).run()
    _okay(at)
    assert any("Tidak ada ulasan untuk filter ini" in warning.value for warning in at.warning)
    assert not at.metric and not at.get("vega_lite_chart")
    at.button(key="bi_reset_empty").click().run()
    _okay(at)  # a reset rebuilds the filter widgets under fresh keys (see test_reset_rebuilds_filter_widgets_under_fresh_keys)
    assert at.multiselect(key="bi_venues_1").value == [] and at.select_slider(key="bi_years_1").value == (2023, 2025)
    assert "bi_venues" not in at.session_state and KPI_LABELS <= {metric.label for metric in at.metric}
    at.multiselect(key="bi_venues_1").set_value(["museum_tekstil"]).run()
    _okay(at)
    assert next(metric for metric in at.metric if metric.label == "Ulasan pelanggan").value == "2"


def test_missing_labels_aspects_and_audit_degrade_to_info(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path, labels=False, business=False)
    (paths["run"] / "audit.json").unlink()
    for report in [*(paths["expansion"] / "airbnb").glob("*.json"), paths["expansion"] / "seeksophie_initial" / "collection_report.json"]:
        report.unlink()
    patch_paths(monkeypatch, paths)
    _clear_caches()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    labels = {metric.label for metric in at.metric}
    assert {"Ulasan pelanggan", "Rating rata-rata"} <= labels and not labels & {"Ulasan berkeluhan", f"Keluhan di balik {STARS_4_5}", "Cakupan per target"}
    infos = [info.value for info in at.info]
    assert any("Belum ada label sentimen" in info for info in infos) and any("Belum ada label aspek" in info for info in infos)
    assert any("Audit belum tersedia" in item.value for item in at.markdown)
    assert any("belum tersedia" in info for info in infos if "audit" in info.lower())
    assert not any(secret in _shown_text(at) for secret in SECRETS)
    _clear_caches()


def test_lead_page_dispatch_through_ui_main(fixture_paths, monkeypatch):
    from temajelajah import ui
    pages = []
    monkeypatch.setattr(ui, "load_settings", lambda: SimpleNamespace(model_dump_json=lambda: "{}"))
    monkeypatch.setattr(ui, "_service", lambda _: (None, None, None))
    monkeypatch.setattr(ui, "_project_sidebar", lambda storage, settings, names=None: pages.append(names))
    script = ("from temajelajah.ui import main\nfrom temajelajah.bi_dashboard import render_dashboard\n"
              "main(lead_pages={'Dashboard BI': render_dashboard}, extra_pages={'Ekstra': lambda *args: None})")
    at = AppTest.from_string(script, default_timeout=30).run()
    _okay(at)
    assert pages[-1] == ["Dashboard BI", *ui.PAGES, "Ekstra"]
    assert KPI_LABELS <= {metric.label for metric in at.metric}
    at.session_state["page"] = "Dashboard BI"
    _okay(at.run())
    assert any(button.label.startswith("Uji model klasifikasi") for button in at.button)


def test_bi_modules_do_not_import_sklearn():
    code = ("import sys, temajelajah.bi_dashboard\n"
            "print(sorted(name for name in ('sklearn', 'altair', 'torch', 'temajelajah.ml_experiments', "
            "'temajelajah.business_experiments') if name in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_bi_sources_utf8_without_mojibake():
    markers = ("\u00c2", "\u00f0\u0178", "\u00e2\u20ac")
    files = sorted((ROOT / "temajelajah").glob("bi_*.py")) + [ROOT / "tests" / "bi_fixture.py", *sorted((ROOT / "tests").glob("test_bi_*.py"))]
    assert len(files) >= 7
    for path in files:
        text = path.read_bytes().decode("utf-8")
        assert not any(marker in text for marker in markers), path


def test_reset_rebuilds_filter_widgets_under_fresh_keys(monkeypatch):
    # Popping the keys only (the old reset) leaves a browser widget on its stale value, which it sends back on the next rerun.
    # AppTest keeps widget state on the server and cannot see that, so the contract is pinned here: new keys on every reset.
    state = {"bi_venues": ["batik_popiler_2"], "bi_sources": ["SeekSophie"], "bi_years": (2024, 2024), "bi_undated": False, "bi_kwic_q": "wax"}
    monkeypatch.setattr(bi_dashboard.st, "session_state", state)
    bi_dashboard._reset_filters()
    assert state == {"bi_filter_gen": 1, "bi_kwic_q": "wax"} and bi_dashboard._filter_key("bi_venues") == "bi_venues_1"
    state["bi_years_1"] = (2025, 2025)
    bi_dashboard._reset_filters()
    assert "bi_years_1" not in state
    assert [bi_dashboard._filter_key(name) for name in bi_dashboard.FILTER_KEYS] == [f"{name}_2" for name in bi_dashboard.FILTER_KEYS]
    assert {"bi_sessions", "bi_rubrics", "bi_prompts", "bi_dates", "bi_models", "bi_verifications"} <= set(bi_dashboard.FILTER_KEYS)


def test_planted_names_never_reach_the_page(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    for query in ("nearby", "kind", "left", "patient", "Thanks"):
        at.text_input(key="bi_kwic_q").set_value(query).run()
        _okay(at)
        shown = _shown_text(at)
        assert not [name for name in NAMES if re.search(rf"(?<![A-Za-z]){name}(?![A-Za-z])", shown)], query  # "Wayang" stays
        assert any("[nama]" in frame.value.to_csv() for frame in at.dataframe), query
    assert any("sebagian bisa terlewat" in caption.value for caption in at.caption)
    assert any("Kolom identitas pengulas tidak dimuat" in item.value for item in at.markdown)


def test_audit_mismatches_turn_the_badge_orange_and_are_listed(fixture_paths):
    audit_path = fixture_paths["run"] / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_path.write_text(json.dumps({**audit, "snapshot": {"sha256": "0" * 64}}), encoding="utf-8")
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert any("Snapshot belum cocok dengan audit" in item.value for item in at.markdown)
    assert any("SHA-256 raw_snapshot.sqlite3 berbeda" in warning.value for warning in at.warning)
    assert not any("cocok dengan hitungan audit" in caption.value for caption in at.caption)
    audit_path.write_text(json.dumps({**audit, "counts": {**audit["counts"], "customer_text": 99}}), encoding="utf-8")
    _clear_caches()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    assert any("korpus 10 ≠ audit 99" in warning.value for warning in at.warning)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    fixture_paths["snapshot"].unlink()
    _clear_caches()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert not any("Snapshot cocok dengan audit" in item.value for item in at.markdown)
    assert any("tidak ditemukan" in warning.value for warning in at.warning)


def test_odd_audit_and_run_rows_degrade_without_errors(fixture_paths):
    older = fixture_paths["expansion"] / "runs" / OLD_RUN_ID / "audit.json"
    older.write_text(json.dumps({"counts": {"stored_unique_source_identities": 9}}), encoding="utf-8")  # no created_at
    con = sqlite3.connect(fixture_paths["snapshot"])
    try:
        con.execute("UPDATE runs SET started_at = NULL WHERE run_id = 'run-b'")
        con.commit()
    finally:
        con.close()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert {"Identitas unik terkumpul", "Run scraping tercatat", "Durasi scraping tercatat"} <= {metric.label for metric in at.metric}
    assert [tab.label for tab in at.tabs] == TABS


def test_matching_audit_counts_without_reference_hash_remain_unverified(fixture_paths):
    audit_path = fixture_paths["run"] / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("snapshot")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert fixture_paths["snapshot"].is_file()  # The file exists; only evidence for verifying its hash is absent.
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert any("Snapshot belum terverifikasi" in item.value for item in at.markdown)
    assert not any("Snapshot cocok dengan audit" in item.value for item in at.markdown)
    assert any("SHA-256 snapshot belum terverifikasi" in warning.value for warning in at.warning)
    assert not any("SHA-256 snapshot juga cocok" in caption.value for caption in at.caption)


def test_narrowed_years_say_undated_reviews_are_included(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert not any("tanpa bulan pasti**" in caption.value for caption in at.caption)
    at.select_slider(key="bi_years").set_value((2025, 2025)).run()
    _okay(at)
    assert any("Menampilkan 5 dari 11 ulasan pelanggan, **termasuk 2 tanpa bulan pasti**" in caption.value for caption in at.caption)
    assert "ikut dihitung" in next(metric for metric in at.metric if metric.label == "Ulasan pelanggan").help


def _metric(at, label: str):
    return next(metric for metric in at.metric if metric.label == label)


QUOTE_LINE = re.compile(r"^:\w+-badge\[[+○±−·] (positif|netral|campuran|negatif|tanpa polaritas)\] ")


def _quotes(at) -> list[str]:
    return [QUOTE_LINE.sub("", item.value) for item in at.markdown if QUOTE_LINE.match(item.value)]


def _move_review(paths, record_id: str, relative_date: str) -> None:
    corpus = paths["run"] / "corpus.csv"
    with corpus.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["relative_date"] = relative_date if row["record_id"] == record_id else row["relative_date"]
    with corpus.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def test_k1_trend_window_follows_the_corpus_calendar_and_the_year_filter(fixture_paths, monkeypatch):
    _move_review(fixture_paths, "R06", "March 2026")  # 1 review in the last 12 months, 2 in the 12 before
    monkeypatch.setattr(bi_dashboard.data, "MIN_WINDOW", 1)
    calls = []
    real = bi_dashboard.data.monthly_series

    def spy(reviews, collected_at, *, basis=None, years=None):
        calls.append({"rows": len(reviews), "basis": None if basis is None else len(basis), "years": years})
        return real(reviews, collected_at, basis=basis, years=years)

    monkeypatch.setattr(bi_dashboard.data, "monthly_series", spy)
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert _metric(at, "Ulasan pelanggan").proto.delta == "-50%" and calls[-1] == {"rows": 12, "basis": 12, "years": None}
    at.multiselect(key="bi_venues").set_value(["batik_popiler_2"]).run()
    _okay(at)
    assert calls[-1] == {"rows": 4, "basis": 12, "years": None}  # one venue still uses the corpus' incomplete month
    at.multiselect(key="bi_venues").set_value([])
    at.select_slider(key="bi_years").set_value((2025, 2026)).run()
    _okay(at)
    k1 = _metric(at, "Ulasan pelanggan")
    assert calls[-1]["years"] == (2025, 2026) and not k1.proto.delta and "filter tahun memotong jendela" in k1.help


def test_quotes_follow_the_source_and_year_filters(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    at.pills(key="bi_sources").set_value(["SeekSophie"]).run()
    _okay(at)
    assert _quotes(at) == ["Wonderful batik painting session in Bali"]  # the only SeekSophie review in that cell
    at.pills(key="bi_sources").set_value([])
    at.select_slider(key="bi_years").set_value((2025, 2025)).run()
    _okay(at)
    assert _quotes(at) == ["the communication was slow"]  # the 2023 quote of the same cell stays out


def test_a_clicked_heatmap_cell_picks_the_quotes(fixture_paths, monkeypatch):
    seen = []
    monkeypatch.setattr(bi_dashboard, "_selected_cell", lambda state: seen.append(state) or ("fasilitas_kenyamanan", "batik_popiler_2"))
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert seen and any("**Popiler 2 × Fasilitas**" in caption.value for caption in at.caption)
    assert _quotes(at) == ["Lovely shop"]


def test_term_and_cell_clicks_are_read_from_the_chart_state(monkeypatch):
    state = {"bi_terms": {"selection": {"pilih_kata": [{"term": "communication"}]}}, "bi_terms_venue": {"selection": {"pilih_kata": []}},
             "bi_kwic_q": "wax"}
    monkeypatch.setattr(bi_dashboard.st, "session_state", state)
    bi_dashboard._pick_term("bi_terms_venue")
    assert state["bi_kwic_q"] == "wax"  # a cleared selection keeps the query
    bi_dashboard._pick_term("bi_terms")
    assert state["bi_kwic_q"] == "communication"
    cell = {"selection": {"pilih_sel": [{"aspek_kode": "fasilitas_kenyamanan", "venue_kode": "batik_popiler_2"}]}}
    assert bi_dashboard._selected_cell(cell) == ("fasilitas_kenyamanan", "batik_popiler_2")
    assert bi_dashboard._selected_cell({"selection": {"pilih_sel": {}}}) is None and bi_dashboard._selected_cell(None) is None


def test_read_errors_show_an_error_without_metrics(fixture_paths):
    latest, corpus = fixture_paths["expansion"] / "latest.json", fixture_paths["run"] / "corpus.csv"
    header, *rows = corpus.read_text(encoding="utf-8-sig").splitlines()
    for broken in ("latest", "run_id", "column"):
        if broken == "latest":
            latest.write_text("{", encoding="utf-8")
        elif broken == "run_id":
            latest.write_text(json.dumps({"run_id": "../x"}), encoding="utf-8")
        else:
            latest.write_text(json.dumps({"run_id": fixture_paths["run"].name}), encoding="utf-8")
            corpus.write_text("\n".join([header.replace('"clean_text"', '"text"'), *rows]), encoding="utf-8-sig")
        _clear_caches()
        at = AppTest.from_string(SCRIPT, default_timeout=30).run()
        _okay(at)
        assert at.error and not at.metric and not at.tabs, broken


def test_print_mode_enlarges_chart_text_and_hides_table_toggles(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30)
    at.query_params["cetak"] = "1"
    at.run()
    _okay(at)
    assert {json.loads(chart.proto.spec)["config"]["axis"]["labelFontSize"] for chart in at.get("vega_lite_chart")} == {15}
    assert any("st-key-bi_table_" in item.value for item in at.markdown if "<style>" in item.value)
    plain = AppTest.from_string(SCRIPT, default_timeout=30).run()
    assert {json.loads(chart.proto.spec)["config"]["axis"]["labelFontSize"] for chart in plain.get("vega_lite_chart")} == {12}


def test_kwic_caption_escapes_the_query(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    at.text_input(key="bi_kwic_q").set_value("*x*").run()
    _okay(at)
    assert any(r"memuat '\*x\*'" in caption.value for caption in at.caption)


def test_identity_card_follows_the_run_on_screen(fixture_paths):
    newer = fixture_paths["expansion"] / "runs" / "20261001T180000000000Z_newer" / "audit.json"
    newer.parent.mkdir(parents=True)
    newer.write_text(json.dumps({"created_at": "2026-10-01T18:00:00+00:00", "counts": {"stored_unique_source_identities": 50}}), encoding="utf-8")
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    card = _metric(at, "Identitas unik terkumpul")
    assert (card.value, card.proto.delta) == ("12", "+3")  # = funnel stage 1 of this run, not the newer run's 50
    (fixture_paths["run"] / "audit.json").unlink()
    _clear_caches()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert "Identitas unik terkumpul" not in {metric.label for metric in at.metric}
    assert any("audit belum tersedia" in info.value for info in at.info)


def test_one_missing_airbnb_report_is_named_in_the_per_target_summary(fixture_paths):
    (fixture_paths["expansion"] / "airbnb" / "five-report.json").unlink()
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    k5 = _metric(at, "Cakupan per target")
    assert k5.value == "—" and "Airbnb · Five Arts" in k5.help
    assert any("Laporan Airbnb · Five Arts tidak tersedia" in info.value for info in at.info)
    at.multiselect(key="bi_venues").set_value(["batik_popiler_2"]).run()
    _okay(at)
    assert _metric(at, "Cakupan per target").value == "1/1 target"  # the missing report is outside this venue


def test_pooled_cards_name_the_older_museum_labels(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert "termasuk 2 label lama Museum Tekstil" in _metric(at, "Ulasan berkeluhan").help
    assert sum("Data termasuk 2 label lama Museum Tekstil." in caption.value for caption in at.caption) == 2  # rating and length cards
    at.multiselect(key="bi_venues").set_value(["five_art_studio"]).run()
    _okay(at)
    assert not any("label lama" in caption.value for caption in at.caption)


def _record_metadata(path: Path, **metadata) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(metadata)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def test_annotation_filters_export_recorded_fields_and_preserve_unknowns(fixture_paths, monkeypatch):
    _record_metadata(fixture_paths["labels_real"], annotation_session="recorded-A", annotation_rubric="rubric-v2",
                     annotation_prompt="prompt-v3", annotation_date="2026-09-28", backend_model_id="recorded-model")
    _record_metadata(fixture_paths["labels_legacy"], annotation_session="recorded-B")
    exports = []
    original = bi_dashboard.st.download_button

    def capture(label, payload, **kwargs):
        exports.append(payload.decode("utf-8-sig"))
        return original(label, payload, **kwargs)

    monkeypatch.setattr(bi_dashboard.st, "download_button", capture)
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    for key, value in (("bi_sessions", "recorded-A"), ("bi_rubrics", "rubric-v2"), ("bi_prompts", "prompt-v3"),
                       ("bi_dates", "2026-09-28"), ("bi_models", "recorded-model"), ("bi_verifications", "unverified")):
        assert value in at.multiselect(key=key).options
    assert {"annotation_session", "annotation_prompt", "annotation_date", "annotation_model", "annotation_verification",
            "presence_verification", "polarity_verification"} <= set(exports[-1].splitlines()[0].split(","))
    assert "unknown" in exports[-1] and not any(secret in exports[-1] for secret in SECRETS)
    at.multiselect(key="bi_sessions").set_value(["recorded-A"]).run()
    _okay(at)
    assert _metric(at, "Ulasan pelanggan").value == "11"  # metadata does not invent a dated-corpus subset
    assert "recorded-A" in exports[-1] and "recorded-B" not in exports[-1]
    assert "rubric-v2" in exports[-1] and "prompt-v3" in exports[-1] and "recorded-model" in exports[-1]


def test_all_triggered_insights_remain_visible(fixture_paths, monkeypatch):
    items = [bi_dashboard.data.Insight(f"rule-{i}", "Info", str(i), f"review-trigger-{i}", f"basis-{i}", i) for i in range(7)]
    monkeypatch.setattr(bi_dashboard.data, "compute_insights", lambda *args: items)
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    for i in range(7):
        assert f"review-trigger-{i}" in _shown_text(at)
        assert f"basis-{i}" in _shown_text(at)


def test_mixed_sessions_split_venue_comparisons_and_never_draw_a_pooled_reference(fixture_paths, monkeypatch):
    seen = []
    original = bi_dashboard.charts.venue_share_spec

    def capture(rows, overall, cfg):
        seen.append((set(rows["annotation_session"]), overall, list(rows["venue_key"])))
        return original(rows, overall, cfg)

    monkeypatch.setattr(bi_dashboard.charts, "venue_share_spec", capture)
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    assert len(seen) >= 2 and all(len(sessions) == 1 and overall != overall for sessions, overall, _ in seen)
    assert all(venues == sorted(venues, key=lambda venue: (bi_dashboard.data.venue_rank(venue), venue)) for _, _, venues in seen)
    assert any("Perbandingan lintas batch" in item.value for item in at.info)


def test_coverage_never_pools_percentages_and_names_unknown_denominators(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    summary = _metric(at, "Cakupan per target")
    assert summary.value == "4/5 target" and "%" not in summary.value
    assert "Giriloyo" in summary.help and "tak diketahui" in summary.help
    assert "0" in summary.help and "tidak digabung" in summary.help
    assert any("nol terkumpul" in caption.value and "tidak memiliki persentase" in caption.value for caption in at.caption)


def test_method_separates_display_masking_from_raw_capture_governance(fixture_paths):
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    shown = _shown_text(at)
    assert "Capture mentah tetap ada" in shown and "jadwal penghapusan belum tercatat" in shown
    assert "Presisi/recall penyamaran" in shown and "belum diukur" in shown
    assert "tidak mencakup bias pengumpulan atau galat anotasi" in shown
    assert "pekat = |z|" not in shown


def test_trend_exposes_calendar_and_heuristic_sensitivity_with_counts(fixture_paths, monkeypatch):
    _move_review(fixture_paths, "R06", "September 2026")
    monkeypatch.setattr(bi_dashboard.data, "MIN_WINDOW", 1)
    monkeypatch.setattr(bi_dashboard.data, "partial_month", lambda *args: bi_dashboard.pd.Timestamp(2026, 9, 1))
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    _okay(at)
    tables = [frame.value for frame in at.dataframe if "Aturan jendela" in frame.value]
    assert len(tables) == 1
    assert tables[0]["Aturan jendela"].tolist() == ["Bulan kalender lengkap (utama)", "Eksklusi bulan rendah (sensitivitas)"]
    assert {"Periode kini", "n kini", "Periode pembanding", "n pembanding", "Perubahan"} <= set(tables[0])
    assert any("pola musiman" in caption.value and "bukan bukti perubahan permintaan" in caption.value for caption in at.caption)
