"""Dashboard BI page: honest KPIs, NLP insight and scraping operations, read-only from local exports."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from temajelajah import bi_charts as charts
from temajelajah import bi_data as data
from temajelajah import bi_nlp as nlp
from temajelajah.bi_data import fmt_ci, fmt_dec, fmt_int, fmt_pct

BI_CSS = """
<style>
.tj-hero{padding:18px 26px;margin-bottom:14px}.tj-hero p{display:none}.tj-hero h1{font-size:1.55rem}
[class*="st-key-bi_card_"],.st-key-bi_filterbar{background:#FFFFFF;border-radius:16px}
[class*="st-key-bi_card_"] h5{font-size:1.02rem;font-weight:650;color:#0F3B45;letter-spacing:-.01em;padding:0}
[data-testid="stMainBlockContainer"].block-container{padding-left:2.4rem;padding-right:2.4rem}
[class*="st-key-bi_insight_"]{background:#EAF4F3;border-radius:12px;padding:12px 14px}
[class*="st-key-bi_insight_warn"]{border-left:4px solid #C2702F}
[class*="st-key-bi_insight_info"]{border-left:4px solid #0B6E99}
[class*="st-key-bi_insight_"] strong{font-size:1.1rem;color:#0F3B45}
[data-testid="stMetricChart"] svg{background-color:transparent !important}
.st-key-bi_kpis [data-testid="stMetricDeltaDescription"],.st-key-bi_kpis [data-testid="stMetricDeltaDescription"] p{white-space:normal;overflow:visible}
.st-key-bi_kpis [data-testid="stMetric"]{padding:16px 15px}
.st-key-bi_filterbar [data-testid="stCaptionContainer"] p{margin:0}
[class*="st-key-bi_"] [data-testid="stMultiSelect"] .react-aria-ComboBox>div,[class*="st-key-bi_"] [data-testid="stTextInputRootElement"]{background-color:#EAF4F3;border-color:#6F8F92}
[class*="st-key-bi_"] [data-testid="stMultiSelect"] .react-aria-ComboBox>div:focus-within,[class*="st-key-bi_"] [data-testid="stTextInputRootElement"]:focus-within{border-color:#0E7C7B}
.st-key-bi_kpis [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],.st-key-bi_kpis [data-testid="stMetricLabel"] p{white-space:normal;overflow:visible;text-overflow:clip}
@media (max-width:1439px){.st-key-bi_kpis [data-testid="stMetricLabel"]{min-height:3.1em;align-items:start}}
[class*="st-key-bi_head_"] [data-testid="stMarkdownContainer"]{margin-bottom:0}
[class*="st-key-bi_card_"] [data-testid="stHeaderActionElements"]{display:none}
[class*="st-key-bi_"] [data-testid="stCheckbox"]:not([data-selected="true"])>label>div:not([data-testid]){background-color:#6F8F92}
</style>
"""
PRINT_CSS = """
<style>
[class*="st-key-bi_table_"]{display:none}
.st-key-bi_badges [data-testid="stMarkdownContainer"] p,.st-key-bi_badges [data-testid="stMarkdownContainer"]{font-size:1rem}
.st-key-bi_kpis [data-testid="stMetricLabel"] p{font-size:1rem}
</style>
"""  # ?cetak=1 (paper figures): the "Lihat tabel" toggles are screen-only; the view switches stay, they name the view shown
PROVENANCE_FILTERS = (("sessions", "annotation_session", "Sesi anotasi"),
                      ("rubrics", "annotation_rubric", "Versi rubrik"),
                      ("prompts", "annotation_prompt", "Versi prompt"),
                      ("dates", "annotation_date", "Tanggal anotasi"),
                      ("models", "annotation_model", "Model anotasi"),
                      ("verifications", "annotation_verification", "Status verifikasi"))
FILTER_KEYS = ("bi_venues", "bi_sources", "bi_years", "bi_undated", *(f"bi_{name}" for name, _, _ in PROVENANCE_FILTERS))
PAIRED_CARDS = frozenset({"trend", "insights", "venue", "rating", "aspect", "quotes", "terms", "length", "funnel", "coverage", "method_left",
                          "method_right"})
TABS = ("Ringkasan", "Wawasan NLP", "Operasi scraping", "Metodologi")
GRAINS = ("Bulanan", "Tahunan")
TONES = {"Perhatian": ("orange", ":material/error:", "warn"), "Info": ("blue", ":material/info:", "info")}
DELTA_COLORS = {"Perhatian": "orange", "Info": "blue"}  # K1 follows the insight tones, so its sparkline is not painted as "bad"
POLARITY_COLORS = {"negatif": "red", "campuran": "orange", "netral": "gray", "positif": "blue"}
NO_LABELS = "Belum ada label sentimen: kartu ini menunggu ekspor ml_experiments/dataset."
NO_ASPECTS = "Belum ada label aspek: kartu ini menunggu business_research/dataset/business_real.csv."
NO_AUDIT = "Laporan scraping atau audit belum tersedia untuk bagian ini."
MASK_NOTE = "nama disamarkan secara heuristik; sebagian bisa terlewat; akurasi penyamaran belum diukur"
UNDATED_HELP = ("Ulasan bertanggal relatif (mis. '3 minggu lalu': semua ulasan Google Maps dan sebagian Airbnb) tidak punya bulan pasti. "
                "Bila disertakan, ulasan ini ikut di setiap rentang tahun, tetapi tidak masuk tren bulanan.")
KWIC_WINDOW = 44  # context characters that fit the Kiri/Kanan columns, so the clipping "…" stays visible


def _paths(sig: tuple) -> dict[str, Path]:
    return {name: Path(path) for name, path, *_ in sig}


@st.cache_data(show_spinner="Memuat data lokal…", max_entries=2)
def _bundle(sig: tuple) -> dict:
    bundle = data.load_bundle(_paths(sig))
    bundle["names"] = nlp.name_vocab(bundle["reviews"], bundle["labels"], bundle["business"])
    return bundle


@st.cache_data(max_entries=16, show_spinner=False)
def _terms(sig: tuple, fkey: tuple, mode: str) -> tuple[pd.DataFrame, list[str]]:
    bundle = _bundle(sig)
    if bundle["labels"] is None:
        return pd.DataFrame(columns=nlp.TERM_COLS), []
    return nlp.distinctive_terms(data.apply_filters(bundle["labels"], data.Filters(*fkey)), mode, frozenset(bundle["names"]))


@st.cache_data(max_entries=2, show_spinner=False)
def _collector(sig: tuple) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    paths = _paths(sig)
    if "raw_snapshot" not in paths:
        return None
    try:
        return data.read_collector_snapshot(paths["raw_snapshot"])
    except (sqlite3.Error, ValueError, OSError):
        return None


def _go_to(page: str) -> None:
    st.session_state["page"] = page


def _filter_key(name: str) -> str:
    generation = st.session_state.get("bi_filter_gen", 0)
    return f"{name}_{generation}" if generation else name


def _reset_filters() -> None:
    """A reset rebuilds the filter widgets under fresh keys, so they start at their declared defaults. Popping the keys only
    leaves the browser widgets showing the old values, which they send back on the next rerun; assigning the defaults through
    session state conflicts with the value= that keeps the year slider a range slider."""
    stale = [_filter_key(name) for name in FILTER_KEYS]
    st.session_state["bi_filter_gen"] = st.session_state.get("bi_filter_gen", 0) + 1
    for key in stale:
        st.session_state.pop(key, None)


def _pick_term(chart_key: str = "bi_terms") -> None:
    terms = data.selected_values(st.session_state.get(chart_key), "pilih_kata", "term")
    if terms:
        st.session_state["bi_kwic_q"] = terms[0]


def _selected_cell(state) -> tuple[str, str] | None:
    """The aspect x venue cell clicked on the heatmap (chart state "bi_heat"), or None for the default cell."""
    aspects, venues = (data.selected_values(state, "pilih_sel", field) for field in ("aspek_kode", "venue_kode"))
    return (aspects[0], venues[0]) if aspects and venues else None


def _printing() -> bool:
    return st.query_params.get("cetak") == "1"


def _card(key: str, title: str | None = None):
    """Bordered card; side-by-side cards share one height on screen. In print mode every card keeps its own height, so a figure
    cut from one card carries no blank band."""
    box = st.container(border=True, key=f"bi_card_{key}", height="stretch" if key in PAIRED_CARDS and not _printing() else "content")
    if title:
        box.markdown(f"##### {title}")
    return box


def _head(key: str):
    """Title row with its local control right after the title: the chart toolbar that appears top-right on hover stays clear of it."""
    return st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="left", gap="medium", key=f"bi_head_{key}")


def _chart(spec: dict, *, key: str | None = None, on_select="ignore", selection_mode: str | None = None, width: str = "stretch"):
    return st.vega_lite_chart(spec=spec, theme=None, width=width, key=key, on_select=on_select, selection_mode=selection_mode)


def _table(frame: pd.DataFrame, key: str, **column_config) -> None:
    with st.expander("Lihat tabel", icon=":material/table_view:", key=f"bi_table_{key}"):
        st.dataframe(frame, hide_index=True, width="stretch", column_config=column_config or None)


def _when(stamp) -> str:
    return "—" if pd.isna(stamp) else f"{stamp.day} {data.MONTHS[stamp.month - 1]} {stamp.year}"


def _legacy_suffix(labels: pd.DataFrame | None) -> str:
    """Caption sentence for cards that pool the older Museum labels with the current labelling round."""
    note = data.legacy_note(labels)
    return f" Data {note}." if note else ""


@st.cache_data(max_entries=8, show_spinner=False)
def _ops_tables(sig: tuple, fkey: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle, f = _bundle(sig), data.Filters(*fkey)
    reviews = data.apply_filters(bundle["reviews"], f, use_years=False)
    labels = data.apply_filters(bundle["labels"], f, use_years=False) if bundle["labels"] is not None else None
    return data.funnel_table(reviews, labels), data.quality_matrix(reviews)


@st.cache_data(max_entries=2, show_spinner=False)
def _audit_messages(sig: tuple) -> list[str] | None:
    """Audit mismatches and unverified SHA-256 evidence; None when there is no audit to check against."""
    bundle = _bundle(sig)
    if not bundle["audit_counts"]:
        return None
    messages = data.reconcile_funnel(_ops_tables(sig, data.Filters().key())[0], bundle["audit_counts"])
    if bundle["snapshot_ok"] is False:
        messages.append("SHA-256 raw_snapshot.sqlite3 berbeda dari audit")
    if bundle["snapshot_missing"]:
        messages.append("raw_snapshot.sqlite3 tidak ditemukan; SHA-256 snapshot tidak diperiksa")
    elif bundle["snapshot_ok"] is None:
        messages.append("SHA-256 snapshot belum terverifikasi: hash acuan audit tidak tersedia atau tidak valid")
    return messages


@st.cache_data(max_entries=2, show_spinner=False)
def _telemetry(sig: tuple) -> pd.DataFrame:
    collector = _collector(sig)
    return data.run_telemetry(collector[0] if collector is not None else None, _bundle(sig)["reports"])


def _waiting() -> None:
    st.info("Menunggu snapshot korpus (exports/paper_expansion/latest.json)…")
    st.caption("Folder exports/ tidak ikut di repositori. Jalankan pengumpulan atau impor data terlebih dahulu; halaman ini hanya membaca berkas lokal.")
    st.button("Buka Cari Tempat", on_click=_go_to, args=("Cari Tempat",), key="bi_open_search", icon=":material/travel_explore:")


def _header(bundle: dict, sig: tuple) -> None:
    reviews, labels = bundle["reviews"], bundle["labels"]
    st.caption(f"Snapshot {_when(bundle['collected_at'])} · {reviews['source'].nunique()} platform · {reviews['venue_key'].nunique()} venue · "
               "berkas lokal, tanpa jaringan")
    with st.container(horizontal=True, key="bi_badges", gap="small"):
        if labels is not None:
            st.badge(f"Label asisten sementara · sentimen/polaritas · {fmt_int(labels['expert_approved'].sum())}/{fmt_int(len(labels))} disetujui ahli",
                     color="orange", icon=":material/warning:")
        else:
            st.badge("Label sentimen belum tersedia", color="gray", icon=":material/hourglass_empty:")
        business = bundle["business"]
        if business is not None and len(business):
            st.badge("Kehadiran aspek · pemeriksaan satu penulis dilaporkan", color="blue", icon=":material/fact_check:")
        st.badge("Kolom identitas pengulas tidak dimuat", color="gray", icon=":material/lock:")
        messages = _audit_messages(sig)
        if messages is None:
            st.badge("Audit belum tersedia", color="gray", icon=":material/help:")
        elif messages:
            st.badge("Snapshot belum terverifikasi" if bundle["snapshot_ok"] is None else "Snapshot belum cocok dengan audit",
                     color="orange", icon=":material/sync_problem:")
        else:
            st.badge("Snapshot cocok dengan audit", color="green", icon=":material/verified:")


def _filter_bar(bundle: dict) -> tuple[data.Filters, object, tuple[int, int] | None]:
    reviews = bundle["reviews"]
    present, platforms = set(reviews["venue_key"]), set(reviews["source"])
    venues = [key for key in data.VENUES if key in present] + sorted(present - set(data.VENUES))
    sources = [name for name in data.SOURCES if name in platforms] + sorted(platforms - set(data.SOURCES))
    years = reviews["year"].dropna().astype(int)
    bounds = (int(years.min()), int(years.max())) if len(years) else None
    with st.container(border=True, key="bi_filterbar"):
        venue_col, source_col, year_col = st.columns([3.3, 3.5, 3.2], vertical_alignment="bottom", gap="medium")
        chosen = venue_col.multiselect("Venue", venues, format_func=data.venue_name, key=_filter_key("bi_venues"), persist_state="session",
                                       placeholder="Semua venue")
        picked = source_col.pills("Sumber", sources, selection_mode="multi", key=_filter_key("bi_sources"), persist_state="session", wrap=True)
        if bounds is not None and bounds[1] > bounds[0]:
            span = year_col.select_slider("Tahun", options=list(range(bounds[0], bounds[1] + 1)), value=bounds, key=_filter_key("bi_years"),
                                          persist_state="session")
        else:
            span = bounds
            year_col.caption(f"Tahun: {bounds[0]}" if bounds else "Tahun: belum ada tanggal pasti")
        status_col, undated_col, reset_col = st.columns([4.2, 4.0, 1.8], vertical_alignment="center", gap="medium")  # fits 1280 px + sidebar
        status = status_col.empty()
        undated = undated_col.toggle("Sertakan ulasan tanpa bulan pasti", value=True, key=_filter_key("bi_undated"), persist_state="session",
                                     help=UNDATED_HELP)
        reset_col.button("Atur ulang", on_click=_reset_filters, key="bi_reset", width="stretch", icon=":material/restart_alt:")
        annotation_filters = {}
        with st.expander("Provenans anotasi: filter label dan perbandingan", key="bi_provenance_filters"):
            available = [frame for frame in (bundle["labels"], bundle["business"]) if frame is not None]
            for offset in range(0, len(PROVENANCE_FILTERS), 3):
                for column, (name, field, title) in zip(st.columns(3), PROVENANCE_FILTERS[offset:offset + 3]):
                    values = sorted({str(value) for frame in available if field in frame for value in frame[field].dropna().unique()})
                    annotation_filters[name] = tuple(column.multiselect(title, values, key=_filter_key(f"bi_{name}"),
                                                                        placeholder="Semua", persist_state="session"))
            st.caption("Sesi = proksi batch dari asal berkas, bukan sesi model yang berhasil direkonstruksi. unknown = tidak tercatat dalam berkas sumber; "
                       "tanggal pengumpulan bukan tanggal anotasi. Filter ini berlaku pada label, "
                       "perbandingan dan ekspor, sementara volume korpus serta operasi scraping memakai filter venue/sumber/tahun.")
    return data.Filters(tuple(chosen or ()), tuple(picked or ()), tuple(span) if span else None, undated, **annotation_filters), status, bounds


def _kpi_chip(key: str, card: dict) -> str | None:
    """Every share card shows its k and n on the card itself; the help tooltip keeps the interval and caveats."""
    if key == "K2":
        return card["help"] if card.get("n") else None
    if key == "K5":
        return (f"{card['targets_unknown']} penyebut unknown"
                if "targets_known" in card else None)
    unit = {"K3": "teks", "K4": "keluhan", "K5": "ulasan"}.get(key)
    if unit is None or "k" not in card:
        return None
    return f"{fmt_int(card['k'])}/{fmt_int(card['n'])} {unit}" if key == "K5" else f"{fmt_int(card['k'])} dari {fmt_int(card['n'])} {unit}"


def _kpi_row(cards: dict) -> None:
    shown = [(key, card) for key in ("K1", "K2", "K3", "K4", "K5") if (card := cards.get(key))]
    with st.container(key="bi_kpis"):
        for column, (key, card) in zip(st.columns([1.3 if key == "K1" else 1 for key, _ in shown]), shown):
            lead = key == "K1"
            column.metric(card["label"], card["value"], delta=card.get("delta") if lead else _kpi_chip(key, card),
                          delta_description=card.get("delta_description") if lead else None,
                          delta_color=DELTA_COLORS.get(card.get("tone"), "normal") if lead else "gray",
                          delta_arrow="auto" if lead else "off", help=card.get("help"), chart_data=card.get("chart_data") or None,
                          chart_type=card.get("chart_type") or "line", height="stretch")


def _trend_card(monthly: pd.DataFrame, windows: dict, cfg: dict, through) -> None:
    with _card("trend"):
        head = _head("trend")
        title = head.empty()
        grain = head.segmented_control("Skala", GRAINS, default=GRAINS[0], key="bi_grain", persist_state="session", required=True,
                                       label_visibility="collapsed") or GRAINS[0]
        title.markdown(f"##### Volume ulasan per {'bulan' if grain == GRAINS[0] else 'tahun'}")
        if monthly.empty:
            st.info("Belum ada ulasan bertanggal pada filter ini.")
            return
        _chart(charts.trend_spec(monthly, grain, cfg, through=through))
        pale = "Batang pucat = bulan rendah yang ditandai heuristik; belum terbukti tidak lengkap. Garis = rata-rata 12 bulan." if grain == GRAINS[0] else "Batang pucat = tahun berjalan, belum lengkap."
        st.caption(f"{fmt_int(windows['dated'])} ulasan bertanggal dipetakan; {fmt_int(windows['undated'])} hanya bertanggal relatif tidak "
                   f"dipetakan. {pale}")
        sensitivity = windows.get("sensitivity")
        if sensitivity is not None:
            alternatives = []
            for label, item in (("Bulan kalender lengkap (utama)", windows), ("Eksklusi bulan rendah (sensitivitas)", sensitivity)):
                alternatives.append({"Aturan jendela": label, "Periode kini": item["current_label"], "n kini": item["current"],
                                     "Periode pembanding": item["previous_label"], "n pembanding": item["previous"],
                                     "Perubahan": data.fmt_change(item["change"]) if item.get("ok") and item.get("change") is not None else "—"})
            st.dataframe(pd.DataFrame(alternatives), hide_index=True, width="stretch")
            st.caption("Jendela utama memakai semua bulan kalender lengkap. Heuristik hanya menandai volume bulan terakhir di bawah 50% median "
                       "12 bulan sebelumnya; pola musiman, penurunan nyata dan nol aktivitas juga dapat memicunya. Kedua hasil adalah sensitivitas "
                       "volume ulasan, bukan bukti perubahan permintaan atau jumlah kunjungan.")
        table = pd.DataFrame({"Bulan": [data.fmt_month(month) for month in monthly["bulan"]], "Ulasan": monthly["n"].astype(int),
                              "Rata-rata 12 bulan": [fmt_dec(value, 1) for value in monthly["rata12"]],
                              "Rating rata-rata": [fmt_dec(value, 2) for value in monthly["rating"]],
                              "Catatan": ["rendah, ditandai heuristik" if flag else "" for flag in monthly["parsial"]]}).iloc[::-1]
        _table(table, "trend")


def _insights_card(insights: list[data.Insight]) -> None:
    with _card("insights", "Sorotan data"):
        if not insights:
            st.caption("Belum ada sorotan yang lolos ambang data pada filter ini.")
        for index, item in enumerate(insights):
            color, icon, tone = TONES.get(item.tone, TONES["Info"])
            with st.container(border=True, key=f"bi_insight_{tone}_{index}", gap="xxsmall"):
                st.markdown(f":{color}-badge[{icon} {item.tone}] **{item.value}** · {item.text}")
                st.caption(item.basis)
        if insights:
            st.caption("IK95% = interval Wilson 95%, bersyarat pada ulasan yang berhasil dikumpulkan dan label sementara. "
                       "Interval tidak mengukur bias ulasan yang tidak terambil atau galat anotasi. Tumpang tindih interval bukan uji hipotesis.")


def _venue_card(labels: pd.DataFrame | None, coverage: pd.DataFrame | None, cfg: dict) -> None:
    with _card("venue", "Porsi keluhan per venue dan sesi anotasi"):
        if labels is None or labels.empty:
            st.info(NO_LABELS if labels is None else "Tidak ada ulasan berlabel pada filter ini.")
            return
        blocker = data.session_comparison_blocker(labels)
        if blocker:
            st.info(blocker)
        tables = []
        for session, group in labels.groupby("annotation_session", sort=False, dropna=False):
            rows = data.venue_complaints(group)
            st.caption(f"Batch/sesi proksi: {nlp.md_escape(str(session))} · kontras deskriptif, tanpa peringkat lintas sesi.")
            _chart(charts.venue_share_spec(rows, float("nan"), cfg))
            tables.append(pd.DataFrame({"Sesi anotasi": str(session), "Venue": rows["venue"], "Ulasan berlabel": rows["n"],
                                        "Berkeluhan": rows["k"], "Porsi": [fmt_pct(value) for value in rows["share"]], "IK95%": rows["ci"]}))
        note = ("Garis hitam = interval Wilson 95%, bersyarat pada korpus terkumpul dan label sementara. "
                "Interval tidak mencakup bias cakupan, galat anotasi atau perbedaan protokol antar sesi. Urutan venue tetap; bukan peringkat kualitas.")
        if labels["cohort"].eq("legacy81").any():
            museum = coverage[coverage["venue_key"].eq("museum_tekstil")] if coverage is not None else pd.DataFrame()
            share = f" dengan cakupan {fmt_pct(museum['coverage'].iloc[0])}" if len(museum) and museum["coverage"].notna().any() else ""
            note += f" Museum memakai label lama Google Maps{share}."
        st.caption(note)
        _table(pd.concat(tables, ignore_index=True), "venue")


def _rating_card(labels: pd.DataFrame | None, cfg: dict) -> None:
    with _card("rating", "Rating bintang vs isi teks"):
        if labels is None or labels.empty:
            st.info(NO_LABELS if labels is None else "Tidak ada ulasan berlabel pada filter ini.")
            return
        cells, agreement = data.rating_text_matrix(labels)
        _chart(charts.rating_text_spec(cells, cfg))
        hidden = int(cells.loc[cells["hidden"], "n"].sum())
        st.caption(f"Selaras {fmt_pct(agreement)}. Kotak berbingkai gelap = {fmt_int(hidden)} ulasan berkeluhan yang tetap memberi "
                   f"{data.STARS_4_5}. Warna = porsi dalam baris rating.{_legacy_suffix(labels)}")
        table = cells.pivot(index="rating", columns="sentimen", values="label").reindex(index=range(5, 0, -1), columns=list(data.SENTIMENTS))
        table.index = [f"★{rating}" for rating in table.index]
        _table(table.reset_index(names="Rating"), "rating")


def _tab_overview(monthly, windows, insights, labels, coverage, cfg, through) -> None:
    with st.container(key="bi_row_top"):
        left, right = st.columns([7, 5])
        with left:
            _trend_card(monthly, windows, cfg, through)
        with right:
            _insights_card(insights)
    with st.container(key="bi_row_mid"):
        left, right = st.columns(2)
        with left:
            _venue_card(labels, coverage, cfg)
        with right:
            _rating_card(labels, cfg)


def _aspect_cards(business: pd.DataFrame | None, evidence: pd.DataFrame | None, cfg: dict, mask) -> None:
    left, right = st.columns([7, 5])
    with left, _card("aspect"):
        head = _head("aspect")
        head.markdown("##### Peta aspek × venue")
        metric = head.segmented_control("Metrik", data.METRICS, default="Keluhan", key="bi_heat_metric", persist_state="session", required=True,
                                        label_visibility="collapsed") or "Keluhan"
        if business is None or business.empty:
            st.info(NO_ASPECTS if business is None else "Tidak ada ulasan berlabel aspek pada filter ini.")
            cell = None
        else:
            sessions = sorted(business["annotation_session"].astype(str).unique())
            if len(sessions) > 1:
                st.info("Aspek ditampilkan dalam satu batch/sesi proksi. Perbandingan lintas batch memerlukan anotasi ulang pada protokol yang sama.")
                session = st.selectbox("Batch aspek untuk perbandingan", sessions, key="bi_aspect_session", persist_state="session")
                business = business[business["annotation_session"].eq(session)]
                evidence = evidence[evidence["record_id"].isin(business["record_id"])] if evidence is not None else None
            else:
                st.caption(f"Batch/sesi proksi: {nlp.md_escape(sessions[0])}")
            matrix = data.aspect_venue_matrix(business, metric, _selected_cell(st.session_state.get("bi_heat")))
            _chart(charts.aspect_heatmap_spec(matrix, metric, cfg), key="bi_heat", on_select="rerun", selection_mode="pilih_sel")
            cell = matrix[matrix["terpilih"]].iloc[0]
            what = "porsi ulasan venue yang menyebut aspek" if metric == "Penyebutan" else "porsi penyebutan aspek yang berkeluhan"
            st.caption(f"Warna = {what}. Label sel menampilkan pembilang/penyebut; NA = penyebut nol, berbeda dari 0. "
                       "Klik sel untuk membaca kutipan bukti; bingkai gelap = sel terpilih. 'n kecil' = penyebut sel < 30. "
                       "Kehadiran aspek: pemeriksaan satu penulis dilaporkan; polaritas tetap sementara dan belum divalidasi.")
            table = matrix.assign(porsi_sebut=[fmt_pct(v) for v in matrix["rate"]], porsi_keluhan=[fmt_pct(v) for v in matrix["complaint_rate"]])
            _table(table.rename(columns={"aspek": "Aspek", "venue": "Venue", "n": "Ulasan venue", "mentions": "Penyebutan",
                                         "polarity_n": "Polaritas tersedia (m)", "missing_polarity": "Polaritas hilang", "complaints": "Berkeluhan (k)",
                                         "porsi_sebut": "Porsi menyebut", "porsi_keluhan": "Porsi berkeluhan"})
                   [["Aspek", "Venue", "Ulasan venue", "Penyebutan", "Polaritas tersedia (m)", "Polaritas hilang", "Berkeluhan (k)",
                     "Porsi menyebut", "Porsi berkeluhan"]], "aspect")
    with right, _card("quotes", "Kutipan bukti"):
        if cell is None or evidence is None:
            st.info(NO_ASPECTS if business is None else "Pilih filter lain untuk melihat kutipan.")
            return
        short = data.ASPECT_SHORT[cell["aspek_kode"]]
        if metric == "Keluhan":
            count = (f"{fmt_int(cell['complaints'])} berkeluhan dari {fmt_int(cell['polarity_n'])} penyebutan dengan polaritas tersedia; "
                     f"{fmt_int(cell['missing_polarity'])} polaritas hilang")
        else:
            count = f"{fmt_int(cell['mentions'])} dari {fmt_int(cell['n'])} ulasan menyebut"
        st.caption(f"**{cell['venue']} × {short}** · {count} · {MASK_NOTE}")
        quote_limit = 3 if _printing() else 8
        quotes = data.cell_quotes(evidence, cell["venue_kode"], cell["aspek_kode"], mask, limit=quote_limit)
        with st.container(height="content" if _printing() else 300, key="bi_quote_list"):
            if quotes.empty:
                st.caption("Belum ada kutipan bukti untuk sel ini.")
            for quote in quotes.itertuples(index=False):
                color = POLARITY_COLORS.get(quote.polarity, "gray")
                st.markdown(f":{color}-badge[{quote.glyph} {quote.polarity or 'tanpa polaritas'}] {quote.quote}")
        st.caption(f"Urutan: negatif → campuran → netral → positif, lalu kutipan terpendek. Maks. {quote_limit} kutipan, ≤240 karakter."
                   + (" Tampilan cetak." if _printing() else ""))


def _terms_card(labels: pd.DataFrame | None, sig: tuple, fkey: tuple, cfg: dict) -> None:
    with _card("terms"):
        head = _head("terms")
        head.markdown("##### Kata pembeda")
        mode = head.segmented_control("Mode", nlp.TERM_MODES, default=nlp.TERM_MODES[0], key="bi_terms_mode", persist_state="session",
                                      required=True, label_visibility="collapsed") or nlp.TERM_MODES[0]
        if labels is None:
            st.info(NO_LABELS)
            return
        blocker = nlp.terms_blocker(labels, mode)
        if blocker:  # an empty or tiny comparison group yields only rounding noise
            st.info(blocker)
            return
        terms, skipped = _terms(sig, fkey, mode)
        english = int(labels["source_locale"].eq("en").sum())
        if terms.empty:
            st.info(f"Belum cukup ulasan berbahasa Inggris pada filter ini ({fmt_int(english)}) untuk menghitung kata pembeda.")
            return
        key = "bi_terms" if mode == nlp.TERM_MODES[0] else "bi_terms_venue"
        _chart(charts.terms_spec(terms, mode, cfg), key=key, on_select=partial(_pick_term, key), selection_mode="pilih_kata",
               width="stretch" if mode == nlp.TERM_MODES[0] else "content")
        sides = "Kanan (oranye) = khas ulasan berkeluhan, kiri (biru) = khas ulasan lain. " if mode == nlp.TERM_MODES[0] else ""
        untagged = labels[labels["source_locale"].eq("")]  # e.g. every older Museum label: no language metadata, so never analysed
        gap = (f" ({fmt_int(len(untagged))} teks tanpa metadata bahasa, {fmt_int(untagged['complaint'].sum())} di antaranya berkeluhan, "
               "tidak dianalisis)" if len(untagged) else "")
        st.caption(f"{sides}Log-odds berbobot, prior Dirichlet informatif (Monroe dkk., 2008) · {fmt_int(english)} ulasan berbahasa Inggris{gap} · "
                   "massa kelompok = jumlah insidensi term–dokumen setelah penyaringan, bukan jumlah ulasan · nama disaring. "
                   "Skor z hanya urutan eksploratif; tanpa ambang signifikansi atau koreksi perbandingan berganda. Klik kata untuk membuka konteksnya di KWIC.")
        if skipped:
            st.caption("Dilewati karena ulasan berbahasa Inggris terlalu sedikit (< 30 per venue): " + ", ".join(skipped) + ".")
        shown = terms.assign(z=[fmt_dec(value, 2) for value in terms["z"]])
        names = {"term": "Kata", "venue": "Venue", "z": "z", "df_a": "Ulasan kelompok", "df_b": "Ulasan pembanding", "sisi": "Lebih khas pada",
                 "n_a": "Massa kelompok", "n_b": "Massa pembanding", "docs_a": "Jumlah ulasan kelompok", "docs_b": "Jumlah ulasan pembanding"}
        _table(shown.rename(columns=names)[[name for key, name in names.items() if key in shown.columns]], "terms")


def _length_card(labels: pd.DataFrame | None, cfg: dict) -> None:
    with _card("length", "Panjang ulasan vs keluhan"):
        if labels is None or labels.empty:
            st.info(NO_LABELS if labels is None else "Tidak ada ulasan berlabel pada filter ini.")
            return
        bins = nlp.length_bins(labels)
        _chart(charts.length_spec(bins, cfg))
        longest, rest = bins.iloc[-1], bins.iloc[:-1]
        if longest["n"] and longest["lo"] > rest["hi"].max():
            st.caption(f"Pada korpus terkumpul, porsi keluhan kelompok {longest['bin']} kata adalah {longest['label']}; "
                       "interval deskriptif tidak tumpang tindih dengan kelompok lain. Ini bukan uji beda formal.")
        else:
            st.caption("Pola panjang ulasan vs keluhan masih indikasi awal: interval kepercayaan tumpang tindih.")
        st.caption(f"Garis hitam = IK95% bersyarat pada korpus dan label sementara; garis tipis = rata-rata {fmt_pct(labels['complaint'].mean())} ulasan berkeluhan. "
                   "Bias ulasan yang tidak terambil dan galat anotasi tidak tercakup. Panjang = jumlah kata; "
                   f"aksara Han/Kana yang ditulis tanpa spasi dihitung ±2 karakter per kata.{_legacy_suffix(labels)}")
        _table(pd.DataFrame({"Panjang (kata)": bins["bin"], "Ulasan": bins["n"], "Berkeluhan": bins["k"], "Porsi": bins["label"],
                             "IK95%": [fmt_ci(lo, hi) for lo, hi in zip(bins["lo"], bins["hi"])]}), "length")


def _kwic_card(labels: pd.DataFrame | None, default_query: str | None, mask) -> None:
    with _card("kwic", "Cari kata dalam konteks (KWIC)"):
        if labels is None:
            st.info(NO_LABELS)
            return
        if "bi_kwic_q" not in st.session_state and default_query:
            st.session_state["bi_kwic_q"] = default_query
        query_col, only_col, _ = st.columns([3, 2, 3], vertical_alignment="bottom")
        query = query_col.text_input("Kata atau frasa", key="bi_kwic_q", max_chars=40, placeholder="mis. wax, english, communication",
                                     persist_state="session", icon=":material/search:")
        only = only_col.toggle("Hanya ulasan berkeluhan", key="bi_kwic_only", persist_state="session")
        rows, stats = nlp.kwic(labels, query or "", mask, window=KWIC_WINDOW, only_complaints=only)
        if len(stats["query"]) < 2:
            st.caption("Ketik minimal 2 huruf, atau klik kata pada grafik Kata pembeda.")
            return
        shown = nlp.md_escape(stats["query"])
        if stats["matches"]:
            st.caption(f"{fmt_int(stats['matches'])} ulasan memuat '{shown}' · {fmt_int(stats['complaints'])} berkeluhan "
                       f"({fmt_pct(stats['share'])}, IK95% {fmt_ci(stats['lo'], stats['hi'])}) · ditampilkan {fmt_int(stats['shown'])}, "
                       f"berkeluhan dan terbaru lebih dulu · {MASK_NOTE}")
        else:
            st.caption(f"Tidak ada ulasan yang memuat '{shown}' pada filter ini.")
        if len(rows):
            st.dataframe(rows, hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 400), column_config={
                "Venue": st.column_config.TextColumn(width=120), "Tahun": st.column_config.NumberColumn(format="%d", width=56),
                "★": st.column_config.NumberColumn(format="%d", width=40), "Sentimen": st.column_config.TextColumn(width=100),
                "Kiri": st.column_config.TextColumn(alignment="right", width=290), "Kata": st.column_config.TextColumn(width=110),
                "Kanan": st.column_config.TextColumn(width=290)})


def _tab_nlp(labels, business, evidence, missing_quotes: int, sig: tuple, fkey: tuple, cfg: dict, mask) -> None:
    flagged = f", termasuk {int(business['needs_adjudication'].sum())} ulasan bertanda adjudikasi" if business is not None and len(business) else ""
    st.markdown(f'<div class="tj-ml-note">Sentimen dan polaritas aspek adalah label asisten AI sementara dan belum divalidasi manusia. '
                f'Kehadiran aspek diperiksa satu penulis menurut laporan penulis{flagged}; audit independen dan kesepakatan antarpenilai belum tersedia.</div>',
                unsafe_allow_html=True)  # Static text; the only dynamic part is an integer.
    if business is not None and len(business):
        present = business[[f"presence_{aspect}" for aspect in data.ASPECTS]].sum(axis=1)
        dropped = f" ({fmt_int(missing_quotes)} kutipan yang tak ditemukan di teksnya dibuang)" if missing_quotes else ""
        st.caption(f"{fmt_int(len(business))} ulasan · {fmt_dec(present.mean(), 2)} aspek/ulasan · "
                   f"{fmt_int(0 if evidence is None else len(evidence))} kutipan bukti{dropped}")
    with st.container(key="bi_row_aspect"):
        _aspect_cards(business, evidence, cfg, mask)
    with st.container(key="bi_row_text"):
        left, right = st.columns([7, 5])
        with left:
            _terms_card(labels, sig, fkey, cfg)
        with right:
            _length_card(labels, cfg)
    complaint_terms, _ = _terms(sig, fkey, nlp.TERM_MODES[0]) if labels is not None else (pd.DataFrame(), [])
    default_query = str(complaint_terms["term"].iloc[0]) if len(complaint_terms) and complaint_terms["z"].iloc[0] > nlp.Z_EPS else None
    _kwic_card(labels, default_query, mask)
    st.button("Uji model klasifikasi → Aspek bisnis", on_click=_go_to, args=("Aspek bisnis",), key="bi_to_business", icon=":material/science:")


def _provenance_card(labels: pd.DataFrame | None, business: pd.DataFrame | None) -> None:
    """Export counts and recorded/reconstructed annotation provenance, without review text or native identities."""
    parts = [data.provenance_table(frame).assign(jenis=kind)
             for kind, frame in (("sentimen", labels), ("aspek", business)) if frame is not None and len(frame)]
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    with st.expander("Provenans anotasi dan ekspor", key="bi_provenance"):
        st.caption("Sesi adalah proksi batch berdasarkan asal berkas, bukan identitas sesi model. Nilai unknown menyatakan metadata tidak tercatat; "
                   "alias permintaan tidak membuktikan model backend. Kehadiran aspek diperiksa satu penulis menurut laporan penulis; "
                   "sentimen/polaritas tetap sementara. Jumlah di bawah mengikuti semua filter anotasi.")
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Unduh provenans dan jumlah terfilter (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="Annotation provenance and filtered counts.csv", mime="text/csv", key="bi_export_provenance")


def _funnel_card(bundle: dict, funnel: pd.DataFrame, filtered: bool, sig: tuple, cfg: dict) -> None:
    with _card("funnel", "Corong data: dari pengumpulan ke data latih"):
        _chart(charts.funnel_spec(funnel, cfg))
        st.caption("Batang pucat = tahap sebelumnya; selisihnya adalah baris yang dikeluarkan, dengan alasannya di kanan.")
        messages = None if filtered else _audit_messages(sig)
        if filtered:
            st.caption("Rekonsiliasi dengan audit hanya dihitung tanpa filter venue atau sumber.")
        elif messages is not None:
            for message in messages:
                st.warning(message)
            if not messages:
                st.caption("Tahap 1–4 cocok dengan hitungan audit snapshot" + ("; SHA-256 snapshot juga cocok." if bundle["snapshot_ok"] else "."))
        _table(funnel.assign(n=[fmt_int(v) for v in funnel["n"]], turun=[fmt_int(v) for v in funnel["turun"]])
               .rename(columns={"urut": "Urut", "tahap": "Tahap", "n": "Jumlah", "turun": "Turun", "catatan": "Catatan"})
               [["Urut", "Tahap", "Jumlah", "Turun", "Catatan"]], "funnel")


def _coverage_card(coverage: pd.DataFrame | None, missing: list[str], cfg: dict) -> None:
    with _card("coverage", "Cakupan per target scraping"):
        if missing:  # say which platform is absent instead of showing a silently partial picture
            st.info(f"Laporan {' dan '.join(missing)} tidak tersedia; ringkasan hanya mencakup target yang memiliki laporan.")
        if coverage is None or coverage.empty:
            if not missing:
                st.info(NO_AUDIT)
            return
        _chart(charts.coverage_spec(coverage, cfg))
        st.caption("Status pengambilan dan ketersediaan penyebut dilaporkan terpisah: nol terkumpul adalah hasil nol, sedangkan total tidak diketahui "
                   "tidak memiliki persentase cakupan. 'Dilaporkan' berbeda per platform: Airbnb = ulasan tertulis, SeekSophie = ulasan opsi, "
                   "Google = total panel. Tidak ada persentase gabungan lintas platform. Semua tahun.")
        _table(pd.DataFrame({"Platform": coverage["platform"], "Target": [data.venue_name(key) for key in coverage["venue_key"]],
                             "Terkumpul": [fmt_int(v) for v in coverage["collected"]], "Dilaporkan": [fmt_int(v) for v in coverage["reported"]],
                             "Cakupan": [fmt_pct(v) for v in coverage["coverage"]],
                             "Status": [f"{data.STATUS[status]} {status}" for status in coverage["status"]], "Alasan berhenti": coverage["reason"]}),
               "coverage")


def _ops_metrics(cards: dict) -> None:
    identities, duplicates, runs, duration = (cards[key] for key in ("identities", "duplicates", "runs", "duration"))
    with st.container(key="bi_ops_metrics"):
        columns = st.columns(4)
        if identities:  # "primary" keeps the sparkline in the page's teal; growth is a fact here, not a verdict
            columns[0].metric(identities["label"], identities["value"], delta=identities["delta"], delta_color="primary",
                              delta_description="vs snapshot sebelumnya" if identities["delta"] else None, chart_data=identities["chart_data"],
                              chart_type="area", help=identities["help"], height="stretch")
        else:
            columns[0].info(NO_AUDIT)
        if duplicates:
            columns[1].metric(duplicates["label"], duplicates["value"], delta=duplicates["delta"], delta_color="orange", delta_arrow="off",
                              help=duplicates["help"], height="stretch")
        else:
            columns[1].info(NO_AUDIT)
        for column, card in ((columns[2], runs), (columns[3], duration)):
            if card:
                column.metric(card["label"], card["value"], delta=card["delta"], delta_color="orange" if card.get("warn") else "gray",
                              delta_arrow="off", help=card["help"], height="stretch")
    st.caption("Empat metrik operasi ini dihitung dari seluruh snapshot dan log run, tanpa filter.")


def _quality_card(matrix: pd.DataFrame, cfg: dict) -> None:
    with _card("quality", "Pemeriksaan kualitas data"):
        _chart(charts.quality_spec(matrix, cfg))
        st.caption("Sel = jumlah baris (porsi dari kolomnya); makin gelap makin besar porsinya. 'Teks ≤3 kata' menghitung aksara Han/Kana "
                   "±2 karakter per kata. Hanya hitungan: tanpa ID dan tanpa teks.")
        table = matrix.pivot(index="cek", columns="kolom", values="label")
        order_rows = list(dict.fromkeys(matrix.sort_values("urut_cek")["cek"]))
        order_cols = list(dict.fromkeys(matrix.sort_values("urut_kolom")["kolom"]))
        _table(table.reindex(index=order_rows, columns=order_cols).reset_index(names="Pemeriksaan"), "quality")


def _scroll_card(collector, coverage: pd.DataFrame | None, cfg: dict) -> None:
    runs, checkpoints = collector
    curves = data.scroll_curves(runs, checkpoints)
    if curves.empty:
        return
    burden = data.google_burden(coverage) or (int(curves.groupby("run_id")["duplikat"].max().sum()), int(curves.groupby("run_id")["unik"].max().sum()))
    duplicates, unique = burden
    note = (f"Semua run Google: {fmt_int(duplicates)} observasi duplikat · {fmt_int(unique)} ulasan unik"
            + (f" ({data.fmt_times(duplicates, unique)})" if unique else ""))
    with _card("scroll", "Efisiensi gulir Google Maps"):
        _chart(charts.scroll_spec(curves, note, cfg))
        st.caption("Kurva datar = feed macet: putaran gulir terus berjalan tanpa ulasan baru. Hanya run dengan checkpoint yang digambar; "
                   "angka di kiri atas menjumlah semua target dan run Google Maps. Tanpa filter.")


def _telemetry_table(telemetry: pd.DataFrame) -> None:
    shown = pd.DataFrame({"Platform": telemetry["platform"], "Target": telemetry["target"],
                          "Mulai (UTC)": [f"{value:%d-%m %H:%M:%S}" if pd.notna(value) else "—" for value in telemetry["mulai"]],
                          "Durasi (detik)": [fmt_dec(value, 1) for value in telemetry["durasi"]],
                          "Putaran": [fmt_int(value) for value in telemetry["putaran"]], "Baru": [fmt_int(value) for value in telemetry["baru"]],
                          "Duplikat": [fmt_int(value) for value in telemetry["duplikat"]],
                          "Hasil": [f"{data.STATUS[value]} {value}" if value in data.STATUS else str(value) for value in telemetry["hasil"]],
                          "Alasan berhenti": [value if isinstance(value, str) and value else "—" for value in telemetry["alasan"]]})
    numbers = {name: st.column_config.TextColumn(alignment="right") for name in ("Durasi (detik)", "Putaran", "Baru", "Duplikat")}
    with st.expander(f"Telemetri run scraping ({fmt_int(len(shown))} run, tanpa filter)", icon=":material/monitoring:", key="bi_runs"):
        st.dataframe(shown, hide_index=True, width="stretch", height=38 + 35 * len(shown), column_config=numbers)
        st.caption("metrics.detail (galat Playwright) tidak dimuat atau ditampilkan oleh dashboard. SQLite dibuka immutable, tanpa membaca tabel ulasan.")


def _tab_ops(bundle: dict, f: data.Filters, missing: list[str], sig: tuple, cfg: dict) -> None:
    st.caption("Filter tahun dan 'tanpa bulan pasti' tidak berlaku di tab ini. Filter venue dan sumber berlaku untuk corong, cakupan dan matriks "
               "kualitas; metrik operasi, kurva gulir dan telemetri run selalu dihitung tanpa filter.")
    coverage = data.apply_filters(bundle["coverage"], f, use_years=False) if bundle["coverage"] is not None else None
    funnel, quality = _ops_tables(sig, data.Filters(f.venues, f.sources).key())
    collector, telemetry = _collector(sig), _telemetry(sig)
    with st.container(key="bi_row_ops"):
        left, right = st.columns([7, 5])
        with left:
            _funnel_card(bundle, funnel, bool(f.venues or f.sources), sig, cfg)
        with right:
            _coverage_card(coverage, missing, cfg)
    _ops_metrics(data.ops_summary(bundle["snapshots"], bundle["coverage"], telemetry))
    _quality_card(quality, cfg)
    if collector is not None:
        _scroll_card(collector, bundle["coverage"], cfg)
    if len(telemetry):
        _telemetry_table(telemetry)


METHOD_LEFT = """
**Definisi**
- *Ulasan pelanggan*: baris korpus berperan `customer_text` atau `rating_only`; balasan pemilik tidak dihitung.
- *Ulasan berkeluhan*: label sentimen teks `campuran` atau `negatif`. Rating bintang tidak menentukan label teks.
- *Keluhan di balik ★4–5*: ulasan berkeluhan yang tetap memberi rating 4–5.
- *Cakupan scraping*: rasio terkumpul/dilaporkan per target. Ringkasan menghitung target dengan laporan dan penyebut diketahui;
  tidak ada persentase gabungan karena makna penyebut berbeda antarplatform. Nol terkumpul dan penyebut tak diketahui dilaporkan terpisah.
  Filter tahun tidak berlaku pada cakupan.
- *Bulan ulasan*: tanggal ISO (tanggal lokal), lalu "Bulan Tahun" Airbnb. Tanggal relatif ("3 minggu lalu": Google Maps dan sebagian Airbnb)
  tidak dikonversi; bila disertakan, ulasan ini ikut di setiap rentang tahun tetapi tidak masuk tren.
- *Jendela tren*: utama = bulan kalender lengkap. Sensitivitas mengeluarkan bulan terakhir bila volumenya < 50% median 12 bulan sebelumnya,
  ditentukan sekali dari seluruh korpus. Penanda ini bukan bukti ketidaklengkapan; pola musiman dan penurunan nyata bisa memicunya.
  Kedua periode dan hitungan ditampilkan; perbandingan disembunyikan bila filter tahun memotong jendela.
- *Jumlah kata*: dipisah spasi; aksara Han/Kana yang ditulis tanpa spasi dihitung ±2 karakter per kata.

**Metode**
- Interval kepercayaan Wilson 95% (IK95%, z = 1,96) untuk porsi utama: ulasan berkeluhan (KPI, per venue, panjang ulasan, KWIC),
  keluhan di balik ★4–5 dan perbandingan pada sorotan. Porsi di peta panas dan tabel aspek tanpa interval; 'n kecil' menandai penyebut sel < 30.
  Interval bersyarat pada korpus terkumpul dan label sementara: tidak mencakup bias pengumpulan atau galat anotasi. Tumpang tindih bukan uji beda formal.
- Kata pembeda: log-odds berbobot dengan prior Dirichlet informatif (Monroe, Colaresi & Quinn, 2008), α₀ = 1000, frekuensi dokumen,
  kosakata dengan df ≥ 5, hanya ulasan berbahasa Inggris, ≥ 5 ulasan di tiap kelompok pembanding; nama orang dan nama venue disaring.
  Massa kelompok adalah jumlah insidensi term–dokumen pada kosakata tersaring, bukan jumlah ulasan. Skor z adalah urutan eksploratif;
  tanpa ambang signifikansi dan tanpa koreksi perbandingan berganda.
- KWIC: pencocokan kata utuh tanpa regex dari pengguna (aksara Han, Kana dan Hangul dicocokkan sebagai potongan teks); konteks dipotong agar
  muat di kolom; maksimal 30 baris, ulasan berkeluhan lebih dulu.
- Sorotan otomatis: semua aturan yang terpicu ditampilkan dengan basisnya. Kontras venue hanya deskriptif dalam batch/sesi proksi;
  peringkat lintas sesi ditahan karena invariansi anotasi belum diuji.
"""
METHOD_RIGHT = """
**Privasi dan tata kelola**
- Loader dashboard mengecualikan ID ulasan asli, URL dan `raw_text`; kutipan/KWIC memakai `clean_text` hasil normalisasi.
  Capture mentah tetap ada dalam ekspor lokal dan snapshot SQLite; pembatasan tampilan tidak menghapus data sumber.
- Nama orang di dalam teks ulasan disamarkan menjadi [nama] secara heuristik: kata berhuruf kapital yang tidak pernah ditulis huruf kecil,
  kata setelah sapaan (Pak, Bu, Mbak, Mr.), alamat e-mail, @akun dan tanda tangan di akhir ulasan, ditambah daftar nama umum.
  Heuristik ini bisa melewatkan nama, terutama di awal kalimat dan pada teks berbahasa Jerman, Rusia atau beraksara non-Latin.
  Presisi/recall penyamaran dan kebocoran identitas belum diukur; penyamaran bukan jaminan anonimitas.
- SQLite dibuka read-only dan immutable; tabel ulasan tidak dibaca, dan metrics.detail (galat Playwright) tidak dimuat atau ditampilkan oleh dashboard.
- Pengguna yang berhak mengakses capture mentah, ACL, enkripsi, log akses serta kebijakan/durasi retensi dan jadwal penghapusan belum tercatat
  dalam bukti yang dimuat dashboard. Status tata kelola tersebut tetap unknown dan memerlukan konfirmasi pemilik data.

**Batasan**
- Sentimen dan polaritas aspek adalah draft asisten AI yang belum divalidasi; persetujuan ahli ditampilkan dari kolom sumber.
  Pemeriksaan kehadiran aspek oleh satu penulis dilaporkan penulis; tidak ada audit independen atau kesepakatan antarpenilai. Sebagian besar ulasan positif.
- Sesi anotasi adalah proksi batch dari asal berkas. Versi rubrik/prompt, tanggal anotasi dan model backend tetap unknown bila tidak tercatat;
  hash protokol dan versi skema yang tersedia bukan pengganti versi rubrik/prompt yang hilang. Perbandingan lintas batch dapat bercampur dengan efek anotasi.
- Museum Tekstil memakai label lama (ikut dihitung dalam porsi gabungan: KPI keluhan, rating vs teks, panjang ulasan) dan cakupan
  Google Maps yang rendah; teksnya tanpa metadata bahasa sehingga tidak masuk kata pembeda. Museum adalah museum umum, bukan kelas batik.
- Mayoritas ulasan berbahasa Inggris dan jumlah ulasan per venue tidak seimbang.
- Volume ulasan bukan ukuran jumlah kunjungan; penurunan volume bisa berasal dari platform, bukan dari venue.
"""


def _tab_method(bundle: dict) -> None:
    left, right = st.columns(2)
    with left, _card("method_left"):
        st.markdown(METHOD_LEFT)
    with right, _card("method_right"):
        st.markdown(METHOD_RIGHT)
    files = bundle["files"].sort_values("file", ignore_index=True)
    with _card("sources", "Berkas sumber"):
        st.caption("Relatif terhadap exports/; semua berkas hanya dibaca. Perubahan berkas otomatis memperbarui cache halaman ini.")
        st.dataframe(pd.DataFrame({"Berkas": files["file"], "Baris": [fmt_int(v) if pd.notna(v) else "—" for v in files["rows"]],
                                   "Diubah (UTC)": [f"{value:%Y-%m-%d %H:%M}" for value in files["modified"]]}),
                     hide_index=True, width="stretch", height=38 + 35 * len(files),
                     column_config={"Berkas": st.column_config.TextColumn(width="large")})


def render_dashboard(settings=None, storage=None, app=None, project=None) -> None:
    st.markdown(BI_CSS, unsafe_allow_html=True)  # Application-owned static CSS only.
    st.header("Dashboard BI · Ulasan wisata belajar batik", anchor=False)
    try:
        paths = data.source_paths()
        if paths is None:
            _waiting()
            return
        sig = data.signature(paths)
        bundle = _bundle(sig)
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        st.error(str(exc))
        return
    printing = _printing()
    cfg = charts.base_config(printing)
    if printing:
        st.markdown(PRINT_CSS, unsafe_allow_html=True)  # Application-owned static CSS only.
    _header(bundle, sig)
    f, status, bounds = _filter_bar(bundle)
    reviews = data.apply_filters(bundle["reviews"], f)
    customers = reviews[reviews["record_role"].isin(data.CUSTOMER_ROLES)]
    labels = data.apply_filters(bundle["labels"], f) if bundle["labels"] is not None else None
    business = data.apply_filters(bundle["business"], f) if bundle["business"] is not None else None
    evidence = bundle["evidence"][bundle["evidence"]["record_id"].isin(business["record_id"])] if business is not None else None
    coverage = data.apply_filters(bundle["coverage"], f, use_years=False) if bundle["coverage"] is not None else None
    missing = data.missing_reports(bundle["coverage_missing"], data.apply_filters(bundle["reviews"], f, use_years=False))  # no year filter
    narrowed = f.years is not None and bounds is not None and tuple(f.years) != tuple(bounds)
    undated = int(customers["month"].isna().sum())
    total = int(bundle["reviews"]["record_role"].isin(data.CUSTOMER_ROLES).sum())
    parts = [f"Menampilkan {fmt_int(len(customers))} dari {fmt_int(total)} ulasan pelanggan"
             + (f", **termasuk {fmt_int(undated)} tanpa bulan pasti**" if narrowed and undated else "")]
    parts += [f"{fmt_int(len(labels))} teks berlabel"] if labels is not None else []
    parts += [f"{fmt_int(len(business))} berlabel aspek"] if business is not None else []
    status.caption(" · ".join(parts))
    if customers.empty:
        st.warning("Tidak ada ulasan untuk filter ini.")
        st.button("Atur ulang filter", on_click=_reset_filters, key="bi_reset_empty", icon=":material/restart_alt:")
        return
    stamp = bundle["collected_at"]
    through = None if pd.isna(stamp) else pd.Timestamp(stamp.year, stamp.month, 1) - pd.DateOffset(months=1)
    monthly, windows = data.monthly_series(reviews, stamp, basis=bundle["reviews"], years=f.years if narrowed else None)
    _kpi_row(data.kpis(reviews, labels, coverage, windows, missing))
    if data.session_comparison_blocker(labels):
        st.caption("Porsi gabungan adalah deskripsi label sementara dari beberapa batch/sesi proksi. Perbandingan venue dipisah per batch; "
                   "perbedaan protokol anotasi belum dapat dipisahkan dari perbedaan venue.")
    _provenance_card(labels, business)
    insights = data.compute_insights(reviews, labels, business, coverage, windows)
    mask = nlp.name_masker(bundle["names"])
    overview, insight, ops, method = st.tabs(list(TABS))
    with overview:
        _tab_overview(monthly, windows, insights, labels, coverage, cfg, through)
    with insight:
        _tab_nlp(labels, business, evidence, bundle["evidence_missing"], sig, f.key(), cfg, mask)
    with ops:
        _tab_ops(bundle, f, missing, sig, cfg)
    with method:
        _tab_method(bundle)
