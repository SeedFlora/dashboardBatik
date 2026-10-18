"""Pure BI data layer: privacy-safe loading, honest aggregates and Indonesian formatting."""
import json
import math
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from bi_fixture import OLD_RUN_ID, SECRETS, patch_paths, write_fixture
from temajelajah import bi_data, bi_nlp


def _load():
    loaded = bi_data.load_bundle(bi_data.source_paths())
    return loaded | {"names": bi_nlp.name_vocab(loaded["reviews"], loaded["labels"], loaded["business"])}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    patch_paths(monkeypatch, write_fixture(tmp_path))
    return _load()


def _windows(bundle):
    return bi_data.monthly_series(bundle["reviews"], bundle["collected_at"])[1]


def test_parse_month_iso_month_year_relative(bundle):
    months = bi_data.parse_month(pd.Series(["2023-08-05T10:07:07.536+08:00", "2023-09-01T01:00:00.000+08:00", "", "", "null"]),
                                 pd.Series(["", "", "August 2023", "3 weeks ago", "null"]))
    assert months[:3].tolist() == [pd.Timestamp("2023-08-01"), pd.Timestamp("2023-09-01"), pd.Timestamp("2023-08-01")]
    assert months[3:].isna().all()
    reviews = bundle["reviews"].set_index("record_id")
    assert str(reviews["year"].dtype) == "Int64"
    assert reviews.loc["R03", "year"] == 2023 and pd.isna(reviews.loc["R10", "year"])
    assert not (reviews["year"] == -1).any()


def test_rating_and_bool_normalisation(bundle):
    ratings = bi_data.to_float(pd.Series(["5", "5.0", "null", ""]))
    assert ratings[:2].tolist() == [5.0, 5.0] and ratings[2:].isna().all()
    assert bi_data.to_bool(pd.Series(["True", "true", "false", "False", ""])).tolist() == [True, True, False, False, False]
    business = bundle["business"].set_index("record_id")
    assert business["presence_belajar_praktik"].dtype == bool and business.loc["R03", "presence_belajar_praktik"]
    assert pd.isna(business.loc["R01", "polarity_hasil_karya_produk"])
    assert bundle["reviews"].set_index("record_id").loc["R02", "rating_num"] == 5.0
    assert bundle["labels"]["expert_approved"].dtype == bool


def test_wilson_reference_values():
    assert bi_data.wilson_interval(16, 81) == pytest.approx((0.1254, 0.2970), abs=1e-3)
    assert bi_data.wilson_interval(0, 15) == pytest.approx((0.0, 0.2039), abs=1e-3)
    assert all(math.isnan(value) for value in bi_data.wilson_interval(0, 0))


def test_bundle_has_no_identity_columns_or_secrets(bundle):
    frames = {name: value for name, value in bundle.items() if isinstance(value, pd.DataFrame)}
    assert {"reviews", "labels", "business", "evidence", "coverage", "snapshots", "files"} <= set(frames)
    for name, frame in frames.items():
        assert not bi_data.FORBIDDEN_COLS & set(frame.columns), name
        dump = frame.to_csv() + repr(frame)
        assert not any(secret in dump for secret in SECRETS), name
    assert not any(secret in repr({k: v for k, v in bundle.items() if k not in frames}) for secret in SECRETS)
    with pytest.raises(ValueError, match="terlarang"):
        bi_data.read_table(bi_data.source_paths()["corpus"], ["record_id", "raw_text"])


def test_source_paths_validate_run_id_and_need_corpus(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path, labels=False, business=False)
    patch_paths(monkeypatch, paths)
    found = bi_data.source_paths()
    assert {"corpus", "audit", "raw_snapshot", "seeksophie"} <= set(found) and "labels_real" not in found
    assert bi_data.load_bundle(found)["labels"] is None
    for bad in ("../outside", "..", "a/b"):
        (paths["expansion"] / "latest.json").write_text(json.dumps({"run_id": bad}), encoding="utf-8")
        with pytest.raises(ValueError, match="run_id"):
            bi_data.source_paths()
    (paths["run"] / "corpus.csv").unlink()
    (paths["expansion"] / "latest.json").write_text(json.dumps({"run_id": paths["run"].name}), encoding="utf-8")
    assert bi_data.source_paths() is None


def test_apply_filters_each_dimension_and_undated_toggle(bundle):
    reviews, F = bundle["reviews"], bi_data.Filters
    ids = lambda frame: set(frame["record_id"])
    assert len(bi_data.apply_filters(reviews, F())) == 12
    assert ids(bi_data.apply_filters(reviews, F(venues=["museum_tekstil"]))) == {"R10", "R11", "R12"}
    assert ids(bi_data.apply_filters(reviews, F(sources=("SeekSophie",)))) == {"R03"}
    assert ids(bi_data.apply_filters(reviews, F(years=(2025, 2025), include_undated=False))) == {"R02", "R04", "R06"}
    assert ids(bi_data.apply_filters(reviews, F(years=(2025, 2025)))) == {"R02", "R04", "R06", "R10", "R11", "R12"}
    assert len(bi_data.apply_filters(reviews, F(years=(2030, 2031), include_undated=False), use_years=False)) == 12
    assert F(venues=["a"]).key() == F(venues=("a",)).key() and hash(F(venues=["a"]))
    assert set(bi_data.apply_filters(bundle["evidence"], F(venues=("batik_popiler_2",)))["venue_key"]) == {"batik_popiler_2"}


def test_kpis_exclude_owner_replies_and_count_hidden_critique(bundle):
    reviews, labels = bundle["reviews"], bundle["labels"]
    cards = bi_data.kpis(reviews, labels, bundle["coverage"], _windows(bundle))
    assert cards["K1"]["value"] == "11" and cards["K1"]["delta"] is None
    assert cards["K1"]["help"].startswith("2 ulasan tanpa bulan pasti") and cards["K1"]["chart_type"] == "bar"
    assert cards["K2"]["value"] == "4,55" and cards["K2"]["help"].startswith("72,7% bintang 5")
    assert cards["K3"]["value"] == "30,0%" and cards["K3"]["help"].startswith("3 dari 10 ulasan teks berlabel")
    assert cards["K4"]["value"] == "—" and (cards["K4"]["k"], cards["K4"]["n"]) == (2, 3)
    assert cards["K5"]["value"] == "4/5 target" and "Airbnb · Five Arts: 4/4 · 100%" in cards["K5"]["help"]
    assert "Google Maps · Giriloyo: 0 · total tak diketahui" in cards["K5"]["help"] and "k" not in cards["K5"]
    many = bi_data.kpis(reviews, pd.concat([labels] * 4, ignore_index=True), None, _windows(bundle))
    assert many["K4"]["value"] == "67%" and many["K4"]["help"].startswith("8 dari 12 ulasan berkeluhan") and many["K5"] is None
    assert bi_data.kpis(reviews, None, None, _windows(bundle))["K3"] is None


def _monthly_rows(counts):
    months = [month for month, n in counts.items() for _ in range(n)]
    return pd.DataFrame({"record_role": "customer_text", "month": pd.to_datetime(months), "rating_num": 5.0})


def test_monthly_series_flags_partial_tail_and_windows():
    counts = {f"2024-{m:02d}-01": 3 for m in range(1, 9) if m != 3}
    counts |= {f"{y}-{m:02d}-01": 10 for y, months in ((2024, range(9, 13)), (2025, range(1, 9))) for m in months}
    counts |= {f"{y}-{m:02d}-01": 5 for y, months in ((2025, range(9, 13)), (2026, range(1, 9))) for m in months}
    counts |= {"2026-09-01": 1, "2026-10-01": 4}
    reviews = pd.concat([_monthly_rows(counts), _monthly_rows({"2026-01-01": 2}).assign(record_role="owner_response")], ignore_index=True)
    frame, windows = bi_data.monthly_series(reviews, pd.Timestamp("2026-10-01T13:00:00+00:00"))
    assert frame["bulan"].max() == pd.Timestamp("2026-09-01") and frame["parsial"].tolist()[-2:] == [False, True]
    assert frame.set_index("bulan").loc["2024-03-01", "n"] == 0 and frame.set_index("bulan").loc["2026-01-01", "n"] == 5
    assert (windows["current"], windows["previous"], windows["ok"]) == (56, 115, True)
    assert windows["change"] == pytest.approx(56 / 115 - 1) and windows["end"] == pd.Timestamp("2026-09-01")
    assert (windows["current_label"], windows["previous_label"]) == ("Okt 2025–Sep 2026", "Okt 2024–Sep 2025")
    sensitivity = windows["sensitivity"]
    assert (sensitivity["current"], sensitivity["previous"], sensitivity["change"]) == (60, 120, -0.5)
    full, steady = bi_data.monthly_series(_monthly_rows({**counts, "2026-09-01": 5}), pd.Timestamp("2026-10-01"))
    assert not full["parsial"].any() and steady["end"] == pd.Timestamp("2026-09-01") and steady["current"] == 60


def test_rating_text_matrix_row_shares_and_hidden_cells(bundle):
    cells, agreement = bi_data.rating_text_matrix(bundle["labels"])
    assert len(cells) == 20 and cells["rating"].unique().tolist() == [5, 4, 3, 2, 1]
    sums = cells.groupby("rating")["row_share"].sum()
    assert all(math.isclose(value, 1.0) or value == 0 for value in sums) and sums[1] == 0 and sums[3] == 0
    hidden = cells[cells["hidden"]]
    assert set(zip(hidden["rating"], hidden["sentimen"])) == {(5, "campuran"), (5, "negatif"), (4, "campuran"), (4, "negatif")}
    assert cells.set_index(["rating", "sentimen"]).loc[(5, "positif"), "label"] == "7 · 88%"
    assert agreement == pytest.approx(0.8)


def _business(rate_a, rate_b, n=100):
    rows = []
    for venue, rate in (("batik_kotagede_nunung", rate_a), ("widya_batik_class", rate_b)):
        for i in range(n):
            rows.append({"venue_key": venue, **{f"presence_{a}": False for a in bi_data.ASPECTS},
                         **{f"polarity_{a}": pd.NA for a in bi_data.ASPECTS}, "presence_fasilitas_kenyamanan": i < round(rate * n)})
    return pd.DataFrame(rows).assign(annotation_session="batch:fixture")


def test_aspect_matrix_rates_small_n_and_quotes_complaints_first_masked(bundle):
    matrix = bi_data.aspect_venue_matrix(bundle["business"], "Penyebutan", None)
    cell = matrix.set_index(["aspek_kode", "venue_kode"])
    assert len(matrix) == 8 and matrix["small"].all()
    assert cell.loc[("belajar_praktik", "five_art_studio"), "rate"] == 1.0
    assert cell.loc[("pengajar_layanan", "batik_popiler_2"), "rate"] == pytest.approx(1 / 3)
    assert cell.loc[("fasilitas_kenyamanan", "five_art_studio"), "complaint_rate"] == 1.0
    assert cell.loc[("belajar_praktik", "five_art_studio"), "label"] == "100%\n3/3\nn kecil"
    assert matrix.loc[matrix["terpilih"], ["aspek_kode", "venue_kode"]].values.tolist() == [["pengajar_layanan", "five_art_studio"]]
    keluhan = bi_data.aspect_venue_matrix(bundle["business"], "Keluhan", None)
    assert keluhan.loc[keluhan["terpilih"], ["aspek_kode", "venue_kode"]].values.tolist() == [["pengajar_layanan", "five_art_studio"]]
    mixed = pd.concat([_business(0.5, 0.0, n=40), _business(1.0, 1.0, n=5).replace("widya_batik_class", "lombok_batik_alwan")], ignore_index=True)
    default = bi_data.aspect_venue_matrix(mixed, "Penyebutan", None)
    assert default.loc[default["terpilih"], ["aspek_kode", "venue_kode"]].values.tolist() == [["pengajar_layanan", "widya_batik_class"]]
    chosen = bi_data.aspect_venue_matrix(bundle["business"], "Keluhan", ("fasilitas_kenyamanan", "batik_popiler_2"))
    assert chosen.loc[chosen["terpilih"], "nilai"].tolist() == [0.0]
    mask = bi_nlp.name_masker(bundle["names"])
    quotes = bi_data.cell_quotes(bundle["evidence"], "five_art_studio", "pengajar_layanan", mask)
    assert quotes["polarity"].tolist() == ["campuran", "positif"] and quotes["glyph"].tolist() == ["±", "+"]
    assert quotes["quote"].iloc[1] == r"our teacher \[nama\], and \[nama\] was patient" and "Wayan" not in quotes.to_csv()
    assert len(bi_data.cell_quotes(bundle["evidence"], "five_art_studio", "belajar_praktik", mask, limit=2)) == 2
    empty = bi_data.aspect_venue_matrix(bundle["business"].iloc[:0], "Keluhan", None)
    assert empty.empty and {"aspek_kode", "venue_kode", "nilai", "terpilih", "label"} <= set(empty.columns)


def test_explode_evidence_drops_and_counts_missing_quotes(bundle):
    evidence, missing = bi_data.explode_evidence(bundle["business"])
    assert missing == 1 == bundle["evidence_missing"] and len(evidence) == 11
    texts = bundle["business"].set_index("record_id")["clean_text"]
    assert all(texts[row.record_id][row.start:row.end] == row.quote for row in evidence.itertuples())


def test_funnel_monotone_and_reconcile_reports_mismatch(bundle):
    funnel = bi_data.funnel_table(bundle["reviews"], bundle["labels"])
    assert funnel["n"].tolist() == [12, 10, 9, 8, 8, 7, 7] and (funnel["n"].diff().dropna() <= 0).all()
    assert funnel["label"].tolist()[1:3] == ["10 · −1 rating saja, −1 balasan pemilik", "9 · −1 luar domain"]
    assert funnel["label"].iloc[4] == "8 · 0 disetujui ahli" and funnel["label"].iloc[5] == "7 · −1 non-batik"
    assert funnel["label"].iloc[-1] == "4 / 2 / 1"
    assert bi_data.reconcile_funnel(funnel, bundle["audit_counts"]) == [] and bundle["snapshot_ok"] is True
    messages = bi_data.reconcile_funnel(funnel, {**bundle["audit_counts"], "main_domain_customer_texts": 99})
    assert len(messages) == 1 and "99" in messages[0]
    assert len(bi_data.funnel_table(bundle["reviews"], None)) == 4


def test_coverage_unknown_denominator_nan_and_status_gagal(bundle):
    rows = bi_data.coverage_rows(
        [{"venue_key": "widya_batik_class", "customer_texts": 272, "stop_reason": "reported_written_review_count_reached",
          "source_totals_metadata": {"writtenReviewCount": "272", "ratingCount": "273"}}], None,
        [{"target_key": "seno", "inserted_total": 0, "reported_total": None, "duplicates_total": 0}],
        {"seno": {"status": "partial", "reason": "reviews_panel_unavailable", "metrics": {"detail": "SECRET_LOG"}}})
    airbnb, seno = rows.iloc[0], rows.iloc[1]
    assert (airbnb["coverage"], airbnb["status"], airbnb["reason"]) == (1.0, "lengkap", "jumlah ulasan tertulis tercapai")
    assert math.isnan(seno["coverage"]) and math.isnan(seno["reported"]) and seno["status"] == "gagal"
    assert seno["label"] == "0 · total tak diketahui" and seno["target"] == "○ Google · Seno" and "SECRET" not in rows.to_csv()
    assert bi_data.coverage_rows([], {"venue_key": "five_art_studio", "actual_unique_reviews_collected": "20",
                                      "option_reviews_total_metadata_only": "89.0"}, [], {})["label"].tolist() == ["20/89 · 22,5%"]
    fixture = bundle["coverage"].set_index("target")
    assert fixture.loc["◐ SeekSophie · Five Arts", "label"] == "20/89 · 22,5%"
    assert fixture.loc["◐ Google · Museum Tekstil", "reason"] == "feed macet"


def test_quality_matrix_fixture_counts(bundle):
    matrix = bi_data.quality_matrix(bundle["reviews"])
    totals = matrix[matrix["kolom_kode"] == "total"].set_index("cek")["n"].to_dict()
    assert totals == {"Rating saja": 1, "Balasan pemilik": 1, "Teks identik lintas ID": 2, "Kemungkinan terpotong": 1,
                      "Bahasa tanpa metadata": 3, "Varian bahasa tak diketahui": 3, "Tanggal relatif saja": 3,
                      "Kelengkapan belum terverifikasi": 4, "Teks ≤3 kata": 2, "Teks dinormalisasi": 2}
    cell = matrix.set_index(["cek_kode", "kolom_kode"])["label"]
    assert cell[("rating_only", "museum_tekstil")] == "1 (33,3%)" and cell[("rating_only", "five_art_studio")] == "·"
    assert matrix["kolom"].unique().tolist() == ["Five Arts", "Popiler 2", "Museum Tekstil", "Total"]


def test_insights_guards_and_cautious_wording(bundle):
    reviews, labels = bundle["reviews"], bundle["labels"]
    keys = [item.key for item in bi_data.compute_insights(reviews, labels, bundle["business"], bundle["coverage"], _windows(bundle))]
    assert "volume_trend" not in keys and "hidden_critique" not in keys and "aspect_gap" not in keys and "coverage_gap" in keys
    for rates, cautious in (((0.30, 0.22), True), ((0.70, 0.26), False)):
        gap = next(item for item in bi_data.compute_insights(reviews, None, _business(*rates), None, None) if item.key == "aspect_gap")
        words = " ".join(map(str, gap)).lower()
        assert ("indikasi awal" in words) is cautious and "signifikan" not in words
        assert gap.text == "Fasilitas disebut di Nunung vs Widya" and gap.tone == "Info"
    many = pd.concat([labels] * 4, ignore_index=True)
    hidden = next(item for item in bi_data.compute_insights(reviews, many, None, None, None) if item.key == "hidden_critique")
    assert (hidden.value, hidden.text, hidden.priority) == ("67%", f"ulasan berkeluhan tetap memberi {bi_data.STARS_4_5}", 1)
    assert hidden.basis == "8/12 ulasan berkeluhan · label asisten sementara · termasuk 8 label lama Museum Tekstil"


def test_selected_values_dict_and_attr_states():
    state = {"selection": {"pilih_sel": [{"aspek_kode": "fasilitas_kenyamanan", "venue_kode": "batik_popiler_2"}]}}
    assert bi_data.selected_values(state, "pilih_sel", "venue_kode") == ["batik_popiler_2"]
    attrs = SimpleNamespace(selection=SimpleNamespace(pilih_kata=[{"term": "wax"}, {"term": "wax"}, {"term": "but"}]))
    assert bi_data.selected_values(attrs, "pilih_kata", "term") == ["wax", "but"]
    assert bi_data.selected_values({"selection": {"p": {"term": ["a", "b"]}}}, "p", "term") == ["a", "b"]
    assert bi_data.selected_values({"selection": {"pilih_sel": {}}}, "pilih_sel", "venue_kode") == []
    assert bi_data.selected_values(None, "pilih_sel", "venue_kode") == [] == bi_data.selected_values({}, "x", "y")


def test_proper_noun_vocab_and_masking_whole_words(bundle):
    texts = ["We met Ardi at the studio.", "Then Ardi helped us and Ardi smiled.", "In Bali we met Ardi in Bali, so Bali stays.", "So I said I like it."]
    vocab = bi_nlp.proper_noun_vocab(texts)
    assert "ardi" in vocab and "bali" not in vocab and "wayan" in vocab and "studio" not in vocab
    assert "wayan" in bundle["names"] and "keliki" not in bi_nlp.proper_noun_vocab(["at Keliki", "in Keliki", "near Keliki"])
    mask = bi_nlp.name_masker(bundle["names"])
    assert mask("Wayan's wax and a Wayang show in Bali, I think WAYAN") == "[nama] wax and a Wayang show in Bali, I think [nama]"
    assert mask("Wayan과 함께, Nunungさんが, 和Wayan会") == "[nama]과 함께, [nama]さんが, 和[nama]会"
    assert bi_nlp.name_masker([])("Wayan") == "Wayan"
    assert bi_nlp.md_escape("*x* <b>") == r"\*x\* \<b\>"


def test_log_odds_symmetric_zero_exclusive_positive():
    docs = [{"wax", "class"}, {"wax", "tea"}, {"class", "tea"}] * 3
    assert (bi_nlp.log_odds_dirichlet(docs, docs, min_df=1)["z"].abs() < 1e-9).all()
    result = bi_nlp.log_odds_dirichlet([{"wax", "class"}] * 5 + [{"class"}] * 5, [{"class"}] * 10 + [{"tea"}] * 5).set_index("term")
    assert result.loc["wax", "z"] > 0 and result.loc["wax", "df_b"] == 0 and result.loc["tea", "z"] < 0
    assert bi_nlp.tokenize("Wayan's class wasn't bad, but no wax!", frozenset({"wayan"})) == {"class", "wasn't", "bad", "but", "no", "wax"}


def test_distinctive_terms_modes_and_skip_small_venues(bundle):
    texts = ["Lovely wax painting with tea at Five Arts"] * 30 + ["Lovely garden with tea"] * 20 + ["Rude Ardi but nice tea"] * 10
    synthetic = pd.DataFrame({"source_locale": ["en"] * 59 + ["fr"], "clean_text": texts, "complaint": [False] * 50 + [True] * 10,
                              "venue_key": ["five_art_studio"] * 30 + ["widya_batik_class"] * 30, "annotation_session": "batch:fixture"})
    terms, skipped = bi_nlp.distinctive_terms(synthetic, "Keluhan vs lainnya", frozenset({"ardi"}), top_pos=3, top_neg=2)
    side = terms.groupby("sisi")["term"].apply(list).to_dict()
    assert set(side["ulasan berkeluhan"]) == {"rude", "but", "nice"} and "lovely" in side["ulasan lain"] and skipped == []
    assert "kuat" not in terms and terms.loc[terms["term"] == "rude", "label"].tolist() == ["9 ulasan"]
    assert not terms["term"].isin({"ardi", "five", "arts"}).any()
    per_venue, skipped = bi_nlp.distinctive_terms(synthetic, "Per venue", frozenset(), per_venue=2)
    assert per_venue.loc[per_venue["venue_key"] == "five_art_studio", "term"].tolist() == ["painting", "wax"]
    assert skipped == ["Widya (29)"] and per_venue["venue"].unique().tolist() == ["Five Arts"]
    empty, _ = bi_nlp.distinctive_terms(bundle["labels"], "Keluhan vs lainnya", frozenset(bundle["names"]))
    assert {"term", "z", "df_a", "df_b", "n_a", "n_b", "sisi", "label"} <= set(empty.columns)
    with pytest.raises(ValueError):
        bi_nlp.distinctive_terms(bundle["labels"], "lain", frozenset())


def test_kwic_regex_escape_and_clipping(bundle):
    labels, mask = bundle["labels"], bi_nlp.name_masker(bundle["names"])
    rows, stats = bi_nlp.kwic(labels, "c++", mask)
    assert rows.empty and stats["matches"] == 0 and bi_nlp.kwic(labels, " x ", mask)[1]["matches"] == 0
    rows, stats = bi_nlp.kwic(labels, "  wax ", mask)
    assert stats["matches"] == 2 and rows.columns.tolist() == ["Venue", "Tahun", "★", "Sentimen", "Kiri", "Kata", "Kanan"]
    assert rows["Kiri"].str.startswith("…").any() and not rows["Kanan"].str.endswith("…").any() and "Wayan" not in rows.to_csv()
    assert rows["Kanan"].tolist() == [" work.", " pen."] or rows["Kanan"].tolist() == [" pen.", " work."]
    narrow, _ = bi_nlp.kwic(labels, "wax", mask, window=5)
    assert sorted(narrow["Kanan"]) == [" pen.", " work…"]
    rows, stats = bi_nlp.kwic(labels, "CLASS", mask, only_complaints=True)
    assert (stats["matches"], stats["complaints"], len(rows)) == (4, 1, 1) and rows["Sentimen"].iloc[0] == "− negatif"


def test_length_bins_sum_to_n(bundle):
    bins = bi_nlp.length_bins(bundle["labels"])
    assert bins["n"].sum() == len(bundle["labels"]) and bins["bin"].tolist() == ["≤20", "21–40", "41–80", "81–160", ">160"]
    assert bins["axis"].iloc[0] == "≤20\n(n=10)" and bins["label"].iloc[0] == "30,0%"
    synthetic = pd.DataFrame({"clean_text": ["w " * 5, "w " * 25, "w " * 200, ""], "complaint": [False, True, True, False]})
    assert bi_nlp.length_bins(synthetic)["n"].tolist() == [2, 1, 0, 0, 1]


def test_collector_snapshot_immutable_no_sidecars(tmp_path):
    paths = write_fixture(tmp_path, labels=False, business=False)
    before = sorted(path.name for path in paths["run"].iterdir())
    runs, checkpoints = bi_data.read_collector_snapshot(paths["snapshot"])
    assert sorted(path.name for path in paths["run"].iterdir()) == before
    assert not {"review_text", "review_id", "detail", "metrics", "url"} & (set(runs.columns) | set(checkpoints.columns))
    assert not any(secret in runs.to_csv() + checkpoints.to_csv() for secret in SECRETS) and len(runs) == 2
    curves = bi_data.scroll_curves(runs, checkpoints)
    assert curves.loc[curves["run_id"].eq("run-a"), "unik"].tolist() == [2, 3]  # inserted, not inserted + updated
    assert curves.loc[curves["run_id"].eq("run-b"), "unik"].tolist() == [4]
    assert curves["run"].unique().tolist() == ["13:30 · 2 putaran", "13:40 · 1 putaran"] and curves["akhir"].sum() == 2


def test_fmt_indonesian():
    assert (bi_data.fmt_int(1687), bi_data.fmt_int(1234567), bi_data.fmt_int(None)) == ("1.687", "1.234.567", "—")
    assert (bi_data.fmt_pct(0.0467), bi_data.fmt_pct(0.894, 0), bi_data.fmt_pct(float("nan"))) == ("4,7%", "89%", "—")
    assert (bi_data.fmt_pct(0.0), bi_data.fmt_pct(1.0), bi_data.fmt_dec(4.928)) == ("0%", "100%", "4,93")
    assert bi_data.fmt_ci(0.1254, 0.2970) == "12,5–29,7%" and bi_data.fmt_ci(0.0, 0.0256) == "0–2,6%"
    assert bi_data.fmt_month(pd.Timestamp("2026-08-01")) == "Agu 2026"


def test_specs_json_serialisable_and_palette_only(bundle, tmp_path):
    from temajelajah import bi_charts as charts
    reviews, labels, business = bundle["reviews"], bundle["labels"], bundle["business"]
    monthly, _ = bi_data.monthly_series(reviews, bundle["collected_at"])
    venues = bi_data.venue_complaints(labels)
    texts = ["Lovely wax painting with tea"] * 30 + ["Lovely garden with tea"] * 30 + ["Rude host but nice tea"] * 10
    synthetic = pd.DataFrame({"source_locale": "en", "clean_text": texts, "complaint": [False] * 60 + [True] * 10,
                              "venue_key": ["five_art_studio"] * 35 + ["widya_batik_class"] * 35, "annotation_session": "batch:fixture"})
    terms = {mode: bi_nlp.distinctive_terms(synthetic, mode, frozenset())[0] for mode in bi_nlp.TERM_MODES}
    assert all(len(frame) for frame in terms.values())
    curves = bi_data.scroll_curves(*bi_data.read_collector_snapshot(bi_data.source_paths()["raw_snapshot"]))
    empty_terms = bi_nlp.distinctive_terms(labels, bi_nlp.TERM_MODES[0], frozenset())[0]
    specs = []
    for print_mode in (False, True):
        cfg = charts.base_config(print_mode)
        specs += [charts.trend_spec(monthly, grain, cfg, through=pd.Timestamp("2025-09-01")) for grain in ("Bulanan", "Tahunan")]
        specs += [charts.trend_spec(monthly.iloc[:0], "Bulanan", cfg), charts.venue_share_spec(venues, venues["overall"].iloc[0], cfg)]
        specs += [charts.rating_text_spec(bi_data.rating_text_matrix(labels)[0], cfg), charts.length_spec(bi_nlp.length_bins(labels), cfg)]
        specs += [charts.aspect_heatmap_spec(bi_data.aspect_venue_matrix(business, metric, None), metric, cfg) for metric in bi_data.METRICS]
        specs += [charts.terms_spec(frame, mode, cfg) for mode, frame in terms.items()] + [charts.terms_spec(empty_terms, "Keluhan vs lainnya", cfg)]
        specs += [charts.funnel_spec(bi_data.funnel_table(reviews, labels), cfg), charts.coverage_spec(bundle["coverage"], cfg)]
        specs += [charts.quality_spec(bi_data.quality_matrix(reviews), cfg), charts.scroll_spec(curves, "6 observasi duplikat", cfg)]
    for spec in specs:
        text = json.dumps(spec, allow_nan=False)
        assert {code.upper() for code in re.findall(r"#[0-9A-Fa-f]{6}", text)} <= charts.PALETTE, text[:200]
        assert spec["config"]["locale"]["number"]["decimal"] == "," and not any(secret in text for secret in SECRETS)
    assert specs[0]["config"]["axis"]["labelFontSize"] == 12 and specs[-1]["config"]["axis"]["labelFontSize"] == 15
    assert "pilih_sel" in json.dumps(specs[6]) and "pilih_kata" in json.dumps(specs[8]) and "pilih_kata" in json.dumps(specs[9])
    yearly = {row["tahun"]: row for row in specs[1]["data"]["values"]}
    assert yearly["2025"]["parsial"] and yearly["2025"]["label"].endswith("Jan–Sep") and not yearly["2023"]["parsial"]


def test_aspect_names_match_business_dashboard():
    from temajelajah import business_dashboard
    assert bi_data.ASPECT_NAMES == business_dashboard.LABEL_NAMES and bi_data.ASPECTS == business_dashboard.LABELS


def _trend_rows():
    counts = {f"2024-{m:02d}-01": 3 for m in range(1, 9) if m != 3}
    counts |= {f"{y}-{m:02d}-01": 10 for y, months in ((2024, range(9, 13)), (2025, range(1, 9))) for m in months}
    counts |= {f"{y}-{m:02d}-01": 5 for y, months in ((2025, range(9, 13)), (2026, range(1, 9))) for m in months}
    rows = _monthly_rows(counts | {"2026-09-01": 1, "2026-10-01": 4})
    return rows.assign(year=rows["month"].dt.year.astype("Int64"))


STAMP = pd.Timestamp("2026-10-01T13:00:00+00:00")


def test_year_filter_that_cuts_a_window_withholds_the_comparison():
    reviews = _trend_rows()
    _, whole = bi_data.monthly_series(reviews, STAMP, years=(2024, 2026))
    assert not whole["cut"] and whole["ok"] and (whole["current"], whole["previous"]) == (56, 115)
    sliced = reviews[reviews["year"] >= 2025]  # what the 2025-2026 year filter leaves: Sep-Dec 2024 is gone
    _, naive = bi_data.monthly_series(sliced, STAMP)
    assert (naive["current"], naive["previous"]) == (56, 85)  # without years, an incomplete comparison could appear valid
    _, cut = bi_data.monthly_series(sliced, STAMP, years=(2025, 2026))
    assert cut["cut"] and not cut["ok"] and cut["span_label"] == "Okt 2024–Sep 2026"
    k1 = bi_data.kpis(sliced, None, None, cut)["K1"]
    assert k1["delta"] is None and "filter tahun memotong jendela Okt 2024–Sep 2026" in k1["help"]
    assert "volume_trend" not in [item.key for item in bi_data.compute_insights(sliced, None, None, None, cut)]


def test_every_slice_shares_the_corpus_calendar_for_the_incomplete_month():
    corpus = _trend_rows()
    four = {f"{y}-{m:02d}-01": 4 for y, months in ((2024, range(9, 13)), (2025, range(1, 13)), (2026, range(1, 9))) for m in months}
    venue = _monthly_rows(four | {"2026-09-01": 3})
    own_frame, own = bi_data.monthly_series(venue, STAMP)
    assert own["partial"] is None and own["end"] == pd.Timestamp("2026-09-01") and (own["current"], own["previous"]) == (47, 48)
    frame, shared = bi_data.monthly_series(venue, STAMP, basis=corpus)
    assert shared["partial"] == pd.Timestamp("2026-09-01") and shared["end"] == pd.Timestamp("2026-09-01")
    assert (shared["current"], shared["previous"]) == (47, 48)
    assert (shared["sensitivity"]["current"], shared["sensitivity"]["previous"], shared["sensitivity"]["change"]) == (48, 48, 0.0)
    assert frame.set_index("bulan").loc["2026-09-01", "parsial"] and not own_frame["parsial"].any()
    quiet = venue[venue["month"] < "2026-09-01"]  # no review at all in the incomplete month
    for basis in (None, corpus):
        _, stopped = bi_data.monthly_series(quiet, STAMP, basis=basis)
        assert stopped["end"] == pd.Timestamp("2026-09-01") and stopped["current"] == 44
        assert stopped["sensitivity"]["end"] == pd.Timestamp("2026-08-01") and stopped["sensitivity"]["current"] == 48
    assert bi_data.partial_month(corpus, STAMP) == pd.Timestamp("2026-09-01") and bi_data.partial_month(venue, STAMP) is None


def test_kpis_delta_tone_and_sparkline_from_valid_windows():
    reviews = _trend_rows()
    _, windows = bi_data.monthly_series(reviews, STAMP)
    k1 = bi_data.kpis(reviews, None, None, windows)["K1"]
    assert (k1["delta"], k1["delta_description"], k1["tone"]) == ("-51%", "12 bln s.d. Sep 2026 vs 12 bln sebelumnya", "Perhatian")
    assert k1["chart_data"] == [61, 100] and "2024–2025, tahun berjalan tidak ditampilkan" in k1["help"]  # 2026 is incomplete
    trend = next(item for item in bi_data.compute_insights(reviews, None, None, None, windows) if item.key == "volume_trend")
    assert (trend.tone, trend.value, trend.text, trend.basis) == ("Perhatian", "−51%", "56 vs 115 ulasan", "Okt 2025–Sep 2026 vs Okt 2024–Sep 2025")


def test_venue_outlier_insight_withholds_cross_cohort_rank():
    rows = [("museum_tekstil", "legacy81", i < 16) for i in range(81)] + [("five_art_studio", "real1524", i < 20) for i in range(900)]
    labels = pd.DataFrame(rows, columns=["venue_key", "cohort", "complaint"]).assign(rating_num=5.0)
    assert "venue_outlier" not in [item.key for item in bi_data.compute_insights(None, labels, None, None, None)]


def test_word_count_counts_han_and_kana_by_characters():
    assert (bi_data.word_count("Great class!"), bi_data.word_count(""), bi_data.word_count(None)) == (2, 0, 0)
    assert bi_data.word_count("체험이 정말 좋았어요") == 3  # Hangul is written with spaces
    chinese = "老师非常耐心地教我们画蜡染图案，整个过程很有趣。"  # 15 + 7 Han characters, no spaces
    assert bi_data.word_count(chinese) == 12 and bi_data.word_count("Gedeの街中にある素敵な場所") == 1 + 6
    long_review = pd.DataFrame({"clean_text": [chinese * 8, "Great class!"], "complaint": [True, False]})
    assert bi_nlp.length_bins(long_review)["n"].tolist() == [1, 0, 0, 1, 0]  # 96 words, no longer a "≤20" complaint


def test_shares_never_round_a_nonzero_count_to_zero():
    assert [bi_data.fmt_share(x) for x in (1 / 1513, 0.996, 0.97, 0, 1)] == ["<1%", ">99%", "97%", "0%", "100%"]
    labels = pd.DataFrame({"rating_num": 5.0, "sentiment_4class": ["positif"] * 300 + ["negatif"], "complaint": [False] * 300 + [True]})
    cells = bi_data.rating_text_matrix(labels)[0].set_index(["rating", "sentimen"])["label"]
    assert (cells[(5, "negatif")], cells[(5, "positif")]) == ("1 · <1%", "300 · >99%")


def test_kpis_withhold_coverage_when_a_platform_report_is_missing(bundle):
    cards = bi_data.kpis(bundle["reviews"], None, bundle["coverage"], _windows(bundle), ["Google Maps"])
    assert cards["K5"]["value"] == "—" and "Google Maps" in cards["K5"]["help"] and "k" not in cards["K5"]
    assert bi_data.kpis(bundle["reviews"], None, None, _windows(bundle), ["Airbnb"])["K5"] is None


def test_load_bundle_names_missing_sources_and_tolerates_odd_audits(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    patch_paths(monkeypatch, paths)
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["coverage_missing"] == [] and loaded["snapshots"]["n"].tolist() == [9, 12] and "names" not in loaded
    assert not any(secret in repr(loaded["reports"]) for secret in SECRETS)
    audit_path = paths["run"] / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    (paths["expansion"] / "runs" / OLD_RUN_ID / "audit.json").write_text(json.dumps({"counts": {"stored_unique_source_identities": 9}}), encoding="utf-8")
    audit_path.write_text(json.dumps({**audit, "snapshot": "sha", "counts": [1]}), encoding="utf-8")
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["snapshot_ok"] is None and not loaded["snapshot_missing"] and loaded["audit_counts"] == {}
    assert loaded["snapshots"]["n"].tolist() == []  # an audit without created_at (or counts) cannot be ordered: left out
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    paths["snapshot"].unlink()
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["snapshot_ok"] is None and loaded["snapshot_missing"]
    audit_path.unlink()
    for report in (paths["expansion"] / "airbnb").glob("*.json"):
        report.unlink()
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["coverage_missing"] == [("Airbnb", None), ("Google Maps", None)] and loaded["coverage"]["platform"].tolist() == ["SeekSophie"]


def test_run_telemetry_and_ops_summary_values(bundle):
    runs, _ = bi_data.read_collector_snapshot(bi_data.source_paths()["raw_snapshot"])
    telemetry = bi_data.run_telemetry(runs, bundle["reports"])
    assert telemetry["platform"].tolist() == ["Google Maps", "Google Maps", "Airbnb", "Airbnb", "SeekSophie"]  # oldest first, undated last
    assert telemetry["target"].tolist() == ["Museum Tekstil", "Museum Tekstil", "Five Arts", "Popiler 2", "Five Arts"]
    assert telemetry["durasi"].tolist()[:4] == [60.0, 30.0, 30.0, 60.0] and math.isnan(telemetry["durasi"].iloc[4])
    assert telemetry["baru"].tolist() == [3.0, 1.0, 4.0, 5.0, 20.0]
    assert telemetry["hasil"].tolist() == ["parsial", "parsial", "lengkap", "lengkap", "parsial"]
    assert telemetry["alasan"].tolist() == ["feed macet", "urutan terbaru tak tersedia", "jumlah ulasan tertulis tercapai",
                                            "jumlah ulasan tertulis tercapai", "muatan awal halaman publik habis"]
    assert not any(secret in telemetry.to_csv() for secret in SECRETS)
    cards = bi_data.ops_summary(bundle["snapshots"], bundle["coverage"], telemetry)
    assert (cards["identities"]["value"], cards["identities"]["delta"], cards["identities"]["chart_data"]) == ("12", "+3", [9, 12])
    assert cards["identities"]["help"] == "2 snapshot audit, 16:00–17:00 UTC"
    assert (cards["duplicates"]["value"], cards["duplicates"]["delta"]) == ("40", "≈13× per ulasan unik")
    assert (cards["runs"]["value"], cards["runs"]["delta"], cards["runs"]["warn"]) == ("5", "semua 2 run Google parsial", True)
    assert (cards["duration"]["value"], cards["duration"]["delta"]) == ("3,0 mnt", "4 run berdurasi")
    assert cards["duration"]["help"] == "Airbnb 1,5 mnt · Google Maps 1,5 mnt; SeekSophie tanpa catatan durasi"
    shrinking = pd.DataFrame({"created_at": pd.to_datetime(["2026-10-01T16:00:00Z", "2026-10-01T17:00:00Z"]), "n": [20, 12]})
    empty = bi_data.ops_summary(shrinking, None, telemetry.iloc[:0])
    assert empty["identities"]["delta"] == "-8" and empty["duplicates"] is None and empty["runs"] is None
    assert [bi_data.fmt_signed(n) for n in (386, 0, -1234, None)] == ["+386", "0", "-1.234", "—"]


def test_scroll_curves_label_runs_without_a_start_time(tmp_path):
    paths = write_fixture(tmp_path, labels=False, business=False)
    runs, checkpoints = bi_data.read_collector_snapshot(paths["snapshot"])
    runs.loc[runs["run_id"].eq("run-b"), "started_at"] = None
    assert bi_data.scroll_curves(runs, checkpoints)["run"].unique().tolist() == ["13:30 · 2 putaran", "run run-b · 1 putaran"]
    assert math.isnan(bi_data.run_telemetry(runs, [])["durasi"].iloc[1])


def test_name_masking_catches_rare_names_handles_honorifics_and_signoffs(bundle):
    texts = ["Our guide, Bimo, started by sharing a bit about the history.", "Kusman welcomed us to his home.", "Dik was a great teacher.",
             "Julie, (one of our teachers) helped me.", "A Wayang show near Ubud in Bali.", "Additionally, we had lunch."]
    vocab = bi_nlp.proper_noun_vocab(texts)
    assert {"bimo", "kusman", "dik", "julie"} <= vocab and not {"wayang", "ubud", "bali", "additionally", "show", "our"} & vocab
    assert {"dika", "bimo", "paige", "alex", "nunu", "wayan"} <= bundle["names"] and "wayang" not in bundle["names"]
    mask = bi_nlp.name_masker(vocab | bundle["names"])
    cases = {"Our guide, Bimo, started": "Our guide, [nama], started",
             "Thanks a lot! Carolin and Lisa": "Thanks a lot! [nama] and [nama]",
             "Aloha!\n-Paige and Alex": "Aloha!\n-[nama] and [nama]",
             "Best, Sophie": "Best, [nama]",
             "Der Kurs war super. liebe Grüße Luisa :)": "Der Kurs war super. liebe Grüße [nama] :)",
             "smile @Wayangkujana @kadek_bali see u": "smile [nama] [nama] see u",
             "mail me at a.b@example.org": "mail me at [nama]",
             "Pak Suseno and Mba Nuning’s tour, thanks Mr. Wayan!": "Pak [nama] and Mba [nama] tour, thanks Mr. [nama]!",
             "Lovely class!\n\nSamantha": "Lovely class!\n\n[nama]",
             "Recommend!\n\nTerimakasih :)": "Recommend!\n\nTerimakasih :)",
             "I miss Bali. Bali and Java": "I miss Bali. Bali and Java",
             "A Wayang show in Ubud with Nunu": "A Wayang show in Ubud with [nama]"}
    for text, expected in cases.items():
        assert mask(text) == expected, text
    review = pd.DataFrame({"clean_text": ["We are happy about the beautiful results. Thanks a lot! Carolin and Lisa"], "complaint": [False],
                           "year": [2024], "venue_key": ["five_art_studio"], "rating_num": [5.0], "sentiment_4class": ["positif"]})
    assert bi_nlp.kwic(review, "results", mask)[0]["Kanan"].iloc[0] == ". Thanks a lot! [nama] and [nama]"
    rows, _ = bi_nlp.kwic(review, "Lisa", mask)  # the sign-off is found on the whole review, not on the cut-off left context
    assert rows["Kata"].iloc[0] == "[nama]" and "Carolin" not in rows.to_csv()


def test_distinctive_terms_refuse_empty_comparison_groups():
    texts = ["Lovely wax painting with tea"] * 40
    one = pd.DataFrame({"source_locale": "en", "clean_text": texts, "complaint": [False] * 36 + [True] * 4, "venue_key": "widya_batik_class"})
    for mode in bi_nlp.TERM_MODES:
        assert bi_nlp.distinctive_terms(one, mode, frozenset())[0].empty and bi_nlp.terms_blocker(one, mode)
    assert "≥2 venue" in bi_nlp.terms_blocker(one, "Per venue") and "ada 4 dan 36" in bi_nlp.terms_blocker(one, "Keluhan vs lainnya")
    both = pd.concat([one, one.assign(venue_key="five_art_studio"), one.assign(venue_key="museum_tekstil", source_locale="fr")], ignore_index=True).assign(annotation_session="batch:fixture")
    terms, skipped = bi_nlp.distinctive_terms(both, "Per venue", frozenset())
    assert terms.empty and skipped == ["Museum Tekstil (0)"]  # identical groups: rounding noise (|z| ~ 1e-15) is not a term
    assert bi_nlp.terms_blocker(both, "Per venue") is None


def test_kwic_matches_cjk_queries_inside_cjk_text():
    labels = pd.DataFrame({"clean_text": ["我们在巴厘岛学习了蜡染，非常有趣", "バティック体験が楽しかった", "batik 체험이 좋았어요", "Great batik"],
                           "complaint": [False, False, True, False], "year": [2024, 2025, 2023, 2022], "venue_key": "widya_batik_class",
                           "rating_num": 5.0, "sentiment_4class": ["positif", "positif", "campuran", "positif"]})
    mask = bi_nlp.name_masker([])
    for query, matches in (("蜡染", 1), ("バティック", 1), ("체험", 1), ("batik", 2), ("bat", 0)):
        assert bi_nlp.kwic(labels, query, mask)[1]["matches"] == matches, query


def test_kwic_matches_either_apostrophe_like_the_term_counts():
    labels = pd.DataFrame({"clean_text": ["I didn’t like the wax", "We didn't get lunch", "It did not rain", "didn’t"],
                           "complaint": [True, True, False, False], "year": [2024, 2023, 2022, 2021], "venue_key": "widya_batik_class",
                           "rating_num": 4.0, "sentiment_4class": ["negatif", "campuran", "positif", "positif"]})
    assert all("didn't" in bi_nlp.tokenize(text) for text in labels["clean_text"].iloc[[0, 1, 3]])  # what the terms chart counts
    for query in ("didn't", "didn’t", "DIDN'T"):  # a clicked term carries the straight apostrophe
        rows, stats = bi_nlp.kwic(labels, query, bi_nlp.name_masker([]))
        assert (stats["matches"], stats["complaints"], stats["query"]) == (3, 2, query)
        assert sorted(rows["Kata"]) == ["didn't", "didn’t", "didn’t"]


def test_a_rare_name_written_once_in_lower_case_stays_masked():
    texts = ["Liz was out, so we were met by her sister Cathy who gave us a tour of her house.",
             "fue excelente atencion de cathy y toda la gente",  # an informal lower-case use elsewhere does not clear the name
             "We painted a Mandala and a fish.", "the mandala took hours",  # a list item, not a name
             "Our Teacher was kind.", "the teacher smiled", "a teacher"]  # a common word: lower case wins
    vocab = bi_nlp.proper_noun_vocab(texts)
    assert {"cathy", "liz"} <= vocab and not {"mandala", "teacher", "fish"} & vocab
    mask = bi_nlp.name_masker(vocab)
    assert mask(texts[0]) == "[nama] was out, so we were met by her sister [nama] who gave us a tour of her house."
    assert mask(texts[1]) == "fue excelente atencion de [nama] y toda la gente"


def test_names_in_cyrillic_and_before_very_are_masked():
    mask = bi_nlp.name_masker(bi_nlp.proper_noun_vocab(["Un’esperienza unica. Qatar molto gentile e competente."]))
    assert mask("Un’esperienza unica. Qatar molto gentile.") == "Un’esperienza unica. [nama] molto gentile."
    text = "Транспорт был комфортным. Густи отличный водитель, а с Кетутом мы рисовали."
    assert mask(text) == "Транспорт был комфортным. [nama] отличный водитель, а с [nama]ом мы рисовали."
    assert mask("тучи сгустились") == "тучи сгустились"  # a name starts at a word start


def test_per_venue_terms_name_venues_without_language_metadata():
    texts = ["Lovely wax painting with tea"] * 40 + ["Great garden with tea"] * 40 + ["Old museum hall"] * 3
    labels = pd.DataFrame({"source_locale": ["en"] * 80 + [""] * 3, "clean_text": texts, "complaint": False,
                           "venue_key": ["five_art_studio"] * 40 + ["widya_batik_class"] * 40 + ["museum_tekstil"] * 3, "annotation_session": "batch:fixture"})
    assert bi_nlp.distinctive_terms(labels, "Per venue", frozenset())[1] == ["Museum Tekstil (0; 3 tanpa metadata bahasa)"]


def test_pooled_shares_name_the_older_museum_labels(bundle):
    labels, real = bundle["labels"], bundle["labels"][bundle["labels"]["cohort"].eq("real1524")]
    assert bi_data.legacy_note(labels) == "termasuk 2 label lama Museum Tekstil"
    assert bi_data.legacy_note(real) == "" == bi_data.legacy_note(None)
    cards = bi_data.kpis(bundle["reviews"], labels, None, _windows(bundle))
    assert cards["K3"]["help"].endswith("label asisten sementara · termasuk 2 label lama Museum Tekstil")
    assert "label lama" not in bi_data.kpis(bundle["reviews"], real, None, _windows(bundle))["K3"]["help"]
    many = bi_data.kpis(bundle["reviews"], pd.concat([labels] * 4, ignore_index=True), None, _windows(bundle))
    assert many["K4"]["help"].endswith("termasuk 8 label lama Museum Tekstil")


def test_coverage_card_and_run_table_share_one_status_rule(bundle):
    reports = [{"platform": "Airbnb", "venue_key": "widya_batik_class", "customer_texts": 272, "reported": "272", "stop_reason": "time_budget"},
               {"platform": "Airbnb", "venue_key": "batik_popiler_2", "customer_texts": 0, "reported": "5", "stop_reason": "collector_error"},
               {"platform": "SeekSophie", "venue_key": "five_art_studio", "actual_unique_reviews_collected": 0,
                "option_reviews_total_metadata_only": 89, "stop_reason": "initial_public_payload_exhausted"}]
    telemetry = bi_data.run_telemetry(None, reports)
    coverage = bi_data.coverage_rows([{**report, "source_totals_metadata": {"writtenReviewCount": report["reported"]}} for report in reports[:2]],
                                     reports[2], [], {})
    expected = {("Airbnb", "widya_batik_class"): "lengkap", ("Airbnb", "batik_popiler_2"): "gagal", ("SeekSophie", "five_art_studio"): "gagal"}
    assert dict(zip(zip(telemetry["platform"], telemetry["venue_key"]), telemetry["hasil"])) == expected
    assert dict(zip(zip(coverage["platform"], coverage["venue_key"]), coverage["status"])) == expected
    assert {report["venue_key"]: report["reported"] for report in bundle["reports"] if report["platform"] == "Airbnb"} == {
        "five_art_studio": "4", "batik_popiler_2": "5"}


def test_report_gaps_name_a_single_missing_target(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    (paths["expansion"] / "airbnb" / "five-report.json").unlink()
    patch_paths(monkeypatch, paths)
    loaded = bi_data.load_bundle(bi_data.source_paths())
    reviews = loaded["reviews"]
    assert loaded["coverage_missing"] == [("Airbnb", "five_art_studio")]
    assert bi_data.missing_reports(loaded["coverage_missing"], reviews) == ["Airbnb · Five Arts"]
    assert bi_data.missing_reports(loaded["coverage_missing"], reviews[reviews["venue_key"].eq("batik_popiler_2")]) == []
    assert bi_data.missing_reports([("Airbnb", None)], reviews[reviews["source"].eq("SeekSophie")]) == []
    cards = bi_data.kpis(reviews, None, loaded["coverage"], _windows(loaded), ["Airbnb · Five Arts"])
    assert cards["K5"]["value"] == "—" and "Airbnb · Five Arts" in cards["K5"]["help"]


def test_identity_snapshots_end_at_the_run_on_screen(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    newer = paths["expansion"] / "runs" / "20261001T180000000000Z_newer" / "audit.json"
    newer.parent.mkdir(parents=True)
    newer.write_text(json.dumps({"created_at": "2026-10-01T18:00:00+00:00", "counts": {"stored_unique_source_identities": 50}}), encoding="utf-8")
    patch_paths(monkeypatch, paths)
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["snapshots"]["n"].tolist() == [9, 12]  # funnel stage 1 of this run is 12; the newer run is not on screen
    (paths["run"] / "audit.json").unlink()
    loaded = bi_data.load_bundle(bi_data.source_paths())
    no_runs = pd.DataFrame(columns=bi_data.TELEMETRY_COLS)
    assert loaded["snapshots"].empty and bi_data.ops_summary(loaded["snapshots"], None, no_runs)["identities"] is None


def _shipped(spec: dict) -> str:
    """The spec string Streamlit sends after moving the rows into Arrow (the frontend keys its view on this string)."""
    from streamlit.elements.vega_charts import _marshall_chart_data, _stabilize_vega_json_spec
    from streamlit.proto.VegaLiteChart_pb2 import VegaLiteChart
    spec = json.loads(json.dumps(spec))
    _marshall_chart_data(VegaLiteChart(), spec)
    return _stabilize_vega_json_spec(json.dumps(spec))


def test_chart_specs_change_with_their_rows_after_streamlit_moves_the_data(bundle):
    from temajelajah import bi_charts as charts
    cfg, labels, business = charts.base_config(), bundle["labels"], bundle["business"]
    flipped = labels.assign(sentiment_4class=labels["sentiment_4class"].replace({"campuran": "netral"}))
    first, second = (_shipped(charts.rating_text_spec(bi_data.rating_text_matrix(frame)[0], cfg)) for frame in (labels, flipped))
    assert first != second and _shipped(charts.rating_text_spec(bi_data.rating_text_matrix(labels)[0], cfg)) == first  # stable on reruns
    picks = [_shipped(charts.aspect_heatmap_spec(bi_data.aspect_venue_matrix(business, "Keluhan", cell), "Keluhan", cfg))
             for cell in (None, ("fasilitas_kenyamanan", "batik_popiler_2"))]
    assert picks[0] != picks[1]  # a click moves the frame: same rows, another selected cell
    heat = charts.aspect_heatmap_spec(bi_data.aspect_venue_matrix(business, "Keluhan", None), "Keluhan", cfg)
    assert heat["layer"][1]["encoding"]["strokeOpacity"] == {"condition": {"test": "datum.terpilih", "value": 1}, "value": 0}
    texts = ["Lovely wax painting with tea"] * 35 + ["Lovely garden with tea"] * 35
    venues = pd.DataFrame({"source_locale": "en", "clean_text": texts, "complaint": False,
                           "venue_key": ["five_art_studio"] * 35 + ["widya_batik_class"] * 35, "annotation_session": "batch:fixture"})
    terms = bi_nlp.distinctive_terms(venues, "Per venue", frozenset())[0]
    assert "rev" in charts.terms_spec(terms, "Per venue", cfg)["usermeta"]


def test_annotation_provenance_keeps_unknowns_and_batch_source_separate(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    patch_paths(monkeypatch, paths)
    supplied = bi_data.source_paths()
    business_path = supplied["business"]
    source = pd.read_csv(business_path, dtype=str, keep_default_na=False)
    source["annotation_component_file"] = "exports/business/annotations/batch_one.jsonl"
    source["annotation_component_sha256"] = "a" * 64
    source["annotator_requested_alias"] = "GPT Sol"  # an alias must never become a backend identifier
    source["backend_model_id"] = "null"
    source.to_csv(business_path, index=False)
    loaded = bi_data.load_bundle(supplied)
    business = loaded["business"]
    assert set(business["annotation_session"]) == {"batch:batch_one.jsonl"}
    assert set(business["annotation_model"]) == {"unknown"}
    assert all(set(business[field]) == {"unknown"} for field in ("annotation_rubric", "annotation_prompt", "annotation_date"))
    assert set(business["presence_verification"]) == {"author_reported_single_author_check"}
    assert set(business["polarity_verification"]) == {"unverified"}
    metadata = bi_data.provenance_table(business)
    assert metadata["rows"].sum() == len(business) and metadata["annotation_source_sha256"].tolist() == ["a" * 64]
    assert not {"clean_text", "raw_text", "source_review_id"} & set(metadata.columns)
    assert set(loaded["evidence"]["annotation_session"]) == {"batch:batch_one.jsonl"}


def test_snapshot_digest_mismatch_is_detected_even_with_valid_csv(tmp_path, monkeypatch):
    paths = write_fixture(tmp_path)
    patch_paths(monkeypatch, paths)
    with paths["snapshot"].open("ab") as handle:
        handle.write(b"digest mismatch")
    loaded = bi_data.load_bundle(bi_data.source_paths())
    assert loaded["snapshot_ok"] is False and not loaded["snapshot_missing"]
    assert len(loaded["reviews"]) == 12  # CSV loading must not turn a mismatch into a verified snapshot


def test_annotation_filters_roundtrip_and_empty_selection_do_not_invent_rows(bundle):
    labels = bundle["labels"].copy()
    labels["annotation_session"] = ["batch:a"] * 5 + ["batch:b"] * 5
    selected = bi_data.Filters(sessions=["batch:a"], rubrics=["unknown"], prompts=["unknown"], dates=["unknown"],
                               models=["unknown"], verifications=["unverified"])
    assert bi_data.Filters(*selected.key()) == selected
    assert len(bi_data.apply_filters(labels, selected)) == 5
    assert len(bi_data.apply_filters(bundle["reviews"], selected)) == 12  # review volume is independent of annotation filters
    empty = bi_data.apply_filters(labels, bi_data.Filters(sessions=["batch:missing"]))
    assert empty.empty and bi_data.provenance_table(empty).empty
    assert bi_data.venue_complaints(empty).empty
    assert all(bi_data.kpis(bundle["reviews"].iloc[:0], empty, None, {})[key]["value"] == "—" for key in ("K2", "K3", "K4"))


def test_cross_batch_contrasts_are_withheld_and_within_batch_stays_descriptive():
    reviews = pd.DataFrame({"source_locale": "en", "clean_text": ["rude hot slow"] * 40 + ["lovely garden tea"] * 40,
                            "complaint": [True] * 40 + [False] * 40,
                            "venue_key": ["widya_batik_class"] * 40 + ["five_art_studio"] * 40,
                            "cohort": "real1524", "annotation_session": ["batch:b"] * 40 + ["batch:a"] * 40})
    assert bi_nlp.distinctive_terms(reviews, "Per venue", frozenset())[0].empty
    assert "lintas batch" in bi_nlp.terms_blocker(reviews, "Per venue")
    shares = bi_data.venue_complaints(reviews)
    assert shares["overall"].isna().all() and shares["annotation_session"].tolist() == ["batch:a", "batch:b"]
    assert "venue_outlier" not in [item.key for item in bi_data.compute_insights(None, reviews.assign(rating_num=5), None, None, None)]
    same = reviews.assign(annotation_session="batch:one")
    assert bi_nlp.terms_blocker(same, "Per venue") is None
    assert not bi_nlp.distinctive_terms(same, "Per venue", frozenset())[0].empty
    assert bi_data.venue_complaints(same)["venue_key"].tolist() == ["five_art_studio", "widya_batik_class"]  # canonical, not score-ranked
    aspects = _business(0.7, 0.2).assign(annotation_session=["batch:a"] * 100 + ["batch:b"] * 100)
    pooled = bi_data.aspect_venue_matrix(aspects, "Keluhan", None)
    assert pooled.empty and "comparison_blocker" in pooled.attrs
    assert "aspect_gap" not in [item.key for item in bi_data.compute_insights(None, None, aspects, None, None)]
    unknown = same.assign(annotation_session="unknown (real1524)")
    assert "tidak diketahui" in bi_data.session_comparison_blocker(unknown)
    assert bi_nlp.distinctive_terms(unknown, "Per venue", frozenset())[0].empty
    assert bi_data.venue_complaints(unknown)["overall"].isna().all()
    unknown_aspects = aspects.assign(annotation_session="unknown (business1515)")
    assert not bi_data.aspect_venue_matrix(unknown_aspects, "Penyebutan", None).empty  # descriptive cells remain readable
    assert "aspect_gap" not in [item.key for item in bi_data.compute_insights(None, None, unknown_aspects, None, None)]


def test_sparse_aspect_cells_expose_mentions_missing_polarity_and_na():
    from temajelajah import bi_charts as charts
    rows = _business(0.0, 0.0, n=100)
    target = "fasilitas_kenyamanan"
    rows.loc[:2, f"presence_{target}"] = True
    rows.loc[0, f"polarity_{target}"] = "negatif"
    rows.loc[1, f"polarity_{target}"] = "positif"
    matrix = bi_data.aspect_venue_matrix(rows, "Keluhan", None)
    cell = matrix.set_index(["aspek_kode", "venue_kode"]).loc[(target, "batik_kotagede_nunung")]
    assert (cell["n"], cell["mentions"], cell["polarity_n"], cell["missing_polarity"], cell["complaints"]) == (100, 3, 2, 1, 1)
    assert cell["nilai"] == 0.5 and cell["small"] and "1/2" in cell["label"] and "1 hilang" in cell["label"]
    empty = matrix[matrix["mentions"].eq(0)]
    assert empty["nilai"].isna().all() and empty["label"].eq("NA\n0/0").all() and empty["missing"].all()
    spec = charts.aspect_heatmap_spec(matrix, "Keluhan", charts.base_config())
    assert any(value["nilai"] is None and value["missing"] for value in spec["data"]["values"])
    assert "datum.missing" in json.dumps(spec) and "Denominator sel" in json.dumps(spec)
    json.dumps(spec, allow_nan=False)


def test_log_odds_uses_document_frequency_mass_not_review_count_and_no_threshold():
    from temajelajah import bi_charts as charts
    docs_a, docs_b = [{"wax", "tea", "red"}] * 5, [{"tea", "blue"}] * 5
    terms = bi_nlp.log_odds_dirichlet(docs_a, docs_b, alpha0=10, min_df=1).set_index("term")
    assert terms["n_a"].unique().tolist() == [15] and terms["n_b"].unique().tolist() == [10]
    assert terms["docs_a"].unique().tolist() == [5] and terms["docs_b"].unique().tolist() == [5]
    alpha = 10 * 5 / 25
    delta = math.log((5 + alpha) / (15 + 10 - 5 - alpha)) - math.log(alpha / (10 + 10 - alpha))
    assert terms.loc["wax", "z"] == pytest.approx(delta / math.sqrt(1 / (5 + alpha) + 1 / alpha))
    labels = pd.DataFrame({"source_locale": "en", "clean_text": ["wax tea red"] * 35 + ["tea blue"] * 35,
                           "complaint": [True] * 35 + [False] * 35, "venue_key": "five_art_studio"})
    shown = bi_nlp.distinctive_terms(labels, "Keluhan vs lainnya", frozenset())[0]
    spec = charts.terms_spec(shown, "Keluhan vs lainnya", charts.base_config())
    assert "kuat" not in shown and "1.96" not in json.dumps(spec)
    assert spec["layer"][0]["encoding"]["opacity"] == {"value": 1}


def test_coverage_zero_collected_is_different_from_unknown_observation():
    rows = bi_data.coverage_rows([], None,
        [{"target_key": "seno", "inserted_total": 0, "reported_total": 20},
         {"target_key": "giriloyo", "inserted_total": 0, "reported_total": None},
         {"target_key": "museum_tekstil", "inserted_total": None, "reported_total": 20}], {})
    zero, unknown_denominator, unknown_observation = (rows.iloc[index] for index in range(3))
    assert zero["coverage"] == 0 and zero["label"] == "0/20 · 0%"
    assert math.isnan(unknown_denominator["coverage"]) and unknown_denominator["label"] == "0 · total tak diketahui"
    assert math.isnan(unknown_observation["collected"]) and math.isnan(unknown_observation["coverage"])
    assert unknown_observation["status"] == "tak diketahui" and unknown_observation["label"] == "—/20 · —"


def test_funnel_excludes_explicit_rejection_and_legacy_from_new_training_scope(bundle):
    labels = bundle["labels"]
    assert set(labels.loc[labels["actual_ml_split"].eq("excluded"), "record_id"]) == {"R07"}
    assert labels[labels["cohort"].eq("legacy81")]["record_id"].tolist() == ["R10", "L01"]
    funnel = bi_data.funnel_table(bundle["reviews"], labels)
    splits = funnel[funnel["tahap"].str.startswith("Split")]
    assert splits["n"].sum() == 7  # 8 new labels minus the explicitly excluded cooking review; legacy stays separate
    real = bi_data.apply_filters(labels, bi_data.Filters(sessions=["unknown (real1524)"]))
    assert len(real) == 8 and not real["cohort"].eq("legacy81").any()
