"""Small on-disk export tree for the BI dashboard; planted secrets must never reach a loaded frame."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

RUN_ID = "20261001T170000000000Z_fixture"
OLD_RUN_ID = "20261001T160000000000Z_older"
SECRETS = ("SECRET", "secret.example")
NAMES = ("Wayan", "Dika", "Bimo", "Nunu", "kadek_bali", "Paige", "Alex")  # planted in review texts; each must reach the page only as [nama]
ASPECTS = ("pengajar_layanan", "belajar_praktik", "hasil_karya_produk", "fasilitas_kenyamanan")
ABS = "absolute_review_date_unavailable"
LEGACY = ["language_variant_unknown", ABS, "completeness_migration_default_unverified"]
TWIN = "identical_clean_text_different_source_ids_retained"
CORPUS = [  # id, venue, source, role, review_date, relative_date, rating, locale, variant, completeness, domain, flags, duplicates, text
    ("R01", "five_art_studio", "Airbnb", "customer_text", "", "August 2023", "5", "en", "original", "complete_as_observed", "True", [ABS], "1",
     "We loved the batik class with our teacher Wayan, and Wayan was patient. Later Wayan showed us the wax pen."),
    ("R02", "five_art_studio", "Airbnb", "customer_text", "", "September 2025", "5.0", "en", "original", "complete_as_observed", "True", [ABS], "1",
     "Fun workshop but the room was a bit hot and the communication was slow. Dika was kind."),
    ("R03", "five_art_studio", "SeekSophie", "customer_text", "2023-08-05T10:07:07.536+08:00", "", "5", "", "displayed", "complete_as_observed", "True", [], "1",
     "Wonderful batik painting session in Bali with great wax work."),
    ("R04", "five_art_studio", "Airbnb", "customer_text", "", "September 2025", "4", "fr", "original", "unverified_legacy_default", "True", [ABS], "1",
     "Atelier de batik très agréable, merci."),
    ("R05", "batik_popiler_2", "Airbnb", "customer_text", "", "August 2023", "2", "en", "original", "complete_as_observed", "True", [ABS], "1",
     "Not good. The communication was poor and the class ended early. Our guide, Bimo, left."),
    ("R06", "batik_popiler_2", "Airbnb", "customer_text", "", "September 2025", "5", "en", "original", "complete_as_observed", "True", [ABS], "1",
     "Lovely shop and we watched a Wayang puppet show nearby with Nunu. Thanks @kadek_bali! -Paige and Alex"),
    ("R07", "batik_popiler_2", "Airbnb", "customer_text", "", "August 2023", "5", "en", "original", "complete_as_observed", "True",
     [ABS, "terminal_ellipsis_possible_truncation_or_style"], "1", "A cooking tour with lunch, no batik at all..."),
    ("R08", "batik_popiler_2", "Airbnb", "customer_text", "", "August 2023", "5", "en", "original", "complete_as_observed", "True", [ABS, TWIN], "2", "Great class!"),
    ("R09", "five_art_studio", "Airbnb", "customer_text", "", "August 2023", "5", "en", "original", "complete_as_observed", "True", [ABS, TWIN], "2", "Great class!"),
    ("R10", "museum_tekstil", "Google Maps", "customer_text", "", "3 weeks ago", "4.0", "", "unknown", "unverified_legacy_default", "False", LEGACY, "1",
     "Nice museum but the batik room was closed."),
    ("R11", "museum_tekstil", "Google Maps", "rating_only", "", "a year ago", "5.0", "", "unknown", "unverified_legacy_default", "False", LEGACY, "0", ""),
    ("R12", "museum_tekstil", "Google Maps", "owner_response", "", "a month ago", "5.0", "", "unknown", "unverified_legacy_default", "False",
     ["known_prior_owner_capture_id", *LEGACY], "1", "Thank you for visiting our museum."),
]
RAW_DIFFERS = {"R01": " SECRET_RAW", "R05": "  "}
LABELS = {  # id: sentiment, confidence, split, excluded_reason
    "R01": ("positif", "tinggi", "train", ""), "R02": ("campuran", "sedang", "train", ""), "R03": ("positif", "tinggi", "val", ""),
    "R05": ("negatif", "tinggi", "test", ""), "R06": ("positif", "tinggi", "train", ""), "R07": ("positif", "tinggi", "excluded", "explicit_cooking_class"),
    "R08": ("positif", "tinggi", "train", ""), "R09": ("positif", "tinggi", "val", ""),
}
LEGACY_LABELS = [("R10", "4.0", "3 weeks ago", "Nice museum but the batik room was closed.", "campuran", "train"),
                 ("L01", "5.0", "Edited 2 years ago", "Beautiful batik collection, worth a visit.", "positif", "val")]
DECISIONS = {  # id: aspect -> (polarity, quotes); the second R06 quote is deliberately absent from its text
    "R01": {"pengajar_layanan": ("positif", ["our teacher Wayan, and Wayan was patient"]), "belajar_praktik": ("positif", ["Later Wayan showed us the wax pen"])},
    "R02": {"pengajar_layanan": ("campuran", ["the communication was slow"]), "belajar_praktik": ("positif", ["Fun workshop"]),
            "fasilitas_kenyamanan": ("negatif", ["the room was a bit hot"])},
    "R03": {"belajar_praktik": ("positif", ["Wonderful batik painting session in Bali"]), "hasil_karya_produk": ("netral", ["great wax work"])},
    "R05": {"pengajar_layanan": ("negatif", ["The communication was poor"]), "belajar_praktik": ("negatif", ["the class ended early"])},
    "R06": {"fasilitas_kenyamanan": ("positif", ["Lovely shop", "Lovely shop with air conditioning"])},
    "R08": {"belajar_praktik": ("positif", ["Great class!"])},
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _csv(path: Path, rows: list[dict], *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def _corpus_row(row: tuple) -> dict:
    rid, venue, source, role, date, relative, rating, locale, variant, completeness, domain, flags, duplicates, text = row
    raw = text + RAW_DIFFERS.get(rid, "")
    return {"record_id": rid, "venue_key": venue, "venue_name": venue.title(), "source": source, "source_review_id": f"SECRET_ID_{rid}",
            "source_url": f"https://secret.example/{rid}", "collected_at": "2026-10-01T16:00:00.000000+00:00", "review_date": date,
            "relative_date": relative, "rating": rating, "source_locale": locale, "raw_text": raw, "clean_text": text, "text_variant": variant,
            "completeness_status": completeness, "record_role": role, "quality_flags": json.dumps(flags), "main_domain_eligible": domain,
            "exact_text_duplicate_count": duplicates, "raw_text_sha256": _sha(raw), "clean_text_sha256": _sha(text)}


def _label_row(row: dict, sentiment: str, confidence: str, split: str, reason: str, locale: str) -> dict:
    return {"index": row["record_id"][1:], "record_id": row["record_id"], "source_review_id": row["source_review_id"], "source": row["source"],
            "venue_key": row["venue_key"], "source_url": row["source_url"], "venue_url": "https://secret.example/venue",
            "source_locale": locale, "review_date": row["review_date"], "relative_date": row["relative_date"], "rating": row["rating"],
            "raw_text": row["raw_text"], "clean_text": row["clean_text"], "sentiment_4class": sentiment, "confidence": confidence,
            "actual_ml_split": split, "excluded_reason": reason, "expert_approved": "false"}


def _business_row(row: dict, decisions: dict) -> dict:
    out = {"record_id": row["record_id"], "source_review_id": row["source_review_id"], "source": row["source"], "venue_key": row["venue_key"],
           "source_locale": row["source_locale"] or "null", "raw_text": row["raw_text"], "clean_text": row["clean_text"], "rating": row["rating"],
           "review_date": row["review_date"] or "null", "relative_date": row["relative_date"] or "null", "source_url": row["source_url"],
           "needs_adjudication": "true" if row["record_id"] == "R02" else "false"}
    full = {}
    for aspect in ASPECTS:
        polarity, quotes = decisions.get(aspect, (None, []))
        out[f"presence_{aspect}"] = "true" if polarity else "false"
        out[f"polarity_{aspect}"] = polarity or "null"
        full[aspect] = {"present": bool(polarity), "polarity": polarity, "evidence_sentence_ids": [], "evidence_quotes": quotes}
    out["aspect_decisions"] = json.dumps(full, ensure_ascii=False)
    return out


def _sqlite(path: Path) -> Path:
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, target_key TEXT, status TEXT, reason TEXT, round_number INTEGER, "
                    "started_at TEXT, finished_at TEXT, metrics TEXT)")
        con.execute("CREATE TABLE checkpoints (run_id TEXT, round_number INTEGER, metrics TEXT)")
        con.execute("CREATE TABLE reviews (review_id TEXT, target_key TEXT, review_text TEXT)")
        con.executemany("INSERT INTO runs VALUES (?, 'museum_tekstil', 'partial', ?, ?, ?, ?, ?)", [
            ("run-a", "feed_stalled", 2, "2026-10-01T13:30:00+00:00", "2026-10-01T13:31:00+00:00",
             json.dumps({"detail": "SECRET_LOG", "inserted": 3, "updated": 1, "duplicates": 40, "rounds": 2})),
            ("run-b", "sort_unavailable", 1, "2026-10-01T13:40:00+00:00", "2026-10-01T13:40:30+00:00",
             json.dumps({"detail": "", "observed_unique": 4, "inserted": 1, "updated": 0, "duplicates": 5, "rounds": 1}))])
        con.executemany("INSERT INTO checkpoints VALUES (?, ?, ?)",  # run-a predates observed_unique; updated re-counts known IDs
                        [("run-a", 1, json.dumps({"inserted": 2, "updated": 0, "duplicates": 10, "detail": "SECRET_LOG"})),
                         ("run-a", 2, json.dumps({"inserted": 3, "updated": 1, "duplicates": 40})),
                         ("run-b", 1, json.dumps({"observed_unique": 4, "inserted": 1, "updated": 0, "duplicates": 5}))])
        con.execute("INSERT INTO reviews VALUES ('SECRET_RID', 'museum_tekstil', 'SECRET_TEXT')")
        con.commit()
    finally:
        con.close()
    return path


def write_fixture(root: Path, *, labels: bool = True, business: bool = True) -> dict[str, Path]:
    exports = root / "exports"
    expansion = exports / "paper_expansion"
    run = expansion / "runs" / RUN_ID
    dataset = exports / "ml_experiments" / "dataset"
    paths = {"workspace": root, "expansion": expansion, "run": run, "labels_real": dataset / "real_labeled1524.csv",
             "labels_legacy": dataset / "legacy_supplement81.csv", "business": exports / "business_research/dataset/business_real.csv"}
    corpus = [_corpus_row(row) for row in CORPUS]
    _csv(run / "corpus.csv", corpus, bom=True)
    paths["snapshot"] = _sqlite(run / "raw_snapshot.sqlite3")
    _json(expansion / "latest.json", {"run_id": RUN_ID, "audit": str(root / "SECRET_elsewhere" / "audit.json")})
    _json(run / "audit.json", {
        "run_id": RUN_ID, "created_at": "2026-10-01T17:00:00+00:00", "run_directory": "C:/SECRET/run",
        "snapshot": {"path": "C:/SECRET/raw_snapshot.sqlite3", "sha256": hashlib.sha256(paths["snapshot"].read_bytes()).hexdigest()},
        "counts": {"stored_unique_source_identities": 12, "customer_text": 10, "main_domain_customer_texts": 9,
                   "paper_core_observed_complete_texts": 8, "rating_only": 1, "owner_response": 1},
        "collector_targets": [
            {"target_key": "museum_tekstil", "name": "Museum Tekstil Jakarta", "url": "https://secret.example/maps", "reported_total": 100,
             "inserted_total": 3, "updated_total": 0, "duplicates_total": 40},
            {"target_key": "giriloyo", "name": "Batik Giriloyo", "url": "https://secret.example/maps2", "reported_total": None,
             "inserted_total": 0, "updated_total": 0, "duplicates_total": 0}],
        "latest_collector_runs": {
            "museum_tekstil": {"status": "partial", "reason": "feed_stalled", "metrics": {"detail": "SECRET_LOG"}},
            "giriloyo": {"status": "partial", "reason": "sort_unavailable", "metrics": {"detail": "SECRET_LOG"}}}})
    _json(expansion / "runs" / OLD_RUN_ID / "audit.json", {"created_at": "2026-10-01T16:00:00+00:00", "counts": {"stored_unique_source_identities": 9}})
    _json(expansion / "airbnb" / "x-report.json", {
        "source_url": "https://secret.example/airbnb", "venue_key": "batik_popiler_2", "customer_texts": 5,
        "source_totals_metadata": {"ratingCount": "6", "writtenReviewCount": "5", "ratingAverage": 4.9},
        "stop_reason": "reported_written_review_count_reached", "started_at": "2026-10-01T16:00:00+00:00", "finished_at": "2026-10-01T16:01:00+00:00"})
    _json(expansion / "airbnb" / "five-report.json", {  # every (platform, venue) in the corpus has its report
        "source_url": "https://secret.example/airbnb-five", "venue_key": "five_art_studio", "customer_texts": 4,
        "source_totals_metadata": {"ratingCount": "4", "writtenReviewCount": "4", "ratingAverage": 5.0},
        "stop_reason": "reported_written_review_count_reached", "started_at": "2026-10-01T15:00:00+00:00", "finished_at": "2026-10-01T15:00:30+00:00"})
    _json(expansion / "seeksophie_initial" / "collection_report.json", {
        "source_url": "https://secret.example/seeksophie", "venue_key": "five_art_studio", "actual_unique_reviews_collected": 20,
        "option_reviews_total_metadata_only": 89, "stop_reason": "initial_public_payload_exhausted"})
    by_id = {row["record_id"]: row for row in corpus}
    if labels:
        _csv(paths["labels_real"], [_label_row(by_id[rid], *value, locale=by_id[rid]["source_locale"] or "null") for rid, value in LABELS.items()])
        legacy = []
        for rid, rating, relative, text, sentiment, split in LEGACY_LABELS:
            base = by_id.get(rid) or {**by_id["R10"], "record_id": rid, "source_review_id": f"SECRET_ID_{rid}", "raw_text": text}
            legacy.append(_label_row({**base, "rating": rating, "relative_date": relative, "clean_text": text}, sentiment, "sedang", split, "", "null"))
        _csv(paths["labels_legacy"], legacy)
    if business:
        _csv(paths["business"], [_business_row(by_id[rid], decisions) for rid, decisions in DECISIONS.items()])
    return paths


def patch_paths(monkeypatch, paths: dict[str, Path]) -> None:
    from temajelajah import bi_data
    monkeypatch.setattr(bi_data, "WORKSPACE", paths["workspace"])
    monkeypatch.setattr(bi_data, "EXPANSION_ROOT", paths["expansion"])
    monkeypatch.setattr(bi_data, "LABELS_REAL", paths["labels_real"])
    monkeypatch.setattr(bi_data, "LABELS_LEGACY", paths["labels_legacy"])
    monkeypatch.setattr(bi_data, "BUSINESS_CSV", paths["business"])
