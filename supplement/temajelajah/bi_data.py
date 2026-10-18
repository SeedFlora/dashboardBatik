"""Read-only data layer of the BI dashboard: privacy-safe loading, filters, honest aggregates and Indonesian formatting."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
EXPANSION_ROOT = WORKSPACE / "exports/paper_expansion"
LABELS_REAL = WORKSPACE / "exports/ml_experiments/dataset/real_labeled1524.csv"
LABELS_LEGACY = WORKSPACE / "exports/ml_experiments/dataset/legacy_supplement81.csv"
BUSINESS_CSV = WORKSPACE / "exports/business_research/dataset/business_real.csv"
UNKNOWN_PROVENANCE = "unknown"
PROVENANCE_COLS = ("annotation_session", "annotation_rubric", "annotation_prompt", "annotation_date", "annotation_model",
                   "annotation_verification", "presence_verification", "polarity_verification", "annotation_method",
                   "annotation_source_sha256", "annotation_schema", "annotation_protocol_sha256")
ANNOTATION_OPTIONAL_COLS = ("annotation_method", "backend_model_id", "annotation_component_file", "annotation_component_sha256",
                            "csv_source_jsonl", "csv_source_jsonl_sha256", "annotation_origin", "annotation_source_sha256",
                            "annotation_session", "annotation_rubric", "annotation_prompt", "annotation_date", "expert_approved")


class Venue(NamedTuple):
    name: str
    region: str = ""
    kind: str = ""


VENUES = {"five_art_studio": Venue("Five Arts", "Keliki, Gianyar, Bali", "kelas"),
          "widya_batik_class": Venue("Widya", "Ubud, Bali", "kelas"),
          "batik_popiler_2": Venue("Popiler 2", "Denpasar Timur, Bali", "kelas"),
          "batik_kotagede_nunung": Venue("Nunung (Kotagede)", "Yogyakarta", "kelas"),
          "lombok_batik_alwan": Venue("Lombok Alwan", "Pringgasela, Lombok", "kelas"),
          "museum_tekstil": Venue("Museum Tekstil", "Jakarta Barat", "museum umum"),
          "giriloyo": Venue("Giriloyo"), "seno": Venue("Seno")}
SOURCES = ("Airbnb", "Google Maps", "SeekSophie")
REPORT_PLATFORMS = ("Airbnb", "SeekSophie", "Google Maps")  # platforms whose scraping reports give a reported total
PLATFORM_SHORT = {"Google Maps": "Google"}
CUSTOMER_ROLES = ("customer_text", "rating_only")
ASPECTS = ("pengajar_layanan", "belajar_praktik", "hasil_karya_produk", "fasilitas_kenyamanan")
ASPECT_NAMES = {"pengajar_layanan": "Pengajar dan layanan", "belajar_praktik": "Pembelajaran dan praktik",
                "hasil_karya_produk": "Hasil karya", "fasilitas_kenyamanan": "Fasilitas dan kenyamanan"}
ASPECT_SHORT = {aspect: name.split(" dan ")[0] for aspect, name in ASPECT_NAMES.items()}
SENTIMENTS = {"positif": "+", "netral": "○", "campuran": "±", "negatif": "−"}
COMPLAINT = ("campuran", "negatif")
CORPUS_COLS = ("record_id", "venue_key", "source", "record_role", "collected_at", "review_date", "relative_date", "rating", "source_locale",
               "text_variant", "completeness_status", "main_domain_eligible", "quality_flags", "exact_text_duplicate_count",
               "raw_text_sha256", "clean_text_sha256", "clean_text")
LABEL_COLS = ("record_id", "venue_key", "source", "rating", "review_date", "relative_date", "source_locale", "clean_text",
              "sentiment_4class", "confidence", "actual_ml_split", "excluded_reason", "expert_approved")
BUSINESS_COLS = ("record_id", "venue_key", "source", "rating", "review_date", "relative_date", "source_locale", "clean_text",
                 *(f"presence_{aspect}" for aspect in ASPECTS), *(f"polarity_{aspect}" for aspect in ASPECTS),
                 "aspect_decisions", "needs_adjudication")
FORBIDDEN_COLS = frozenset({"source_review_id", "source_url", "venue_url", "raw_text", "review_url", "review_id", "review_text", "detail"})
REASONS = {"reported_written_review_count_reached": "jumlah ulasan tertulis tercapai",
           "initial_public_payload_exhausted": "muatan awal halaman publik habis", "reported_total_observed": "total laporan teramati",
           "sort_unavailable": "urutan terbaru tak tersedia", "feed_stalled": "feed macet",
           "reviews_panel_unavailable": "panel ulasan tak tersedia", "collector_error": "galat kolektor",
           "feed_end_count_mismatch": "feed habis, jumlah tak cocok", "feed_end_total_unknown": "feed habis, total tak diketahui",
           "review_budget": "batas ulasan tercapai", "round_budget": "batas putaran tercapai", "time_budget": "batas waktu tercapai",
           "login_required": "perlu login", "consent_required": "perlu persetujuan", "captcha": "captcha",
           "offsite_redirect": "dialihkan ke luar situs", "parse_error": "galat parsing", "cancelled": "dibatalkan",
           "not_started": "belum dimulai"}
QUALITY_CHECKS = (("rating_only", "Rating saja"), ("owner_response", "Balasan pemilik"), ("duplicate_text", "Teks identik lintas ID"),
                  ("truncated", "Kemungkinan terpotong"), ("no_locale", "Bahasa tanpa metadata"),
                  ("variant_unknown", "Varian bahasa tak diketahui"), ("relative_date", "Tanggal relatif saja"),
                  ("unverified", "Kelengkapan belum terverifikasi"), ("short_text", "Teks ≤3 kata"), ("normalized", "Teks dinormalisasi"))
STATUS = {"lengkap": "●", "parsial": "◐", "gagal": "○", "tak diketahui": "?"}
RUN_STATUS = {"partial": "parsial", "completed": "lengkap", "failed": "gagal", "cancelled": "dibatalkan", "running": "berjalan"}
REPORT_KEYS = {"Airbnb": ("venue_key", "started_at", "finished_at", "customer_texts", "stop_reason"),  # allowlist: no URLs
               "SeekSophie": ("venue_key", "collected_at", "actual_unique_reviews_collected", "option_reviews_total_metadata_only",
                              "stop_reason", "duplicate_objects")}
TELEMETRY_COLS = ["platform", "target", "venue_key", "mulai", "durasi", "putaran", "baru", "duplikat", "hasil", "alasan"]
METRICS = ("Penyebutan", "Keluhan")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
AUDIT_STAGES = {1: "stored_unique_source_identities", 2: "customer_text", 3: "main_domain_customer_texts",
                4: "paper_core_observed_complete_texts"}
POLARITY_ORDER = {"negatif": 0, "campuran": 1, "netral": 2, "positif": 3}
RUN_METRICS = ("rounds", "observed_unique", "inserted", "updated", "duplicates", "invalid", "missing_ids", "elapsed_seconds")
MIN_WINDOW, MIN_COMPLAINTS, SMALL_N, MIN_ASPECT_N, MIN_OUTLIER_N, QUOTE_CHARS = 30, 10, 30, 100, 50, 240
STARS_4_5 = "★4–\u20605"  # U+2060 word joiner: "★4–5" never breaks after the dash
RUN_ID_RE = re.compile(r"[\w.-]+")
MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")
CJK_RUN = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")  # Han and kana: words are not space-separated
CJK_CHARS_PER_WORD = 2


def _missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}".replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_int(n: Any) -> str:
    return "—" if _missing(n) else _number(round(float(n)), 0)


def fmt_pct(x: Any, digits: int = 1) -> str:
    if _missing(x):
        return "—"
    return _number(float(x) * 100, 0 if float(x) in (0.0, 1.0) else digits) + "%"


def fmt_dec(x: Any, digits: int = 2) -> str:
    return "—" if _missing(x) else _number(float(x), digits)


def fmt_ci(lo: Any, hi: Any) -> str:
    if _missing(lo) or _missing(hi):
        return "—"
    return "–".join(_number(float(v) * 100, 0 if float(v) in (0.0, 1.0) else 1) for v in (lo, hi)) + "%"


def fmt_month(month: Any) -> str:
    return "—" if _missing(month) else f"{MONTHS[month.month - 1]} {month.year}"


def fmt_change(x: Any, minus: str = "-") -> str:
    """Signed whole percent. st.metric needs the ASCII minus to read the direction; running text passes minus="−"."""
    return "—" if _missing(x) else f"{float(x) * 100:+.0f}%".replace("-", minus)


def fmt_signed(n: Any) -> str:
    """Signed whole number with an ASCII minus, so st.metric reads the direction: +386, -8, 0."""
    if _missing(n):
        return "—"
    value = round(float(n))
    return ("+" if value > 0 else "-" if value < 0 else "") + fmt_int(abs(value))


def fmt_share(x: Any) -> str:
    """Whole-percent share that never rounds a non-zero count to 0% (or a partial row to 100%)."""
    if _missing(x):
        return "—"
    x = float(x)
    return "<1%" if 0 < x < 0.005 else ">99%" if 0.995 <= x < 1 else fmt_pct(x, 0)


def _pct_short(x: float) -> str:
    return fmt_pct(x, 1 if x < 0.1 else 0)


def md_escape(text: str) -> str:
    return MD_SPECIAL.sub(r"\\\1", text)


def venue_name(key: str) -> str:
    return VENUES[key].name if key in VENUES else str(key)


def venue_short(key: str) -> str:
    return venue_name(key).split(" (")[0]


def venue_rank(key: str) -> int:
    return list(VENUES).index(key) if key in VENUES else len(VENUES)


def word_count(text: Any) -> int:
    """Whitespace words; a run of Han or kana (written without spaces) counts one word per two characters. Hangul is spaced."""
    if not isinstance(text, str):
        return 0
    if not CJK_RUN.search(text):
        return len(text.split())
    rest = [token for token in CJK_RUN.sub(" ", text).split() if any(ch.isalnum() for ch in token)]
    return len(rest) + sum(math.ceil(len(run) / CJK_CHARS_PER_WORD) for run in CJK_RUN.findall(text))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p, denominator = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def to_float(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def to_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold().eq("true").astype(bool)


def parse_month(review_date: pd.Series, relative_date: pd.Series) -> pd.Series:
    local_day = review_date.astype(str).str[:10]
    iso = pd.to_datetime(local_day.where(local_day.str.fullmatch(r"\d{4}-\d{2}-\d{2}")), format="%Y-%m-%d", errors="coerce")
    month_year = pd.to_datetime(relative_date.astype(str).str.strip(), format="%B %Y", errors="coerce")
    return iso.fillna(month_year).dt.to_period("M").dt.to_timestamp()


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"{path.name} terlalu besar")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} harus berupa objek JSON")
    return value


def source_paths() -> dict[str, Path] | None:
    latest = EXPANSION_ROOT / "latest.json"
    if not latest.is_file():
        return None
    run_id, runs = _read_json(latest).get("run_id"), EXPANSION_ROOT / "runs"
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id) or (runs / run_id).resolve().parent != runs.resolve():
        raise ValueError("run_id snapshot tidak valid")
    run = runs / run_id
    if not (run / "corpus.csv").is_file():
        return None
    optional = {"audit": run / "audit.json", "raw_snapshot": run / "raw_snapshot.sqlite3", "labels_real": LABELS_REAL,
                "labels_legacy": LABELS_LEGACY, "business": BUSINESS_CSV,
                "labels_manifest": LABELS_REAL.parent / "manifest.json", "labels_report": LABELS_REAL.parent / "annotation_report.json",
                "business_manifest": BUSINESS_CSV.parent / "manifest.json",
                "business_report": BUSINESS_CSV.parent.parent / "audits/annotation_report.json",
                "seeksophie": EXPANSION_ROOT / "seeksophie_initial/collection_report.json"}
    paths = {"latest": latest, "corpus": run / "corpus.csv", **{name: path for name, path in optional.items() if path.is_file()}}
    paths.update({f"airbnb:{path.stem}": path for path in sorted((EXPANSION_ROOT / "airbnb").glob("*-report.json"))})
    paths.update({f"snapshot:{path.parent.name}": path for path in sorted(runs.glob("*/audit.json"))})
    if "labels_report" in paths:
        for item in _read_json(paths["labels_report"]).get("annotation_inputs", []):
            path = WORKSPACE / str(item.get("path", ""))
            if path.is_relative_to(WORKSPACE) and path.is_file():
                paths[f"label_batch:{path.stem}"] = path
    return paths


def signature(paths: dict[str, Path]) -> tuple:
    stats = {name: path.stat() for name, path in paths.items()}
    return tuple((name, str(paths[name]), stats[name].st_mtime_ns, stats[name].st_size) for name in sorted(paths))


def read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    wanted = list(columns)
    leaked = FORBIDDEN_COLS.intersection(wanted)
    if leaked:
        raise ValueError(f"Kolom terlarang tidak boleh dimuat: {', '.join(sorted(leaked))}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=wanted)[wanted]


def _annotation_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Load only safe, requested fields; historical fixtures may not contain provenance columns."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    return read_table(path, [*columns, *(name for name in ANNOTATION_OPTIONAL_COLS if name in header and name not in columns)])


def _recorded(value: Any) -> str:
    return UNKNOWN_PROVENANCE if _missing(value) or str(value).strip().lower() in ("", "null", "none") else str(value).strip()


def _provenance(frame: pd.DataFrame, *, kind: str, cohort: str, manifest: dict | None = None,
                batches: Mapping[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    manifest = manifest or {}
    for name in ANNOTATION_OPTIONAL_COLS:
        if name not in frame:
            frame[name] = ""
    def session(row: pd.Series) -> str:
        recorded = _recorded(row["annotation_session"])
        if recorded != UNKNOWN_PROVENANCE:
            return recorded
        if batches and row["record_id"] in batches:
            return "batch:" + batches[row["record_id"]][0]
        component = _recorded(row["annotation_component_file"])
        if component != UNKNOWN_PROVENANCE:
            return "batch:" + Path(component.replace("\\", "/")).name
        origin = _recorded(row["annotation_origin"])
        return "origin:" + origin if origin != UNKNOWN_PROVENANCE else f"unknown ({cohort})"
    frame["annotation_session"] = frame.apply(session, axis=1) if len(frame) else pd.Series(index=frame.index, dtype=str)
    frame["annotation_rubric"] = frame["annotation_rubric"].map(_recorded)
    frame["annotation_schema"] = _recorded(manifest.get("schema_version"))
    frame["annotation_protocol_sha256"] = _recorded(manifest.get("protocol_sha256"))
    for name in ("annotation_prompt", "annotation_date"):
        frame[name] = frame[name].map(_recorded)
    frame["annotation_model"] = frame["backend_model_id"].map(_recorded)  # requested aliases do not identify the backend
    frame["annotation_method"] = frame["annotation_method"].map(_recorded)
    approved = to_bool(frame["expert_approved"])
    frame["annotation_verification"] = np.where(approved, "expert_approved", "unverified")
    frame["presence_verification"] = np.where(approved, "expert_approved", "author_reported_single_author_check" if kind == "aspect" else "not_applicable")
    frame["polarity_verification"] = np.where(approved, "expert_approved", "unverified")
    hashes = []
    for _, row in frame.iterrows():
        value = batches[row["record_id"]][1] if batches and row["record_id"] in batches else row["annotation_component_sha256"]
        value = _recorded(value)
        if value == UNKNOWN_PROVENANCE:
            value = _recorded(row["csv_source_jsonl_sha256"])
        hashes.append(_recorded(row["annotation_source_sha256"]) if value == UNKNOWN_PROVENANCE else value)
    frame["annotation_source_sha256"] = hashes
    return frame


def provenance_table(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Metadata-only export. Batch file IDs are provenance proxies, not verified annotation sessions."""
    if frame is None or frame.empty or "annotation_session" not in frame:
        return pd.DataFrame(columns=[*PROVENANCE_COLS, "rows", "venues"])
    columns = [name for name in PROVENANCE_COLS if name in frame]
    grouped = frame.groupby(columns, dropna=False, sort=True)
    return grouped.agg(rows=("record_id", "size"), venues=("venue_key", lambda values: ", ".join(sorted(set(values))))).reset_index()


def session_comparison_blocker(frame: pd.DataFrame | None, *, allow_unknown_descriptive: bool = False) -> str | None:
    if frame is not None and (("annotation_session" in frame and frame["annotation_session"].nunique() > 1) or
                              ("annotation_session" not in frame and "cohort" in frame and frame["cohort"].nunique() > 1)):
        return "Perbandingan lintas batch anotasi tidak didukung; pilih satu batch. ID batch adalah proksi provenance sesi."
    if frame is not None and not allow_unknown_descriptive and "venue_key" in frame and frame["venue_key"].nunique() > 1:
        sessions = frame["annotation_session"].astype(str) if "annotation_session" in frame else pd.Series([UNKNOWN_PROVENANCE])
        if sessions.str.startswith(("unknown", "origin:")).any():
            return "Provenance batch tidak diketahui; kontras lintas venue ditahan sampai asal batch dapat diperiksa."
    return None


def _dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame["rating_num"] = to_float(frame["rating"])
    frame["month"] = parse_month(frame["review_date"], frame["relative_date"])
    frame["year"] = frame["month"].dt.year.astype("Int64")
    frame["dated"] = frame["month"].notna()
    frame["word_count"] = frame["clean_text"].map(word_count).astype(int)
    frame["source_locale"] = frame["source_locale"].replace("null", "")
    return frame


def _flags(text: str) -> list[str]:
    value = json.loads(text) if text.strip() else []
    return [str(flag) for flag in value] if isinstance(value, list) else []


def _labels(paths: dict[str, Path]) -> pd.DataFrame | None:
    batches = {}
    for name, path in paths.items():
        if name.startswith("label_batch:"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                row = json.loads(line)
                if isinstance(row, dict) and row.get("record_id"):
                    batches[str(row["record_id"])] = (path.name, digest)
    manifest = _read_json(paths["labels_manifest"]) if "labels_manifest" in paths else {}
    parts = [_provenance(_annotation_table(paths[name], LABEL_COLS).assign(cohort=cohort), kind="sentiment", cohort=cohort,
                         manifest=manifest if cohort == "real1524" else {}, batches=batches if cohort == "real1524" else None)
             for name, cohort in (("labels_real", "real1524"), ("labels_legacy", "legacy81")) if name in paths]
    if not parts:
        return None
    labels = _dates(pd.concat(parts, ignore_index=True))
    labels["complaint"] = labels["sentiment_4class"].isin(COMPLAINT)
    labels["expert_approved"] = to_bool(labels["expert_approved"])
    labels["excluded_reason"] = labels["excluded_reason"].replace("null", "")
    return labels


def _business(frame: pd.DataFrame) -> pd.DataFrame:
    for aspect in ASPECTS:
        frame[f"presence_{aspect}"] = to_bool(frame[f"presence_{aspect}"])
        frame[f"polarity_{aspect}"] = frame[f"polarity_{aspect}"].replace({"null": pd.NA, "": pd.NA})
    frame["needs_adjudication"] = to_bool(frame["needs_adjudication"])
    return _dates(frame)


def explode_evidence(business: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows, missing = [], 0
    for record_id, venue, source, year, text, raw in business[["record_id", "venue_key", "source", "year", "clean_text",
                                                                "aspect_decisions"]].itertuples(index=False):
        decisions = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        for aspect in ASPECTS:
            decision = (decisions.get(aspect) if isinstance(decisions, dict) else None) or {}
            for quote in decision.get("evidence_quotes") or []:
                start = text.find(quote) if isinstance(quote, str) and quote else -1
                if start < 0:
                    missing += 1
                    continue
                rows.append((record_id, venue, source, year, aspect, decision.get("polarity"), quote, start, start + len(quote)))
    evidence = pd.DataFrame(rows, columns=["record_id", "venue_key", "source", "year", "aspect", "polarity", "quote", "start", "end"])
    evidence["year"] = evidence["year"].astype("Int64")
    return evidence, missing


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _utc(value: Any) -> pd.Timestamp | None:
    stamp = pd.to_datetime(value, utc=True, errors="coerce") if isinstance(value, str) and value.strip() else None
    return None if stamp is None or pd.isna(stamp) else stamp


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _snapshots(paths: dict[str, Path], audit: dict) -> list[dict]:
    """Stored identities per audit, ending at the run on screen: later runs and undated audits are left out, and without the
    run's own audit there is no series at all (the card then says the audit is missing)."""
    anchor, current = _utc(audit.get("created_at")), _dict(audit.get("counts")).get("stored_unique_source_identities")
    if "audit" not in paths or anchor is None or not _is_count(current):
        return []
    own, rows = f"snapshot:{paths['audit'].parent.name}", []
    for name, path in paths.items():
        if name.startswith("snapshot:") and name != own:
            older = _read_json(path)
            stored, created = _dict(older.get("counts")).get("stored_unique_source_identities"), _utc(older.get("created_at"))
            if _is_count(stored) and created is not None and created < anchor:
                rows.append({"created_at": created, "n": stored})
    return [*rows, {"created_at": anchor, "n": current}]


def _report_gaps(reviews: pd.DataFrame, coverage: pd.DataFrame | None, airbnb: list, seeksophie: dict | None,
                 targets: list | None) -> list[tuple[str, str | None]]:
    """Scraping reports the corpus needs but the exports lack: a whole platform (venue None) or one of its targets, such as a
    single missing Airbnb report, so a partial sum never passes for the whole collection."""
    gaps = [(platform, None) for platform, found in (("Airbnb", airbnb), ("SeekSophie", seeksophie), ("Google Maps", targets is not None))
            if not found]
    whole = {platform for platform, _ in gaps}
    covered = set(zip(coverage["source"], coverage["venue_key"])) if coverage is not None else set()
    used = set(zip(reviews["source"], reviews["venue_key"]))
    loose = [(platform, venue) for platform, venue in used - covered if platform in REPORT_PLATFORMS and platform not in whole]
    return gaps + sorted(loose, key=lambda gap: (REPORT_PLATFORMS.index(gap[0]), venue_rank(gap[1]), gap[1]))


def missing_reports(gaps: Sequence[tuple[str, str | None]], scoped: pd.DataFrame) -> list[str]:
    """The report gaps inside the current venue/source scope, named for the page: "Airbnb" or "Airbnb · Widya"."""
    sources, pairs = set(scoped["source"]), set(zip(scoped["source"], scoped["venue_key"]))
    return [platform if venue is None else f"{platform} · {venue_name(venue)}" for platform, venue in gaps
            if (platform, venue) in pairs or (venue is None and platform in sources)]


def load_bundle(paths: dict[str, Path]) -> dict:
    reviews = _dates(read_table(paths["corpus"], CORPUS_COLS))
    reviews["flags"] = reviews["quality_flags"].map(_flags)
    reviews["main_domain_eligible"] = to_bool(reviews["main_domain_eligible"])
    reviews["exact_text_duplicate_count"] = to_float(reviews["exact_text_duplicate_count"]).fillna(0).astype(int)
    labels = _labels(paths)
    manifest = _read_json(paths["business_manifest"]) if "business_manifest" in paths else {}
    if "business_report" in paths:
        manifest = {**manifest, "protocol_sha256": _read_json(paths["business_report"]).get("protocol_sha256")}
    business = (_provenance(_business(_annotation_table(paths["business"], BUSINESS_COLS)), kind="aspect", cohort="business1515",
                            manifest=manifest) if "business" in paths else None)
    evidence, missing = explode_evidence(business) if business is not None else (None, 0)
    if evidence is not None:
        evidence = evidence.merge(business[["record_id", *PROVENANCE_COLS]], how="left", on="record_id", validate="many_to_one")
    audit = _read_json(paths["audit"]) if "audit" in paths else {}
    airbnb = [_read_json(path) for name, path in paths.items() if name.startswith("airbnb:")]
    seeksophie = _read_json(paths["seeksophie"]) if "seeksophie" in paths else None
    targets = audit.get("collector_targets") if isinstance(audit.get("collector_targets"), list) else None
    runs = _dict(audit.get("latest_collector_runs"))
    coverage = coverage_rows(airbnb, seeksophie, targets or [], runs) if airbnb or seeksophie or targets else None
    gaps = _report_gaps(reviews, coverage, airbnb, seeksophie, targets)
    expected = _dict(audit.get("snapshot")).get("sha256")
    expected = expected.lower() if isinstance(expected, str) and expected.strip() else None
    snapshot_ok = (hashlib.sha256(paths["raw_snapshot"].read_bytes()).hexdigest() == expected
                   if "raw_snapshot" in paths and expected else None)
    snapshots = _snapshots(paths, audit)
    reports = [{"platform": "Airbnb", **{key: report.get(key) for key in REPORT_KEYS["Airbnb"]},
                "reported": _dict(report.get("source_totals_metadata")).get("writtenReviewCount")} for report in airbnb]
    reports += [{"platform": "SeekSophie", **{key: seeksophie.get(key) for key in REPORT_KEYS["SeekSophie"]}}] if seeksophie else []
    counts = {"corpus": len(reviews), "business": None if business is None else len(business)}
    if labels is not None:
        counts |= {"labels_real": int(labels["cohort"].eq("real1524").sum()), "labels_legacy": int(labels["cohort"].eq("legacy81").sum())}
    files, seen = [], set()
    for name, path in paths.items():
        shown = path.relative_to(EXPANSION_ROOT.parent).as_posix() if path.is_relative_to(EXPANSION_ROOT.parent) else path.name
        if shown not in seen:
            seen.add(shown)
            files.append({"file": shown, "rows": counts.get(name), "modified": pd.Timestamp(path.stat().st_mtime_ns, unit="ns", tz="UTC")})
    collected = pd.to_datetime(reviews["collected_at"], utc=True, errors="coerce", format="ISO8601").max()
    return {"reviews": reviews, "labels": labels, "business": business, "evidence": evidence, "evidence_missing": missing,
            "coverage": coverage, "coverage_missing": gaps, "reports": reports,
            "snapshots": pd.DataFrame(snapshots, columns=["created_at", "n"]).sort_values("created_at", ignore_index=True),
            "audit_counts": {key: value for key, value in _dict(audit.get("counts")).items() if isinstance(value, int)},
            "snapshot_ok": snapshot_ok, "snapshot_missing": bool(expected) and "raw_snapshot" not in paths,
            "files": pd.DataFrame(files, columns=["file", "rows", "modified"]).astype({"rows": "Int64"}), "collected_at": collected}


@dataclass(frozen=True)
class Filters:
    venues: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    years: tuple[int, int] | None = None
    include_undated: bool = True
    sessions: tuple[str, ...] = ()
    rubrics: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    verifications: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "venues", tuple(self.venues or ()))
        object.__setattr__(self, "sources", tuple(self.sources or ()))
        object.__setattr__(self, "years", tuple(int(year) for year in self.years) if self.years else None)
        object.__setattr__(self, "include_undated", bool(self.include_undated))
        for name in ("sessions", "rubrics", "prompts", "dates", "models", "verifications"):
            object.__setattr__(self, name, tuple(getattr(self, name) or ()))

    def key(self) -> tuple:
        return (self.venues, self.sources, self.years, self.include_undated, self.sessions, self.rubrics, self.prompts, self.dates,
                self.models, self.verifications)


def apply_filters(frame: pd.DataFrame, f: Filters, *, use_years: bool = True) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    if f.venues:
        keep &= frame["venue_key"].isin(f.venues)
    if f.sources:
        keep &= frame["source"].isin(f.sources)
    for field, values in (("annotation_session", f.sessions), ("annotation_rubric", f.rubrics), ("annotation_prompt", f.prompts),
                          ("annotation_date", f.dates), ("annotation_model", f.models), ("annotation_verification", f.verifications)):
        if values and field in frame:
            keep &= frame[field].isin(values)
    if use_years:
        year = frame["year"]
        in_range = year.notna() if f.years is None else year.between(*f.years).fillna(False).astype(bool)
        keep &= in_range | (year.isna() & f.include_undated)
    return frame[keep.astype(bool)]


def _month_start(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if _missing(value) else pd.Timestamp(value)
    stamp = stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp
    return pd.Timestamp(stamp.year, stamp.month, 1)


def _dated_counts(reviews: pd.DataFrame, stop: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    rows = reviews[reviews["record_role"].isin(CUSTOMER_ROLES)]
    dated = rows[rows["month"].notna() & (rows["month"] < stop)]
    return dated, dated.groupby("month").size()


def partial_month(reviews: pd.DataFrame, collected_at: Any) -> pd.Timestamp | None:
    """Flag a low-volume tail (<half the preceding 12-month median); this does not prove collection incompleteness.
    Decide on the whole corpus so every filtered slice uses the same descriptive sensitivity rule."""
    stop = _month_start(collected_at)
    last = stop - pd.DateOffset(months=1)
    _, counts = _dated_counts(reviews, stop)
    if counts.empty or counts.index.min() > last:
        return None
    n = counts.reindex(pd.date_range(counts.index.min(), last, freq="MS"), fill_value=0)  # a trailing empty month still counts
    prior = n.iloc[-13:-1]
    return last if len(prior) >= 6 and n.iloc[-1] < 0.5 * prior.median() else None


def monthly_series(reviews: pd.DataFrame, collected_at: pd.Timestamp, *, basis: pd.DataFrame | None = None,
                   years: tuple[int, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Calendar-complete trailing windows and a low-volume-tail sensitivity. `basis` fixes the heuristic for every slice;
    `years` withholds either comparison when the selection cuts its window. The heuristic is not a completeness test."""
    stop = _month_start(collected_at)
    dated, counts = _dated_counts(reviews, stop)
    rows = reviews[reviews["record_role"].isin(CUSTOMER_ROLES)]
    last = stop - pd.DateOffset(months=1)
    partial = partial_month(reviews if basis is None else basis, collected_at)
    windows = {"dated": len(dated), "undated": int(rows["month"].isna().sum()), "partial": partial}
    frame = pd.DataFrame(columns=["bulan", "n", "rata12", "rating", "tahun", "parsial"])
    if len(counts):
        months = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
        n = counts.reindex(months, fill_value=0)
        frame = pd.DataFrame({"bulan": months, "n": n.to_numpy(), "rata12": n.rolling(12, min_periods=6).mean().to_numpy(),
                              "rating": dated.groupby("month")["rating_num"].mean().reindex(months).to_numpy(),
                              "tahun": months.year, "parsial": months == partial if partial is not None else False})
    def comparison(end: pd.Timestamp) -> dict:
        start = end - pd.DateOffset(months=23)
        span = counts.reindex(pd.date_range(start, end, freq="MS"), fill_value=0)
        current, previous = int(span.iloc[12:].sum()), int(span.iloc[:12].sum())
        cut = years is not None and (int(years[0]) > start.year or int(years[1]) < end.year)
        return {"end": end, "current": current, "previous": previous, "change": current / previous - 1 if previous else None,
                "cut": cut, "ok": not cut and current >= MIN_WINDOW and previous >= MIN_WINDOW,
                "span_label": f"{fmt_month(start)}–{fmt_month(end)}",
                "current_label": f"{fmt_month(end - pd.DateOffset(months=11))}–{fmt_month(end)}",
                "previous_label": f"{fmt_month(start)}–{fmt_month(end - pd.DateOffset(months=12))}"}
    _, base_counts = _dated_counts(reviews if basis is None else basis, stop)
    tail_prior = base_counts.reindex(pd.date_range(last - pd.DateOffset(months=12), last - pd.DateOffset(months=1), freq="MS"), fill_value=0)
    median = float(tail_prior.median())
    windows |= {**comparison(last), "primary_basis": "calendar_complete_months",
                "sensitivity": comparison(last - pd.DateOffset(months=1)) if partial is not None else None,
                "sensitivity_reason": "low_volume_tail_heuristic" if partial is not None else None,
                "tail_n": int(base_counts.get(last, 0)), "tail_prior_median": median, "tail_threshold": 0.5 * median}
    return frame, windows


def legacy_note(labels: pd.DataFrame | None) -> str:
    """Caveat for shares that pool the older labelling round (81 Museum Tekstil labels from Google Maps) with the current one."""
    legacy = labels[labels["cohort"].eq("legacy81")] if labels is not None and "cohort" in labels else None
    if legacy is None or legacy.empty:
        return ""
    venues = sorted(set(legacy["venue_key"]), key=lambda key: (venue_rank(key), key))
    return f"termasuk {fmt_int(len(legacy))} label lama {', '.join(venue_name(key) for key in venues)}"


def kpis(reviews: pd.DataFrame, labels: pd.DataFrame | None, coverage: pd.DataFrame | None, windows: dict,
         missing: Sequence[str] = ()) -> dict[str, dict | None]:
    """K1-K5 cards. `missing` names the scraping reports (a platform, or one target) absent for the current venue/source scope."""
    rows = reviews[reviews["record_role"].isin(CUSTOMER_ROLES)]
    years, end = rows["year"].dropna().astype(int), (windows or {}).get("end")
    if end is not None:  # only complete years on the sparkline: the running year would read as a collapse
        years = years[years < (end.year + 1 if end.month == 12 else end.year)]
    yearly = years.value_counts().reindex(range(years.min(), years.max() + 1), fill_value=0) if len(years) else pd.Series(dtype=int)
    undated = int(rows["month"].isna().sum())
    notes = [f"{fmt_int(undated)} ulasan tanpa bulan pasti ikut dihitung di sini tetapi tidak masuk tren" if undated
             else "Semua ulasan di sini bertanggal pasti"]
    span = f"{years.min()}–{years.max()}" if len(years) and years.min() < years.max() else str(years.min()) if len(years) else ""
    notes += [f"grafik: ulasan per tahun {span}, tahun berjalan tidak ditampilkan"] if span else []
    if windows and windows.get("cut"):
        notes.append(f"perbandingan 12 bulan disembunyikan: filter tahun memotong jendela {windows['span_label']}")
    cards: dict[str, dict | None] = {"K1": {
        "label": "Ulasan pelanggan", "value": fmt_int(len(rows)), "n": len(rows), "delta": None, "delta_description": None,
        "help": " · ".join(notes), "chart_data": [int(v) for v in yearly.to_numpy()], "chart_type": "bar"}}
    if windows and windows.get("ok"):
        cards["K1"] |= {"delta": fmt_change(windows["change"]), "delta_description": f"12 bln s.d. {fmt_month(windows['end'])} vs 12 bln sebelumnya",
                        "tone": "Perhatian" if windows["change"] <= -0.2 else "Info"}
    rated = rows["rating_num"].dropna()
    cards["K2"] = {"label": "Rating rata-rata", "value": fmt_dec(rated.mean()) if len(rated) else "—", "n": len(rated),
                   "help": f"{fmt_pct((rated == 5).mean())} bintang 5" if len(rated) else "Belum ada rating"}
    cards["K3"] = cards["K4"] = None
    if labels is not None:
        n, k = len(labels), int(labels["complaint"].sum())
        note = legacy_note(labels)
        legacy = f" · {note}" if note else ""
        cards["K3"] = {"label": "Ulasan berkeluhan", "value": fmt_pct(k / n) if n else "—", "k": k, "n": n,
                       "help": f"{fmt_int(k)} dari {fmt_int(n)} ulasan teks berlabel (campuran/negatif) · IK95% {fmt_ci(*wilson_interval(k, n))}"
                               f" · label asisten sementara{legacy}"}
        complaints = labels[labels["complaint"]]
        hidden, total = int((complaints["rating_num"] >= 4).sum()), len(complaints)
        enough = total >= MIN_COMPLAINTS
        cards["K4"] = {"label": f"Keluhan di balik {STARS_4_5}", "value": fmt_pct(hidden / total, 0) if enough else "—", "k": hidden,
                       "n": total, "help": (f"{fmt_int(hidden)} dari {fmt_int(total)} ulasan berkeluhan tetap memberi 4–5 bintang"
                                            f" · IK95% {fmt_ci(*wilson_interval(hidden, total))}{legacy}") if enough
                       else f"Butuh ≥{MIN_COMPLAINTS} ulasan berkeluhan; saat ini {fmt_int(total)}"}
    cards["K5"] = None
    if coverage is not None and missing:  # a partial sum would pass for the whole collection
        cards["K5"] = {"label": "Cakupan per target", "value": "—",
                       "help": f"Belum bisa dihitung: laporan {' dan '.join(missing)} tidak tersedia untuk venue/sumber ini."}
    elif coverage is not None and len(coverage):
        known = coverage[coverage["reported"].notna() & (coverage["reported"] > 0)]
        unknown = len(coverage) - len(known)
        parts = [f"{row['platform']} · {venue_name(row['venue_key'])}: {row['label']}" for _, row in coverage.iterrows()]
        cards["K5"] = {"label": "Cakupan per target", "value": f"{len(known)}/{len(coverage)} target",
                       "targets_known": len(known), "targets_unknown": unknown,
                       "help": " · ".join([*parts, "denominator platform berbeda; tidak digabung", "semua tahun (filter tahun tidak berlaku)"])}
    return cards


def venue_complaints(labels: pd.DataFrame) -> pd.DataFrame:
    multiple = session_comparison_blocker(labels) is not None
    rows = []
    grouping = ["annotation_session", "venue_key"] if "annotation_session" in labels else ["venue_key"]
    for keys, group in labels.groupby(grouping, sort=True):
        venue = keys[-1] if isinstance(keys, tuple) else keys
        session = group["annotation_session"].iloc[0] if "annotation_session" in group else UNKNOWN_PROVENANCE
        n, k = len(group), int(group["complaint"].sum())
        legacy = bool(group["cohort"].eq("legacy81").all()) if "cohort" in group else False
        lo, hi = wilson_interval(k, n)
        rows.append({"venue_key": venue, "venue": venue_name(venue) + (" · label lama" if legacy else ""),
                     "annotation_session": session, "venue_urut": venue_rank(venue),
                     "cohort": "legacy81" if legacy else "real1524", "n": n, "k": k, "share": k / n, "lo": lo, "hi": hi,
                     "label": f"{fmt_pct(k / n)} · {fmt_int(k)}/{fmt_int(n)}", "ci": fmt_ci(lo, hi),
                     "overall": math.nan if multiple else float(labels["complaint"].mean())})
    frame = pd.DataFrame(rows, columns=["venue_key", "venue", "annotation_session", "venue_urut", "cohort", "n", "k", "share", "lo", "hi", "label", "ci", "overall"])
    return frame.sort_values(["annotation_session", "venue_urut", "venue_key"], ignore_index=True)


def rating_text_matrix(labels: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rated = labels[labels["rating_num"].notna() & labels["sentiment_4class"].isin(list(SENTIMENTS))]
    stars = rated["rating_num"].round().clip(1, 5).astype(int)
    table = (rated.groupby([stars, rated["sentiment_4class"]]).size().unstack(fill_value=0)
             .reindex(index=range(5, 0, -1), columns=list(SENTIMENTS), fill_value=0))
    rows, agree = [], 0
    for rating, counts in table.iterrows():
        total = int(counts.sum())
        for sentiment, n in counts.items():
            share = n / total if total else 0.0
            fits = (sentiment == "positif") if rating >= 4 else (sentiment in ("netral", "campuran")) if rating == 3 else (sentiment == "negatif")
            agree += int(n) if fits else 0
            rows.append({"rating": rating, "sentimen": sentiment, "n": int(n), "row_total": total, "row_share": share,
                         "hidden": rating >= 4 and sentiment in COMPLAINT, "label": f"{fmt_int(n)} · {fmt_share(share)}" if n else "0"})
    total = int(table.to_numpy().sum())
    return pd.DataFrame(rows), agree / total if total else math.nan


def _venue_order(frame: pd.DataFrame) -> list[str]:
    counts = frame["venue_key"].value_counts()
    return sorted(counts.index, key=lambda key: (-counts[key], venue_rank(key), key))


def aspect_venue_matrix(business: pd.DataFrame, metric: str, selected: tuple[str, str] | None) -> pd.DataFrame:
    if metric not in METRICS:
        raise ValueError(f"Metrik aspek tidak dikenal: {metric}")
    blocker = session_comparison_blocker(business, allow_unknown_descriptive=True)
    if blocker:
        empty = aspect_venue_matrix(business.iloc[:0], metric, selected)
        empty.attrs["comparison_blocker"] = blocker
        return empty
    venues, rows = sorted(business["venue_key"].unique(), key=lambda venue: (venue_rank(venue), venue)), []
    for aspect in ASPECTS:
        for rank, venue in enumerate(venues):
            group = business[business["venue_key"].eq(venue)]
            present = group[f"presence_{aspect}"].astype(bool)
            mentions = int(present.sum())
            known_polarity = present & group[f"polarity_{aspect}"].isin(SENTIMENTS)
            polarity_n = int(known_polarity.sum())
            complaints = int((known_polarity & group[f"polarity_{aspect}"].isin(COMPLAINT)).sum())
            denominator = len(group) if metric == "Penyebutan" else polarity_n
            rows.append({"aspek_kode": aspect, "aspek": ASPECT_NAMES[aspect], "venue_kode": venue, "venue": venue_name(venue),
                         "venue_urut": rank, "n": len(group), "mentions": mentions, "complaints": complaints, "rate": mentions / len(group),
                         "complaint_rate": complaints / polarity_n if polarity_n else math.nan, "small": denominator < SMALL_N,
                         "polarity_n": polarity_n, "missing_polarity": mentions - polarity_n, "denominator": denominator,
                         "missing": metric == "Keluhan" and polarity_n == 0})
    frame = pd.DataFrame(rows, columns=["aspek_kode", "aspek", "venue_kode", "venue", "venue_urut", "n", "mentions", "complaints", "rate",
                                        "complaint_rate", "small", "polarity_n", "missing_polarity", "denominator", "missing"])
    frame["nilai"] = frame["rate"] if metric == "Penyebutan" else frame["complaint_rate"]
    if frame.empty:
        return frame.assign(terpilih=pd.Series(dtype=bool), label=pd.Series(dtype=object))
    cells = list(zip(frame["aspek_kode"], frame["venue_kode"]))
    available = frame[frame["nilai"].notna()]
    first = available.index[0] if len(available) else frame.index[0]
    pick = tuple(selected) if selected and tuple(selected) in cells else cells[first]
    frame["terpilih"] = [cell == pick for cell in cells]
    frame["label"] = [(("NA" if _missing(row.nilai) else fmt_pct(row.nilai)) +
                       f"\n{fmt_int(row.mentions if metric == 'Penyebutan' else row.complaints)}/{fmt_int(row.denominator)}" +
                       (f" · {row.missing_polarity} hilang" if metric == "Keluhan" and row.missing_polarity else "") +
                       ("\nn kecil" if row.small and row.denominator else "")) for row in frame.itertuples()]
    return frame


def _clip(text: str, size: int) -> str:
    return text if len(text) <= size else text[:size - 1].rstrip() + "…"


def cell_quotes(evidence: pd.DataFrame, venue_key: str, aspect: str, mask: Callable[[str], str], limit: int = 8) -> pd.DataFrame:
    cell = evidence[evidence["venue_key"].eq(venue_key) & evidence["aspect"].eq(aspect)]
    cell = cell.assign(_rank=cell["polarity"].map(POLARITY_ORDER).fillna(len(POLARITY_ORDER)), _length=cell["quote"].str.len())
    cell = cell.sort_values(["_rank", "_length"], kind="stable").head(limit)
    return pd.DataFrame({"polarity": cell["polarity"].tolist(), "glyph": cell["polarity"].map(SENTIMENTS).fillna("·").tolist(),
                         "quote": [md_escape(_clip(" ".join(mask(quote).split()), QUOTE_CHARS)) for quote in cell["quote"]],
                         "venue": [venue_name(key) for key in cell["venue_key"]]})


def selected_values(state: Any, param: str, field: str) -> list[str]:
    def get(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    chosen = get(get(state, "selection"), param) if state is not None else None
    items = chosen if isinstance(chosen, list) else [chosen] if chosen is not None else []
    values: list[str] = []
    for item in items:
        found = get(item, field)
        for value in found if isinstance(found, list) else [found]:
            if value is not None and str(value) not in values:
                values.append(str(value))
    return values


def funnel_table(reviews: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    role = reviews["record_role"]
    customer = reviews[role.eq("customer_text")]
    domain = customer[customer["main_domain_eligible"].astype(bool)]
    complete = domain[domain["completeness_status"].eq("complete_as_observed")]
    stages = [("Terkumpul", len(reviews), []),
              ("Teks pelanggan", len(customer), [(int(role.eq("rating_only").sum()), "rating saja"), (int(role.eq("owner_response").sum()), "balasan pemilik")]),
              ("Domain inti batik", len(domain), [(len(customer) - len(domain), "luar domain")]),
              ("Lengkap-teramati", len(complete), [(len(domain) - len(complete), "kelengkapan belum terverifikasi")])]
    if labels is not None:
        real = labels[labels["cohort"].eq("real1524") & labels["record_id"].isin(complete["record_id"])]
        usable = real[real["actual_ml_split"].isin(("train", "val", "test"))]
        dropped = real.loc[~real.index.isin(usable.index), "excluded_reason"].astype(str)
        ellipsis = int(dropped.str.contains("ellipsis", case=False).sum())
        split = usable["actual_ml_split"].value_counts()
        stages += [("Berlabel asisten (sementara)", len(real), [(len(complete) - len(real), "belum berlabel")],
                    f"{fmt_int(real['expert_approved'].sum())} disetujui ahli"),
                   ("Layak ML", len(usable), [(len(dropped) - ellipsis, "non-batik"), (ellipsis, "elipsis")]),
                   ("Split latih / validasi / uji", len(usable), [], " / ".join(fmt_int(split.get(part, 0)) for part in ("train", "val", "test")))]
    rows, previous = [], None
    for order, (name, n, drops, *extra) in enumerate(stages, start=1):
        note = ", ".join([f"−{fmt_int(k)} {why}" for k, why in drops if k] + list(extra))
        label = extra[0] if name.startswith("Split") else f"{fmt_int(n)} · {note}" if note else fmt_int(n)
        rows.append({"urut": order, "tahap": name, "n": n, "n_prev": n if previous is None else previous, "turun": 0 if previous is None else previous - n,
                     "catatan": note, "label": label})
        previous = n
    return pd.DataFrame(rows)


def reconcile_funnel(funnel: pd.DataFrame, audit_counts: dict) -> list[str]:
    messages = []
    for order, key in AUDIT_STAGES.items():
        stage = funnel[funnel["urut"].eq(order)]
        if stage.empty:
            continue
        name, got, expected = stage["tahap"].iloc[0], int(stage["n"].iloc[0]), audit_counts.get(key)
        if not isinstance(expected, int):
            messages.append(f"{name}: audit tidak mencatat {key}")
        elif expected != got:
            messages.append(f"{name}: korpus {fmt_int(got)} ≠ audit {fmt_int(expected)}")
    return messages


def _int_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return float(int(number)) if math.isfinite(number) else math.nan


def coverage_status(collected: Any, reported: Any) -> str:
    """One rule for the coverage card and the run table: lengkap once the platform total is reached, parsial with some reviews,
    gagal with none."""
    collected, reported = _int_or_nan(collected), _int_or_nan(reported)
    if math.isnan(collected):
        return "tak diketahui"
    known = not math.isnan(reported) and reported > 0
    return "lengkap" if known and collected >= reported else "parsial" if collected > 0 else "gagal"


def _coverage_row(platform: str, venue: str, collected: Any, reported: Any, reason: Any, duplicates: Any = None) -> dict:
    status = coverage_status(collected, reported)
    collected, reported = _int_or_nan(collected), _int_or_nan(reported)
    known = not math.isnan(reported) and reported > 0
    coverage = collected / reported if known and not math.isnan(collected) else math.nan
    label = (f"{fmt_int(collected)}/{fmt_int(reported)} · {fmt_pct(coverage)}" if known else
             f"{fmt_int(collected)}/0 · NA (total dilaporkan nol)" if reported == 0 else f"{fmt_int(collected)} · total tak diketahui")
    return {"platform": platform, "source": platform, "venue_key": venue,
            "target": f"{STATUS[status]} {PLATFORM_SHORT.get(platform, platform)} · {venue_name(venue)}",
            "collected": collected, "reported": reported, "coverage": coverage, "status": status,
            "reason": REASONS.get(str(reason), str(reason)) if reason else "—", "duplicates": _int_or_nan(duplicates),
            "label": label}


def coverage_rows(airbnb: list[dict], seeksophie: dict | None, targets: list[dict], latest_runs: dict) -> pd.DataFrame:
    rows = [_coverage_row("Airbnb", report.get("venue_key"), report.get("customer_texts"),
                          (report.get("source_totals_metadata") or {}).get("writtenReviewCount"), report.get("stop_reason"))
            for report in sorted(airbnb, key=lambda report: venue_rank(report.get("venue_key")))]
    if seeksophie:
        rows.append(_coverage_row("SeekSophie", seeksophie.get("venue_key"), seeksophie.get("actual_unique_reviews_collected"),
                                  seeksophie.get("option_reviews_total_metadata_only"), seeksophie.get("stop_reason")))
    for target in targets:
        run = latest_runs.get(target.get("target_key")) or {}
        rows.append(_coverage_row("Google Maps", target.get("target_key"), target.get("inserted_total"), target.get("reported_total"),
                                  run.get("reason"), target.get("duplicates_total")))
    columns = ["platform", "source", "venue_key", "target", "collected", "reported", "coverage", "status", "reason", "duplicates", "label"]
    return pd.DataFrame(rows, columns=columns)


def quality_matrix(reviews: pd.DataFrame) -> pd.DataFrame:
    text = reviews["clean_text"].ne("")
    checks = {"rating_only": reviews["record_role"].eq("rating_only"), "owner_response": reviews["record_role"].eq("owner_response"),
              "duplicate_text": reviews["exact_text_duplicate_count"].astype(int) > 1,
              "truncated": reviews["flags"].map(lambda flags: any(flag.startswith("terminal_ellipsis") for flag in flags)).astype(bool),
              "no_locale": reviews["source_locale"].eq("") & text, "variant_unknown": reviews["text_variant"].eq("unknown"),
              "relative_date": reviews["month"].isna(), "unverified": reviews["completeness_status"].ne("complete_as_observed"),
              "short_text": text & (reviews["word_count"] <= 3),
              "normalized": text & reviews["raw_text_sha256"].ne(reviews["clean_text_sha256"])}
    columns = [(venue, venue_name(venue)) for venue in sorted(reviews["venue_key"].unique(), key=lambda key: (venue_rank(key), key))]
    rows = []
    for check_order, (code, name) in enumerate(QUALITY_CHECKS):
        for column_order, (column, title) in enumerate([*columns, ("total", "Total")]):
            scope = reviews["venue_key"].eq(column) if column != "total" else pd.Series(True, index=reviews.index)
            n, total = int((checks[code] & scope).sum()), int(scope.sum())
            pct = n / total if total else math.nan
            rows.append({"cek_kode": code, "cek": name, "urut_cek": check_order, "kolom_kode": column, "kolom": title, "urut_kolom": column_order,
                         "n": n, "total": total, "pct": pct, "label": f"{fmt_int(n)} ({fmt_pct(pct)})" if n else "·"})
    return pd.DataFrame(rows)


class Insight(NamedTuple):
    key: str
    tone: str
    value: str
    text: str
    basis: str
    priority: int


def _qualifier(first: tuple[float, float], second: tuple[float, float]) -> str:
    disjoint = first[0] > second[1] or second[0] > first[1]
    return "IK95% tidak tumpang tindih" if disjoint else "indikasi awal (IK95% tumpang tindih)"


def compute_insights(reviews: pd.DataFrame, labels: pd.DataFrame | None, business: pd.DataFrame | None,
                     coverage: pd.DataFrame | None, windows: dict | None) -> list[Insight]:
    found = []
    if labels is not None:
        complaints = labels[labels["complaint"]]
        hidden = int((complaints["rating_num"] >= 4).sum())
        if len(complaints) >= MIN_COMPLAINTS:
            note = legacy_note(labels)
            legacy = f" · {note}" if note else ""
            found.append(Insight("hidden_critique", "Perhatian", fmt_pct(hidden / len(complaints), 0), f"ulasan berkeluhan tetap memberi {STARS_4_5}",
                                 f"{fmt_int(hidden)}/{fmt_int(len(complaints))} ulasan berkeluhan · label asisten sementara{legacy}", 1))
    if windows and windows.get("ok"):
        found.append(Insight("volume_trend", "Perhatian" if windows["change"] <= -0.2 else "Info", fmt_change(windows["change"], "−"),
                             f"{fmt_int(windows['current'])} vs {fmt_int(windows['previous'])} ulasan",
                             f"{windows['current_label']} vs {windows['previous_label']}", 2))
    known = coverage[coverage["coverage"].notna()] if coverage is not None else pd.DataFrame()
    if len(known) and known["coverage"].min() < 0.5:
        low = known.loc[known["coverage"].idxmin()]
        found.append(Insight("coverage_gap", "Perhatian", fmt_pct(low["coverage"]),
                             f"cakupan {low['platform']} · {venue_name(low['venue_key'])} ({fmt_int(low['collected'])}/{fmt_int(low['reported'])})",
                             f"{low['status']} · {low['reason']} · semua tahun (filter tahun tidak berlaku)", 3))
    if business is not None and not session_comparison_blocker(business):
        sizes = business["venue_key"].value_counts()
        big = [venue for venue in _venue_order(business) if sizes[venue] >= MIN_ASPECT_N]
        best = None
        for aspect in ASPECTS:
            stats = {venue: (int(business.loc[business["venue_key"].eq(venue), f"presence_{aspect}"].astype(bool).sum()), int(sizes[venue]))
                     for venue in big}
            if len(stats) >= 2:
                high = max(stats, key=lambda v: stats[v][0] / stats[v][1])
                low_venue = min(stats, key=lambda v: stats[v][0] / stats[v][1])
                gap = stats[high][0] / stats[high][1] - stats[low_venue][0] / stats[low_venue][1]
                if best is None or gap > best[0]:
                    best = (gap, aspect, (high, *stats[high]), (low_venue, *stats[low_venue]))
        if best is not None and best[0] > 0:
            _, aspect, (high, k_high, n_high), (low_venue, k_low, n_low) = best
            found.append(Insight("aspect_gap", "Info", f"{fmt_pct(k_high / n_high)} vs {fmt_pct(k_low / n_low)}",
                                 f"{ASPECT_SHORT[aspect]} disebut di {venue_short(high)} vs {venue_short(low_venue)}",
                                 f"{fmt_int(k_high)}/{fmt_int(n_high)} vs {fmt_int(k_low)}/{fmt_int(n_low)} ulasan · "
                                 f"{_qualifier(wilson_interval(k_high, n_high), wilson_interval(k_low, n_low))} · pemeriksaan satu penulis dilaporkan; bukan audit independen", 4))
    if labels is not None and len(labels) and not session_comparison_blocker(labels):
        overall = float(labels["complaint"].mean())
        shares = venue_complaints(labels)
        outliers = shares[(shares["n"] >= MIN_OUTLIER_N) & (shares["lo"] > overall)]
        if len(outliers):
            top = outliers.iloc[0]
            rest = labels[labels["venue_key"].ne(top["venue_key"])]
            k_rest, n_rest = int(rest["complaint"].sum()), len(rest)
            rest_ci = wilson_interval(k_rest, n_rest)
            legacy = " (label lama)" if top["cohort"] == "legacy81" else ""
            found.append(Insight("venue_outlier", "Perhatian", fmt_pct(top["share"]), f"ulasan berkeluhan di {venue_name(top['venue_key'])}{legacy}",
                                 f"{fmt_int(top['k'])}/{fmt_int(top['n'])} · IK95% {top['ci']} vs venue lain "
                                 f"{fmt_pct(k_rest / n_rest if n_rest else math.nan)} (IK95% {fmt_ci(*rest_ci)}) · "
                                 f"{_qualifier((top['lo'], top['hi']), rest_ci)}", 5))
    return sorted(found, key=lambda insight: insight.priority)


def _run_metrics(text: str) -> dict:
    value = json.loads(text) if isinstance(text, str) and text.strip() else {}
    return {key: (value.get(key) if isinstance(value, dict) else None) for key in RUN_METRICS}


def read_collector_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read run telemetry through an immutable URI: no sidecar files, no review rows, no free-text detail."""
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        runs = con.execute("SELECT run_id, target_key, status, reason, round_number, started_at, finished_at, metrics FROM runs").fetchall()
        points = con.execute("SELECT run_id, round_number, metrics FROM checkpoints").fetchall()
    finally:
        con.close()
    run_cols, point_cols = ["run_id", "target_key", "status", "reason", "round_number", "started_at", "finished_at"], ["run_id", "round_number"]
    runs_frame = pd.DataFrame([{**dict(zip(run_cols, row[:-1])), **_run_metrics(row[-1])} for row in runs], columns=[*run_cols, *RUN_METRICS])
    points_frame = pd.DataFrame([{**dict(zip(point_cols, row[:-1])), **_run_metrics(row[-1])} for row in points], columns=[*point_cols, *RUN_METRICS])
    return runs_frame.sort_values("started_at", ignore_index=True), points_frame.sort_values(point_cols, ignore_index=True)


def _clock(stamp: Any) -> str:
    return "—" if _missing(stamp) else f"{stamp:%H:%M}"


def _seconds(start: pd.Timestamp | None, end: pd.Timestamp | None) -> float | None:
    return (end - start).total_seconds() if start is not None and end is not None else None


def minutes(seconds: float) -> str:
    return f"{fmt_dec(seconds / 60, 1)} mnt" if seconds >= 90 else f"{fmt_int(seconds)} dtk"


def scroll_curves(runs: pd.DataFrame, checkpoints: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for run in runs.sort_values("started_at", na_position="last").itertuples(index=False):
        points = checkpoints[checkpoints["run_id"].eq(run.run_id)].sort_values("round_number")
        if points.empty:
            continue
        observed = pd.to_numeric(points["observed_unique"], errors="coerce")
        unique = observed.fillna(pd.to_numeric(points["inserted"], errors="coerce"))  # first runs lack observed_unique; updated re-counts known IDs
        started = _utc(run.started_at)
        when = _clock(started) if started is not None else f"run {str(run.run_id)[:8]}"  # unique legend entry even without a start time
        frames.append(pd.DataFrame({"run_id": run.run_id, "run": f"{when} · {fmt_int(points['round_number'].max())} putaran",
                                    "putaran": points["round_number"].astype(int).to_numpy(), "unik": unique.fillna(0).astype(int).to_numpy(),
                                    "duplikat": pd.to_numeric(points["duplicates"], errors="coerce").fillna(0).astype(int).to_numpy(),
                                    "akhir": [False] * (len(points) - 1) + [True]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["run_id", "run", "putaran", "unik", "duplikat", "akhir"])


def run_telemetry(runs: pd.DataFrame | None, reports: Sequence[Mapping]) -> pd.DataFrame:
    """One row per scraping run, oldest first: Google Maps runs from the collector snapshot, Airbnb and SeekSophie from their reports."""
    rows = []
    for run in runs.itertuples(index=False) if runs is not None else ():
        started = _utc(run.started_at)
        rows.append({"platform": "Google Maps", "venue_key": run.target_key, "mulai": started, "durasi": _seconds(started, _utc(run.finished_at)),
                     "putaran": run.rounds, "baru": run.inserted, "duplikat": run.duplicates,
                     "hasil": RUN_STATUS.get(run.status, run.status), "alasan": REASONS.get(run.reason, run.reason)})
    for report in reports:  # the result follows the coverage card's rule (coverage_status), not the stop reason
        reason = report.get("stop_reason")
        if report.get("platform") == "Airbnb":
            started = _utc(report.get("started_at"))
            rows.append({"platform": "Airbnb", "venue_key": report.get("venue_key"), "mulai": started,
                         "durasi": _seconds(started, _utc(report.get("finished_at"))), "putaran": None, "baru": report.get("customer_texts"),
                         "duplikat": None, "hasil": coverage_status(report.get("customer_texts"), report.get("reported")),
                         "alasan": REASONS.get(reason, reason)})
        else:
            collected = _int_or_nan(report.get("actual_unique_reviews_collected"))
            rows.append({"platform": report.get("platform"), "venue_key": report.get("venue_key"), "mulai": _utc(report.get("collected_at")),
                         "durasi": None, "putaran": None, "baru": collected, "duplikat": report.get("duplicate_objects"),
                         "hasil": coverage_status(collected, report.get("option_reviews_total_metadata_only")),
                         "alasan": REASONS.get(reason, reason)})
    frame = pd.DataFrame(rows, columns=TELEMETRY_COLS)
    frame["target"] = [venue_name(key) if key else "—" for key in frame["venue_key"]]
    frame["mulai"] = pd.to_datetime(frame["mulai"], utc=True)
    for column in ("durasi", "putaran", "baru", "duplikat"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame.sort_values("mulai", ignore_index=True, na_position="last")


def google_burden(coverage: pd.DataFrame | None) -> tuple[int, int] | None:
    """Google Maps duplicate observations and unique reviews, summed over every target and run in the audit."""
    google = coverage[coverage["platform"].eq("Google Maps")] if coverage is not None else None
    if google is None or google.empty or not google["duplicates"].notna().any():
        return None
    return int(google["duplicates"].sum()), int(google["collected"].sum())


def fmt_times(duplicates: int, unique: int) -> str:
    return f"≈{fmt_int(duplicates / unique)}×" if unique else "—"


def ops_summary(snapshots: pd.DataFrame, coverage: pd.DataFrame | None, telemetry: pd.DataFrame) -> dict[str, dict | None]:
    """The four unfiltered operations cards; None where the source file is absent."""
    cards: dict[str, dict | None] = dict.fromkeys(("identities", "duplicates", "runs", "duration"))
    if len(snapshots):
        counts = [int(v) for v in snapshots["n"]]
        cards["identities"] = {"label": "Identitas unik terkumpul", "value": fmt_int(counts[-1]), "chart_data": counts,
                               "delta": fmt_signed(counts[-1] - counts[-2]) if len(counts) > 1 else None,
                               "help": f"{len(counts)} snapshot audit, {_clock(snapshots['created_at'].iloc[0])}–"
                                       f"{_clock(snapshots['created_at'].iloc[-1])} UTC"}
    burden = google_burden(coverage)
    if burden:
        duplicates, unique = burden
        cards["duplicates"] = {"label": "Observasi duplikat Google Maps", "value": fmt_int(duplicates),
                               "delta": f"{fmt_times(duplicates, unique)} per ulasan unik" if unique else None,
                               "help": f"{fmt_int(duplicates)} kali feed membaca ulang ulasan yang sudah tersimpan, untuk {fmt_int(unique)} ulasan unik."}
    if len(telemetry):
        google = telemetry[telemetry["platform"].eq("Google Maps")]
        partial = int(google["hasil"].eq("parsial").sum())
        chip = (f"semua {fmt_int(len(google))} run Google parsial" if partial and partial == len(google)
                else f"{fmt_int(partial)} run Google parsial") if len(google) else None
        cards["runs"] = {"label": "Run scraping tercatat", "value": fmt_int(len(telemetry)), "delta": chip, "warn": bool(partial),
                         "help": " · ".join(f"{name} {fmt_int(count)}" for name, count in telemetry["platform"].value_counts().items())}
        timed = telemetry.dropna(subset=["durasi"])
        untimed = sorted(set(telemetry["platform"]) - set(timed["platform"]))
        cards["duration"] = {"label": "Durasi scraping tercatat", "value": minutes(float(timed["durasi"].sum())),
                             "delta": f"{fmt_int(len(timed))} run berdurasi" if len(timed) else None,
                             "help": " · ".join(f"{name} {minutes(float(value))}" for name, value in timed.groupby("platform")["durasi"].sum().items())
                                     + "".join(f"; {name} tanpa catatan durasi" for name in untimed)}
    return cards
