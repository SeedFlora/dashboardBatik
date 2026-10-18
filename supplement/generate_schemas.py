"""Export schema documentation from the actual bundled data-module allowlists."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("schemas.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo.resolve()))
    data = importlib.import_module("temajelajah.bi_data")
    result = {
        "description": "Documentation of dashboard input allowlists, optional provenance and telemetry. CSV files may contain other columns, but they are not loaded unless explicitly allowlisted.",
        "csv_storage_type": "Input CSV values are initially strings; dates, numeric ratings and boolean fields are normalized by the data module.",
        "csv_required_allowlists": {name: list(getattr(data, name)) for name in ("CORPUS_COLS", "LABEL_COLS", "BUSINESS_COLS")},
        "optional_annotation_columns": list(getattr(data, "ANNOTATION_OPTIONAL_COLS", ())),
        "normalized_provenance_columns": list(getattr(data, "PROVENANCE_COLS", ())),
        "forbidden_dashboard_columns": sorted(data.FORBIDDEN_COLS),
        "enums": {"customer_record_roles": list(data.CUSTOMER_ROLES), "sentiment_and_polarity": list(data.SENTIMENTS),
                  "complaint_labels": list(data.COMPLAINT), "aspect_codes": list(data.ASPECTS),
                  "sources": list(data.SOURCES), "unknown_provenance": getattr(data, "UNKNOWN_PROVENANCE", "unknown")},
        "input_tree": {
            "exports/paper_expansion/latest.json": {"run_id": "validated relative child of runs/"},
            "exports/paper_expansion/runs/<run_id>/corpus.csv": "CORPUS_COLS", 
            "exports/paper_expansion/runs/<run_id>/audit.json": {
                "run_id": "string", "created_at": "ISO timestamp", "snapshot": {"sha256": "SHA-256 hex digest"},
                "counts": {key: "nonnegative integer" for key in data.AUDIT_STAGES.values()},
                "collector_targets": [{"target_key": "venue key", "reported_total": "number or null",
                                       "inserted_total": "number", "updated_total": "number", "duplicates_total": "number"}],
                "latest_collector_runs": {"<venue_key>": {"status": "string", "reason": "stop-reason code", "metrics": "allowlisted numeric metrics"}}},
            "exports/paper_expansion/runs/<run_id>/raw_snapshot.sqlite3": "Only runs/checkpoints metadata tables are queried; reviews table is not queried.",
            "exports/paper_expansion/airbnb/*-report.json": {"allowlisted_fields": list(data.REPORT_KEYS["Airbnb"]),
                                                           "source_totals_metadata": {"writtenReviewCount": "number or numeric string; per-listing denominator"}},
            "exports/paper_expansion/seeksophie_initial/collection_report.json": {"allowlisted_fields": list(data.REPORT_KEYS["SeekSophie"])},
            "exports/ml_experiments/dataset/real_labeled1524.csv": "LABEL_COLS plus optional annotation metadata",
            "exports/ml_experiments/dataset/legacy_supplement81.csv": "LABEL_COLS plus optional annotation metadata; separate legacy cohort",
            "exports/business_research/dataset/business_real.csv": "BUSINESS_COLS plus optional annotation metadata",
        },
        "aspect_decisions_json": {"<aspect_code>": {"present": "boolean", "polarity": "sentiment enum or null",
                                                    "evidence_sentence_ids": ["optional source-local sentence reference"],
                                                    "evidence_quotes": ["literal text substring; correctness of polarity is not established"]}},
        "sqlite_telemetry_tables": {
            "runs": ["run_id", "target_key", "status", "reason", "round_number", "started_at", "finished_at", "metrics"],
            "checkpoints": ["run_id", "round_number", "metrics"],
            "metrics_numeric_allowlist": list(data.RUN_METRICS),
        },
        "generated_telemetry_columns": list(data.TELEMETRY_COLS),
        "missingness": {"unknown_model_prompt_rubric_date": "unknown; never derived from filesystem modification times",
                        "missing_expected_snapshot_hash": "unknown snapshot identity; count agreement does not imply a hash match",
                        "zero_retrieval": "observed numerator, not a missing denominator",
                        "coverage_unknown_or_zero_denominator": "NA; no pooled denominator across platforms",
                        "aspect_missing_polarity": "missing/NA; not a zero complaint"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Schema documentation written.")


if __name__ == "__main__":
    main()
