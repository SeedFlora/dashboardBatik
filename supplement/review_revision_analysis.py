"""Read-only reanalysis for ICOT manuscript revision; no source texts/identifiers exported."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="Compatible source repository and private export tree supplied separately.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    import pandas as pd
    from temajelajah import bi_data as data

    bundle = data.load_bundle(data.source_paths())
    reviews, stamp = bundle["reviews"], bundle["collected_at"]
    monthly, windows = data.monthly_series(reviews, stamp, basis=reviews)
    stop = data._month_start(stamp)
    dated, counts = data._dated_counts(reviews, stop)

    def window_pair(end):
        current_start = end - pd.DateOffset(months=11)
        previous_end = end - pd.DateOffset(months=12)
        previous_start = end - pd.DateOffset(months=23)
        span = counts.reindex(pd.date_range(previous_start, end, freq="MS"), fill_value=0)
        previous, current = int(span.iloc[:12].sum()), int(span.iloc[12:].sum())
        return {"current_start": str(current_start.date()), "current_end_month": str(end.date()),
                "previous_start": str(previous_start.date()), "previous_end_month": str(previous_end.date()),
                "current_count": current, "previous_count": previous,
                "relative_change": current / previous - 1 if previous else None}

    last = stop - pd.DateOffset(months=1)
    prior_months = counts.reindex(pd.date_range(last - pd.DateOffset(months=12), last, freq="MS"), fill_value=0)
    out = {
        "purpose": "Revision analysis; observed results and synthetic adversarial examples are separate.",
        "snapshot_collection_timestamp": str(stamp),
        "trend": {
            "heuristic": window_pair(last - pd.DateOffset(months=1) if windows["partial"] is not None else last),
            "calendar_complete": window_pair(last),
            "flagged_final_month": str(windows["partial"].date()) if windows["partial"] is not None else None,
            "final_month_count": int(counts.get(last, 0)),
            "preceding_12_month_median": float(prior_months.iloc[:-1].median()),
            "low_month_threshold": float(0.5 * prior_months.iloc[:-1].median()),
            "dated_customer_records": len(dated),
            "undated_customer_records": windows["undated"],
            "interpretation": "Low activity alone cannot establish incomplete collection. Both comparisons describe observed review volume, not demand/bookings/performance.",
        },
    }

    synthetic_months = pd.date_range("2024-10-01", "2026-09-01", freq="MS")
    sequences = {
        "stable_activity": ([10] * 24, "Stable complete activity should not flag a month."),
        "genuine_seasonal_dip": ([10] * 11 + [2] + [10] * 11 + [2], "A fully observed recurring seasonal dip is still flagged; lower activity is not evidence of missingness."),
        "genuine_recent_decline": ([10] * 22 + [4, 2], "A fully observed recent decline is still flagged."),
        "incomplete_collection_identical_counts": ([10] * 23 + [2], "The heuristic cannot distinguish these counts from a genuine final-month decline."),
        "genuine_zero_activity": ([10] * 23 + [0], "A fully observed zero-activity month is flagged even though the denominator of collection completeness is unknown."),
    }
    examples = {}
    for key, (values, interpretation) in sequences.items():
        rows = [{"record_role": "customer_text", "month": month, "rating_num": 5}
                for month, count in zip(synthetic_months, values) for _ in range(count)]
        synthetic = pd.DataFrame(rows)
        _, result = data.monthly_series(synthetic, pd.Timestamp("2026-10-01", tz="UTC"), basis=synthetic)
        comparison = result.get("sensitivity") or result
        examples[key] = {"monthly_counts": values,
                         "flagged_month": str(result["partial"].date()) if result["partial"] is not None else None,
                         "current_count": comparison["current"], "previous_count": comparison["previous"],
                         "relative_change": comparison["change"], "interpretation": interpretation}
    out["synthetic_adversarial_examples"] = examples

    facts = json.loads((args.repo / "outputs/paper_figures/facts.json").read_text(encoding="utf-8-sig"))
    historical = facts["tests"]
    out["historical_test_relationship"] = {
        "source": "outputs/paper_figures/facts.json", "recorded_at": facts["generated_at"],
        "baseline": historical["baseline_before_this_work"], "new_tests": historical["new_tests"],
        "full_suite": historical["full_suite_run_once"], "collected_total": historical["collected_total_now"],
        "new_tests_included_in_full_suite": historical["baseline_before_this_work"]["passed"] + historical["new_tests"]["total"] == historical["full_suite_run_once"]["passed"],
        "caveat": "Historical recorded results; this read-only analysis does not rerun or establish current suite status.",
    }

    batch_names = ("collector_a_0001_0800.jsonl", "root_0801_1524.jsonl", "ml_1141_1252.jsonl", "ui_1253_1524.jsonl")
    expected_fields = ("annotation_session", "session_id", "prompt_version", "rubric_version", "annotation_date", "annotated_at")
    safe_fields = ("annotation_method", "annotator_requested_alias", "backend_model_id", "expert_approved", "rating_used_for_annotation")
    batches = {}
    for name in batch_names:
        path = args.repo / "exports/ml_experiments/annotations" / name
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        keys = set().union(*(set(row) for row in records))
        venue_by_id = dict(zip(reviews["record_id"], reviews["venue_key"]))
        batches[name] = {"records": len(records), "index_min": min(row["index"] for row in records),
                         "index_max": max(row["index"] for row in records),
                         "venue_counts": dict(Counter(venue_by_id.get(row.get("record_id"), "unknown") for row in records)),
                         "metadata": {field: dict(Counter(json.dumps(row.get(field), ensure_ascii=False) for row in records)) for field in safe_fields},
                         "missing_dedicated_fields": [field for field in expected_fields if field not in keys]}
    out["annotation_batch_provenance"] = batches
    out["annotation_metadata_caveat"] = "Batch filename provides a reconstructed batch/session proxy, not a recorded session identity. File modification timestamps and collection timestamps must not be relabeled annotation dates. Requested assistant alias does not identify the backend model. Prompt/rubric version, annotation date, and backend model are unknown when no separate evidence exists."

    for file_key, relative in (("sentiment_real_csv", "exports/ml_experiments/dataset/real_labeled1524.csv"),
                               ("sentiment_legacy_csv", "exports/ml_experiments/dataset/legacy_supplement81.csv"),
                               ("aspect_csv", "exports/business_research/dataset/business_real.csv")):
        with (args.repo / relative).open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            records = list(reader)
            keys = reader.fieldnames
        selected_fields = [field for field in safe_fields if field in keys]
        out[file_key] = {"rows": len(records), "metadata": {field: dict(Counter(row.get(field) or "unknown" for row in records)) for field in selected_fields},
                         "component_filenames": dict(Counter(Path(row.get("annotation_component_file", "")).name or "unknown" for row in records)) if "annotation_component_file" in keys else {},
                         "missing_dedicated_fields": [field for field in expected_fields if field not in keys]}
        if "annotation_component_file" in keys:
            components = sorted(set(Path(row.get("annotation_component_file", "")).name or "unknown" for row in records))
            out[file_key]["component_venue_counts"] = {
                component: dict(Counter(row["venue_key"] for row in records
                                        if (Path(row.get("annotation_component_file", "")).name or "unknown") == component))
                for component in components}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trend": out["trend"],
                      "historical_69_included_in_627": out["historical_test_relationship"]["new_tests_included_in_full_suite"],
                      "batch_venue_counts": {key: value["venue_counts"] for key, value in batches.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
