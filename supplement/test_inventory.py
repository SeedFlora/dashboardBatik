"""List test function definitions; this does not execute or count parametrized cases."""
import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("test_inventory.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    excluded = {
        "test_aspect_names_match_business_dashboard": "Requires the separate full business-model dashboard.",
        "test_lead_page_dispatch_through_ui_main": "Requires the full application's navigation/services.",
    }
    files = []
    for path in sorted((root / "tests").glob("test_bi_*.py")):
        entries = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                entries.append({"name": node.name, "line": node.lineno,
                                "selected": node.name not in excluded,
                                "exclusion_reason": excluded.get(node.name),
                                "parametrized": any("parametrize" in ast.unparse(decorator) for decorator in node.decorator_list)})
        files.append({"file": path.relative_to(root).as_posix(), "test_functions": sorted(entries, key=lambda item: item["line"])})
    result = {"kind": "static inventory; not execution results", "files": files,
              "function_count": sum(len(item["test_functions"]) for item in files),
              "selected_function_count": sum(test["selected"] for item in files for test in item["test_functions"]),
              "notes": ["Parametrization can produce multiple collected cases per function.",
                        "Historical 69 new tests are included in 627 passes (558 baseline + 69); 11 historical skips.",
                        "Current selected standalone results must be read from a new run, not inferred from this inventory."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"function_count": result["function_count"], "selected_function_count": result["selected_function_count"]}))


if __name__ == "__main__":
    main()
