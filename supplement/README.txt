REPRODUCIBILITY SUPPLEMENT: COVERAGE-AWARE TOURISM REVIEW DASHBOARD

This local supplement accompanies the revised manuscript. It contains dashboard
source, synthetic fixtures, rules, schema documentation, analysis code and testing
tools. It is not evidence of a public code release. It contains no collected
review texts, source identifiers, raw captures, credentials or private exports.
The fixture contains deliberately fabricated review sentences and planted names,
URLs and secret markers for negative tests; these are not real observations.

SETUP
Use Python 3.13 and install the supplied requirements.txt in an isolated virtual
environment. Browser timing additionally requires Playwright's Chromium:
  python -m playwright install chromium
The numerical historical environment was Python 3.13.5, Streamlit 1.64.0,
pandas 3.0.6, NumPy 2.5.3 and Playwright 1.61.0 / Chromium 149. Versions in
requirements.txt define the supplied package environment; installation and a
successful run are separate from the manuscript's historical measurements.

SYNTHETIC TESTS
  python test_inventory.py --output test_inventory.json
  python run_tests.py --junitxml=results/synthetic-tests.xml
The selected test files use temporary synthetic export trees and no collectors.
run_tests.py explicitly deselects two integration tests requiring the larger
application: full application navigation, and consistency with the separate
business-model dashboard. No replacement stubs are used to make them pass.
Passing tests establish implemented behavior on their inputs; they do not
validate sentiment accuracy, anonymization recall or managerial benefits.

SCHEMAS AND RULES
schemas.json records the source's CSV allowlists and optional annotation fields,
the audit/report structure and telemetry tables. It can be regenerated after
source changes with:
  python generate_schemas.py --output schemas.json
rules.txt describes aggregation guards, interpretation and missing-data rules.

DASHBOARD DEMONSTRATION
  python -m streamlit run launch_dashboard.py --server.address=127.0.0.1
The launch script uses only a synthetic fixture unless BI_DATA_ROOT is explicitly
set to a private, compatible export tree supplied separately by its owner. The
generated fixture is clearly a small functional demonstration. Its performance
is not interchangeable with timing on the manuscript's 1,702-record collection.

TIMING
  python timing.py --output results/timing-synthetic.json
This starts its own loopback Streamlit process and measures its first observed
page load, five fresh browser contexts after server-cache warm-up, and five filter
changes. timing.py is a reconstruction of the documented procedure; the original
historical timing script was not recovered. It creates new measurements when run,
does not reproduce historical arrays by assertion, and records its actual protocol.
historical_timing.json preserves the already recorded historical timing arrays
and distinguishes the single first-after-start observation from warm-server-cache
browser sessions. The minimal supplement launcher differs from the full original
application shell. Use --data-root only for separately authorized private inputs.

TREND AND PROVENANCE ANALYSIS
review_revision_analysis.json contains aggregate observed results and distinctly
labeled synthetic adversarial examples. review_revision_analysis.py requires
--repo pointing to a separately available compatible private source/export tree:
  python review_revision_analysis.py --repo PATH --output results/analysis.json
The real corpus is intentionally absent, so the aggregate historical analysis
cannot be regenerated from the synthetic fixture alone. Neither the analysis
nor a quotation-string match validates the labels' substantive correctness.

RIGHTS AND GOVERNANCE
Only software and synthetic fixtures are supplied here. Collection access and
content redistribution rights are separate from a software license. No collected
content is licensed for redistribution by this package. The synthetic demo creates
temporary files locally. Private raw captures must remain outside this package;
the dashboard's telemetry reader does not query its SQLite reviews table. Name
masking is heuristic; multilingual recall has not been measured.

The exact current test results and source hashes are recorded separately in the
package's results/ artifacts when the accompanying verification run is performed.
