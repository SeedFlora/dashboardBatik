# dashboardBatik

Reproducibility supplement for **A Coverage-Aware Business Intelligence Dashboard for Web-Scraped Tourism Reviews With Evidence-Grounded Text Analytics**, ICOT 2026, Paper #317.

Authors: Budi Juarto, Iston Dwija Utama, and Artha Sanjaya, Bina Nusantara University.

## Available materials

The [supplement](supplement/) contains the dashboard's four business intelligence modules, synthetic test fixtures, input schemas, aggregation and interpretation rules, test inventory, timing reconstruction tools, and recorded aggregate analyses. The software and synthetic fixtures are supplied under the [MIT License](supplement/LICENSE).

The collected tourism review texts, platform review identifiers, source URLs, raw captures, and private exports are not distributed. Platform access and content redistribution permissions have not been independently verified. The synthetic fixture demonstrates implemented behavior; it cannot reconstruct the manuscript's real-corpus aggregate results.

## Run the synthetic demonstration

Use Python 3.13 in an isolated environment. From the repository root:

```sh
cd supplement
python -m pip install -r requirements.txt
python run_tests.py
python -m streamlit run launch_dashboard.py --server.address=127.0.0.1
```

The launcher uses synthetic data by default and does not collect remote reviews. The fixture intentionally includes fabricated names, `secret.example` URLs, `SECRET` markers, and fabricated paths for negative tests. These are test inputs, not private observations or credentials.

For the optional browser timing reconstruction:

```sh
python -m playwright install chromium
python timing.py --output results/new-synthetic-timing.json
```

This produces new measurements on synthetic inputs. It does not reproduce the historical real-corpus performance measurements.

## Validation scope and integrity

[results/validation.json](supplement/results/validation.json) lists the SHA-256 hashes of the unchanged supplement files and distinguishes these recorded runs:

- The definitive standalone package run passed **82 tests**, with **2 integration tests explicitly deselected** because they require the larger application.
- The coordinated source-repository verification passed **84 targeted BI tests**: 55 data tests and 29 UI tests.
- The earlier full-project result of **627 passed and 11 skipped** includes the earlier 69 BI tests and is a historical result, not a new full-project run.

Passing synthetic tests establishes behavior on their inputs. It does not establish sentiment accuracy, anonymization recall, collection rights, or managerial benefit. The [rules](supplement/rules.txt) document these boundaries.

The bundled README and validation artifacts retain the wording of the local package at its validation date; the public release is recorded by this repository's commit history. See [supplement/README.txt](supplement/README.txt) for schemas, timing details, and the analysis procedure requiring a separately authorized private input tree.

## Data availability statement

The dashboard source code, synthetic fixtures, schemas, rules, aggregate analysis records, and validation artifacts are publicly available in this repository. The collected review texts and raw captures are not publicly redistributed because content redistribution permissions remain unverified and the material may contain identifying information. Synthetic fixtures cannot reproduce the real-corpus aggregates reported in the paper.
