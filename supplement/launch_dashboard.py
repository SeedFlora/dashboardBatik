"""Minimal supplement host; defaults to synthetic data, never collects remote reviews."""
from pathlib import Path
import os
import sys
import tempfile

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
from temajelajah import bi_data

private_root = os.environ.get("BI_DATA_ROOT")
if private_root:
    data_root = Path(private_root).resolve()
else:
    from bi_fixture import RUN_ID, write_fixture
    if "supplement_fixture_root" not in st.session_state:
        st.session_state["supplement_fixture_root"] = os.environ.get("BI_SYNTHETIC_ROOT") or tempfile.mkdtemp(prefix="bi-synthetic-")
    data_root = Path(st.session_state["supplement_fixture_root"])
    if not (data_root / "exports/paper_expansion/runs" / RUN_ID / "corpus.csv").is_file():
        write_fixture(data_root)

bi_data.WORKSPACE = data_root
bi_data.EXPANSION_ROOT = data_root / "exports/paper_expansion"
bi_data.LABELS_REAL = data_root / "exports/ml_experiments/dataset/real_labeled1524.csv"
bi_data.LABELS_LEGACY = data_root / "exports/ml_experiments/dataset/legacy_supplement81.csv"
bi_data.BUSINESS_CSV = data_root / "exports/business_research/dataset/business_real.csv"
st.set_page_config(page_title="Coverage-aware dashboard supplement", layout="wide")
st.caption("Supplement demonstration: " + ("separately supplied private inputs" if private_root else "synthetic fixture only"))
from temajelajah.bi_dashboard import render_dashboard
render_dashboard()
