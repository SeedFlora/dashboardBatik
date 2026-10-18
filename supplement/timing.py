"""New reconstruction of the recorded browser protocol; original timing script not recovered.

Default synthetic runs are functional timing checks, not replication of the real corpus.
The script starts and stops only its own Streamlit process. Browser requests are
restricted to loopback. No historical performance values are inserted into results.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen


READY_JS = r"""({metrics, charts, sparkline, redraw}) => {
  const visible = el => { const r=el.getBoundingClientRect(), s=getComputedStyle(el);
    return r.width>0 && r.height>0 && s.display!=='none' && s.visibility!=='hidden'; };
  const app=document.querySelector('[data-testid="stApp"]');
  const state=app?.getAttribute('data-test-script-state');
  const metricsShown=[...document.querySelectorAll('[data-testid="stMetric"]')].filter(visible);
  const plotted=[...document.querySelectorAll('[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"]')].filter(visible);
  const generic=[...document.querySelectorAll('.vega-embed')].filter(visible);
  const plots=plotted.length ? plotted : generic;
  const drawn=plots.length>=charts && plots.every(p=>p.querySelector('svg.marks'));
  const spinning=[...document.querySelectorAll('[data-testid="stSpinner"]')].some(visible);
  const newSvg=!redraw || (window.__biTimingSvgBefore || []).every(svg=>!document.contains(svg));
  const spark=!sparkline || metricsShown.some(el=>el.querySelector('svg,canvas'));
  return state==='notRunning' && metricsShown.length>=metrics && drawn && !spinning && newSvg && spark;
}"""


def summary(values):
    return {"n": len(values), "runs_s": values, "median_s": statistics.median(values),
            "min_s": min(values), "max_s": max(values)}


def redact_local_paths(value):
    """Keep distributable diagnostics free of machine-specific filesystem locations."""
    return re.sub(r'[A-Za-z]:[\\/][^"\r\n]*', '<local-path>', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/timing-synthetic.json"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--data-root", type=Path, help="Separately supplied private data; never bundled or uploaded.")
    parser.add_argument("--min-visible-charts", type=int, default=2, help="Synthetic revised default=2; historical original criterion was at least 3.")
    parser.add_argument("--expected-metrics", type=int, default=5)
    parser.add_argument("--require-sparkline", action="store_true", help="Original protocol required a KPI sparkline; optional for the tiny fixture.")
    parser.add_argument("--filter-label", default="Airbnb")
    parser.add_argument("--filter-selector", default=".st-key-bi_sources button")
    args = parser.parse_args()
    if args.runs < 1 or not 1024 <= args.port <= 65535:
        parser.error("runs must be positive and port must be 1024-65535")
    from playwright.sync_api import sync_playwright
    root = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    start_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    command = [sys.executable, "-m", "streamlit", "run", "launch_dashboard.py", "--server.address=127.0.0.1",
               f"--server.port={args.port}", "--server.headless=true", "--server.fileWatcherType=none",
               "--browser.gatherUsageStats=false"]
    external_requests, page_errors = [], []
    process = None
    result = {
        "kind": "newly measured reconstruction; not the original historical script",
        "data": "separately supplied private inputs" if args.data_root else "synthetic fixture",
        "historical_equivalence": False,
        "protocol": {"server": "new dedicated loopback Streamlit process; minimal supplement launcher",
                     "fresh_sessions": "first measured navigation after server readiness; then fresh contexts with warm server data cache",
                     "filter_changes": "further fresh context, alternating one source pill on/off",
                     "completion": "first animation frame after script notRunning, metric/chart counts reached, all visible chart SVG marks drawn and no spinner",
                     "minimum_chart_count": args.min_visible_charts, "expected_metric_count": args.expected_metrics,
                     "require_kpi_sparkline": args.require_sparkline,
                     "historical_differences": ["Minimal launcher instead of complete original app shell.",
                                                 "Default synthetic fixture is smaller than the real collection.",
                                                 "Default chart minimum is 2; original recorded criterion was 3.",
                                                 "KPI sparkline is optional unless --require-sparkline is provided."]},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "packages": {name: importlib.metadata.version(name) for name in ("streamlit", "pandas", "numpy", "playwright")}},
    }
    options = {"metrics": args.expected_metrics, "charts": args.min_visible_charts,
               "sparkline": args.require_sparkline, "redraw": False}
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))  # refuse to measure an unrelated process on an occupied port
        with tempfile.TemporaryDirectory(prefix="bi-synthetic-timing-") as scratch:
            if args.data_root:
                environment["BI_DATA_ROOT"] = str(args.data_root.resolve())
            else:
                environment.pop("BI_DATA_ROOT", None)
                environment["BI_SYNTHETIC_ROOT"] = scratch
            with (args.output.parent / "timing-server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, **start_options)
                deadline = time.monotonic() + args.timeout_s
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Dedicated Streamlit process exited; inspect timing-server.log")
                    try:
                        with urlopen(url + "_stcore/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Streamlit readiness timed out")
                    time.sleep(0.1)
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True)
                    result["environment"]["browser"] = browser.version

                    def new_context():
                        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                                                      service_workers="block")
                        def route_request(route):
                            parsed = urlparse(route.request.url)
                            if parsed.scheme in ("about", "blob", "data") or parsed.hostname in ("127.0.0.1", "localhost", "::1"):
                                route.continue_()
                            else:
                                external_requests.append({"scheme": parsed.scheme, "host": parsed.hostname})
                                route.abort()
                        context.route("**/*", route_request)
                        context.on("page", lambda page: page.on("pageerror", lambda error: page_errors.append(str(error))))
                        return context

                    def load(context):
                        page = context.new_page()
                        page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_s * 1000)
                        page.wait_for_function(READY_JS, arg=options, timeout=args.timeout_s * 1000)
                        measured = page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve(performance.now()/1000)))")
                        return page, measured

                    context = new_context()
                    _, first = load(context)
                    result["first_after_server_start_s"] = {"n": 1, "value": first}
                    context.close()
                    sessions = []
                    for _ in range(args.runs):
                        context = new_context()
                        _, measured = load(context)
                        sessions.append(measured)
                        context.close()
                    result["fresh_browser_sessions_warm_server_cache"] = summary(sessions)
                    context = new_context()
                    page, _ = load(context)
                    reruns = []
                    for _ in range(args.runs):
                        pill = page.locator(args.filter_selector).filter(has_text=re.compile(r"^" + re.escape(args.filter_label) + r"$")).first
                        pill.wait_for(state="visible", timeout=args.timeout_s * 1000)
                        page.evaluate("() => { window.__biTimingSvgBefore=[...document.querySelectorAll('.vega-embed svg.marks')]; window.__biTimingClick=performance.now(); }")
                        pill.click()
                        page.wait_for_function(READY_JS, arg={**options, "redraw": True}, timeout=args.timeout_s * 1000)
                        measured = page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve((performance.now()-window.__biTimingClick)/1000)))")
                        reruns.append(measured)
                    result["filter_reruns"] = summary(reruns)
                    context.close()
                    browser.close()
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = redact_local_paths(f"{type(error).__name__}: {error}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        result["blocked_external_requests"] = external_requests
        result["page_errors"] = page_errors
        log_path = args.output.parent / "timing-server.log"
        if log_path.is_file():
            log_path.write_text(redact_local_paths(log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": args.output.name}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
