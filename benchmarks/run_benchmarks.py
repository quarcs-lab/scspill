"""Run the scspill validation benchmarks against the frozen R results.

Usage::

    python benchmarks/run_benchmarks.py --all --report
    python benchmarks/run_benchmarks.py --case california_sar

Each case computes a metrics dict and is judged against its module's
``EXPECTED`` inclusive intervals. ``--report`` writes ``benchmarks/REPORT.md``
and ``benchmarks/report.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from registry import CASES  # noqa: E402


def run_case(name: str) -> dict:
    """Run one registered case and judge its metrics."""
    module = importlib.import_module(CASES[name])
    t0 = time.time()
    try:
        metrics = module.run()
        error = None
    except Exception:
        metrics = {}
        error = traceback.format_exc()
    elapsed = time.time() - t0

    rows = []
    ok_all = error is None
    expected = getattr(module, "EXPECTED", {})
    for metric, (lo, hi) in expected.items():
        got = metrics.get(metric)
        ok = got is not None and lo <= got <= hi
        ok_all = ok_all and ok
        rows.append({"metric": metric, "got": got, "lo": lo, "hi": hi, "ok": bool(ok)})
    extra = {k: v for k, v in metrics.items() if k not in expected}
    return {
        "case": name,
        "ok": bool(ok_all),
        "elapsed_s": round(elapsed, 1),
        "rows": rows,
        "extra": extra,
        "error": error,
    }


def write_report(results: list[dict]) -> None:
    """Write REPORT.md and report.json under benchmarks/."""
    lines = [
        "# scspill validation report",
        "",
        "Cross-validation of the Python port against the frozen results of the",
        "R replication package (see `reference/values.json` for provenance).",
        "",
        "| Case | Status | Time |",
        "| --- | --- | --- |",
    ]
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        lines.append(f"| {r['case']} | {status} | {r['elapsed_s']}s |")
    for r in results:
        lines += ["", f"## {r['case']}", ""]
        if r["error"]:
            lines += ["```", r["error"].strip(), "```"]
            continue
        lines += ["| Metric | Got | Accepted range | OK |", "| --- | --- | --- | --- |"]
        for row in r["rows"]:
            got = "-" if row["got"] is None else f"{row['got']:.4g}"
            lines.append(
                f"| {row['metric']} | {got} | [{row['lo']:.4g}, {row['hi']:.4g}] | "
                f"{'yes' if row['ok'] else 'NO'} |"
            )
        if r["extra"]:
            lines += ["", "Additional metrics:", ""]
            for k, v in r["extra"].items():
                lines.append(f"- `{k}`: {v}")
    (BENCH_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    (BENCH_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {BENCH_DIR / 'REPORT.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="case name (repeatable)")
    parser.add_argument("--all", action="store_true", help="run every registered case")
    parser.add_argument("--report", action="store_true", help="write REPORT.md/report.json")
    args = parser.parse_args()

    names = list(CASES) if args.all else (args.case or [])
    if not names:
        parser.error("pass --all or --case NAME")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {unknown}; known: {list(CASES)}")

    results = []
    for name in names:
        print(f"=== {name} ...", flush=True)
        result = run_case(name)
        results.append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"=== {name}: {status} ({result['elapsed_s']}s)")
        for row in result["rows"]:
            got = "-" if row["got"] is None else f"{row['got']:.4g}"
            print(
                f"    {row['metric']}: {got} in [{row['lo']:.4g}, {row['hi']:.4g}] "
                f"-> {'ok' if row['ok'] else 'FAIL'}"
            )
        if result["error"]:
            print(result["error"])
    if args.report:
        write_report(results)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
