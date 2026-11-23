#!/usr/bin/env python
"""Collect benchmark result JSONs into the three-way table published / parts / baseline.

    python scripts/collect_benchmarks.py RESULTS_DIR [more dirs or files] \\
        --manifest manifests/repro-fast.tsv --manifest ... --logs RESULTS_DIR/logs \\
        --out report [--tolerance 0.02] [--published-tolerance 0.05]

writes ``report.md`` (summary, flagged rows, all rows), ``report.csv`` (all rows,
numbers unformatted) and ``report.rst`` (a ``list-table`` to include in the
documentation, e.g. as ``docs/source/benchmarks/_three_way.rst``). See
:mod:`mowl.benchmarks.collect` for the flags.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", nargs="+", help="result JSONs, directories or globs")
    p.add_argument("--manifest", action="append", default=[],
                   help="manifest(s) the results come from (to report missing runs)")
    p.add_argument("--logs", default=None,
                   help="directory of the run logs (a run with a log and no result failed)")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="relative difference parts vs baseline that is flagged")
    p.add_argument("--abs-tolerance", type=float, default=0.0)
    p.add_argument("--published-tolerance", type=float, default=0.05)
    p.add_argument("--plugin", action="append", default=[],
                   help="modules to import first (e.g. tests.benchmarks.toy_benchmarks)")
    p.add_argument("--title", default="Published / mOWL (parts) / mOWL (direct)")
    args = p.parse_args(argv)

    from mowl.benchmarks.collect import (collect, read_manifest, read_results,
                                         status_of_manifest, summary_counts,
                                         three_way_table, write_csv)
    from mowl.benchmarks.implementations import load_plugins
    load_plugins(args.plugin)
    results, bad = read_results(args.results)
    runs = [r for m in args.manifest for r in read_manifest(m)]
    status = {}
    if runs:
        dirs = [d for d in args.results if os.path.isdir(d)] or [os.path.dirname(
            args.results[0])]
        for d in dirs:
            for tag, state in status_of_manifest(runs, d, args.logs).items():
                if status.get(tag) != "done":
                    status[tag] = state
    rows = collect(results, runs, status, rel_tolerance=args.tolerance,
                   published_tolerance=args.published_tolerance,
                   abs_tolerance=args.abs_tolerance)
    counts = summary_counts(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lines = [f"# {args.title}", "",
             f"{len(results)} result files, {len(rows)} rows (benchmark, method, recipe, "
             f"subset, metric).", ""]
    if runs:
        states = {}
        for s in status.values():
            states[s] = states.get(s, 0) + 1
        lines += [f"Manifest runs: {len(runs)} ({', '.join(f'{k}: {v}' for k, v in sorted(states.items()))}).", ""]
    if bad:
        lines += [f"Unreadable files: {len(bad)}", ""]
    lines += ["| flag | rows |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "## Flagged rows", "", three_way_table(rows, "markdown", only_flagged=True),
              "## All rows", "", three_way_table(rows, "markdown")]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines))
    write_csv(rows, args.out + ".csv")
    with open(args.out + ".rst", "w") as f:
        f.write(three_way_table(rows, "rst", title=args.title))
    print("\n".join(lines[:8 + len(counts)]))
    print(f"wrote {args.out}.md, {args.out}.csv, {args.out}.rst")
    return 0


if __name__ == "__main__":
    sys.exit(main())
