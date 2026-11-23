"""Collect benchmark results into a three-way table: published / mOWL (parts) / mOWL (direct).

The full-scale benchmark runs every method on every benchmark with published numbers
twice, with the same recipe and seed: as the combination of parts that users get
(implementation ``"parts"``) and as the direct implementation kept for comparison
(``"baseline"``, :mod:`mowl.baselines`). :func:`collect` groups the result files by
(benchmark, method, recipe), averages over seeds, puts the published value of the
recipe's publication next to them and flags what needs a look:

* ``parts≠baseline``: the two implementations differ by more than the tolerance (by
  default 2% of the baseline value with one seed; with several seeds the larger of that
  and the seed standard deviation);
* ``parts far from published`` / ``baseline far from published``: more than 5% (relative)
  from the published value;
* ``missing parts`` / ``missing baseline``: a run of the manifest has no result;
* ``failed parts`` / ``failed baseline``: a run has a log but no result.

.. testcode::

   from mowl.benchmarks.collect import collect, three_way_table

   rows = collect([])            # paths of result JSONs, or BenchmarkResult objects
   print(three_way_table(rows, fmt="markdown").splitlines()[0])

.. testoutput::

   | benchmark | method | recipe | subset | metric | published | parts | baseline | parts − baseline | parts vs published | baseline vs published | flags |

From the command line: ``python scripts/collect_benchmarks.py results/ --manifest m.tsv
--out report`` writes ``report.md``, ``report.csv`` and ``report.rst``.
"""

import collections
import csv
import glob
import io
import json
import math
import os

from mowl.benchmarks.report import _fmt, table

__all__ = ["collect", "three_way_table", "write_csv", "read_results", "read_manifest",
           "status_of_manifest", "summary_counts", "to_csv_string", "HEADER"]

#: Columns of :func:`three_way_table`.
HEADER = ["benchmark", "method", "recipe", "subset", "metric", "published", "parts",
          "baseline", "parts − baseline", "parts vs published", "baseline vs published",
          "flags"]

IMPLEMENTATIONS = ("parts", "baseline")


def read_results(paths):
    """Result dicts from JSON files, directories of them (``*.json``) or globs. Files
    that cannot be read are reported in the second return value."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        elif any(c in p for c in "*?["):
            files += sorted(glob.glob(p))
        else:
            files.append(p)
    out, bad = [], {}
    for f in files:
        try:
            with open(f) as handle:
                data = json.load(handle)
        except (OSError, ValueError) as error:
            bad[f] = str(error)
            continue
        data.setdefault("_path", f)
        out.append(data)
    return out, bad


def read_manifest(path):
    """The runs of a manifest: dicts with ``tag``, ``benchmark``, ``method``, ``epochs``,
    ``seed``, ``extra``, and ``recipe`` and ``implementation`` read from ``extra``."""
    runs = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t") + [""] * 6
            tag, bench, method, epochs, seed, extra = fields[:6]
            args = extra.split()
            recipe = None
            impl = "parts"
            for i, a in enumerate(args):
                if a == "--recipe":
                    nxt = args[i + 1] if i + 1 < len(args) else ""
                    recipe = nxt if nxt and not nxt.startswith("--") else True
                elif a == "--implementation" and i + 1 < len(args):
                    impl = args[i + 1]
            runs.append({"tag": tag, "benchmark": bench, "method": method,
                         "epochs": epochs, "seed": int(seed) if seed else 0,
                         "extra": extra, "recipe": _recipe_label(recipe),
                         "implementation": impl})
    return runs


def status_of_manifest(manifest_runs, results_dir, logs_dir=None):
    """``{tag: "done" | "failed" | "missing"}``: a run is done if ``<tag>.json`` is in
    ``results_dir``, failed if only its log (``<logs_dir>/<tag>.log`` or
    ``run-<tag>.log``) exists, missing otherwise."""
    out = {}
    for run in manifest_runs:
        tag = run["tag"]
        if os.path.exists(os.path.join(results_dir, tag + ".json")):
            out[tag] = "done"
            continue
        logs = [os.path.join(d, n) for d in filter(None, (logs_dir,))
                for n in (tag + ".log", f"run-{tag}.log")]
        out[tag] = "failed" if any(os.path.exists(p) for p in logs) else "missing"
    return out


def _recipe_label(recipe):
    if recipe is None or recipe is False:
        return "-"
    if recipe is True:
        return "recipe"
    if isinstance(recipe, dict):
        return recipe.get("source", "") + (f":{recipe['name']}" if recipe.get("name")
                                           else "")
    return str(recipe)


def _implementation(result):
    impl = result.get("implementation") or {}
    if isinstance(impl, dict):
        return impl.get("name", "parts")
    return str(impl)


def _stats(values):
    values = [v for v in values if isinstance(v, (int, float)) and not
              (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None, len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var), len(values)


def _rel(a, b):
    if a is None or b is None:
        return None
    if b == 0:
        return 0.0 if a == 0 else math.inf
    return (a - b) / abs(b)


def collect(results, manifest_runs=None, status=None, rel_tolerance=0.02,
            published_tolerance=0.05, abs_tolerance=0.0):
    """One row per (benchmark, method, recipe, subset, metric).

    :param results: Result dicts (as written by
        :meth:`~mowl.benchmarks.runner.BenchmarkResult.save`), paths, or
        :class:`~mowl.benchmarks.runner.BenchmarkResult` objects.
    :param manifest_runs: The runs of the manifests (:func:`read_manifest`), to report
        missing and failed runs.
    :param status: ``{tag: status}`` (:func:`status_of_manifest`).
    :param rel_tolerance: Relative difference parts vs baseline that is flagged (with
        several seeds, the seed standard deviation if larger).
    :param published_tolerance: Relative difference from the published value that is
        flagged.
    :param abs_tolerance: Absolute differences up to this are never flagged (0: flag any
        difference above the relative tolerance).
    :return: list of dicts with the keys of :data:`HEADER` (numbers unformatted) and
        ``seeds`` (``{implementation: n}``), ``sd`` and the raw differences.
    """
    from mowl.benchmarks.registry import get_benchmark, published_results
    loaded = []
    for r in results:
        if isinstance(r, str):
            with open(r) as f:
                r = json.load(f)
        elif hasattr(r, "to_dict"):
            r = r.to_dict()
        loaded.append(r)

    groups = collections.defaultdict(lambda: {i: [] for i in IMPLEMENTATIONS})
    for r in loaded:
        key = (r["benchmark"], r.get("method"), _recipe_label(r.get("recipe")))
        groups[key].setdefault(_implementation(r), []).append(r)

    expected = collections.defaultdict(set)
    run_state = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in manifest_runs or ():
        key = (run["benchmark"], run["method"], run["recipe"])
        expected[key].add(run["implementation"])
        state = (status or {}).get(run["tag"], "unknown")
        run_state[key][run["implementation"]].append(state)
        groups.setdefault(key, {i: [] for i in IMPLEMENTATIONS})

    rows = []
    for (bench_key, method, recipe), by_impl in sorted(groups.items(),
                                                       key=lambda kv: tuple(map(str, kv[0]))):
        try:
            bench = get_benchmark(bench_key)
            subsets, metrics = list(bench.subsets), list(bench.metrics)
        except KeyError:
            bench, subsets, metrics = None, [], []
        source = recipe.split(":")[0] if recipe not in ("-", "recipe") else None
        published = {}
        if bench is not None and method:
            for pub in published_results(benchmark=bench_key, method=method):
                if source is None or pub.source.key == source:
                    for m, v in pub.metrics.items():
                        published.setdefault((pub.subset, m), v)
        group_flags = []
        for impl in sorted(expected.get((bench_key, method, recipe), ())):
            states = run_state[(bench_key, method, recipe)][impl]
            if "failed" in states:
                group_flags.append(f"failed {impl}")
            elif any(s in ("missing", "unknown") for s in states) and not by_impl.get(impl):
                group_flags.append(f"missing {impl}")
        present = [s for s in subsets if any(s in (r.get("metrics") or {})
                                             for rs in by_impl.values() for r in rs)]
        if published:
            present = [s for s in present if any((s, m) in published for m in metrics)] \
                or present
        if not present:
            rows.append(_row(bench_key, method, recipe, "-", "-", None, (None, None, 0),
                             (None, None, 0), group_flags, rel_tolerance,
                             published_tolerance, abs_tolerance))
            continue
        for subset in present:
            for metric in metrics:
                if published and (subset, metric) not in published:
                    continue
                stats = {impl: _stats([(r.get("metrics") or {}).get(subset, {}).get(metric)
                                       for r in by_impl.get(impl, [])])
                         for impl in IMPLEMENTATIONS}
                rows.append(_row(bench_key, method, recipe, subset, metric,
                                 published.get((subset, metric)), stats["parts"],
                                 stats["baseline"], group_flags, rel_tolerance,
                                 published_tolerance, abs_tolerance))
    return rows


def _row(bench, method, recipe, subset, metric, published, parts, baseline, group_flags,
         rel_tolerance, published_tolerance, abs_tolerance):
    p, p_sd, p_n = parts
    b, b_sd, b_n = baseline
    flags = list(group_flags)
    diff = None if p is None or b is None else p - b
    if diff is not None:
        tol = rel_tolerance * abs(b)
        sds = [s for s in (p_sd, b_sd) if s is not None]
        if sds:
            tol = max(tol, max(sds))
        if abs(diff) > max(tol, abs_tolerance):
            flags.append("parts≠baseline")
    for name, value in (("parts", p), ("baseline", b)):
        rel = _rel(value, published)
        if rel is not None and abs(rel) > published_tolerance:
            flags.append(f"{name} far from published")
    return {"benchmark": bench, "method": method, "recipe": recipe, "subset": subset,
            "metric": metric, "published": published, "parts": p, "baseline": b,
            "parts − baseline": diff, "parts vs published": _rel(p, published),
            "baseline vs published": _rel(b, published),
            "flags": ", ".join(flags), "seeds": {"parts": p_n, "baseline": b_n},
            "sd": {"parts": p_sd, "baseline": b_sd},
            "parts − published": None if p is None or published is None else p - published,
            "baseline − published": None if b is None or published is None
            else b - published}


def _pct(value):
    if value is None:
        return None
    if math.isinf(value):
        return "inf"
    return f"{100 * value:+.1f}%"


def _with_sd(mean, sd, n):
    if mean is None:
        return None
    text = _fmt(float(mean))
    if sd is not None:
        text += f" ± {_fmt(float(sd))}"
    if n and n > 1:
        text += f" (n={n})"
    return text


def three_way_table(rows, fmt="text", title=None, only_flagged=False):
    """Render the rows of :func:`collect` (``fmt``: ``"text"``, ``"markdown"``,
    ``"rst"``)."""
    body = []
    for r in rows:
        if only_flagged and not r["flags"]:
            continue
        body.append([r["benchmark"], r["method"], r["recipe"], r["subset"], r["metric"],
                     r["published"],
                     _with_sd(r["parts"], r["sd"]["parts"], r["seeds"]["parts"]),
                     _with_sd(r["baseline"], r["sd"]["baseline"], r["seeds"]["baseline"]),
                     r["parts − baseline"], _pct(r["parts vs published"]),
                     _pct(r["baseline vs published"]), r["flags"]])
    return table(HEADER, body, fmt, title)


def write_csv(rows, path_or_file):
    """Write the rows of :func:`collect` as CSV (numbers unformatted)."""
    columns = HEADER[:-1] + ["parts − published", "baseline − published", "seeds parts",
                             "seeds baseline", "sd parts", "sd baseline", "flags"]
    handle = open(path_or_file, "w", newline="") if isinstance(path_or_file, str) \
        else path_or_file
    try:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in rows:
            extra = {"seeds parts": r["seeds"]["parts"],
                     "seeds baseline": r["seeds"]["baseline"],
                     "sd parts": r["sd"]["parts"], "sd baseline": r["sd"]["baseline"]}
            writer.writerow([extra.get(c, r.get(c)) if r.get(c) is not None or c in extra
                             else "" for c in columns])
    finally:
        if isinstance(path_or_file, str):
            handle.close()


def summary_counts(rows):
    """``{flag: number of rows}`` and the number of rows without a flag."""
    counts = collections.Counter()
    for r in rows:
        if not r["flags"]:
            counts["agree"] += 1
        for f in filter(None, (x.strip() for x in r["flags"].split(","))):
            counts[f] += 1
    return dict(counts)


def to_csv_string(rows):
    buffer = io.StringIO()
    write_csv(rows, buffer)
    return buffer.getvalue()
