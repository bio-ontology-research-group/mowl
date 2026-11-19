"""Tables of benchmark results, published results and comparisons as plain text,
Markdown or reStructuredText.

:func:`published_to_rst` renders the whole registry of published results; the
documentation calls it from ``docs/source/conf.py``.
"""

from mowl.catalog.render import rst_escape
from mowl.benchmarks.registry import get_benchmark, list_benchmarks, published_results

FORMATS = ("text", "markdown", "rst")


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if abs(value) >= 1000 or value == int(value):
            return f"{value:.0f}"
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_fmt(v[0])} vs {_fmt(v[1])}" if isinstance(v, tuple)
                         else f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "+".join(_fmt(v) for v in value)
    return str(value)


def table(header, rows, fmt="text", title=None):
    """Render a table.

    :param header: Column names.
    :param rows: Rows (sequences of cells; cells are formatted with a compact number
        format).
    :param fmt: ``"text"``, ``"markdown"`` or ``"rst"`` (an RST ``list-table``).
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    cells = [[_fmt(c) for c in r] for r in rows]
    header = [str(h) for h in header]
    if fmt == "markdown":
        out = [f"**{title}**", ""] if title else []
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join("---" for _ in header) + "|")
        out += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in cells]
        return "\n".join(out) + "\n"
    if fmt == "rst":
        out = [f".. list-table:: {title}" if title else ".. list-table::",
               "   :header-rows: 1", ""]
        for r in [header] + cells:
            r = [rst_escape(c) for c in r]
            out.append("   * - " + (r[0] or " "))
            out += [f"     - {c or ' '}" for c in r[1:]]
        return "\n".join(out) + "\n"
    widths = [max(len(x) for x in col) for col in zip(header, *cells)] if cells else \
        [len(h) for h in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [title, ""] if title else []
    out += [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join(out) + "\n"


def result_table(result, fmt="text"):
    """The metrics of a :class:`~mowl.benchmarks.runner.BenchmarkResult`, one row per
    subset."""
    bench = get_benchmark(result.benchmark)
    rows = [[s] + [result.metrics.get(s, {}).get(m) for m in bench.metrics]
            for s in bench.subsets if s in result.metrics]
    title = (f"{result.benchmark} / {result.method or 'custom model'} (seed {result.seed}, "
             f"{result.epochs} epochs, evaluation regime {result.evaluation_regime}, "
             f"training regime {_fmt(result.training_regime)})")
    return table(["subset"] + list(bench.metrics), rows, fmt, title)


def published_table(benchmark, fmt="text", method=None):
    """The published results of one benchmark: one row per (subset, method, source)."""
    bench = get_benchmark(benchmark)
    results = published_results(benchmark=bench.key, method=method)
    order = {s: i for i, s in enumerate(bench.subsets)}
    results = sorted(results, key=lambda r: (order.get(r.subset, 99), r.source.key,
                                             r.locator, r.method))
    rows = []
    for r in results:
        regime = r.settings.get("training_regime")
        label = r.method + (f" [{_fmt(regime)}]" if regime else "")
        rows.append([r.subset, label, *[r.metrics.get(m) for m in bench.metrics],
                     f"{r.source.key}, {r.locator}"])
    return table(["subset", "method"] + list(bench.metrics) + ["source"], rows, fmt,
                 f"{bench.key}: {bench.name}")


def comparison_table(rows, fmt="text"):
    """Render the rows of :func:`mowl.benchmarks.runner.compare`."""
    body = []
    for r in rows:
        regime = "" if r["regime_match"] else \
            f"ours {_fmt(r['training_regime'][0])} vs {_fmt(r['training_regime'][1])}"
        body.append([r["subset"], r["method"], r["metric"], r["ours"], r["published"],
                     r["difference"], f"{r['source']}, {r['locator']}",
                     r["settings_differences"] or "", regime])
    return table(["subset", "method", "metric", "ours", "published", "diff", "source",
                  "settings differences", "training regime differs"], body, fmt)


def published_to_rst(heading_char="~"):
    """All benchmarks with their published results as reStructuredText (one section per
    benchmark: description, notes and a table of published numbers)."""
    out = []
    for bench in list_benchmarks():
        title = f"{bench.key}"
        out += [title, heading_char * len(title), "", f"**{rst_escape(bench.name)}** "
                f"(task ``{bench.task.value}``). {rst_escape(bench.description)}", ""]
        if bench.notes:
            out += [f"*Notes.* {rst_escape(bench.notes)}", ""]
        if published_results(benchmark=bench.key):
            out.append(published_table(bench.key, fmt="rst"))
        else:
            out += ["No published results are recorded for this benchmark.", ""]
    return "\n".join(out)
