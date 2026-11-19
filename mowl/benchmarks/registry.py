"""Registry of benchmarks and published results.

* Benchmarks live in :mod:`mowl.benchmarks.entries`, one module per benchmark family,
  exposing ``BENCHMARK`` or ``BENCHMARKS``.
* Published results live in :mod:`mowl.benchmarks.published`, one module per source
  publication, exposing ``RESULTS``.

Both packages are discovered automatically.
"""

import importlib
import pkgutil

from mowl.benchmarks.spec import Benchmark, PublishedResult, Task

_BENCHMARKS = None
_RESULTS = None


def _modules(package):
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if not info.name.startswith("_"):
            yield importlib.import_module(f"{package.__name__}.{info.name}")


def _benchmarks():
    global _BENCHMARKS
    if _BENCHMARKS is None:
        from mowl.benchmarks import entries
        found = {}
        for module in _modules(entries):
            items = list(getattr(module, "BENCHMARKS", ()))
            if hasattr(module, "BENCHMARK"):
                items.append(module.BENCHMARK)
            if not items:
                raise ImportError(f"{module.__name__} defines no BENCHMARK(S).")
            for b in items:
                if not isinstance(b, Benchmark):
                    raise TypeError(f"{module.__name__} exposes a non-Benchmark object.")
                if b.key in found:
                    raise ValueError(f"Duplicate benchmark key {b.key!r}.")
                found[b.key] = b
        _BENCHMARKS = found
    return _BENCHMARKS


def _results():
    global _RESULTS
    if _RESULTS is None:
        from mowl.benchmarks import published
        found = []
        known = _benchmarks()
        for module in _modules(published):
            for r in getattr(module, "RESULTS", ()):
                if not isinstance(r, PublishedResult):
                    raise TypeError(f"{module.__name__} exposes a non-PublishedResult.")
                if r.benchmark not in known:
                    raise ValueError(
                        f"{module.__name__}: unknown benchmark {r.benchmark!r}.")
                found.append(r)
        _RESULTS = found
    return _RESULTS


def get_benchmark(key):
    """Return the benchmark registered under ``key``."""
    benchmarks = _benchmarks()
    if key not in benchmarks:
        raise KeyError(f"Unknown benchmark {key!r}. Known: {sorted(benchmarks)}")
    return benchmarks[key]


def list_benchmarks(task=None):
    """Return all benchmarks, optionally only those of one :class:`Task`."""
    task = Task(task) if task is not None else None
    return sorted((b for b in _benchmarks().values() if task is None or b.task is task),
                  key=lambda b: b.key)


def published_results(benchmark=None, method=None, subset=None):
    """Return published results, optionally filtered by benchmark, method and subset."""
    return [r for r in _results()
            if (benchmark is None or r.benchmark == benchmark)
            and (method is None or r.method == method)
            and (subset is None or r.subset == subset)]
