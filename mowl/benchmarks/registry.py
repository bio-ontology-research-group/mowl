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


def list_benchmarks(task=None, completion_task=None, inference_task=None, target=None,
                    assumption=None,
                    method=None):
    """Return the benchmarks, optionally only those that match every given filter.

    :param task: A :class:`Task` (coarse group) or its value. For convenience, the key of
        a completion task or of an inference task is accepted here as well.
    :param completion_task: Key of a completion task (:data:`mowl.logic.COMPLETION_TASKS`)
        the benchmark scores, directly or in another form.
    :param inference_task: Key of an inference task (:data:`mowl.logic.TASKS`).
    :param target: A :class:`mowl.logic.Target` or its value.
    :param assumption: A :class:`mowl.logic.WorldAssumption` or its value: keep the
        benchmarks whose evaluation regime makes this assumption.
    :param method: Catalogue key of a method: only benchmarks whose inference tasks the
        method declares (:func:`supports`).
    """
    from mowl.logic.tasks import COMPLETION_TASKS, TASKS, Target
    group = None
    if task is not None:
        try:
            group = Task(task)
        except ValueError:
            if task in COMPLETION_TASKS:
                completion_task = task
            elif task in TASKS:
                inference_task = task
            else:
                raise ValueError(
                    f"Unknown task {task!r}. Benchmark groups: {[t.value for t in Task]}; "
                    f"completion tasks: {sorted(COMPLETION_TASKS)}; inference tasks: "
                    f"{sorted(TASKS)}") from None
    if completion_task is not None and completion_task not in COMPLETION_TASKS:
        raise ValueError(f"Unknown completion task {completion_task!r}.")
    if inference_task is not None and inference_task not in TASKS:
        raise ValueError(f"Unknown inference task {inference_task!r}.")
    target = Target(target) if target is not None else None
    if assumption is not None:
        from mowl.logic import WorldAssumption
        assumption = WorldAssumption(assumption)
    out = []
    for b in _benchmarks().values():
        if group is not None and b.task is not group:
            continue
        if completion_task is not None and not b.measures(completion_task):
            continue
        if inference_task is not None and not b.measures(inference_task):
            continue
        if target is not None and b.target is not target:
            continue
        if assumption is not None and b.assumption is not assumption:
            continue
        if method is not None and not supports(method, b):
            continue
        out.append(b)
    return sorted(out, key=lambda b: b.key)


def missing_support(method, benchmark):
    """What a method lacks for a benchmark, as a list of strings (empty: nothing).

    Compares data only: the ``tasks`` and the expressivity of the method card with the
    :attr:`~mowl.benchmarks.Benchmark.inference_tasks` of the benchmark and the constructs
    its candidates use (``needs`` of its completion tasks). Whether the benchmark's
    protocol can obtain scores from the model is a property of the protocol. Tasks a
    method reaches only through a scoring head on its vectors (``head_tasks``) count.

    :param method: A :class:`mowl.catalog.MethodCard` or its key.
    :param benchmark: A :class:`Benchmark` or its key.
    """
    from mowl.catalog import get_method
    from mowl.logic import Construct
    from mowl.logic.tasks import COMPLETION_TASKS
    card = get_method(method) if isinstance(method, str) else method
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    tasks = tuple(card.tasks) + tuple(card.head_tasks)
    if not tasks:
        return ["the method card declares no inference task"]
    out = [f"inference task {t!r}" for t in benchmark.inference_tasks if t not in tasks]
    try:
        expressivity = card.supported_expressivity
    except ValueError:
        expressivity = None
    if expressivity is not None:
        for t in benchmark.completion_tasks:
            for name in COMPLETION_TASKS[t].needs:
                if Construct[name] not in expressivity:
                    out.append(f"construct {Construct[name].value} (for {t})")
    for ability in benchmark.requires:
        if ability == "unseen names":
            rep = card.representation
            if rep is None or not rep.inductive:
                out.append("embeddings for names not seen in training (the method looks "
                           "its embeddings up in a table)")
        else:
            out.append(f"ability {ability!r}")
    return list(dict.fromkeys(out))


def supports(method, benchmark):
    """Whether a method declares every inference task of a benchmark and has the
    constructs its candidates use (:func:`missing_support` is empty)."""
    return not missing_support(method, benchmark)


def methods_for(benchmark, implemented=True):
    """The method cards that declare every inference task of a benchmark.

    :param implemented: Only methods that have an implementation in mOWL.
    """
    from mowl.catalog import list_methods
    return [m for m in list_methods()
            if supports(m, benchmark) and (not implemented or m.implemented)]


def benchmarks_for(method):
    """The benchmarks whose inference tasks a method declares."""
    return list_benchmarks(method=method)


def published_results(benchmark=None, method=None, subset=None):
    """Return published results, optionally filtered by benchmark, method and subset."""
    return [r for r in _results()
            if (benchmark is None or r.benchmark == benchmark)
            and (method is None or r.method == method)
            and (subset is None or r.subset == subset)]
