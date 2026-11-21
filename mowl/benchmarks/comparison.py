"""Run several methods on one benchmark and tabulate them side by side.

.. code-block:: python

   from mowl.benchmarks.comparison import compare_methods, methods_table

   outcome = compare_methods("common-go-subsumption-prediction", epochs=500,
                             seeds=(0, 1, 2), device="cuda", out_dir="results")
   print(methods_table(outcome.results))
   for method, reason in outcome.skipped.items():
       print(method, reason)

The same from the command line: ``python -m mowl.benchmarks compare-methods BENCHMARK``
and ``python -m mowl.benchmarks table results/*.json``.
"""

from dataclasses import dataclass, field
import logging
import os

from mowl.benchmarks.registry import get_benchmark, methods_for, missing_support

logger = logging.getLogger(__name__)

__all__ = ["Comparison", "runnable_methods", "compare_methods", "methods_table"]


@dataclass
class Comparison:
    """Outcome of :func:`compare_methods`.

    :param benchmark: Benchmark key.
    :param results: The :class:`~mowl.benchmarks.runner.BenchmarkResult` objects.
    :param skipped: Methods that were not run, with the reason.
    :param failed: ``{(method, seed): error message}`` for runs that raised.
    """

    benchmark: str
    results: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)


def runnable_methods(benchmark):
    """``(run, skipped)``: the keys of the methods that can be trained and ranked on the
    benchmark, and for every other implemented method the reason it cannot.

    A method is run if its card declares the inference tasks of the benchmark and the
    constructs of its candidates (:func:`mowl.benchmarks.missing_support`), and it is a
    trainable implementation. Exact constructions are skipped: they do not rank
    candidates, they decide them.
    """
    from mowl.catalog import list_methods
    from mowl.catalog.card import Status
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    run, skipped = [], {}
    supported = {m.key for m in methods_for(benchmark)}
    for card in list_methods():
        if not card.implemented:
            continue
        if card.key not in supported:
            skipped[card.key] = "; ".join(missing_support(card, benchmark))
        elif card.status is not Status.LEARNABLE:
            skipped[card.key] = ("an exact construction, not a trained model; it decides "
                                 "the candidates instead of ranking them")
        else:
            run.append(card.key)
    return run, skipped


def compare_methods(benchmark, methods=None, epochs=None, seeds=(0,), device="cpu",
                    model_kwargs=None, out_dir=None, dataset=None, stop_on_error=False,
                    recipes=True, **run_kwargs):
    """Train and evaluate several methods on one benchmark.

    :param benchmark: A :class:`~mowl.benchmarks.Benchmark` or its key.
    :param methods: Catalogue keys (default: :func:`runnable_methods`).
    :param epochs: Training epochs for every method (default: the runner's).
    :param seeds: Random seeds; every method is run once per seed.
    :param model_kwargs: Model arguments: one dict for all methods, or a dict from
        method key to a dict (``"*"`` for the others). Arguments a method does not take
        are an error of that run, which is recorded and does not stop the others.
    :param out_dir: Directory to which each result is written as
        ``<benchmark>.<method>.s<seed>.json``; existing files are loaded instead of
        being recomputed, so an interrupted comparison can be continued.
    :param dataset: An already loaded dataset shared by all runs.
    :param stop_on_error: Raise the first error instead of recording it.
    :param recipes: Use the recipe of a method on the benchmark where there is exactly
        one (model arguments, training regime, epochs); ``epochs`` and ``model_kwargs``
        given here take precedence. The joint benchmarks have default recipes for every
        method (:mod:`mowl.benchmarks.recipe_sets.common_defaults`).
    :param run_kwargs: Passed to :func:`mowl.benchmarks.runner.run_benchmark`.
    :rtype: :class:`Comparison`
    """
    from mowl.benchmarks.runner import BenchmarkResult, run_benchmark
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)
    default, skipped = runnable_methods(benchmark)
    if methods is None:
        methods = default
    else:
        methods = list(methods)
        skipped = {k: v for k, v in skipped.items() if k in methods}
    outcome = Comparison(benchmark.key, skipped=skipped)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_method = model_kwargs or {}
    keyed = bool(per_method) and all(isinstance(v, dict) for v in per_method.values())
    for method in methods:
        if method in skipped:
            continue
        kwargs = dict(per_method.get(method, per_method.get("*", {}))) if keyed \
            else dict(per_method)
        for seed in seeds:
            path = out_dir and os.path.join(out_dir, f"{benchmark.key}.{method}.s{seed}.json")
            if path and os.path.exists(path):
                outcome.results.append(BenchmarkResult.load(path))
                continue
            recipe = None
            if recipes and "recipe" not in run_kwargs:
                from mowl.benchmarks.recipes import list_recipes
                if len(list_recipes(benchmark.key, method)) == 1:
                    recipe = True
            try:
                result = run_benchmark(benchmark, method=method, model_kwargs=kwargs,
                                       epochs=epochs, seed=seed, device=device,
                                       dataset=dataset,
                                       **{"recipe": recipe, **run_kwargs})
            except Exception as error:                              # noqa: BLE001
                if stop_on_error:
                    raise
                logger.warning("%s on %s (seed %s) failed: %s", method, benchmark.key,
                               seed, error)
                outcome.failed[(method, seed)] = f"{type(error).__name__}: {error}"
                continue
            if path:
                result.save(path)
            outcome.results.append(result)
    return outcome


def methods_table(results, metrics=None, subset=None, fmt="text", benchmark=None):
    """One row per method: the mean of every metric over the seeds, the number of seeds,
    and how the scores were obtained.

    :param results: :class:`~mowl.benchmarks.runner.BenchmarkResult` objects of one
        benchmark.
    :param metrics: Metric names (default: those of the benchmark).
    :param subset: Result subset (default: ``"combined"`` if the benchmark has it,
        otherwise its last subset).
    :param benchmark: The :class:`~mowl.benchmarks.Benchmark`, if it is not registered.
    """
    from mowl.benchmarks import report
    results = list(results)
    if not results:
        return ""
    keys = {r.benchmark for r in results}
    if len(keys) > 1:
        raise ValueError(f"The results are of several benchmarks: {sorted(keys)}")
    bench = benchmark if benchmark is not None else get_benchmark(results[0].benchmark)
    metrics = list(metrics or bench.metrics)
    if subset is None:
        subset = "combined" if "combined" in bench.subsets else bench.subsets[-1]
    by_method = {}
    for r in results:
        by_method.setdefault(r.method or "custom model", []).append(r)
    rows = []
    for method, runs in by_method.items():
        row = [method]
        for m in metrics:
            values = [r.metrics.get(subset, {}).get(m) for r in runs]
            values = [v for v in values if v is not None]
            row.append(sum(values) / len(values) if values else None)
        scoring = sorted({str((r.protocol_info or {}).get("scoring", "")) for r in runs})
        epochs = sorted({r.epochs for r in runs})
        row += [len(runs), ", ".join(str(e) for e in epochs), ", ".join(scoring)]
        rows.append(row)
    first = metrics[0]
    better = bench.higher_is_better.get(first, True)
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0) if better else (r[1] or 0)))
    title = (f"{bench.key} ({', '.join(bench.completion_tasks)}; {bench.target.value}; "
             f"subset {subset}; mean over seeds)")
    return report.table(["method"] + metrics + ["seeds", "epochs", "scoring"], rows, fmt,
                        title)
