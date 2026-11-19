"""Protocol of the BoxLitE Family benchmarks (Lourenço et al., KR 2026, Section 6 and
Appendix G.4), and what is needed to run BoxLitE with the benchmark runner.

The protocol binds :class:`mowl.evaluation.assertion_ranking.RoleAssertionRankingRegime`
(filtered head and tail ranking of the test role assertions among all individuals) to a
benchmark.

BoxLitE is solved by a conic solver, not trained for a number of epochs, so
:func:`mowl.benchmarks.runner.run_benchmark` has no built-in model family for it.
:class:`SolverModelFamily` is that family: register it with
``mowl.benchmarks.runner.register_family(SolverModelFamily())`` and ``run_benchmark``
works with ``method="boxlite"`` (``epochs`` is ignored). :func:`run_family_benchmark`
does the same without the runner and adds the paper's model selection (grid search of
the regularisation weights on the validation MRR).
"""

import itertools
import time

from mowl.benchmarks.protocols.base import Protocol
from mowl.evaluation.assertion_ranking import RoleAssertionRankingRegime

#: The grid of Appendix G.3: every lambda_i in this set.
PAPER_LAMBDA_GRID = (0.0, 0.001, 0.003, 0.1, 0.3, 1.0, 3.0)


class BoxLitEFamilyProtocol(Protocol):
    """Filtered MRR and Hits@{1,3,5,10} of the test role assertions.

    :param ties: Tie policy, see :class:`RoleAssertionRankingRegime`.
    """

    key = RoleAssertionRankingRegime.key

    def __init__(self, benchmark, ties="ordinal", **kwargs):
        super().__init__(benchmark, **kwargs)
        self.regime = RoleAssertionRankingRegime(split="test", ties=ties,
                                                 hits=(1, 3, 5, 10))
        self.info = {}

    def evaluate(self, model, dataset):
        result = self.regime.evaluate(model, dataset)
        self.info = dict(self.regime.info)
        return result


class SolverModelFamily:
    """Benchmark-runner family for models solved in one call of ``model.train()``
    (BoxLitE). Duck-types :class:`mowl.benchmarks.runner.ModelFamily`."""

    name = "solver"

    def matches(self, cls):
        from mowl.models.boxlite.model import BoxLitE
        return isinstance(cls, type) and issubclass(cls, BoxLitE)

    def build(self, cls, dataset, model_kwargs, device, model_filepath):
        kwargs = dict(model_kwargs)
        kwargs.setdefault("model_filepath", model_filepath)
        return cls(dataset, **kwargs)

    def train(self, model, epochs, protocol, validate_every=None, select_best=True):
        result = model.train()
        return {"selected": "single solve", "status": result.status,
                "objective": result.value, "solver": result.solver,
                "solve_time": result.solve_time, "lambdas": list(result.lambdas)}


def select_lambdas(model, dataset, grid=PAPER_LAMBDA_GRID, ties="ordinal", callback=None):
    """The paper's model selection (Appendix G.3-G.4): solve for every
    :math:`(\\lambda_1, \\lambda_2, \\lambda_3) \\in` ``grid``:sup:`3` and keep the
    embedding with the best filtered validation MRR. The programme is compiled once.

    :param grid: Values of each :math:`\\lambda_i`, or an iterable of triples.
    :param callback: Called as ``callback(lambdas, metrics)`` after each solve.
    :return: ``(best lambdas, best validation metrics, all results)``; the model keeps
        the best embedding.
    """
    triples = list(grid) if grid and isinstance(next(iter(grid)), (tuple, list)) \
        else list(itertools.product(grid, repeat=3))
    regime = RoleAssertionRankingRegime(split="valid", ties=ties, hits=(1, 3, 5, 10))
    best, results = None, []
    for lambdas in triples:
        try:
            model.train(lambdas=lambdas)
        except RuntimeError as exc:
            results.append((tuple(lambdas), {"error": str(exc)}))
            continue
        metrics = regime.evaluate(model, dataset)["all"]
        results.append((tuple(lambdas), metrics))
        if callback is not None:
            callback(lambdas, metrics)
        if best is None or metrics["mrr"] > best[1]["mrr"]:
            best = (tuple(lambdas), metrics, model.embedding)
    if best is None:
        raise RuntimeError("No configuration was solved.")
    model.embedding, model.lambdas = best[2], best[0]
    return best[0], best[1], results


def run_family_benchmark(benchmark, model_kwargs=None, grid=None, dataset=None,
                         dataset_kwargs=None, ties="ordinal"):
    """Run BoxLitE on a ``boxlite-family-v*`` benchmark without the runner.

    :param benchmark: Benchmark key or object.
    :param model_kwargs: Keyword arguments of :class:`mowl.models.boxlite.BoxLitE`
        (``embed_dim``, ``lambda1``, ..., ``variant``, ``solver``).
    :param grid: ``None`` (solve once with the given lambdas) or a grid for
        :func:`select_lambdas`.
    :return: A :class:`mowl.benchmarks.runner.BenchmarkResult`.
    """
    from mowl.benchmarks.registry import get_benchmark
    from mowl.benchmarks.runner import BenchmarkResult, environment, _jsonable
    from mowl.benchmarks.spec import Benchmark
    from mowl.models.boxlite.model import BoxLitE

    t0 = time.time()
    if not isinstance(benchmark, Benchmark):
        benchmark = get_benchmark(benchmark)
    d_kwargs = {**benchmark.dataset_kwargs, **(dataset_kwargs or {})}
    if dataset is None:
        dataset = Benchmark._resolve(benchmark.dataset)(**d_kwargs)
    model_kwargs = dict(model_kwargs or {})
    model = BoxLitE(dataset, **model_kwargs)
    t_load = time.time()
    if grid is None:
        result = model.train()
        selection = {"status": result.status, "objective": result.value,
                     "solver": result.solver, "solve_time": result.solve_time}
    else:
        lambdas, metrics, results = select_lambdas(model, dataset, grid, ties=ties)
        selection = {"best_lambdas": list(lambdas), "validation": metrics,
                     "configurations": len(results)}
    t_train = time.time()
    protocol = BoxLitEFamilyProtocol(benchmark, ties=ties)
    metrics = protocol.evaluate(model, dataset)
    t_eval = time.time()
    settings = dict(model_kwargs)
    settings.update({"embed_dim": model.embed_dim, "s_omega": model.s_omega,
                     "epsilon": model.epsilon, "lambda_1": model.lambdas[0],
                     "lambda_2": model.lambdas[1], "lambda_3": model.lambdas[2],
                     "variant": model.variant,
                     "model_class": "mowl.models.boxlite.model.BoxLitE"})
    info = dict(protocol.info)
    info["training"] = selection
    info["skipped_axioms"] = dict(model.kb.skipped)
    return BenchmarkResult(
        benchmark=benchmark.key, method="boxlite", metrics=_jsonable(metrics),
        settings=_jsonable(settings), epochs=0, seed=0, device="cpu",
        evaluation_regime=protocol.key, training_regime=None,
        dataset_kwargs=_jsonable(d_kwargs), protocol_kwargs={"ties": ties},
        protocol_info=_jsonable(info),
        wall_time={"load": t_load - t0, "train": t_train - t_load,
                   "evaluate": t_eval - t_train, "total": t_eval - t0},
        environment=environment())
