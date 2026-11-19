"""Base class of benchmark evaluation protocols."""

from mowl.evaluation.regime import EvaluationRegime


class Protocol(EvaluationRegime):
    """An :class:`~mowl.evaluation.regime.EvaluationRegime` bound to a benchmark.

    The evaluation logic itself should live in an ``EvaluationRegime`` in
    :mod:`mowl.evaluation` that can be used without a benchmark; a protocol only binds
    it to a :class:`~mowl.benchmarks.Benchmark` (dataset, keyword arguments) so the
    runner can call it. Small protocols may implement :meth:`evaluate` directly.

    :param benchmark: The benchmark this protocol evaluates.
    :type benchmark: :class:`mowl.benchmarks.Benchmark`
    """

    def __init__(self, benchmark, **kwargs):
        self.benchmark = benchmark
        self.kwargs = kwargs
        self.metrics = tuple(benchmark.metrics)
        self.subsets = tuple(benchmark.subsets)
