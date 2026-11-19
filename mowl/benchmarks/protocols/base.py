"""Base class of benchmark evaluation protocols."""


class Protocol:
    """Evaluates a trained model on a benchmark.

    A protocol fixes everything a publication leaves implicit when it says "we evaluate
    on X": which axioms are queries, what the candidate set is, whether and how known
    positives are filtered, and which metrics are reported.

    Subclasses implement :meth:`evaluate`. They may override :meth:`prepare` to adapt a
    freshly loaded dataset (for example to restrict the evaluation classes).

    :param benchmark: The benchmark this protocol evaluates.
    :type benchmark: :class:`mowl.benchmarks.Benchmark`
    """

    def __init__(self, benchmark, **kwargs):
        self.benchmark = benchmark
        self.kwargs = kwargs

    def prepare(self, dataset):
        """Hook called once on the loaded dataset, before a model is built. Returns the
        dataset to train on."""
        return dataset

    def evaluate(self, model, dataset):
        """Evaluate ``model``, already trained on ``dataset``.

        :return: ``{subset: {metric: value}}`` with the subsets and metric names declared
            by the benchmark (:attr:`~mowl.benchmarks.Benchmark.subsets`,
            :attr:`~mowl.benchmarks.Benchmark.metrics`).
        :rtype: dict
        """
        raise NotImplementedError
