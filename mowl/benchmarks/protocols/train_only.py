"""The protocol of a run that only trains (:func:`mowl.pipeline.pipeline` without an
evaluation)."""

from mowl.benchmarks.protocols.base import Protocol


class TrainOnlyProtocol(Protocol):
    """Measures nothing: :meth:`evaluate` returns no metrics."""

    key = ""

    def evaluate(self, model, dataset):
        return {}
