"""Evaluation regimes: how a trained model is evaluated, separately from the model and
from the data.

An evaluation regime fixes everything a publication leaves implicit when it says "we
evaluate on X": which axioms are queries, what the candidate set is, whether and how
known or entailed positives are filtered, and which metrics are reported. The same
regime can be applied to any compatible model and dataset; a benchmark
(:mod:`mowl.benchmarks`) is a dataset together with an evaluation regime and the results
published for that combination.

Regimes are described in the catalogue by a :class:`mowl.catalog.RegimeCard`, whether
they were published on their own or as the evaluation section of a method paper.
"""


class EvaluationRegime:
    """Base class of evaluation regimes.

    Subclasses implement :meth:`evaluate` and declare what they report.
    """

    #: Key of the :class:`mowl.catalog.RegimeCard` describing this regime.
    key = ""
    #: Names of the reported metrics, in display order.
    metrics = ()
    #: Names of the reported result subsets (for example normal forms and
    #: ``"combined"``). ``("all",)`` if there is only one.
    subsets = ("all",)

    def prepare(self, dataset):
        """Hook called once on a loaded dataset before a model is built (for example to
        restrict the evaluation classes). Returns the dataset to train on."""
        return dataset

    def initialise_model(self, model, dataset):
        """Hook called once on the built, untrained model before training (after a
        training regime, if any, is attached). A regime whose task fixes part of the
        starting point of the model uses it: the masked grounding of the DF-ALC revision
        task, the probabilistic conditionals of the BoxSEL task. Default: nothing."""

    def evaluate(self, model, dataset):
        """Evaluate ``model``, already trained on ``dataset``.

        :return: ``{subset: {metric: value}}``.
        :rtype: dict
        """
        raise NotImplementedError
