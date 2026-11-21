"""Protocols of the OnT benchmarks (Yang et al., ISWC 2025) and a runner family for the
language-model based models.

:class:`OnTPredictionProtocol` and :class:`OnTInferenceProtocol` bind the regimes of
:mod:`mowl.evaluation.ont_ranking` to the ``ont-*`` benchmarks.

:class:`LanguageModelFamily` lets :func:`mowl.benchmarks.runner.run_benchmark` build and
train :class:`~mowl.models.hit.HiT` and :class:`~mowl.models.ont.OnT`. The runner does
not know it by default; register it before running::

    from mowl.benchmarks.runner import register_family
    from mowl.benchmarks.protocols.ont import LanguageModelFamily
    register_family(LanguageModelFamily())

or use :func:`main`, the command line interface of :mod:`mowl.benchmarks` with the family
registered (the JVM must be started first, because :mod:`mowl.benchmarks.protocols`
imports :mod:`mowl.evaluation`)::

    python -c "import mowl; mowl.init_jvm('16g'); \\
    from mowl.benchmarks.protocols.ont import main; main()" \\
        run ont-galen-prediction --method ont --epochs 1 --device cuda \\
        --param verbalisation_style=release --out result.json
"""

from mowl.benchmarks.protocols.bindings import RegimeProtocol


class OnTPredictionProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.ont_ranking.OnTRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.ont_ranking import OnTRankingRegime
        return OnTRankingRegime(**kwargs)


class OnTInferenceProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.ont_ranking.OnTInferenceRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.ont_ranking import OnTInferenceRankingRegime
        return OnTInferenceRankingRegime(**kwargs)


class LanguageModelFamily:
    """Runner family (:class:`mowl.benchmarks.runner.ModelFamily` interface) of
    :class:`~mowl.models.hit.HiT` and its subclasses.

    ``epochs`` is passed to ``model.train``; the model selects :math:`\\lambda` and the
    best epoch on the dataset's validation set itself, so ``select_best`` and
    ``validate_every`` have no further effect.
    """

    name = "language_model"

    def matches(self, cls):
        from mowl.models.hit.model import HiT
        return isinstance(cls, type) and issubclass(cls, HiT)

    def build(self, cls, dataset, model_kwargs, device, model_filepath):
        kwargs = dict(model_kwargs)
        kwargs.setdefault("device", device)
        kwargs.setdefault("model_filepath", model_filepath)
        return cls(dataset, **kwargs)

    def train(self, model, epochs, protocol, validate_every=None, select_best=True):
        model.train(epochs=epochs, validate=select_best)
        return {"selected": ("best validation MRR" if model.best_validation else
                             "last epoch"),
                "centri_weight": model.centri_weight,
                "best_validation": model.best_validation}


def main(argv=None):
    """The command line interface of :mod:`mowl.benchmarks` (``list``, ``show``,
    ``published``, ``run``, ``compare``) with :class:`LanguageModelFamily` registered."""
    from mowl.benchmarks.__main__ import main as cli
    from mowl.benchmarks.runner import register_family
    register_family(LanguageModelFamily())
    return cli(argv)
