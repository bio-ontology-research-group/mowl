"""Protocols that bind the evaluation regimes of :mod:`mowl.evaluation` to benchmarks.

Each protocol builds its regime from the benchmark's ``protocol_kwargs`` and reports
exactly the subsets and metrics the benchmark declares.
"""

from mowl.benchmarks.protocols.base import Protocol


class RegimeProtocol(Protocol):
    """Base class: build a regime in :meth:`build` and delegate to it.

    After :meth:`evaluate`, :attr:`info` holds the regime's details of the run (scoring
    used, numbers of queries, ...), which the runner stores with the result.
    """

    def __init__(self, benchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        self.regime = self.build(**kwargs)
        self.key = self.regime.key
        self.info = {}

    def build(self, **kwargs):
        raise NotImplementedError

    def prepare(self, dataset):
        return self.regime.prepare(dataset)

    def evaluate(self, model, dataset):
        results = self.regime.evaluate(model, dataset)
        self.info = dict(getattr(self.regime, "info", {}))
        out = {}
        for subset in self.subsets:
            if subset not in results:
                raise KeyError(f"Regime {self.key!r} did not report subset {subset!r}.")
            missing = [m for m in self.metrics if m not in results[subset]]
            if missing:
                raise KeyError(f"Regime {self.key!r} did not report {missing} for "
                               f"subset {subset!r}.")
            out[subset] = {m: results[subset][m] for m in self.metrics}
        return out


class Box2ELPredictionProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.normal_form_ranking.NormalFormRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
        return NormalFormRankingRegime(**kwargs)


class TransBoxPredictionProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.normal_form_ranking.TransBoxNormalFormRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.normal_form_ranking import TransBoxNormalFormRankingRegime
        return TransBoxNormalFormRankingRegime(**kwargs)


class InferredSubsumptionProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.normal_form_ranking.InferredSubsumptionRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.normal_form_ranking import InferredSubsumptionRankingRegime
        return InferredSubsumptionRankingRegime(**kwargs)


class ELKEmSubsumptionProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.normal_form_ranking.ELKEmSubsumptionRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.normal_form_ranking import ELKEmSubsumptionRankingRegime
        return ELKEmSubsumptionRankingRegime(**kwargs)


class PPIRankingProtocol(RegimeProtocol):
    """:func:`mowl.evaluation.filtered_ranking.ppi_ranking`."""

    def build(self, relation="http://interacts", **kwargs):
        from mowl.evaluation.filtered_ranking import ppi_ranking
        return ppi_ranking(relation, **kwargs)


class DELEProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.deductive.DELEEvaluation` (the DELE evaluation regime,
    with DELE's deductive closure)."""

    def build(self, **kwargs):
        from mowl.evaluation.deductive import DELEEvaluation
        return DELEEvaluation(**kwargs)


class CatEProtocol(RegimeProtocol):
    """:func:`mowl.evaluation.filtered_ranking.cate_ranking`."""

    def build(self, gci_name="gci0", relation=None, closure=False, queries=None,
              exclude_head=False, **kwargs):
        from mowl.evaluation.filtered_ranking import cate_ranking
        return cate_ranking(gci_name, relation=relation, closure=closure,
                            queries=queries, exclude_head=exclude_head, **kwargs)


class CatEPPIProtocol(RegimeProtocol):
    """:func:`mowl.evaluation.filtered_ranking.cate_ppi_ranking`."""

    def build(self, relation="http://interacts", **kwargs):
        from mowl.evaluation.filtered_ranking import cate_ppi_ranking
        return cate_ppi_ranking(relation, **kwargs)


class RegDProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.sampled_f1.SampledF1Regime`."""

    def build(self, **kwargs):
        from mowl.evaluation.sampled_f1 import SampledF1Regime
        return SampledF1Regime(**kwargs)


class EmELppProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.normal_form_ranking.SubsumeeRankingRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.normal_form_ranking import SubsumeeRankingRegime
        return SubsumeeRankingRegime(**kwargs)


class ExactModelProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.exact_model.ExactModelRegime`."""

    def build(self, **kwargs):
        from mowl.evaluation.exact_model import ExactModelRegime
        return ExactModelRegime(**kwargs)


class CommonRankingProtocol(RegimeProtocol):
    """:func:`mowl.evaluation.filtered_ranking.common_ranking`."""

    def build(self, gci_name="gci0", relation=None, **kwargs):
        from mowl.evaluation.filtered_ranking import common_ranking
        return common_ranking(gci_name=gci_name, relation=relation, **kwargs)


class EdgeClassificationProtocol(RegimeProtocol):
    """:class:`mowl.evaluation.edge_classification.EdgeClassificationF1Regime`."""

    def build(self, **kwargs):
        from mowl.evaluation.edge_classification import EdgeClassificationF1Regime
        return EdgeClassificationF1Regime(**kwargs)
