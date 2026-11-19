"""Rank-based evaluation of one axiom type with raw and filtered metrics.

This regime covers the link-prediction style evaluations of the
:math:`\\mathcal{EL}^{++}` embedding literature that rank one side of one axiom type
(:math:`C \\sqsubseteq D`, :math:`\\{P_1\\} \\sqsubseteq \\exists r.\\{P_2\\}`, or a class
assertion :math:`C(a)`) over a candidate set and optionally filter known positives:

* the PPI evaluation of Box\\ :sup:`2`\\ EL (Jackermeier et al., WWW 2024, Section 4.3;
  :data:`PPI_RANKING`),
* the evaluations of CatE (Zhapa-Camacho and Hoehndorf, Section 4.1).

The configurations used by the benchmarks are built with :func:`ppi_ranking` and
:func:`cate_ranking`. The evaluation of DELE, which also reports per-head ("micro")
metrics and filters with DELE's deductive closure, is
:class:`mowl.evaluation.deductive.DELEEvaluation`; the ``micro`` option here follows
the same definition.
"""

import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import (el_rows, candidate_indices, rank_queries,
                                           build_filter, closure_rows, COLUMNS)
from mowl.evaluation.el_scoring import resolve_scorer
from mowl.evaluation.ranking_metrics import summarise_ranks, micro_metrics


def _metric_names(hits, micro, raw, filtered):
    base = [f"hits@{k}" for k in hits] + ["med", "mrr", "mr", "auc"]
    if micro:
        base += ["micro_mr", "micro_auc"]
    out = []
    if raw:
        out += base
    if filtered:
        out += [f"f_{m}" for m in base]
    return tuple(out)


class FilteredRankingRegime(EvaluationRegime):
    """Rank one entity of the test axioms of one normal form among a candidate set,
    with raw and filtered metrics.

    :param gci_name: Normal form of the test axioms (``"gci0"``, ``"gci2"``,
        ``"class_assertion"``).
    :param target: ``"tail"`` (rank :math:`D` in :math:`C \\sqsubseteq D` /
        :math:`C \\sqsubseteq \\exists r.D`, or :math:`C` in :math:`C(a)`) or ``"head"``
        (rank :math:`C`).
    :param relation: For ``gci2``: only test axioms with this role IRI are queries.
    :param candidates: ``"evaluation_classes"`` (the dataset's
        ``evaluation_classes``: tails for ``target="tail"``, heads otherwise),
        ``"classes"`` (all classes of the dataset) or an iterable of IRIs.
    :param filter_by: Which known positives are filtered for the filtered metrics, any of
        ``"train"`` (training axioms of the same normal form), ``"valid"``, ``"test"``
        (the other test axioms) and ``"closure"`` (the deductive closure, see
        ``closure``). Empty: no filtered metrics.
    :param closure: How the deductive closure is obtained, see
        :func:`mowl.evaluation.axiom_ranking.closure_rows`.
    :param ties: ``"ordinal"`` or ``"average"``.
    :param auc: AUC variant (:data:`mowl.evaluation.ranking_metrics.AUC_FUNCTIONS`).
    :param hits: Values of :math:`k` for Hits@\\ :math:`k`.
    :param micro: Also report DELE's micro (per head entity) mean rank and AUC.
    :param raw: Report the raw metrics.
    :param scoring: See :func:`mowl.evaluation.el_scoring.resolve_scorer`.
    :param split: ``"test"`` or ``"valid"``.
    :param key: Catalogue key of the regime this configuration implements.
    """

    def __init__(self, gci_name="gci2", target="tail", relation=None,
                 candidates="evaluation_classes", filter_by=("train",), closure="auto",
                 ties="ordinal", auc="mowl", hits=(1, 10, 100), micro=False, raw=True,
                 scoring="loss", split="test", key="", max_rows=2 ** 22):
        if gci_name not in ("gci0", "gci2", "class_assertion"):
            raise ValueError(f"Unsupported normal form {gci_name!r}.")
        if target not in ("head", "tail"):
            raise ValueError("target must be 'head' or 'tail'.")
        self.gci_name = gci_name
        self.target = target
        self.relation = relation
        self.candidates = candidates
        self.filter_by = tuple(filter_by)
        self.closure = closure
        self.ties = ties
        self.auc = auc
        self.hits = tuple(hits)
        self.micro = micro
        self.raw = raw
        self.scoring = scoring
        self.split = split
        self.max_rows = max_rows
        if key:
            self.key = key
        self.metrics = _metric_names(self.hits, micro, raw, bool(self.filter_by))
        self.subsets = ("all",)
        self.info = {}

    @property
    def target_col(self):
        cols = COLUMNS[self.gci_name]
        if self.gci_name == "class_assertion":
            return 1 if self.target == "tail" else 0
        return len(cols) - 1 if self.target == "tail" else 0

    @property
    def head_col(self):
        return 0 if self.target_col != 0 else len(COLUMNS[self.gci_name]) - 1

    def candidate_iris(self, dataset):
        if self.candidates == "classes":
            return list(dataset.classes.as_str)
        if self.candidates == "evaluation_classes":
            heads, tails = dataset.evaluation_classes
            return list((tails if self.target == "tail" else heads).as_str)
        return list(self.candidates)

    def _restrict(self, model, rows):
        if rows is None or self.relation is None or self.gci_name != "gci2":
            return rows
        rel = model.object_property_index_dict.get(self.relation)
        if rel is None:
            return rows[:0]
        return rows[rows[:, 1] == rel]

    def _filter_rows(self, model, dataset):
        parts = []
        sources = {"train": dataset.ontology, "valid": dataset.validation,
                   "test": dataset.testing}
        for src in self.filter_by:
            if src == "closure":
                r = closure_rows(model, dataset, self.gci_name, closure=self.closure)
            elif src in sources:
                r = el_rows(model, sources[src])[0].get(self.gci_name)
            else:
                raise ValueError(f"Unknown filter source {src!r}.")
            r = self._restrict(model, r)
            if r is not None and len(r):
                parts.append(r)
        if not parts:
            return None
        return th.cat(parts)

    def evaluate(self, model, dataset):
        scorer, scoring_name = resolve_scorer(model, self.scoring)
        onto = dataset.testing if self.split == "test" else dataset.validation
        rows, skipped = el_rows(model, onto)
        width = len(COLUMNS[self.gci_name])
        rows = self._restrict(model, rows.get(self.gci_name,
                                              th.zeros((0, width), dtype=th.long)))
        cands = candidate_indices(model, self.candidate_iris(dataset))
        filters = None
        if self.filter_by:
            filters = build_filter(self._filter_rows(model, dataset), self.target_col)
        raw, filt, kept = rank_queries(model, self.gci_name, rows, self.target_col, cands,
                                       scorer, ties=self.ties, filters=filters,
                                       max_rows=self.max_rows)
        heads = rows[kept][:, self.head_col].tolist()
        out = {}
        n = len(cands)
        if self.raw:
            out.update(summarise_ranks(raw.tolist(), n, hits=self.hits, auc=self.auc))
            if self.micro:
                out.update(micro_metrics(raw.tolist(), heads, n, auc=self.auc))
        if filt is not None:
            out.update(summarise_ranks(filt.tolist(), n, hits=self.hits, auc=self.auc,
                                       prefix="f_"))
            if self.micro:
                out.update(micro_metrics(filt.tolist(), heads, n, auc=self.auc,
                                         prefix="f_"))
        self.info = {"scoring": scoring_name, "num_candidates": int(n),
                     "queries": int(kept.sum()), "dropped_queries": int((~kept).sum()),
                     "skipped_axioms": skipped}
        return {"all": {m: out[m] for m in self.metrics}}


def ppi_ranking(relation, scoring="auto", **kwargs):
    """The PPI evaluation of Box\\ :sup:`2`\\ EL (WWW 2024, Section 4.3; reference
    ``evaluate_ppi.py``): for every test axiom :math:`\\{P_1\\} \\sqsubseteq
    \\exists \\mathit{interacts}.\\{P_2\\}`, rank :math:`P_1` among all proteins, ties by
    position; filtered metrics remove training interactions; AUC of
    :func:`~mowl.evaluation.ranking_metrics.auc_rank_curve`."""
    return FilteredRankingRegime(gci_name="gci2", target="head", relation=relation,
                                 candidates="evaluation_classes", filter_by=("train",),
                                 ties="ordinal", auc="rank_curve", hits=(1, 10, 100),
                                 scoring=scoring, key="ppi-ranking", **kwargs)


def cate_ranking(gci_name, relation=None, closure=False, **kwargs):
    """The evaluation of CatE (Zhapa-Camacho and Hoehndorf, Section 4.1): rank the tail of
    every test axiom among the named concepts (or the evaluation classes), filtered
    metrics only, removing the training axioms (and, for class assertions, the deductive
    closure of the training set). Reported: MR, MRR, Hits@3/10/100, AUC."""
    filter_by = ("train", "closure") if closure else ("train",)
    return FilteredRankingRegime(gci_name=gci_name, target="tail", relation=relation,
                                 filter_by=filter_by, ties="ordinal", auc="mowl",
                                 hits=(3, 10, 100), raw=False, scoring="loss",
                                 key="cate-filtered-ranking", **kwargs)
