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
    :param strict_hits: Count Hits@\\ :math:`k` with ``rank < k`` instead of
        :math:`\\mathit{rank} \\le k`, as the reference code of CatE does
        (``src/cat_completion.py``: ``if rank < k``; a Hits@1 is then always 0).
    :param queries: ``None`` (default: the axioms of the ``split`` ontology) or
        ``"dataset:<attribute>"``: read the query entities from that attribute of the
        dataset, a list of IRI tuples in the column layout of ``gci_name``. Entities the
        model does not know are skipped and counted in ``info['dropped_queries']``.
    :param exclude_head: Remove the head entity of every query from its own candidate
        list (the reference code of CatE sets the head's score to the worst value for
        subsumption queries, ``preds[head] = -10000``).
    :param key: Catalogue key of the regime this configuration implements.
    """

    def __init__(self, gci_name="gci2", target="tail", relation=None,
                 candidates="evaluation_classes", filter_by=("train",), closure="auto",
                 ties="ordinal", auc="mowl", hits=(1, 10, 100), micro=False, raw=True,
                 scoring="loss", split="test", key="", max_rows=2 ** 22,
                 strict_hits=False, queries=None, exclude_head=False):
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
        self.strict_hits = strict_hits
        self.queries = queries
        self.exclude_head = exclude_head
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

    def _explicit_rows(self, model, dataset):
        """Query rows from ``self.queries == "dataset:<attr>"``, indexed with the
        model's entity dictionaries; a list of IRI tuples in the row layout of the
        normal form. Rows with an unknown entity are dropped and counted."""
        attr = self.queries.split(":", 1)[1]
        pairs = list(getattr(dataset, attr) or [])
        cid = model.class_index_dict
        iid = getattr(model, "individual_index_dict", None) or {}
        rid = model.object_property_index_dict
        width = len(COLUMNS[self.gci_name])
        rows, dropped = [], 0
        for pair in pairs:
            pair = [str(x) for x in pair]
            if self.gci_name == "class_assertion":
                row = [iid.get(pair[0]), cid.get(pair[1])]
            elif self.gci_name == "gci0":
                row = [cid.get(pair[0]), cid.get(pair[1])]
            elif self.gci_name == "gci2":
                row = [cid.get(pair[0]), rid.get(pair[1]), cid.get(pair[2])]
            else:
                raise ValueError(f"Explicit queries are not supported for {self.gci_name}.")
            if any(x is None for x in row):
                dropped += 1
                continue
            rows.append(row)
        rows = th.tensor(rows, dtype=th.long).reshape(-1, width) if rows \
            else th.zeros((0, width), dtype=th.long)
        return rows, dropped

    def evaluate(self, model, dataset):
        declared = getattr(model, "scored_forms", None)
        if declared is not None and self.gci_name not in declared \
                and not callable(self.scoring):
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(
                f"{type(model).__name__} has no score for {self.gci_name!r} axioms "
                f"(it scores: {', '.join(declared)}).")
        scorer, scoring_name = resolve_scorer(model, self.scoring)
        onto = dataset.testing if self.split == "test" else dataset.validation
        width = len(COLUMNS[self.gci_name])
        if self.queries is None:
            rows, skipped = el_rows(model, onto)
            rows = self._restrict(model, rows.get(self.gci_name,
                                                  th.zeros((0, width), dtype=th.long)))
        else:
            rows, dropped = self._explicit_rows(model, dataset)
            skipped = dropped
        cands = candidate_indices(model, self.candidate_iris(dataset))
        filters = None
        if self.filter_by:
            filters = build_filter(self._filter_rows(model, dataset), self.target_col)
        raw, filt, kept = rank_queries(model, self.gci_name, rows, self.target_col, cands,
                                       scorer, ties=self.ties, filters=filters,
                                       max_rows=self.max_rows,
                                       exclude_cols=(0,) if self.exclude_head else ())
        heads = rows[kept][:, self.head_col].tolist()
        out = {}
        n = len(cands)
        if self.raw:
            out.update(summarise_ranks(raw.tolist(), n, hits=self.hits, auc=self.auc,
                                       strict_hits=self.strict_hits))
            if self.micro:
                out.update(micro_metrics(raw.tolist(), heads, n, auc=self.auc))
        if filt is not None:
            out.update(summarise_ranks(filt.tolist(), n, hits=self.hits, auc=self.auc,
                                       prefix="f_", strict_hits=self.strict_hits))
            if self.micro:
                out.update(micro_metrics(filt.tolist(), heads, n, auc=self.auc,
                                         prefix="f_"))
        self.info = {"scoring": scoring_name, "num_candidates": int(n),
                     "queries": int(kept.sum()), "dropped_queries": int((~kept).sum()),
                     "skipped_axioms": skipped}
        #: Ranks of the last :meth:`evaluate` call, for composite regimes.
        self.last_ranks = {"raw": raw, "filtered": filt, "kept": kept}
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


def cate_ranking(gci_name, relation=None, closure=False, queries=None, **kwargs):
    """The evaluation of CatE (Zhapa-Camacho and Hoehndorf, arXiv:2305.07163v3,
    Section 4.1 and the reference code): rank the tail of every test axiom among the
    named concepts (or the evaluation classes), filtered metrics only. Reported: MR,
    MRR, Hits@3/10/100, AUC.

    The details follow the reference code, not the paper's prose:

    * filtering removes the deductive closure of the training ontology for
      subsumptions and PPI axioms (``inferred_ancestors.txt`` of the data release;
      pass ``closure=True``), and the training axioms for class assertions
      (``get_filtering_labels`` of ``src/cat_membership.py``; the paper, Section 4.1,
      additionally states the deductive closure for :math:`C(a)`, which the code does
      not do);
    * a query's head is removed from its own candidate list for subsumptions
      (``preds[head] = -10000`` of ``src/cat_completion.py``); pass ``exclude_head``;
    * Hits@\\ :math:`k` counts ranks *strictly* below :math:`k` (``if rank < k``);
    * the AUC is the trapezoid over the cumulative rank curve
      (:func:`~mowl.evaluation.ranking_metrics.auc_rank_curve`, ``compute_rank_roc`` of
      the code), and ties are broken by position.

    :param closure: Filter by the deductive closure of the training ontology as well as
        the training axioms.
    :param queries: ``None`` (the test axioms) or ``"dataset:<attribute>"`` (the
        released query lists, see :class:`FilteredRankingRegime`).
    """
    filter_by = ("train", "closure") if closure else ("train",)
    return FilteredRankingRegime(gci_name=gci_name, target="tail", relation=relation,
                                 filter_by=filter_by, ties="ordinal", auc="rank_curve",
                                 hits=(3, 10, 100), raw=False, scoring="loss",
                                 strict_hits=True, queries=queries,
                                 key="cate-filtered-ranking", **kwargs)


class CatePPIRankingRegime(EvaluationRegime):
    """The PPI evaluation of CatE (Zhapa-Camacho and Hoehndorf, Table 4, left): for every
    test interaction :math:`P_1 \\sqsubseteq \\exists \\mathit{interacts}.P_2`, rank
    :math:`P_1` among the proteins with :math:`P_2` fixed and :math:`P_2` (as
    :math:`\\exists \\mathit{interacts}.P_2`) with :math:`P_1` fixed, and average the
    metrics over the two directions (``src/evaluators.py``, ``PPIEvaluator``: the mOWL
    PPI evaluation the reference code calls).

    Ties by position, Hits@\\ :math:`k` with :math:`\\mathit{rank} \\le k`, AUC of
    :func:`~mowl.evaluation.ranking_metrics.auc_rank_curve` (the trapezoid of
    ``compute_rank_roc``), filtered metrics remove the training *and* validation
    interactions.
    """

    def __init__(self, relation, candidates="evaluation_classes", hits=(1, 3, 10, 100),
                 key="cate-ppi-ranking", **kwargs):
        self.relation = relation
        self.candidates = candidates
        self.hits = tuple(hits)
        self.key = key
        self.tail = FilteredRankingRegime(
            gci_name="gci2", target="tail", relation=relation, candidates=candidates,
            filter_by=("train", "valid"), ties="ordinal", auc="rank_curve",
            hits=self.hits, raw=True, scoring="loss", key="")
        self.head = FilteredRankingRegime(
            gci_name="gci2", target="head", relation=relation, candidates=candidates,
            filter_by=("train", "valid"), ties="ordinal", auc="rank_curve",
            hits=self.hits, raw=True, scoring="loss", key="")
        self.metrics = tuple(
            f"f_{name}" for name in ("mr", "mrr", *[f"hits@{k}" for k in self.hits],
                                     "auc"))
        self.subsets = ("all",)
        self.info = {}

    def prepare(self, dataset):
        self.tail.prepare(dataset)
        self.head.prepare(dataset)
        return dataset

    def evaluate(self, model, dataset):
        self.tail.evaluate(model, dataset)
        self.head.evaluate(model, dataset)
        n = self.tail.info.get("num_candidates")
        ranks = []
        for part in (self.tail, self.head):
            filt = part.last_ranks["filtered"]
            ranks.extend(filt.tolist() if filt is not None else [])
        out = summarise_ranks(ranks, n, hits=self.hits, auc="rank_curve", prefix="f_")
        self.info = {"scoring": self.tail.info.get("scoring"),
                     "num_candidates": n,
                     "queries": int(len(ranks) / 2),
                     "directions": 2}
        return {"all": {m: out[m] for m in self.metrics}}


def cate_ppi_ranking(relation, **kwargs):
    """The PPI evaluation of CatE (Zhapa-Camacho and Hoehndorf, Table 4, left, and the
    reference ``src/evaluators.py``): both directions of every test interaction, ranked
    among the proteins; filtered MR, MRR, Hits@1/3/10/100, AUC."""
    return CatePPIRankingRegime(relation, **kwargs)


def common_ranking(gci_name="gci0", relation=None, candidates="classes", target="tail",
                   filter_by=("train", "valid", "test"), scoring="auto", **kwargs):
    """The ranking of the joint mOWL benchmarks, the same for every model family.

    For every test axiom :math:`C \\sqsubseteq D` (``gci0``) or
    :math:`C \\sqsubseteq \\exists r.D` (``gci2``), the class :math:`D` is ranked among all
    candidates (:math:`C` itself removed). Only filtered metrics are reported: the other
    classes known to be correct for the query (training, validation and test axioms of
    the same form) are removed. Tied candidates share the average rank, so a model that
    gives every candidate the same score gets chance-level ranks. Scores come from
    ``scoring="auto"``: what each model declares
    (:mod:`mowl.evaluation.model_scoring`), recorded with the result.
    """
    return FilteredRankingRegime(
        gci_name=gci_name, target=target, relation=relation, candidates=candidates,
        filter_by=filter_by, ties="average", auc="mowl", hits=(1, 10, 100), raw=False,
        scoring=scoring, exclude_head=True, key="common-filtered-ranking", **kwargs)
