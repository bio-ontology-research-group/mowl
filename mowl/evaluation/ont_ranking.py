"""The axiom-prediction and inference evaluation of OnT (Yang, Chen, He, Gao, Horrocks,
ISWC 2025, Section 5.1 "Evaluation Metrics"; reference code ``ont/evaluation/``), as
evaluation regimes.

The protocol ranks the same side of every normal form as the Box\\ :sup:`2`\\ EL protocol
(:class:`mowl.evaluation.normal_form_ranking.NormalFormRankingRegime`): NF1
:math:`C \\sqsubseteq ?D`, NF2 :math:`C \\sqcap D \\sqsubseteq ?E`, NF3 :math:`?C
\\sqsubseteq \\exists r.D`, NF4 :math:`\\exists r.C \\sqsubseteq ?D`, with raw ranks (ties by
candidate position), Hits@1/10/100, median rank, MRR, mean rank and the same ROC AUC.
It differs in three points:

1. **No trivial answers are removed.** Box2EL removes :math:`C` from the candidates of
   :math:`C \\sqsubseteq ?D` and :math:`C, D` from those of :math:`C \\sqcap D
   \\sqsubseteq ?E`; OnT's ``OnTEvaluator`` ranks every concept.
2. **Candidate set.** Every concept of the normalised ontology, including the concepts
   that OnT's normaliser introduces (and names) for complex concepts, but not
   :math:`\\top` / :math:`\\bot`. The dataset's ``candidate_classes`` are used when it
   has them.
3. **Scores.** The subsumption score of OnT, Eq. 8, :math:`s(C \\sqsubseteq D) =
   -(d_\\kappa(x_C, x_D) + \\lambda(\\lVert x_D\\rVert_\\kappa - \\lVert x_C\\rVert_\\kappa))`,
   where the complex side of NF2-NF4 (:math:`C \\sqcap D`, :math:`\\exists r.D`,
   :math:`\\exists r.C`) is embedded from its verbalisation, and :math:`\\lambda` is chosen
   on validation data. Models providing ``score_candidates`` (HiT, OnT) are scored that
   way; other models (EL embedding models) are scored with
   :func:`mowl.evaluation.el_scoring.resolve_scorer`, which allows running the
   geometric baselines under OnT's ranking.

The test sets are part of the data (:mod:`mowl.datasets.builtin.yang2025ont`): a random
80/10/10 split of the normalised axioms for prediction, and all inferred NF1 axioms for
inference.
"""

import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import el_rows, rank_queries
from mowl.evaluation.el_scoring import resolve_scorer
from mowl.evaluation.normal_form_ranking import BOX2EL_SIDES, BOX2EL_METRICS
from mowl.evaluation.ranking_metrics import summarise_ranks
from mowl.owlapi.defaults import TOP, BOT


def ordinal_ranks_by_counting(scores, targets):
    """The ranks of :func:`mowl.evaluation.ranking_metrics.ordinal_ranks` (lower score is
    better, ties broken by candidate position) computed by counting instead of sorting:
    :math:`1 + |\\{j : s_j < s_t\\}| + |\\{j < t : s_j = s_t\\}|`. Linear in the number of
    candidates and computed on the device of ``scores``.

    :param scores: ``(n, num_candidates)`` tensor.
    :param targets: ``(n,)`` tensor of candidate positions.
    """
    targets = targets.to(scores.device).reshape(-1, 1)
    target_scores = th.take_along_dim(scores, targets, dim=1)
    position = th.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    better = (scores < target_scores).sum(dim=1)
    tied_before = ((scores == target_scores) & (position < targets)).sum(dim=1)
    return better + tied_before + 1


class OnTRankingRegime(EvaluationRegime):
    """Per-normal-form axiom ranking of OnT (see the module docstring).

    :param normal_forms: Normal forms to evaluate (``gci0`` = NF1, ..., ``gci3`` = NF4).
    :param split: ``"test"`` or ``"valid"``.
    :param candidates: IRIs of the candidate classes (default: the dataset's
        ``candidate_classes``, otherwise all classes except :math:`\\top, \\bot`).
    :param centri_weight: :math:`\\lambda` of Eq. 8; default: the model's (selected on
        validation data during training).
    :param scoring: Scoring of models without ``score_candidates`` (see
        :func:`~mowl.evaluation.el_scoring.resolve_scorer`).
    :param combined: Report the ``"combined"`` subset (ranks of all normal forms).
    :param query_batch: Queries scored at once.
    """

    key = "ont-normal-form-ranking"
    metrics = BOX2EL_METRICS

    def __init__(self, normal_forms=("gci0", "gci1", "gci2", "gci3"), split="test",
                 candidates=None, centri_weight=None, scoring="auto", combined=True,
                 query_batch=256, max_rows=2 ** 22):
        for nf in normal_forms:
            if nf not in BOX2EL_SIDES:
                raise ValueError(f"Unsupported normal form {nf!r}.")
        self.normal_forms = tuple(normal_forms)
        self.split = split
        self.candidates = candidates
        self.centri_weight = centri_weight
        self.scoring = scoring
        self.combined = combined
        self.query_batch = query_batch
        self.max_rows = max_rows
        self.subsets = self.normal_forms + (("combined",) if combined else ())
        self.info = {}

    def candidate_iris(self, dataset):
        if self.candidates is not None:
            return list(self.candidates)
        cands = getattr(dataset, "candidate_classes", None)
        if cands is not None:
            return list(cands)
        return [c for c in dataset.classes.as_str if c not in (TOP, BOT)]

    def _ontology(self, dataset):
        if self.split == "test":
            return dataset.testing
        if self.split == "valid":
            return dataset.validation
        raise ValueError(f"Unknown split {self.split!r}.")

    def _rank_text_model(self, model, nf, rows, target, cands):
        position = {int(c): i for i, c in enumerate(cands.tolist())}
        kept = th.tensor([int(r[target]) in position for r in rows.tolist()], dtype=th.bool)
        rows = rows[kept]
        ranks = []
        for s in range(0, len(rows), self.query_batch):
            b = rows[s:s + self.query_batch]
            scores = model.score_candidates(nf, b, target, cands, self.centri_weight)
            targets = th.tensor([position[int(t)] for t in b[:, target].tolist()])
            ranks.append(ordinal_ranks_by_counting(-scores, targets).double().cpu())
        return (th.cat(ranks) if ranks else th.zeros(0)), kept

    def evaluate(self, model, dataset):
        rows, skipped = el_rows(model, self._ontology(dataset))
        cid = model.class_index_dict
        iris = self.candidate_iris(dataset)
        cands = th.tensor(sorted({cid[i] for i in iris if i in cid}), dtype=th.long)
        text_model = hasattr(model, "score_candidates")
        scorer, scoring_name = (None, "ont_eq8") if text_model else \
            resolve_scorer(model, self.scoring)
        results, all_ranks, counts = {}, [], {}
        for nf in self.normal_forms:
            target, _ = BOX2EL_SIDES[nf]
            nf_rows = rows.get(nf, th.zeros((0, 2 if nf == "gci0" else 3), dtype=th.long))
            if text_model:
                ranks, kept = self._rank_text_model(model, nf, nf_rows, target, cands)
            else:
                ranks, _, kept = rank_queries(model, nf, nf_rows, target, cands, scorer,
                                              ties="ordinal", exclude_cols=(),
                                              max_rows=self.max_rows)
            ranks = ranks.tolist()
            counts[nf] = {"queries": int(kept.sum()), "dropped": int((~kept).sum())}
            all_ranks.extend(ranks)
            results[nf] = summarise_ranks(ranks, len(cands), hits=(1, 10, 100), auc="box2el")
        if self.combined:
            results["combined"] = summarise_ranks(all_ranks, len(cands), hits=(1, 10, 100),
                                                  auc="box2el")
        lam = self.centri_weight if self.centri_weight is not None else \
            getattr(model, "centri_weight", None)
        self.info = {"scoring": scoring_name, "centri_weight": lam,
                     "num_candidates": int(len(cands)), "skipped_axioms": skipped,
                     "queries": counts}
        return results


class OnTInferenceRankingRegime(OnTRankingRegime):
    """OnT's inference task (Section 5.3): the model is trained on the whole normalised
    ontology and ranks the superclass of every inferred atomic subsumption
    :math:`C \\sqsubseteq D` (``dataset.testing``) among all concepts, as NF1 of
    :class:`OnTRankingRegime`. Reported subset: ``"all"``."""

    key = "ont-inferred-subsumption-ranking"

    def __init__(self, split="test", candidates=None, centri_weight=None, scoring="auto",
                 query_batch=256, max_rows=2 ** 22):
        super().__init__(normal_forms=("gci0",), split=split, candidates=candidates,
                         centri_weight=centri_weight, scoring=scoring, combined=False,
                         query_batch=query_batch, max_rows=max_rows)
        self.subsets = ("all",)

    def evaluate(self, model, dataset):
        return {"all": super().evaluate(model, dataset)["gci0"]}
