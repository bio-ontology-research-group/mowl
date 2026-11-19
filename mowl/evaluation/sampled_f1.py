"""Sampled binary classification of subsumptions with an F1 score, as in RegD (Yang, Chen,
Sattler, "RegD: Hierarchical Embeddings via Distances over Geometric Regions",
Sections 5.1-5.3).

RegD evaluates ontology embeddings as classifiers: 1,000 subsumptions are sampled from
the test set, negatives are generated "by randomly replacing the child of each positive
pair 10 times", a pair is predicted to hold iff its energy is at most a threshold
:math:`t` "that achieved the best performance on the evaluation set", and the F1 score
is reported.
"""

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import el_rows, candidate_indices, COLUMNS
from mowl.evaluation.el_scoring import resolve_scorer
from mowl.evaluation.ranking_metrics import binary_f1, best_threshold

#: Column that is corrupted to make negatives, per normal form, as in the reference code
#: (``EXP_ontology/evaluate.py`` of HuiYang1997/RegD): for ``C ⊑ D`` (``gci0``,
#: ``compute_nf1_ranks``) the superclass ``D`` is replaced; for ``A ⊑ ∃r.B`` (``gci2``,
#: ``compute_nf3_ranks``) the subclass ``A``. The paper's text says "replacing the child"
#: (with ``C ⊑ D`` read as ``D ≺ C``, the child is ``C``), which agrees with the code for
#: ``gci2`` but not for ``gci0``; the code is followed.
CORRUPT_COLUMN = {"gci0": 1, "gci2": 0}


class SampledF1Regime(EvaluationRegime):
    """Threshold classification of sampled positives and random negatives.

    :param gci_name: ``"gci0"`` (:math:`C \\sqsubseteq D`, RegD's inference task) or
        ``"gci2"`` (:math:`A \\sqsubseteq \\exists r.B`, written :math:`\\exists r.B
        \\prec ?A` in RegD's prediction task).
    :param corrupt_column: Column replaced in negatives (default
        :data:`CORRUPT_COLUMN`).
    :param num_positives: Positives sampled from the test (and from the validation) set.
    :param num_negatives: Negatives per positive.
    :param candidates: IRIs the replacement concepts are drawn from (default: all
        classes of the dataset, as the reference code samples from all classes).
    :param scoring: See :func:`mowl.evaluation.el_scoring.resolve_scorer`; the default
        ``"loss"`` uses the model's own loss as the energy.
    :param seed: Seed of the sampling (independent of the training seed).
    """

    key = "regd-sampled-f1"
    metrics = ("f1", "precision", "recall")
    subsets = ("all",)

    def __init__(self, gci_name="gci0", num_positives=1000, num_negatives=10,
                 candidates=None, scoring="loss", seed=0, corrupt_column=None):
        if gci_name not in CORRUPT_COLUMN:
            raise ValueError(f"Unsupported normal form {gci_name!r}.")
        self.gci_name = gci_name
        self.corrupt_column = (CORRUPT_COLUMN[gci_name] if corrupt_column is None
                               else corrupt_column)
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.candidates = candidates
        self.scoring = scoring
        self.seed = seed
        self.info = {}

    def _sample(self, rows, cands, gen):
        if len(rows) == 0:
            return rows, rows
        n = min(self.num_positives, len(rows))
        idx = th.randperm(len(rows), generator=gen)[:n]
        pos = rows[idx]
        neg = pos.repeat_interleave(self.num_negatives, dim=0).clone()
        col = self.corrupt_column
        repl = cands[th.randint(len(cands), (len(neg),), generator=gen)]
        same = repl == neg[:, col]
        while bool(same.any()):
            repl[same] = cands[th.randint(len(cands), (int(same.sum()),), generator=gen)]
            same = repl == neg[:, col]
        neg[:, col] = repl
        return pos, neg

    def _scores(self, model, scorer, rows):
        if len(rows) == 0:
            return np.zeros(0)
        module = getattr(model, "module", None)
        if module is not None:
            module.eval()
        with th.no_grad():
            out = scorer(model, self.gci_name, rows.to(getattr(model, "device", "cpu")))
        return out.detach().cpu().double().numpy()

    def evaluate(self, model, dataset):
        scorer, scoring_name = resolve_scorer(model, self.scoring)
        iris = self.candidates
        if iris is None:
            iris = list(dataset.classes.as_str)
        cands = candidate_indices(model, iris)
        gen = th.Generator().manual_seed(self.seed)
        width = len(COLUMNS[self.gci_name])
        empty = th.zeros((0, width), dtype=th.long)
        valid_rows = el_rows(model, dataset.validation)[0].get(self.gci_name, empty)
        test_rows = el_rows(model, dataset.testing)[0].get(self.gci_name, empty)
        vpos, vneg = self._sample(valid_rows, cands, gen)
        tpos, tneg = self._sample(test_rows, cands, gen)
        if len(vpos):
            vs = np.concatenate([self._scores(model, scorer, vpos),
                                 self._scores(model, scorer, vneg)])
            vl = np.concatenate([np.ones(len(vpos)), np.zeros(len(vneg))])
            threshold, _ = best_threshold(vs, vl)
            threshold_source = "validation"
        else:
            threshold, threshold_source = None, "test"
        ts = np.concatenate([self._scores(model, scorer, tpos),
                             self._scores(model, scorer, tneg)])
        tl = np.concatenate([np.ones(len(tpos)), np.zeros(len(tneg))]).astype(bool)
        if threshold is None:
            threshold, _ = best_threshold(ts, tl)
        pred = ts <= threshold
        tp = float(np.sum(pred & tl))
        precision = tp / max(float(np.sum(pred)), 1.0)
        recall = tp / max(float(np.sum(tl)), 1.0)
        self.info = {"scoring": scoring_name, "threshold": float(threshold),
                     "threshold_source": threshold_source, "test_positives": int(len(tpos)),
                     "test_negatives": int(len(tneg)), "valid_positives": int(len(vpos))}
        return {"all": {"f1": binary_f1(tl, pred), "precision": precision,
                        "recall": recall}}
