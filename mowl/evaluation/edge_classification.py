"""Threshold classification of subsumptions against negatives, with F1.

The evaluation of the hierarchy embedding literature (order embeddings, hyperbolic
entailment cones, disk embeddings, shadow cones): every held-out subsumption ``C ⊑ D`` and
a number of false pairs are scored, the score threshold with the best F1 on the validation
pairs is chosen, and precision, recall and F1 of that threshold are reported on the test
pairs.

The threshold search and the F1 follow ``eval.py`` of the code of Ganea et al.
(https://github.com/dalab/hyperbolic_cones): every position of the sorted validation
scores is tried, the first with the largest F1 gives the threshold, and a pair is
classified positive iff its score is at most the threshold.

World assumption: closed over stated axioms. A pair is false because it is not in the
(transitive closure of the) data; with sampled negatives a pair entailed by the ontology
can be drawn as a negative.
"""

import numpy as np
import torch as th

from mowl.evaluation.axiom_ranking import el_rows
from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.sampled_f1 import _apply_threshold, reference_best_threshold


def _pair_rows(model, pairs):
    """Index ``(subclass IRI, superclass IRI)`` pairs with the model's class indices;
    pairs with an unknown class are dropped. Returns ``(rows, dropped)``."""
    index = model.class_index_dict
    rows = [(index[c], index[d]) for c, d in pairs if c in index and d in index]
    out = th.tensor(rows, dtype=th.long) if rows else th.zeros((0, 2), dtype=th.long)
    return out, len(pairs) - len(rows)


class EdgeClassificationF1Regime(EvaluationRegime):
    """Classify held-out atomic subsumptions against negatives by a score threshold.

    Scores are the model's costs of ``C ⊑ D`` through the scoring interface of
    :mod:`mowl.evaluation.model_scoring` (lower: more plausible), so any model that scores
    the ``gci0`` form can be evaluated.

    :param negatives: ``"given"`` uses the negatives that come with the dataset
        (``dataset.validation_negatives`` and ``dataset.testing_negatives``, lists of
        ``(subclass IRI, superclass IRI)``, as
        :class:`~mowl.datasets.builtin.wordnet_closure.WordNetNounClosureDataset` has);
        ``"sampled"`` draws ``num_negatives`` corruptions per positive, each replacing the
        subclass or the superclass (chosen uniformly) by a uniformly drawn class, and
        discards those that are training, validation or test axioms. ``"auto"`` (default)
        uses the given ones if the dataset has them.
    :param num_negatives: Negatives per positive when they are sampled.
    :param seed: Seed of the sampling.
    :param batch_size: Pairs scored at a time.
    """

    key = "hierarchy-threshold-f1"
    metrics = ("f1", "precision", "recall")
    subsets = ("all",)

    def __init__(self, negatives="auto", num_negatives=10, seed=0, batch_size=65536):
        super().__init__()
        if negatives not in ("auto", "given", "sampled"):
            raise ValueError("negatives must be 'auto', 'given' or 'sampled'.")
        self.negatives, self.num_negatives = negatives, int(num_negatives)
        self.seed, self.batch_size = seed, int(batch_size)
        self.info = {}

    # -- data ---------------------------------------------------------------------------
    def _positives(self, model, ontology):
        rows = el_rows(model, ontology)[0].get("gci0")
        return th.zeros((0, 2), dtype=th.long) if rows is None else rows.cpu()

    def _sample(self, positives, known, num_classes, gen):
        n, k = len(positives), self.num_negatives
        rows = positives.repeat_interleave(k, dim=0).clone()
        column = th.randint(0, 2, (n * k,), generator=gen)
        rows[th.arange(n * k), column] = th.randint(0, num_classes, (n * k,), generator=gen)
        keep = [i for i, (c, d) in enumerate(rows.tolist()) if c != d and (c, d) not in known]
        return rows[keep]

    def _negatives(self, model, dataset, vpos, tpos):
        given = hasattr(dataset, "validation_negatives") and hasattr(dataset,
                                                                    "testing_negatives")
        if self.negatives == "given" and not given:
            raise ValueError("The dataset has no validation_negatives / testing_negatives; "
                             "use negatives='sampled'.")
        if given and self.negatives != "sampled":
            vneg, vdrop = _pair_rows(model, dataset.validation_negatives)
            tneg, tdrop = _pair_rows(model, dataset.testing_negatives)
            return vneg, tneg, "given", vdrop + tdrop
        gen = th.Generator().manual_seed(self.seed)
        train = self._positives(model, dataset.ontology)
        known = set(map(tuple, th.cat([train, vpos, tpos]).tolist()))
        num_classes = len(model.class_index_dict)
        return (self._sample(vpos, known, num_classes, gen),
                self._sample(tpos, known, num_classes, gen), "sampled", 0)

    @th.no_grad()
    def _scores(self, model, scorer, rows):
        device = getattr(model, "device", "cpu")
        out = [scorer(model, "gci0", rows[i:i + self.batch_size].to(device)).detach().cpu()
               for i in range(0, len(rows), self.batch_size)]
        return th.cat(out).numpy().astype(float) if out else np.zeros(0)

    # -- evaluation ---------------------------------------------------------------------
    def evaluate(self, model, dataset):
        from mowl.evaluation.el_scoring import resolve_scorer
        scorer, name = resolve_scorer(model, "model")
        module = getattr(model, "module", None)
        if module is not None and hasattr(module, "eval"):
            module.eval()
        vpos = self._positives(model, dataset.validation)
        tpos = self._positives(model, dataset.testing)
        vneg, tneg, source, dropped = self._negatives(model, dataset, vpos, tpos)
        s_vpos, s_vneg = self._scores(model, scorer, vpos), self._scores(model, scorer, vneg)
        s_tpos, s_tneg = self._scores(model, scorer, tpos), self._scores(model, scorer, tneg)
        threshold, valid_f1 = reference_best_threshold(s_vpos, s_vneg)
        if threshold is None:
            f1 = precision = recall = 0.0
        else:
            f1, precision, recall = _apply_threshold(s_tpos, s_tneg, threshold)
        self.info = {"scoring": name, "negatives": source, "threshold": threshold,
                     "valid_f1": valid_f1, "valid_positives": int(len(vpos)),
                     "valid_negatives": int(len(vneg)), "test_positives": int(len(tpos)),
                     "test_negatives": int(len(tneg)), "dropped_negatives": int(dropped)}
        return {"all": {"f1": f1, "precision": precision, "recall": recall}}
