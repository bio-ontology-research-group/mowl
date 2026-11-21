"""Triple classification with one score threshold per relation, tuned on validation pairs.

The evaluation of the knowledge graph embedding literature on TransH, TransC, TransEllipsoid
and EIKE (Wang et al., 2024, Section 5.3): a held-out axiom and a negative of the same form
are scored; a relation-specific threshold :math:`\\delta_r` is chosen to maximise accuracy on
the validation axioms and negatives, an axiom is classified as true iff its cost is smaller
than the threshold of its relation, and accuracy, precision, recall and F1 of the test axioms
and negatives are reported (counted over all relations together).

The threshold search follows ``classification_test_normal.py`` of the authors' code: with
``grid=100``, the thresholds tried for a relation are ``min + (max - min) * i / 100`` for
``i = 0..99``, where ``min`` and ``max`` are the lowest and highest validation cost of the
relation, and the first threshold with the best validation accuracy is kept. A relation
without validation axioms classifies everything as false. ``grid=None`` tries every midpoint
between consecutive validation costs instead.

The forms ``class_assertion`` (InstanceOf: one relation), ``gci0`` (SubClassOf: one relation)
and ``object_property_assertion`` (one relation per role) are supported. Costs come from the
model scoring interface (:mod:`mowl.evaluation.model_scoring`), so every model that scores the
form can be evaluated.

World assumption: closed over stated axioms. A negative is false because it is not stated;
the released negatives of YAGO39K are corruptions that are not in the knowledge base, but a
corruption that is true in the world is possible, and sampled negatives are only checked
against the stated axioms.
"""

import numpy as np
import torch as th

from mowl.evaluation.axiom_ranking import el_rows
from mowl.evaluation.regime import EvaluationRegime

#: The forms the regime classifies, with the columns that hold entities of each kind.
_LAYOUT = {
    "class_assertion": {"entities": ("individual", "class"), "relation": None},
    "gci0": {"entities": ("class", "class"), "relation": None},
    "object_property_assertion": {"entities": ("individual", None, "individual"),
                                  "relation": 1},
}


def best_threshold(positive, negative, grid=100):
    """The threshold of one relation: the cost below which an axiom is classified true.

    :param positive: Costs of the validation axioms.
    :param negative: Costs of the validation negatives.
    :param grid: Number of equal steps between the lowest and the highest cost (the
        reference code uses 100), or ``None`` for every midpoint of consecutive costs.
    :return: ``(threshold, accuracy)``; ``(-inf, 0.0)`` if there are no costs.
    """
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    scores = np.concatenate([positive, negative])
    if len(scores) == 0:
        return float("-inf"), 0.0
    low, high = float(scores.min()), float(scores.max())
    if grid is None:
        ordered = np.unique(scores)
        candidates = np.concatenate([[low - 1.0], (ordered[:-1] + ordered[1:]) / 2,
                                     [high + 1.0]])
    else:
        candidates = np.array([low + (high - low) * i / grid for i in range(grid)])
    best, best_accuracy = float("-inf"), -1.0
    for threshold in candidates:
        correct = np.sum(positive < threshold) + np.sum(negative >= threshold)
        accuracy = correct / len(scores)
        if accuracy > best_accuracy:
            best, best_accuracy = float(threshold), float(accuracy)
    return best, best_accuracy


class TripleClassificationRegime(EvaluationRegime):
    """Classify held-out axioms of one form against negatives with per-relation thresholds.

    :param form: ``"class_assertion"``, ``"gci0"`` or ``"object_property_assertion"``.
    :param negatives: ``"given"`` uses ``dataset.negatives(split, form)``, tuples of IRIs in
        the column layout of the form (as :class:`~mowl.datasets.builtin.yago39k.YAGO39KDataset`
        provides them); ``"sampled"`` draws ``num_negatives`` corruptions per positive,
        replacing the subject or the object (chosen uniformly) by a uniformly drawn entity
        of the same kind, and drops those that are training, validation or test axioms.
        ``"auto"`` uses the given ones if the dataset has them.
    :param num_negatives: Negatives per positive when they are sampled.
    :param grid: See :func:`best_threshold`.
    :param seed: Seed of the sampling.
    :param batch_size: Axioms scored at a time.
    """

    key = "triple-classification-thresholds"
    metrics = ("accuracy", "precision", "recall", "f1")
    subsets = ("all",)

    def __init__(self, form="class_assertion", negatives="auto", num_negatives=1, grid=100,
                 seed=0, batch_size=65536):
        super().__init__()
        if form not in _LAYOUT:
            raise ValueError(f"form must be one of {sorted(_LAYOUT)}.")
        if negatives not in ("auto", "given", "sampled"):
            raise ValueError("negatives must be 'auto', 'given' or 'sampled'.")
        self.form, self.negatives, self.num_negatives = form, negatives, int(num_negatives)
        self.grid, self.seed, self.batch_size = grid, seed, int(batch_size)
        self.info = {}

    # -- data ----------------------------------------------------------------------------
    def _index(self, model, kind):
        if kind == "class":
            return model.class_index_dict
        if kind == "individual":
            return getattr(model, "individual_index_dict", None) or {}
        return model.object_property_index_dict

    def _positives(self, model, ontology):
        rows = el_rows(model, ontology)[0].get(self.form)
        width = len(_LAYOUT[self.form]["entities"])
        return th.zeros((0, width), dtype=th.long) if rows is None else rows.cpu()

    def _given(self, model, dataset, split):
        kinds = _LAYOUT[self.form]["entities"]
        indices = [self._index(model, k or "role") for k in kinds]
        rows, dropped = [], 0
        for item in dataset.negatives(split, self.form):
            row = [index.get(x) for index, x in zip(indices, item)]
            if any(v is None for v in row):
                dropped += 1
            else:
                rows.append(row)
        width = len(kinds)
        out = th.tensor(rows, dtype=th.long) if rows else th.zeros((0, width), dtype=th.long)
        return out, dropped

    def _sample(self, model, positives, known, gen):
        kinds = _LAYOUT[self.form]["entities"]
        columns = [i for i, k in enumerate(kinds) if k is not None]
        n, k = len(positives), self.num_negatives
        rows = positives.repeat_interleave(k, dim=0).clone()
        pick = th.randint(0, len(columns), (n * k,), generator=gen)
        for position, column in enumerate(columns):
            size = len(self._index(model, kinds[column]))
            mask = pick == position
            rows[mask, column] = th.randint(0, size, (int(mask.sum()),), generator=gen)
        keep = [i for i, r in enumerate(rows.tolist()) if tuple(r) not in known]
        return rows[keep]

    def _all_negatives(self, model, dataset, vpos, tpos):
        has_given = hasattr(dataset, "negatives")
        if self.negatives == "given" and not has_given:
            raise ValueError("The dataset has no negatives(split, form); use "
                             "negatives='sampled'.")
        if has_given and self.negatives != "sampled":
            vneg, vdrop = self._given(model, dataset, "valid")
            tneg, tdrop = self._given(model, dataset, "test")
            return vneg, tneg, "given", vdrop + tdrop
        gen = th.Generator().manual_seed(self.seed)
        train = self._positives(model, dataset.ontology)
        known = set(map(tuple, th.cat([train, vpos, tpos]).tolist()))
        return (self._sample(model, vpos, known, gen), self._sample(model, tpos, known, gen),
                "sampled", 0)

    @th.no_grad()
    def _scores(self, model, scorer, rows):
        device = getattr(model, "device", "cpu")
        out = [scorer(model, self.form, rows[i:i + self.batch_size].to(device)).detach().cpu()
               for i in range(0, len(rows), self.batch_size)]
        return th.cat(out).numpy().astype(float) if out else np.zeros(0)

    def _relations(self, rows):
        column = _LAYOUT[self.form]["relation"]
        if column is None:
            return np.zeros(len(rows), dtype=np.int64)
        return rows[:, column].numpy()

    # -- evaluation ----------------------------------------------------------------------
    def evaluate(self, model, dataset):
        from mowl.evaluation.el_scoring import resolve_scorer
        scorer, name = resolve_scorer(model, "model")
        module = getattr(model, "module", None)
        if module is not None and hasattr(module, "eval"):
            module.eval()
        vpos = self._positives(model, dataset.validation)
        tpos = self._positives(model, dataset.testing)
        vneg, tneg, source, dropped = self._all_negatives(model, dataset, vpos, tpos)
        costs = {key: (rows, self._scores(model, scorer, rows), self._relations(rows))
                 for key, rows in (("vpos", vpos), ("vneg", vneg), ("tpos", tpos),
                                   ("tneg", tneg))}
        relations = set(costs["vpos"][2]) | set(costs["vneg"][2]) | set(costs["tpos"][2]) \
            | set(costs["tneg"][2])
        thresholds = {}
        for r in relations:
            pick = {k: v[1][v[2] == r] for k, v in costs.items()}
            thresholds[int(r)] = best_threshold(pick["vpos"], pick["vneg"], self.grid)[0]

        def decide(key):
            _, s, r = costs[key]
            limit = np.array([thresholds[int(x)] for x in r]) if len(r) else np.zeros(0)
            return s < limit

        tp = int(decide("tpos").sum())
        fn = len(tpos) - tp
        fp = int(decide("tneg").sum())
        tn = len(tneg) - fp
        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        self.info = {"scoring": name, "form": self.form, "negatives": source,
                     "relations": len(thresholds), "grid": self.grid,
                     "valid_positives": int(len(vpos)), "valid_negatives": int(len(vneg)),
                     "test_positives": int(len(tpos)), "test_negatives": int(len(tneg)),
                     "dropped_negatives": int(dropped)}
        return {"all": {"accuracy": accuracy, "precision": precision, "recall": recall,
                        "f1": f1}}
