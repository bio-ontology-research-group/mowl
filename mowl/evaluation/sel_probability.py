"""Approximate probabilistic inference in Statistical EL: the evaluation of BoxSEL.

The evaluation of Section "Experiments" of Zhu et al. (2024, arXiv:2407.11821): a model is
trained on the conditionals of a learning set; for every query conditional ``(E | C)`` with
premises ``(D | C)[l1, u1]`` and ``(E | C ⊓ D)[l2, u2]`` in the learning set, the
probabilistic modus ponens gives the entailed interval
``[l1 l2, min(1, u1 u2 + 1 - l1)]`` (see
:meth:`~mowl.datasets.builtin.sel_conditionals.SELConditionalDataset.pmp_queries`), and the
model estimates the probability of the query as the volume ratio of its embedding. With
``N`` embeddings (different seeds) the interval estimate is ``[min p_i, max p_i]``; with one
it is the point ``[p, p]``.

Metrics (the paper's, per query ``i`` with entailed interval ``[l_i, u_i]`` and estimate
``[l'_i, u'_i]``):

* ``soundness_accuracy`` (SA): fraction of queries with ``[l', u'] ⊆ [l, u]``;
* ``soundness_error`` (SE): mean of ``[l - l']+ + [u' - u]+``;
* ``approximation_gap`` (AG): mean of ``|l - l'| + |u - u'|``;
* ``mean_absolute_error`` (MAE): mean distance of the embedded probabilities of the learning
  set to their intervals (embedding quality).

The paper's baselines (a fixed estimator that always answers ``[0, 1]`` and a random
estimator) are reported in ``info["baselines"]`` on the same queries; its KDE estimator is
not implemented (the density of the learning-set intervals is degenerate for point
probabilities, ``l = u``).

World assumption: open world. The entailed interval is what the learning set determines;
probabilities outside it are not wrong, only unsupported.
"""

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime


def batch_probabilities(model, conditionals):
    """The embedded probability ``Vol(D ∩ C) / Vol(C)`` of every ``(conclusion, premise, ...)``
    in ``conditionals`` (a sequence whose first two items are the expressions), as a NumPy
    array in the order given; ``nan`` where the premise has no volume. Structures are
    grouped so that each group is one batch."""
    groups = {}
    for i, c in enumerate(conditionals):
        (sd, ld), (sc, lc) = model._compile(c[0]), model._compile(c[1])
        groups.setdefault((sd, sc), []).append((i, ld, lc))
    out = np.full(len(conditionals), np.nan)
    for (sd, sc), items in groups.items():
        idx = [i for i, _, _ in items]
        ld = th.tensor([d for _, d, _ in items], dtype=th.long, device=model.device)
        lc = th.tensor([c for _, _, c in items], dtype=th.long, device=model.device)
        out[idx] = model.module.probability(sd, sc, ld, lc).detach().cpu().numpy()
    return out


def interval_metrics(truth, lower, upper):
    """SA, SE and AG of interval estimates ``[lower, upper]`` against the entailed
    intervals ``truth`` (array ``(q, 2)``). ``lower`` and ``upper`` have length ``q``."""
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    l, u = truth[:, 0], truth[:, 1]
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    eps = 1e-12
    inside = (lower >= l - eps) & (upper <= u + eps)
    return {"soundness_accuracy": float(inside.mean()),
            "soundness_error": float((np.maximum(l - lower, 0) + np.maximum(upper - u, 0)).mean()),
            "approximation_gap": float((np.abs(l - lower) + np.abs(u - upper)).mean())}


def fixed_estimator(q):
    """The paper's fixed baseline: ``[0, 1]`` for every query."""
    return np.zeros(q), np.ones(q)


def random_estimator(q, rng):
    """The paper's random baseline: two uniform draws from ``[0, 1]``, ordered."""
    draws = np.sort(rng.uniform(size=(q, 2)), axis=1)
    return draws[:, 0], draws[:, 1]


class SELProbabilityRegime(EvaluationRegime):
    """Soundness accuracy, soundness error and approximation gap of the interval estimates
    of probabilistic conditionals, on the queries of a
    :class:`~mowl.datasets.builtin.sel_conditionals.SELConditionalDataset`.

    :meth:`initialise_model` hands the learning-set conditionals to the model
    (``model.add_conditionals``; the model must have ``probability`` and ``_compile``, as
    :class:`~mowl.models.boxsel.model.BoxSEL`).

    :param baselines_seed: Seed of the random baseline.
    """

    key = "sel-pmp-intervals"
    metrics = ("soundness_accuracy", "soundness_error", "approximation_gap",
               "mean_absolute_error")
    subsets = ("all",)

    def __init__(self, baselines_seed=0):
        super().__init__()
        self.baselines_seed = baselines_seed
        self.info = {}

    def initialise_model(self, model, dataset):
        if not hasattr(model, "add_conditionals") or not hasattr(model, "_compile"):
            raise TypeError(
                f"{type(model).__name__} cannot learn probabilistic conditionals; the "
                f"statistical EL regime needs a model with add_conditionals, such as BoxSEL.")
        if not hasattr(dataset, "conditionals"):
            raise TypeError("The dataset has no conditionals (SELConditionalDataset).")
        model.add_conditionals(dataset.conditionals)

    def _estimates(self, models, queries):
        ps = np.stack([batch_probabilities(m, queries) for m in models])
        return np.nanmin(ps, axis=0), np.nanmax(ps, axis=0), ps

    def evaluate(self, model, dataset):
        """Evaluate one embedding (the interval estimate is the point ``[p, p]``)."""
        return self.evaluate_ensemble([model], dataset)

    def evaluate_ensemble(self, models, dataset):
        """Evaluate the interval estimate ``[min p_i, max p_i]`` of several embeddings of the
        same learning set (the paper: ``N = 60`` seeds)."""
        queries = dataset.pmp_queries()
        truth = np.array([[q[2], q[3]] for q in queries], dtype=float).reshape(-1, 2)
        if len(queries) == 0:
            raise ValueError("The dataset has no evaluable queries (no query has the premises "
                             "of the probabilistic modus ponens in the learning set).")
        lower, upper, _ = self._estimates(models, queries)
        valid = ~np.isnan(lower)
        metrics = interval_metrics(truth[valid], lower[valid], upper[valid])
        learning = dataset.conditionals
        lo = np.array([c[2] for c in learning])
        up = np.array([c[3] for c in learning])
        errors = [np.nanmean(np.maximum(np.maximum(lo - p, p - up), 0.0))
                  for p in (batch_probabilities(m, learning) for m in models)] \
            if len(learning) else [0.0]
        metrics["mean_absolute_error"] = float(np.mean(errors))
        rng = np.random.default_rng(self.baselines_seed)
        q = int(valid.sum())
        self.info = {
            "embeddings": len(models), "queries": len(queries), "evaluated_queries": q,
            "learning_conditionals": len(learning),
            "queries_without_estimate": int((~valid).sum()),
            "mean_interval_width": float((truth[:, 1] - truth[:, 0]).mean()),
            "baselines": {
                "fixed": interval_metrics(truth[valid], *fixed_estimator(q)),
                "random": interval_metrics(truth[valid], *random_estimator(q, rng))}}
        return {"all": metrics}
