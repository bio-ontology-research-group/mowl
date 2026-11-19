"""Geometric inclusion accuracy: the fraction of subsumptions whose regions are nested.

EmEL++ (Mondal et al. 2021, Section 4.5, Table 4) checks, "for a test instance
:math:`C \\sqsubseteq D`, ... that the :math:`n`-ball of class :math:`C` lies inside the
:math:`n`-ball of class :math:`D`" and reports the fraction of training, testing and
inferred axioms for which this holds, calling it "a much stricter criterion" than ranking.
BoxEL (Xiong et al. 2022, Section 5.2, Table 3) reports the same "accuracy based measure"
for boxes ("the prediction is true if and only if the subclass box is exactly inside the
superclass box").

Unlike the exact-model check (:class:`mowl.evaluation.exact_model.ExactModelRegime`),
which asks whether *all* axioms of an ontology hold, this regime reports the fraction of
the atomic subsumptions of a split that hold, for any split (training, validation,
testing): on held-out axioms it measures inductive generalisation of the geometry.

Regions: balls for modules with ``class_embed`` and ``class_rad`` (ELEmbeddings, EmEL++,
EmEL(var)): :math:`\\|c_C - c_D\\| + |r_C| \\le |r_D|`; single boxes for the modules in
:data:`mowl.evaluation.exact_model.REGION_EXTRACTORS` (ELBE, Box\\ :sup:`2`\\ EL,
TransBox): :math:`l_D \\le l_C` and :math:`u_C \\le u_D`. An empty box (some
:math:`l_i > u_i`) is included in every box. A tolerance ``tol`` is allowed on every
comparison.
"""

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import el_rows


def _ball_inclusion(module, rows, tol):
    c = module.class_embed.weight.detach()
    r = module.class_rad.weight.detach().abs().reshape(-1)
    sub, sup = rows[:, 0], rows[:, 1]
    dist = th.linalg.norm(c[sub] - c[sup], dim=1)
    return (dist + r[sub] <= r[sup] + tol).cpu().numpy()


def _box_inclusion(model, rows, tol):
    from mowl.evaluation.exact_model import REGION_EXTRACTORS
    comps = REGION_EXTRACTORS[type(model.module).__name__](model.module)
    lo = np.stack([np.asarray(c[0][0], dtype=float) for c in comps])
    hi = np.stack([np.asarray(c[0][1], dtype=float) for c in comps])
    sub, sup = rows[:, 0].numpy(), rows[:, 1].numpy()
    empty = (lo[sub] > hi[sub] + tol).any(axis=1)
    inside = ((lo[sup] <= lo[sub] + tol) & (hi[sub] <= hi[sup] + tol)).all(axis=1)
    return empty | inside


def inclusion_mask(model, rows, tol=1e-6):
    """Boolean array: for every ``gci0`` row ``[C, D]``, whether the region of ``C`` lies
    inside the region of ``D``.

    :raises NotImplementedError: If the regions of the model's module cannot be read.
    """
    from mowl.evaluation.exact_model import REGION_EXTRACTORS
    module = model.module
    rows = rows.cpu()
    if len(rows) == 0:
        return np.zeros(0, dtype=bool)
    if hasattr(module, "class_embed") and hasattr(module, "class_rad"):
        return _ball_inclusion(module, rows.to(module.class_embed.weight.device), tol)
    if type(module).__name__ in REGION_EXTRACTORS:
        return _box_inclusion(model, rows, tol)
    raise NotImplementedError(
        f"Cannot read ball or box regions from {type(module).__name__}.")


class InclusionAccuracyRegime(EvaluationRegime):
    """Fraction of atomic subsumptions :math:`C \\sqsubseteq D` whose regions are nested.

    :param splits: Any of ``"train"``, ``"valid"``, ``"test"`` (the subsets reported).
    :param tol: Tolerance of the comparisons.
    """

    key = "region-inclusion-accuracy"
    metrics = ("accuracy", "num_axioms")

    def __init__(self, splits=("train", "test"), tol=1e-6):
        self.splits = tuple(splits)
        self.subsets = self.splits
        self.tol = float(tol)
        self.info = {}

    def evaluate(self, model, dataset):
        sources = {"train": dataset.ontology, "valid": dataset.validation,
                   "test": dataset.testing}
        out = {}
        for split in self.splits:
            onto = sources[split]
            rows = el_rows(model, onto)[0].get("gci0") if onto is not None else None
            if rows is None or len(rows) == 0:
                out[split] = {"accuracy": float("nan"), "num_axioms": 0}
                continue
            bot = model.class_index_dict.get("http://www.w3.org/2002/07/owl#Nothing")
            if bot is not None:
                rows = rows[rows[:, 1] != bot]
            mask = inclusion_mask(model, rows, self.tol)
            out[split] = {"accuracy": float(mask.mean()) if len(mask) else float("nan"),
                          "num_axioms": int(len(mask))}
        return out
