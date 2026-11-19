"""Exact-model check of a trained region embedding on a small propositional ontology.

Expressivity benchmarks (such as the Helly family :math:`\\mathcal{T}_k`) do not ask how
well an embedding ranks held-out axioms but whether the trained embedding *is* a model
of the ontology. This regime reads the class regions off a trained model and decides,
with the exact checks of :mod:`mowl.geometric`:

* which normal-form axioms of the ontology the regions satisfy
  (:class:`mowl.geometric.checker.ModelChecker`, box family),
* whether every satisfiable concept name has a non-empty region
  (:meth:`~mowl.geometric.checker.ModelChecker.is_nondegenerate`, with satisfiability
  from :class:`mowl.geometric.completion.CanonicalModel`),
* how many Helly certificates of the ontology the embedding violates
  (:func:`mowl.geometric.helly.helly_check`): a certificate :math:`(C_1, C_2, C_3)` is
  a triple of conjunctions that are pairwise satisfiable but jointly unsatisfiable; the
  embedding violates it when the regions of all names in
  :math:`C_1 \\cup C_2 \\cup C_3` have a common point, i.e. when it does not satisfy the
  entailed axiom :math:`C_1 \\sqcap C_2 \\sqcap C_3 \\sqsubseteq \\bot`.

Regions are read from the model by an *extractor* (:data:`REGION_EXTRACTORS`): single
boxes for ELBE, Box\\ :sup:`2`\\ EL and TransBox, unions of boxes for modules that provide
``all_class_regions()`` (MultiBoxEL). For single boxes all propositional normal forms are
decided exactly by :class:`~mowl.geometric.checker.ModelChecker`. For unions of boxes the
:math:`\\bot` axioms (:math:`C \\sqsubseteq \\bot`, :math:`C \\sqcap D \\sqsubseteq \\bot`) and
inclusions into a single box are decided exactly; an inclusion into a union of two or
more boxes is not decided and is counted in ``info["unchecked_axioms"]``. Role axioms are
outside this regime.
"""

import math

import numpy as np

from mowl.evaluation.regime import EvaluationRegime


def _boxes_from_centre_offset(centre, offset):
    centre = centre.detach().cpu().double().numpy()
    offset = np.abs(offset.detach().cpu().double().numpy())
    return [[(c - o, c + o)] for c, o in zip(centre, offset)]


def _elbe_regions(module):
    return _boxes_from_centre_offset(module.class_embed.weight, module.class_offset.weight)


def _box2el_regions(module):
    return _boxes_from_centre_offset(module.class_center.weight, module.class_offset.weight)


def _transbox_regions(module):
    return _boxes_from_centre_offset(module.class_center_embedding.weight,
                                     module.class_offset_embedding.weight)


def _union_regions(module):
    r = module.all_class_regions()
    cen = r.cen.double().numpy()
    w = r.w.double().numpy()
    active = r.active.numpy()
    out = []
    for i in range(cen.shape[0]):
        comps = []
        for k in range(cen.shape[1]):
            if active[i, k]:
                comps.append((cen[i, k] - w[i, k], cen[i, k] + w[i, k]))
        out.append(comps)
    return out


#: Region extractors by module class name. Each returns, for every class index, a list
#: of ``(lo, hi)`` component boxes (one for single-box models).
REGION_EXTRACTORS = {
    "ELBEModule": _elbe_regions,
    "BoxSquaredELModule": _box2el_regions,
    "TransBoxModule": _transbox_regions,
}


def extract_regions(model):
    """``{class IRI: [Box, ...]}`` for the classes of ``model``.

    :raises NotImplementedError: If no extractor applies to the model's module.
    """
    from mowl.geometric.regions import Box
    module = model.module
    name = type(module).__name__
    if name in REGION_EXTRACTORS:
        comps = REGION_EXTRACTORS[name](module)
    elif hasattr(module, "all_class_regions"):
        comps = _union_regions(module)
    else:
        raise NotImplementedError(
            f"The exact-model check cannot read class regions from {name}. Register an "
            f"extractor in mowl.evaluation.exact_model.REGION_EXTRACTORS.")
    return {iri: [Box(lo, hi) for lo, hi in comps[idx]]
            for iri, idx in model.class_index_dict.items() if idx < len(comps)}


def _intersect_unions(a, b):
    out = []
    for x in a:
        for y in b:
            z = x.intersection(y)
            if not z.is_empty:
                out.append(z)
    return out


def _nonempty(region):
    return any(not b.is_empty for b in region)


class ExactModelRegime(EvaluationRegime):
    """Whether a trained embedding is an exact model of the (training) ontology.

    Metrics (subset ``"all"``):

    * ``axiom_satisfaction``: fraction of the checked propositional normal-form axioms
      that the regions satisfy;
    * ``is_model``: 1.0 if every checked axiom is satisfied;
    * ``empty_fraction``: fraction of satisfiable concept names with an empty region
      (degeneracy);
    * ``exact_model``: 1.0 if ``is_model`` and no satisfiable name is empty (a
      non-degenerate model) and no axiom was left unchecked;
    * ``helly_violation_rate``: fraction of the Helly certificates of the ontology
      (:func:`mowl.geometric.helly.helly_check`) whose jointly unsatisfiable names have
      a common point in the embedding (``nan`` if the ontology has none).

    :param tol: Numerical tolerance of the inclusion checks.
    :param k_max: Largest minimal unsatisfiable set size enumerated by
        :func:`~mowl.geometric.helly.helly_check`.
    :param max_certificates: Cap on the certificates enumerated.
    """

    key = "exact-model-check"
    metrics = ("axiom_satisfaction", "is_model", "empty_fraction", "exact_model",
               "helly_violation_rate")
    subsets = ("all",)

    def __init__(self, tol=1e-6, k_max=None, max_certificates=1000):
        self.tol = tol
        self.k_max = k_max
        self.max_certificates = max_certificates
        self.info = {}

    def evaluate(self, model, dataset):
        from mowl.geometric.checker import DictInterpretation, ModelChecker
        from mowl.geometric.completion import CanonicalModel
        from mowl.geometric.helly import helly_check
        from mowl.geometric.normal_forms import NF1, NF2, BOTTOM, TOP
        from mowl.geometric.regions import Box

        cm = CanonicalModel.from_dataset(dataset)
        regions = extract_regions(model)
        boxes = [b for comps in regions.values() for b in comps if not b.is_empty]
        d = len(boxes[0].lo) if boxes else 1
        if boxes:
            universe = Box(np.min([b.lo for b in boxes], axis=0) - 1.0,
                           np.max([b.hi for b in boxes], axis=0) + 1.0)
        else:
            universe = Box(np.zeros(d), np.ones(d))
        single = all(len(c) <= 1 for c in regions.values())

        def region(name):
            if name == TOP:
                return [universe]
            if name == BOTTOM:
                return []
            return regions.get(name)

        checked, satisfied, unchecked, unknown = 0, 0, 0, 0
        checker = None
        if single:
            classes = {n: (c[0] if c else Box.empty(d)) for n, c in regions.items()}
            checker = ModelChecker(DictInterpretation("box", universe, classes=classes),
                                   tol=self.tol)
        for ax in cm.axioms:
            if not isinstance(ax, (NF1, NF2)):
                unchecked += 1
                continue
            names = [ax.sub, ax.sup] if isinstance(ax, NF1) else [ax.sub1, ax.sub2, ax.sup]
            if any(region(n) is None for n in names):
                unknown += 1
                continue
            if checker is not None:
                ok = checker.satisfies(ax)
            else:
                left = (region(ax.sub) if isinstance(ax, NF1)
                        else _intersect_unions(region(ax.sub1), region(ax.sub2)))
                sup = region(ax.sup)
                if not _nonempty(left):
                    ok = True
                elif ax.sup == BOTTOM:
                    ok = False
                elif len(sup) == 1:
                    ok = all(b.is_subset(sup[0], tol=self.tol) for b in left if not b.is_empty)
                else:
                    unchecked += 1
                    continue
            checked += 1
            satisfied += int(bool(ok))

        names = [n for n in cm.names if n not in (TOP, BOTTOM) and n in regions]
        empty = [n for n in names if not _nonempty(regions[n])]
        report = helly_check(cm, k_max=self.k_max or max(3, len(names)),
                             max_certificates=self.max_certificates)
        certs = report.certificates_strong
        violated = 0
        for cert in certs:
            joint = [universe]
            for n in set(cert.c1) | set(cert.c2) | set(cert.c3):
                joint = _intersect_unions(joint, regions.get(n, []))
            violated += int(_nonempty(joint))
        is_model = float(checked > 0 and satisfied == checked)
        self.info = {"checked_axioms": checked, "unchecked_axioms": unchecked,
                     "axioms_with_unknown_names": unknown,
                     "satisfiable_names": len(names), "empty_names": sorted(empty),
                     "helly_certificates": len(certs), "helly_violations": violated,
                     "region_kind": "single box" if single else "union of boxes"}
        return {"all": {
            "axiom_satisfaction": satisfied / checked if checked else math.nan,
            "is_model": is_model,
            "empty_fraction": len(empty) / len(names) if names else math.nan,
            "exact_model": float(is_model == 1.0 and not (empty or unchecked or unknown)),
            "helly_violation_rate": violated / len(certs) if certs else math.nan,
        }}
