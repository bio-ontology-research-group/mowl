"""Exact-model check of a trained region embedding on a small propositional ontology.

Expressivity benchmarks (such as the Helly family :math:`\\mathcal{T}_k`) do not ask how
well an embedding ranks held-out axioms but whether the trained embedding *is* a model
of the ontology. This regime reads the class regions off a trained model as
:class:`~mowl.evaluation.regions.Region` objects and decides, exactly and independently of
the model's loss:

* which propositional normal-form axioms of the ontology the regions satisfy
  (:math:`C \\sqsubseteq D`, :math:`C \\sqcap D \\sqsubseteq E` and their :math:`\\bot`
  forms; :func:`mowl.evaluation.regions.satisfies`),
* whether every satisfiable concept name (satisfiability from
  :class:`mowl.geometric.completion.CanonicalModel`) has a non-empty region, and whether
  it has a region with non-empty interior,
* how many Helly certificates of the ontology the embedding violates
  (:func:`mowl.geometric.helly.helly_check`): a certificate :math:`(C_1, C_2, C_3)` is
  a triple of conjunctions that are pairwise satisfiable but jointly unsatisfiable; the
  embedding violates it when the regions of all names in
  :math:`C_1 \\cup C_2 \\cup C_3` have a common point, i.e. when it does not satisfy the
  entailed axiom :math:`C_1 \\sqcap C_2 \\sqcap C_3 \\sqsubseteq \\bot`.

Regions are read from the model's module by a *reader* registered in
:data:`REGION_READERS` (keyed by module class name; subclasses inherit the reader of
their closest registered base class). Supported families and what is decided exactly are
listed in :mod:`mowl.evaluation.regions`: single boxes (ELBE, Box\\ :sup:`2`\\ EL,
TransBox, ELK-Em, RegD-ELBE, BoxEL), balls (ELEmbeddings, EmEL++, EmEL(var),
RegD-ELEm), prisms (PrismEL), products of octagons (OctagonEL) and unions of boxes
(MultiBoxEL, MultiELBE, the volume-based MultiBoxEL). For these families every
propositional normal form with binary conjunctions is decided; an axiom that falls outside
the implemented procedures is counted in ``info["unchecked_axioms"]`` and makes
``exact_model`` zero. Role axioms are outside this regime.

**Degeneracy.** A region can be non-empty without interior (a ball of radius 0, a box
with a zero side, a triangle with :math:`\\sum_k h_k = 0`). Both are reported:
``empty_fraction`` (exact set emptiness) and ``no_interior_fraction`` (no inscribed ball
of radius larger than ``tol / 2``). ``exact_model`` requires every satisfiable name to
have **non-empty interior**, for every family; ``exact_model_nonempty`` only requires
non-empty regions. The distinction matters for balls: three pairwise externally tangent
balls :math:`X_1, X_2, X_3` with the tangency points as :math:`A_1, A_2, A_3` (balls of
radius 0) are a model of :math:`\\mathcal{T}_3` with non-empty names, but not one with
non-empty interiors.
"""

import math

import numpy as np

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.regions import (BallRegion, BoxRegion, EmptyRegion, OctagonRegion,
                                     PrismRegion, UndecidedInclusion, UnionRegion,
                                     Universe, satisfies)


def _np(t):
    return t.detach().cpu().double().numpy()


def _boxes(center, offset):
    center, offset = _np(center), np.abs(_np(offset))
    return [BoxRegion(c - o, c + o) for c, o in zip(center, offset)]


def _all_idx(module, n=None):
    import torch as th
    n = module.nb_ont_classes if n is None else n
    device = next(module.parameters()).device
    return th.arange(n, device=device)


def _elkem_regions(module):
    import torch as th
    with th.no_grad():
        lower, upper = module.class_box(_all_idx(module))
    return [BoxRegion(lo, hi) for lo, hi in zip(_np(lower), _np(upper))]


def _boxel_regions(module):
    import torch as th
    lo = _np(module.min_embedding.weight)
    hi = lo + _np(th.exp(module.delta_embedding.weight))
    return [BoxRegion(a, b) for a, b in zip(lo, hi)]


def _component_unions(cen, off, active=None):
    cen, off = _np(cen), _np(off)
    out = []
    for i in range(cen.shape[0]):
        comps = [BoxRegion(cen[i, k] - off[i, k], cen[i, k] + off[i, k])
                 for k in range(cen.shape[1]) if active is None or active[i, k]]
        out.append(UnionRegion(comps, dim=cen.shape[2]))
    return out


def _structural_union_regions(module):
    r = module.all_class_regions()
    return _component_unions(r.cen, r.w, r.active.cpu().numpy())


def _multielbe_regions(module):
    import torch as th
    with th.no_grad():
        cen, off = module.class_boxes(_all_idx(module))
    return _component_unions(cen, off)


def _volume_multibox_regions(module):
    import torch as th
    with th.no_grad():
        cen, off = module.concept_boxes(_all_idx(module))
    return _component_unions(cen, off)


def _prism_regions(module):
    r = module.all_class_regions()
    h, cen, w = _np(r.h), _np(r.cen), _np(r.w)
    return [PrismRegion(h[i], cen[i] - w[i], cen[i] + w[i]) for i in range(h.shape[0])]


def _octagon_regions(module):
    h = _np(module.all_class_bounds())
    return [OctagonRegion(h[i]) for i in range(h.shape[0])]


#: Region readers by module class name (reference readers that read the parameters as the
#: module's losses do; modules without one fall back to their geometry, see
#: :func:`find_reader`). Each takes the module and returns one
#: :class:`~mowl.evaluation.regions.Region` per class index, reading the parameters as
#: the module's losses do (absolute values of offsets and radii, materialised defined
#: classes of the structural objectives, fixed empty / full boxes of ELK-Em).
REGION_READERS = {
    "ELKEmModule": _elkem_regions,
    "BoxELModule": _boxel_regions,
    "MultiBoxELModule": _structural_union_regions,
    "MultiELBEModule": _multielbe_regions,
    "VolumeMultiBoxELModule": _volume_multibox_regions,
    "PrismModule": _prism_regions,
    "OctagonELModule": _octagon_regions,
}

#: Modules that subclass a registered module but are not region models of that kind.
NOT_REGION_MODULES = {
    "CanonicalMixtureModule": (
        "classes are columns of a 0/1 membership matrix over the finite canonical domain "
        "(a set-based model, faithful by construction), not regions of R^d"),
}


def _geometry_regions(module):
    """Balls or boxes of every class, read through the geometry of the module."""
    import torch as th
    from mowl.evaluation.region_access import all_class_regions, centre_and_size
    with th.no_grad():
        region, _ = all_class_regions(module, "The exact-model check")
        centre, size, is_ball = centre_and_size(region, "The exact-model check")
    centre, size = _np(centre), _np(size)
    if is_ball:
        return [BallRegion(c, r) for c, r in zip(centre, size.reshape(-1))]
    return [BoxRegion(c - o, c + o) for c, o in zip(centre, size)]


def _has_ball_or_box_geometry(module):
    from mowl.representation import geometry_of
    from mowl.representation.regions import Ball, Box
    try:
        geometry = geometry_of(module)
        probe = geometry.classes(_all_idx(module, 1))
    except (NotImplementedError, TypeError, AttributeError, RuntimeError, StopIteration):
        return False
    return isinstance(probe, (Ball, Box))


def find_reader(module):
    """The reader of :data:`REGION_READERS` for ``module`` (by class name, walking the
    method resolution order); failing that, the reader of the geometry of the module if
    its classes are balls or boxes (:func:`mowl.representation.geometry_of`); else
    ``None``."""
    for cls in type(module).__mro__:
        if cls.__name__ in NOT_REGION_MODULES:
            return None
        if cls.__name__ in REGION_READERS:
            return REGION_READERS[cls.__name__]
    if hasattr(module, "all_class_regions"):
        r = module.all_class_regions()
        if hasattr(r, "active"):
            return _structural_union_regions
    if _has_ball_or_box_geometry(module):
        return _geometry_regions
    return None


def extract_regions(model):
    """``{class IRI: Region}`` for the classes of ``model``.

    :raises NotImplementedError: If no reader applies to the model's module.
    """
    module = model.module
    reader = find_reader(module)
    if reader is None:
        name = type(module).__name__
        why = NOT_REGION_MODULES.get(name)
        hint = (f": {why}." if why else
                ". Register a reader in mowl.evaluation.exact_model.REGION_READERS.")
        raise NotImplementedError(
            f"The exact-model check cannot read class regions from {name}{hint}")
    regions = reader(module)
    return {iri: regions[idx] for iri, idx in model.class_index_dict.items()
            if idx < len(regions)}


def region_kind(regions):
    """Human-readable family of a collection of regions."""
    kinds = set()
    for r in regions:
        if isinstance(r, UnionRegion):
            fams = {p.family for p in r.pieces} or {"box"}
            kinds.add("union of boxes" if fams == {"box"} else "union")
        else:
            kinds.add({"box": "single box", "ball": "ball", "prism": "prism",
                       "octagon": "product of octagons"}.get(r.family, r.family))
    return ", ".join(sorted(kinds)) if kinds else "none"


def _axiom_str(ax):
    from mowl.geometric.normal_forms import NF1

    def short(n):
        return n.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    if isinstance(ax, NF1):
        return f"{short(ax.sub)} ⊑ {short(ax.sup)}"
    return f"{short(ax.sub1)} ⊓ {short(ax.sub2)} ⊑ {short(ax.sup)}"


class ExactModelRegime(EvaluationRegime):
    """Whether a trained embedding is an exact model of the (training) ontology.

    Metrics (subset ``"all"``):

    * ``axiom_satisfaction``: fraction of the decided propositional normal-form axioms
      that the regions satisfy;
    * ``is_model``: 1.0 if every decided axiom is satisfied;
    * ``empty_fraction``: fraction of satisfiable concept names with an empty region;
    * ``no_interior_fraction``: fraction of satisfiable concept names whose region
      contains no ball of radius larger than ``tol / 2`` (empty interior);
    * ``exact_model``: 1.0 if ``is_model``, no axiom was left undecided, and every
      satisfiable name has a region with non-empty interior (a non-degenerate model);
    * ``exact_model_nonempty``: as ``exact_model`` but only requiring non-empty regions;
    * ``helly_violation_rate``: fraction of the Helly certificates of the ontology
      (:func:`mowl.geometric.helly.helly_check`) whose jointly unsatisfiable names have
      a common point in the embedding (``nan`` if the ontology has none). The common
      point is decided exactly, without tolerance.

    :param tol: Numerical tolerance of the inclusion and disjointness checks (see
        :mod:`mowl.evaluation.regions`) and of the interior test.
    :param k_max: Largest minimal unsatisfiable set size enumerated by
        :func:`~mowl.geometric.helly.helly_check`.
    :param max_certificates: Cap on the certificates enumerated.
    """

    key = "exact-model-check"
    metrics = ("axiom_satisfaction", "is_model", "empty_fraction", "no_interior_fraction",
               "exact_model", "exact_model_nonempty", "helly_violation_rate")
    subsets = ("all",)

    def __init__(self, tol=1e-6, k_max=None, max_certificates=1000):
        self.tol = tol
        self.k_max = k_max
        self.max_certificates = max_certificates
        self.info = {}

    def evaluate(self, model, dataset):
        from mowl.geometric.completion import CanonicalModel
        return self.evaluate_regions(extract_regions(model), CanonicalModel.from_dataset(dataset))

    def evaluate_regions(self, regions, cm):
        """Evaluate ``{class IRI: Region}`` against the canonical model ``cm`` of the
        ontology (:class:`mowl.geometric.completion.CanonicalModel`)."""
        from mowl.geometric.helly import helly_check
        from mowl.geometric.normal_forms import NF1, NF2, BOTTOM, TOP

        dim = next(iter(regions.values())).dim if regions else 1

        def region(name):
            if name == TOP:
                return Universe(dim)
            if name == BOTTOM:
                return EmptyRegion(dim)
            return regions.get(name)

        checked, satisfied, unchecked, unknown = 0, 0, 0, 0
        violated_axioms, undecided_axioms = [], []
        for ax in cm.axioms:
            if not isinstance(ax, (NF1, NF2)):
                unchecked += 1
                undecided_axioms.append(type(ax).__name__)
                continue
            names = [ax.sub, ax.sup] if isinstance(ax, NF1) else [ax.sub1, ax.sub2, ax.sup]
            if any(region(n) is None for n in names):
                unknown += 1
                continue
            try:
                ok = satisfies(ax, region, tol=self.tol)
            except UndecidedInclusion:
                unchecked += 1
                undecided_axioms.append(_axiom_str(ax))
                continue
            checked += 1
            satisfied += int(ok)
            if not ok:
                violated_axioms.append(_axiom_str(ax))

        names = [n for n in cm.names if n not in (TOP, BOTTOM) and n in regions]
        empty = [n for n in names if regions[n].is_empty()]
        no_interior = [n for n in names if not regions[n].has_interior(self.tol)]
        report = helly_check(cm, k_max=self.k_max or max(3, len(names)),
                             max_certificates=self.max_certificates)
        certs = report.certificates_strong
        violated, helly_undecided = 0, 0
        for cert in certs:
            joint = Universe(dim)
            members = set(cert.c1) | set(cert.c2) | set(cert.c3)
            try:
                for n in sorted(members):
                    joint = joint.intersect(region(n) or EmptyRegion(dim), 0.0)
                violated += int(not joint.is_empty())
            except UndecidedInclusion:
                helly_undecided += 1
        is_model = float(checked > 0 and satisfied == checked)
        complete = is_model == 1.0 and not (unchecked or unknown)
        self.info = {"checked_axioms": checked, "unchecked_axioms": unchecked,
                     "axioms_with_unknown_names": unknown,
                     "violated_axioms": violated_axioms[:50],
                     "undecided_axioms": undecided_axioms[:50],
                     "satisfiable_names": len(names), "empty_names": sorted(empty),
                     "no_interior_names": sorted(no_interior),
                     "helly_certificates": len(certs), "helly_violations": violated,
                     "helly_undecided": helly_undecided,
                     "region_kind": region_kind(regions.values()),
                     "exact_model_requires": "non-empty interior of every satisfiable name"}
        n_certs = len(certs) - helly_undecided
        return {"all": {
            "axiom_satisfaction": satisfied / checked if checked else math.nan,
            "is_model": is_model,
            "empty_fraction": len(empty) / len(names) if names else math.nan,
            "no_interior_fraction": len(no_interior) / len(names) if names else math.nan,
            "exact_model": float(complete and not no_interior),
            "exact_model_nonempty": float(complete and not empty),
            "helly_violation_rate": violated / n_certs if n_certs else math.nan,
        }}
