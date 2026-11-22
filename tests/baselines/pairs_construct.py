"""Constructed methods: the combination of a source and a reading against the direct
construction.

Nothing is trained, so there are no losses and no training run to compare; the checks
are the constructed regions (as tensors), the truth values of a set of test axioms, and
the geometry (built from parts) on the side of the combination. ``scores`` returns both
the regions and the truth values; it reads the public interface the two classes share.
"""
import itertools

import numpy as np
import torch as th

from tests.baselines.harness import Pair

_NOT_TRAINED = ("The method is constructed (completion, closed form, search), not trained "
                "by gradient descent: there is no loss and no training run.")
_SKIP = {"losses": _NOT_TRAINED, "training": _NOT_TRAINED}


def _abox():
    from tests.models.kg_region_data import abox_dataset
    return abox_dataset()


def _cm(dataset):
    from mowl.geometric.completion import CanonicalModel
    return CanonicalModel.from_dataset(dataset)


def _no_copy(old, new):
    """Nothing to copy: both are built from the same data."""


def _nf_axioms(cm):
    """Normal-form test axioms over the names of the canonical model."""
    from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, NF3, NF4
    names = sorted(cm.names)
    axioms = [NF1(a, b) for a, b in itertools.product(names, repeat=2)]
    axioms += [NF2(a, b, c) for a, b, c in itertools.product(names[:6], repeat=3)]
    axioms += [NF2(a, b, BOTTOM) for a, b in itertools.product(names, repeat=2)]
    for r in sorted(cm.role_names):
        axioms += [NF3(a, r, b) for a, b in itertools.product(names, repeat=2)]
        axioms += [NF4(r, a, b) for a, b in itertools.product(names, repeat=2)]
    return axioms


def _truth(model, axioms):
    return th.tensor([float(model.satisfies(ax)) for ax in axioms])


# -- interval union -------------------------------------------------------------------------
def _interval_scores(model):
    cm = model.canonical_model
    names = sorted(model.class_regions)
    width = len(cm.names)
    regions = th.full((len(names), width, 2), float("nan"), dtype=th.float64)
    for i, n in enumerate(names):
        for k, iv in enumerate(model.class_intervals(n)):
            regions[i, k] = th.tensor(iv, dtype=th.float64)
    roles = th.full((len(cm.role_names), width * width, 4), float("nan"), dtype=th.float64)
    for i, r in enumerate(sorted(cm.role_names)):
        for k, (x, y) in enumerate(model.role_region(r).rects):
            roles[i, k] = th.tensor([*x, *y], dtype=th.float64)
    return {"regions": regions, "roles": roles, "truth": _truth(model, _nf_axioms(cm))}


def _new_intervals(dataset):
    from mowl.geometric.intervals import IntervalUnionModel
    return IntervalUnionModel.from_canonical_model(_cm(dataset))


def _old_intervals(dataset):
    from mowl.baselines.geometric.intervals import IntervalUnionModel
    return IntervalUnionModel.from_canonical_model(_cm(dataset))


# -- faces of the simplex ---------------------------------------------------------------------
def _face_scores(model, names, roles, axioms):
    n = model._points.shape[0]
    vertices = th.zeros(len(names), n)
    for i, name in enumerate(names):
        for v in model.class_region(name).vertex_ids:
            vertices[i, v] = 1.0
    hulls = [th.as_tensor(model.role_region(r).points, dtype=th.float64) for r in roles]
    out = {"vertices": vertices, "points": th.as_tensor(model._points),
           "truth": _truth(model, axioms)}
    for r, hull in zip(roles, hulls):
        out[f"role {r}"] = hull
    return out


def _polyhedral_scores(model):
    cm = model.canonical_model
    return _face_scores(model, sorted(model.class_regions), sorted(cm.role_names),
                        _nf_axioms(cm))


def _new_polyhedral(dataset, thicken=False):
    from mowl.geometric.polyhedral import ConvexPolyhedralModel
    return ConvexPolyhedralModel.from_canonical_model(_cm(dataset), thicken=thicken)


def _old_polyhedral(dataset, thicken=False):
    from mowl.baselines.geometric.polyhedral import ConvexPolyhedralModel
    return ConvexPolyhedralModel.from_canonical_model(_cm(dataset), thicken=thicken)


def _family():
    from mowl.datasets.builtin import FamilyDataset
    return FamilyDataset()


def _propositional():
    """A propositional ontology (no roles) of the geometric fixtures."""
    import os
    from mowl.datasets import PathDataset
    here = os.path.dirname(os.path.abspath(__file__))
    return PathDataset(os.path.join(here, "..", "geometric", "fixtures", "helly_ok.owl"))


# -- convex regions of a finite model ------------------------------------------------------------
def _convex_scores(model):
    fm = model.finite_model
    cm_axioms = model._test_axioms
    out = _face_scores(model, sorted(fm.concepts), sorted(fm.roles), cm_axioms)
    phi = model.phi()
    out["phi"] = th.tensor([float(phi.same_as(fm))])
    return out


def _convex(module):
    def build(dataset):
        import importlib
        cm = _cm(dataset)
        cls = importlib.import_module(module).ConvexRegionModel
        model = cls.from_canonical_model(cm)
        model._test_axioms = _nf_axioms(cm)
        return model
    return build


# -- al-cones ----------------------------------------------------------------------------------
def _alc_axioms(names, roles, individuals):
    from mowl.geometric.cones import (And, Atomic, ConceptAssertion, Exists, Forall, GCI, Not,
                                      Or)
    atoms = [Atomic(n) for n in names]
    out = [GCI(a, b) for a, b in itertools.product(atoms, repeat=2)]
    out += [GCI(Not(a), Or((b, Not(a)))) for a, b in itertools.product(atoms[:4], repeat=2)]
    out += [GCI(And((a, b)), Not(c)) for a, b, c in itertools.product(atoms[:4], repeat=3)]
    for r in roles:
        out += [GCI(a, Exists(r, b)) for a, b in itertools.product(atoms, repeat=2)]
        out += [GCI(Exists(r, a), b) for a, b in itertools.product(atoms[:5], repeat=2)]
        out += [GCI(Forall(r, a), b) for a, b in itertools.product(atoms[:4], repeat=2)]
    out += [ConceptAssertion(a, i) for i in individuals for a in atoms]
    out += [ConceptAssertion(Not(a), i) for i in individuals for a in atoms[:4]]
    return out


def _cone_scores(model):
    from mowl.geometric.cones import ALConeChecker
    names, roles = sorted(model.concepts), sorted(model.roles)
    individuals = sorted(model.individuals)
    checker = ALConeChecker(model)
    out = {"bits": th.tensor(np.stack([model.concepts[n].bits for n in names]),
                             dtype=th.float64),
           "truth": th.tensor([float(checker.satisfies(ax))
                               for ax in _alc_axioms(names, roles, individuals)]),
           "dim": th.tensor([float(model.dim)])}
    if individuals:
        out["individuals"] = th.tensor(np.stack([model.individuals[a] for a in individuals]))
    for r in roles:
        out[f"role {r}"] = th.tensor(model.roles[r], dtype=th.float64)
    return out


def _cones(module, how):
    def build(dataset):
        import importlib
        cls = importlib.import_module(module).ALConeInterpretation
        if how == "canonical":
            return cls.from_canonical_model(_cm(dataset))
        return cls.from_boolean_ontology(dataset, faithful=how == "faithful")
    return build


def _boolean_tbox():
    from mowl.geometric.cones import And, Atomic, ConceptAssertion, GCI, Not, Or
    A, B, C, D = (Atomic(x) for x in "ABCD")
    return [GCI(A, B), GCI(And((B, C)), Not(D)), GCI(D, Or((A, C))),
            ConceptAssertion(A, "a"), ConceptAssertion(Or((C, D)), "b"),
            ConceptAssertion(Not(B), "c")]


# -- temporal al-cones --------------------------------------------------------------------------
def _temporal_ontology():
    from mowl.geometric.temporal_cones import TemporalOntology
    return TemporalOntology.parse(tbox=["A <= X B", "B <= X A", "A <= ~B", "C <= F A"],
                                  abox=["A(a)", "C(c)", "~C(b)"])


def _temporal(module):
    def build(onto):
        import importlib
        cls = importlib.import_module(module).TemporalALConeInterpretation
        return cls.from_ontology(onto, max_prefix=1, max_period=2)
    return build


def _temporal_scores(model):
    from mowl.geometric.temporal_cones import parse_axiom
    names = sorted(model.atoms)
    bits = th.tensor(np.stack([[c.bits for c in model.atoms[n]] for n in names]),
                     dtype=th.float64)
    texts = [f"{a} <= {op}{b}" for a, b in itertools.product(names, repeat=2)
             for op in ("", "X ", "F ", "G ", "~")]
    texts += [f"{a} <= {b} U {c}" for a, b, c in itertools.product(names, repeat=3)]
    truth = th.tensor([float(model.satisfies(parse_axiom(t))) for t in texts])
    individuals = sorted(model.individuals)
    members = th.tensor([[float(model.instance_of(i, parse_axiom(f"{n} <= {n}").sub, t))
                          for n in names for t in range(model.length)] for i in individuals])
    return {"bits": bits, "truth": truth, "members": members,
            "individuals": th.tensor(np.stack([model.individuals[i] for i in individuals])),
            "shape": th.tensor([float(model.dim), float(model.length),
                                float(model.loop_start), float(len(model.traces))])}


# -- attributed ---------------------------------------------------------------------------------
def _attributed_ontology():
    from mowl.geometric.attributed import (AttributedOntology, Basic, ConceptFact, ConceptInc,
                                           Specifier)
    s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
    return AttributedOntology([
        ConceptFact("A", "a", s), ConceptFact("A", "b", t), ConceptFact("C", "b", s),
        ConceptInc((Basic("A", s),), Basic("B", s)),
        ConceptInc((Basic("B", s), Basic("C", s)), Basic("D", s))])


def _attributed(module):
    def build(onto):
        import importlib
        return importlib.import_module(module).AttributedConvexModel(onto)
    return build


def _attributed_scores(model):
    from mowl.geometric.attributed import ANY, Specifier
    specs = [Specifier.closed(source="s"), Specifier.closed(source="t"), ANY]
    names = [(c, s) for c in "ABCD" for s in specs]
    n = model.model._points.shape[0]
    vertices = th.zeros(len(names), n)
    for i, (c, s) in enumerate(names):
        for v in model.concept_region(c, s).vertex_ids:
            vertices[i, v] = 1.0
    holds = th.tensor([[float(model.holds_concept(a, c, s)) for (c, s) in names]
                       for a in "ab"])
    included = th.tensor([[float(model.included(c, s, d, t)) for (d, t) in names]
                          for (c, s) in names])
    return {"vertices": vertices, "holds": holds, "included": included,
            "is_model": th.tensor([float(model.is_model())]),
            "points": th.tensor(np.stack([model.point(a) for a in "ab"]))}


PAIRS = [
    Pair("interval_union", _new_intervals, _old_intervals, _no_copy, _interval_scores,
         dataset=_abox, skip=_SKIP),
    Pair("interval_union/family", _new_intervals, _old_intervals, _no_copy,
         _interval_scores, dataset=_family, skip=_SKIP),
    Pair("convex_polyhedral", _new_polyhedral, _old_polyhedral, _no_copy,
         _polyhedral_scores, dataset=_abox, skip=_SKIP),
    Pair("convex_polyhedral/thicken", lambda d: _new_polyhedral(d, True),
         lambda d: _old_polyhedral(d, True), _no_copy, _polyhedral_scores,
         dataset=_propositional,
         skip=_SKIP),
    Pair("convex_regions", _convex("mowl.geometric.convex_regions"),
         _convex("mowl.baselines.geometric.convex_regions"), _no_copy, _convex_scores,
         dataset=_abox, skip=_SKIP),
    Pair("alcone-exact", _cones("mowl.geometric.cones", "canonical"),
         _cones("mowl.baselines.geometric.cones", "canonical"), _no_copy, _cone_scores,
         dataset=_abox, skip=_SKIP),
    Pair("alcone-exact/boolean", _cones("mowl.geometric.cones", "boolean"),
         _cones("mowl.baselines.geometric.cones", "boolean"), _no_copy, _cone_scores,
         dataset=_boolean_tbox, skip=_SKIP),
    Pair("alcone-exact/boolean-faithful", _cones("mowl.geometric.cones", "faithful"),
         _cones("mowl.baselines.geometric.cones", "faithful"), _no_copy, _cone_scores,
         dataset=_boolean_tbox, skip=_SKIP),
    Pair("temporal_cones", _temporal("mowl.geometric.temporal_cones"),
         _temporal("mowl.baselines.geometric.temporal_cones"), _no_copy, _temporal_scores,
         dataset=_temporal_ontology, skip=_SKIP),
    Pair("attributed_dl", _attributed("mowl.geometric.attributed"),
         _attributed("mowl.baselines.geometric.attributed"), _no_copy, _attributed_scores,
         dataset=_attributed_ontology, skip=_SKIP),
]

EXEMPT = {
    "canonical_model": "The canonical model is the source part itself (completion, "
                       "mowl.geometric.completion); reading it as element sets is the "
                       "identity, so there is no second, direct implementation. Its "
                       "geometry is checked against the canonical model's own entailment and "
                       "instance checks in tests/representation/test_recomp_construct.py.",
    "least_model": "The least model is the source part itself (mowl.geometric.least_model); "
                   "reading it as element sets is the identity, so there is no direct "
                   "implementation to compare with. Its geometry is checked against "
                   "LeastModel.holds and instances in test_recomp_construct.py.",
    "box_milp": "The box model is found by the solver, which is the source part itself "
                "(mowl.geometric.boxes.BoxModelBuilder); reading its boxes is the identity. "
                "The geometry's truth values are checked against BoxModel.satisfies in "
                "test_recomp_construct.py.",
}
