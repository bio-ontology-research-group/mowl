"""Constructed methods: the combination of a source and a reading against the direct
construction.

Nothing is trained, so there are no losses and no training run to compare; the checks
are everything a construction produces, element by element: the constructed regions and
points (as tensors), the truth values of test axioms of every axiom form the checker knows
(subsumption, conjunction, bottom and top forms, existential forms on both sides, role
inclusion and role chains; an axiom a checker refuses is recorded as such, so both sides must
refuse the same ones), and the guarantees the construction promises (model, non-degeneracy,
strong faithfulness). ``scores`` reads the public interface the two classes share.
"""
import itertools

import numpy as np
import torch as th

from tests.baselines.harness import Pair

_NOT_TRAINED = ("The method is constructed (completion, closed form, search), not trained "
                "by gradient descent: there is no loss and no training run.")
_SKIP = {"losses": _NOT_TRAINED, "training": _NOT_TRAINED}


def _ns(x):
    """The package of the geometric syntax and mathematics that the object ``x`` (or the module
    name ``x``) belongs to: the library's, or the frozen copy of the baselines."""
    name = x if isinstance(x, str) else type(x).__module__
    return "mowl.baselines.geometric.core" if name.startswith("mowl.baselines") \
        else "mowl.geometric"


def _imp(x, module):
    import importlib
    return importlib.import_module(f"{_ns(x)}.{module}")


class _Sided:
    """Input data made of classes of the geometric package: each side gets objects of its
    own package (the baselines have their own copy of the classes)."""

    def __init__(self, make):
        self.make = make

    def get(self, side):
        return self.make(_ns(side))


def _abox():
    from tests.models.kg_region_data import abox_dataset
    return abox_dataset()


def _cm(dataset, side="mowl.geometric"):
    return _imp(side, "completion").CanonicalModel.from_dataset(dataset)


def _no_copy(old, new):
    """Nothing to copy: both are built from the same data."""


def _nf_axioms(cm):
    """Normal-form test axioms of every form over the names of the canonical model."""
    nf = _imp(cm, "normal_forms")
    BOTTOM, NF1, NF2, NF3, NF4, NF6, NF7, TOP = (nf.BOTTOM, nf.NF1, nf.NF2, nf.NF3, nf.NF4,
                                                nf.NF6, nf.NF7, nf.TOP)
    names = sorted(cm.names)
    axioms = [NF1(a, b) for a, b in itertools.product(names, repeat=2)]
    axioms += [NF2(a, b, c) for a, b, c in itertools.product(names[:6], repeat=3)]
    axioms += [NF2(a, b, BOTTOM) for a, b in itertools.product(names, repeat=2)]
    for r in sorted(cm.role_names):
        axioms += [NF3(a, r, b) for a, b in itertools.product(names, repeat=2)]
        axioms += [NF4(r, a, b) for a, b in itertools.product(names, repeat=2)]
        axioms += [NF3(a, r, BOTTOM) for a in names] + [NF4(r, a, BOTTOM) for a in names]
        axioms += [NF3(a, r, TOP) for a in names] + [NF4(r, TOP, a) for a in names]
    axioms += [NF1(TOP, a) for a in names] + [NF1(a, TOP) for a in names]
    axioms += [NF1(a, BOTTOM) for a in names]
    roles = sorted(cm.role_names)
    axioms += [NF6(r, s) for r, s in itertools.product(roles, repeat=2)]
    axioms += [NF7(r, s, t) for r, s, t in itertools.product(roles, repeat=3)]
    return axioms


def _truth(model, axioms):
    """1 if the model satisfies the axiom, 0 if not, 2 if its checker refuses the axiom
    (both sides must refuse the same axioms)."""
    out = []
    for axiom in axioms:
        try:
            out.append(float(model.satisfies(axiom)))
        except Exception:
            out.append(2.0)
    return th.tensor(out)


def _guarantees(model):
    """The guarantees of a construction that a model offers: a model of its ontology,
    non-degenerate (every satisfiable name has a non-empty region), strongly faithful on
    names (region inclusion is entailment)."""
    return th.tensor([float(getattr(model, name)()) for name in
                      ("is_model", "is_nondegenerate", "is_strongly_faithful_on_names")])


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
    return {"regions": regions, "roles": roles, "truth": _truth(model, _nf_axioms(cm)),
            "guarantees": _guarantees(model)}


def _new_intervals(dataset):
    from mowl.geometric.intervals import IntervalUnionModel
    return IntervalUnionModel.from_canonical_model(_cm(dataset))


def _old_intervals(dataset):
    from mowl.baselines.geometric.intervals import IntervalUnionModel
    return IntervalUnionModel.from_canonical_model(_cm(dataset, "mowl.baselines"))


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
    if hasattr(model, "is_strongly_faithful_on_names"):
        out["guarantees"] = _guarantees(model)
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
    return ConvexPolyhedralModel.from_canonical_model(_cm(dataset, "mowl.baselines"),
                                                      thicken=thicken)


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
    out["violations"] = th.tensor([float(len(model.check().violations(cm_axioms)))])
    out["is_model of the ontology"] = th.tensor(
        [float(model.check().is_model(model._ontology_axioms))])
    return out


def _convex(module):
    def build(dataset):
        import importlib
        cm = _cm(dataset, module)
        cls = importlib.import_module(module).ConvexRegionModel
        model = cls.from_canonical_model(cm)
        model._test_axioms = _nf_axioms(cm)
        model._ontology_axioms = list(cm.axioms)
        return model
    return build


# -- al-cones ----------------------------------------------------------------------------------
def _alc_axioms(names, roles, individuals, side="mowl.geometric"):
    c = _imp(side, "cones")
    And, Atomic, ConceptAssertion, Exists, Forall, GCI, Not, Or, RoleAssertion = (
        c.And, c.Atomic, c.ConceptAssertion, c.Exists, c.Forall, c.GCI, c.Not, c.Or,
        c.RoleAssertion)
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
    out += [RoleAssertion(r, i, j) for r in roles for i in individuals for j in individuals]
    return out


def _alc_concepts(names, roles, side="mowl.geometric"):
    """Compound concepts whose al-cones are compared entry by entry."""
    c = _imp(side, "cones")
    And, Atomic, Exists, Forall, Not, Or = c.And, c.Atomic, c.Exists, c.Forall, c.Not, c.Or
    atoms = [Atomic(n) for n in names]
    out = list(atoms) + [Not(a) for a in atoms]
    out += [And((a, b)) for a, b in itertools.product(atoms[:4], repeat=2)]
    out += [Or((a, b)) for a, b in itertools.product(atoms[:4], repeat=2)]
    for r in roles:
        out += [Exists(r, a) for a in atoms] + [Forall(r, a) for a in atoms]
        out += [Exists(r, Not(a)) for a in atoms[:4]]
    return out


def _cone_scores(model):
    ALConeChecker = _imp(model, "cones").ALConeChecker
    names, roles = sorted(model.concepts), sorted(model.roles)
    individuals = sorted(model.individuals)
    checker = ALConeChecker(model)
    out = {"bits": th.tensor(np.stack([model.concepts[n].bits for n in names]),
                             dtype=th.float64),
           "truth": th.tensor([float(checker.satisfies(ax))
                               for ax in _alc_axioms(names, roles, individuals, model)]),
           "dim": th.tensor([float(model.dim)])}
    concepts = _alc_concepts(names, roles, model)
    out["concept cones"] = th.tensor(np.stack([model.cone(c).bits for c in concepts]),
                                     dtype=th.float64)
    if hasattr(model, "_source_axioms"):
        out["is_model of the ontology"] = th.tensor(
            [float(checker.is_model(model._source_axioms))])
        out["violations"] = th.tensor([float(len(checker.violations(model._source_axioms)))])
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
            return cls.from_canonical_model(_cm(dataset, module))
        axioms = dataset.get(module)
        model = cls.from_boolean_ontology(axioms, faithful=how == "faithful")
        model._source_axioms = list(axioms)
        return model
    return build


def _boolean_tbox():
    return _Sided(_boolean_tbox_of)


def _boolean_tbox_of(ns):
    import importlib
    c = importlib.import_module(ns + ".cones")
    And, Atomic, ConceptAssertion, GCI, Not, Or = (c.And, c.Atomic, c.ConceptAssertion, c.GCI,
                                                  c.Not, c.Or)
    A, B, C, D = (Atomic(x) for x in "ABCD")
    return [GCI(A, B), GCI(And((B, C)), Not(D)), GCI(D, Or((A, C))),
            ConceptAssertion(A, "a"), ConceptAssertion(Or((C, D)), "b"),
            ConceptAssertion(Not(B), "c")]


# -- temporal al-cones --------------------------------------------------------------------------
def _temporal_ontology():
    return _Sided(_temporal_ontology_of)


def _temporal_ontology_of(ns):
    import importlib
    return importlib.import_module(ns + ".temporal_cones").TemporalOntology.parse(
        tbox=["A <= X B", "B <= X A", "A <= ~B", "C <= F A"],
        abox=["A(a)", "C(c)", "~C(b)"])


def _temporal(module):
    def build(onto):
        import importlib
        cls = importlib.import_module(module).TemporalALConeInterpretation
        onto = onto.get(module)
        model = cls.from_ontology(onto, max_prefix=1, max_period=2)
        model._source_ontology = onto
        return model
    return build


def _temporal_scores(model):
    parse_axiom = _imp(model, "temporal_cones").parse_axiom
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
    onto = model._source_ontology
    concepts = [parse_axiom(f"{n} <= {op}{m}").sup for n in names for m in names
                for op in ("", "X ", "F ", "G ", "~")]
    cones = th.tensor(np.stack([model.cone(c, t).bits for c in concepts
                                for t in range(model.length)]), dtype=th.float64)
    return {"bits": bits, "truth": truth, "members": members, "concept cones": cones,
            "is_model of the ontology": th.tensor([float(model.is_model(onto))]),
            "violations": th.tensor([float(len(model.violations(onto.axioms)))]),
            "successor": th.tensor([float(model.successor(t)) for t in range(model.length)]),
            "individuals": th.tensor(np.stack([model.individuals[i] for i in individuals])),
            "shape": th.tensor([float(model.dim), float(model.length),
                                float(model.loop_start), float(len(model.traces))])}


# -- attributed ---------------------------------------------------------------------------------
def _attributed_ontology():
    return _Sided(_attributed_ontology_of)


def _attributed_ontology_of(ns):
    import importlib
    a = importlib.import_module(ns + ".attributed")
    AttributedOntology, Basic, ConceptFact, ConceptInc, RoleFact, RoleInc, Specifier = (
        a.AttributedOntology, a.Basic, a.ConceptFact, a.ConceptInc, a.RoleFact, a.RoleInc,
        a.Specifier)
    s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
    return AttributedOntology([
        ConceptFact("A", "a", s), ConceptFact("A", "b", t), ConceptFact("C", "b", s),
        RoleFact("R", "a", "b", s), RoleFact("R", "b", "a", t), RoleInc("R", s, "Q"),
        ConceptInc((Basic("R", s, exists=True),), Basic("C", s)),
        ConceptInc((Basic("A", s),), Basic("B", s)),
        ConceptInc((Basic("B", s), Basic("C", s)), Basic("D", s))])


def _attributed(module):
    def build(onto):
        import importlib
        return importlib.import_module(module).AttributedConvexModel(onto.get(module))
    return build


def _attributed_scores(model):
    a = _imp(model, "attributed")
    ANY, Specifier = a.ANY, a.Specifier
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
    roles = [(r, sp) for r in "RQ" for sp in specs]
    role_regions = {f"role {r}@{k}": th.as_tensor(model.role_region(r, sp).points,
                                                  dtype=th.float64)
                    for k, (r, sp) in enumerate(roles)}
    holds_role = th.tensor([[float(model.holds_role(a, r, b, sp)) for (r, sp) in roles]
                            for a in "ab" for b in "ab"])
    entailed = th.tensor([[float(model.entails_instance(a, c, sp)) for (c, sp) in names]
                          for a in "ab"])
    satisfiable = th.tensor([float(model.is_satisfiable(c, sp)) for (c, sp) in names])
    axioms = th.tensor([float(model.satisfies(k)) for k in range(len(model.ontology.axioms))])
    return {"vertices": vertices, "holds": holds, "included": included,
            "is_model": th.tensor([float(model.is_model())]),
            "violations": th.tensor([float(len(model.violations()))]),
            "axioms satisfied": axioms, "holds role": holds_role, "entailed": entailed,
            "satisfiable": satisfiable, **role_regions,
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
