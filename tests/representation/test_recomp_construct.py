"""The constructed methods are combinations of a source, a reading and the exact measure
(mowl.representation.construction): read the combinations back, swap their parts, and
check that the parts agree with the catalogue cards."""
import itertools
import math
import os

import pytest
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.datasets import PathDataset  # noqa: E402
from mowl.geometric.completion import CanonicalModel  # noqa: E402
from mowl.representation import ConstructorUnavailable, geometry_of  # noqa: E402
from mowl.representation.construction import (  # noqa: E402
    READINGS, RECIPES, SOURCES, ConstructionGeometry, ExactMeasure, FiniteModel, construct)
from mowl.representation.methods import METHOD_KINDS, METHODS  # noqa: E402
from tests.models.kg_region_data import AXIOMS, NS, abox_dataset  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometric",
                        "fixtures")
KEYS = ["canonical_model", "least_model", "interval_union", "convex_polyhedral",
        "convex_regions", "alcone-exact", "temporal_cones", "attributed_dl", "box_milp"]
#: The combination of every constructed method (catalogue key -> source, reading).
PARTS = {"canonical_model": ("canonical model", "element sets"),
         "least_model": ("least model", "element sets"),
         "interval_union": ("canonical model", "intervals"),
         "convex_polyhedral": ("canonical model", "simplex"),
         "convex_regions": ("finite model", "simplex"),
         "alcone-exact": ("canonical model", "half-axes"),
         "temporal_cones": ("lasso model", "half-axes"),
         "attributed_dl": ("attributed", "simplex"),
         "box_milp": ("box milp", "boxes")}
CELL_READINGS = ["element sets", "half-axes", "intervals", "simplex"]


@pytest.fixture(scope="module")
def cm():
    return CanonicalModel.from_dataset(abox_dataset())


@pytest.fixture(scope="module")
def ranked(tmp_path_factory):
    """The ontology of kg_region_data with a testing ontology of three subsumptions."""
    folder = tmp_path_factory.mktemp("constructed")

    def write(name, axioms):
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
                    f"Ontology(<http://mowl.borg/kgr/{name}>\n")
            f.write("\n".join(a.format(ns=NS) for a in axioms) + "\n)\n")
        return path

    test = ["SubClassOf(<{ns}A> <{ns}C>)", "SubClassOf(<{ns}E> <{ns}C>)",
            "SubClassOf(<{ns}B> <{ns}D>)", "Declaration(Class(<{ns}A>))"]
    return PathDataset(write("train.owl", AXIOMS), testing_path=write("test.owl", test))


def _temporal():
    from mowl.geometric.temporal_cones import TemporalOntology
    return TemporalOntology.parse(tbox=["A <= X B", "B <= X A", "A <= ~B", "C <= F A"],
                                  abox=["A(a)", "C(c)"])


def _attributed():
    from mowl.geometric.attributed import (AttributedOntology, Basic, ConceptFact,
                                           ConceptInc, Specifier)
    s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
    return AttributedOntology([
        ConceptFact("A", "a", s), ConceptFact("A", "b", t),
        ConceptInc((Basic("A", s),), Basic("B", s))])


def _helly():
    return CanonicalModel.from_dataset(PathDataset(os.path.join(FIXTURES, "helly_ok.owl")))


def _inclusion(g):
    n = g.num_classes
    a, b = zip(*itertools.product(range(n), repeat=2))
    return g.inside(g.classes(th.tensor(a)), g.classes(th.tensor(b))).reshape(n, n)


def _boxes_or_skip(cm_):
    from mowl.geometric.errors import MissingDependency
    try:
        return RECIPES["box_milp"].build(cm_)
    except MissingDependency:
        pytest.skip("no MILP solver for the box construction")


# -- (a) the combinations, read back ------------------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_every_constructed_method_is_a_registered_combination(key):
    assert METHOD_KINDS[key] == "construction"
    assert METHODS[key] == RECIPES[key].build
    parts = RECIPES[key].parts
    assert (parts["source"], parts["reading"]) == PARTS[key]
    assert parts["source"] in SOURCES and parts["reading"] in READINGS


def test_the_classes_users_get_are_the_combinations(cm):
    from mowl.geometric.attributed import AttributedConvexModel
    from mowl.geometric.cones import ALConeInterpretation
    from mowl.geometric.convex_regions import ConvexRegionModel
    from mowl.geometric.intervals import IntervalUnionModel
    from mowl.geometric.least_model import least_model
    from mowl.geometric.polyhedral import ConvexPolyhedralModel
    from mowl.geometric.temporal_cones import TemporalALConeInterpretation
    models = {"interval_union": IntervalUnionModel(cm),
              "convex_polyhedral": ConvexPolyhedralModel(cm),
              "convex_regions": ConvexRegionModel.from_canonical_model(cm),
              "alcone-exact": ALConeInterpretation.from_canonical_model(cm),
              "temporal_cones": TemporalALConeInterpretation.from_ontology(_temporal()),
              "attributed_dl": AttributedConvexModel(_attributed()),
              "canonical_model": cm,
              "least_model": least_model(3, gci0=[[0, 1]])}
    for key, model in models.items():
        g = geometry_of(model)
        assert isinstance(g, ConstructionGeometry), key
        source, reading = PARTS[key]
        if key == "alcone-exact":      # from_canonical_model goes through the finite model
            source = "finite model"
        assert (g.parts["source"], g.parts["reading"]) == (source, reading), key
        assert g.parts["measure"] == "exact"
        assert isinstance(g.measure, ExactMeasure)
        assert g.guarantees, key
        if hasattr(model, "module"):
            assert model.module is g


def test_the_constructed_model_class(ranked):
    from mowl.models import ConstructedModel
    model = ConstructedModel(ranked, method="interval_union")
    assert model.parts["source"] == "canonical model"
    assert model.parts["reading"] == "intervals"
    assert model.module.num_classes == len(model.class_index_dict)
    model.train(epochs=1)                       # nothing to train
    c = model.class_index_dict
    rows = th.tensor([[c[NS + "A"], c[NS + "C"]], [c[NS + "C"], c[NS + "A"]]])
    assert model.module(rows, "gci0").flatten().tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        ConstructedModel(ranked, method="elbe")


# -- the meaning of the regions --------------------------------------------------------------------
@pytest.mark.parametrize("reading", CELL_READINGS)
def test_every_reading_of_the_canonical_model_is_faithful_on_names(cm, reading):
    g = construct(cm, "canonical model", reading)
    expected = th.tensor([[0.0 if cm.entails(a, b) else 1.0 for b in g.names]
                          for a in g.names])
    assert th.equal(_inclusion(g), expected)
    # instance checking through the points of the individuals (the geometry of the family)
    names = [n for n in g.names]
    for k, a in enumerate(g.individual_names):
        ids = th.full((len(names),), k)
        cost = g.member(ids, g.classes(th.arange(len(names)))).flatten()
        assert cost.tolist() == [0.0 if cm.instance_of(a, n) else 1.0 for n in names]


@pytest.mark.parametrize("reading", CELL_READINGS)
def test_the_points_of_the_regions_are_the_cells(cm, reading):
    """``contains`` (the geometry of the family) agrees with the cells of the reading."""
    g = construct(cm, "canonical model", reading)
    regions = g.read.classes
    n = regions.n_cells
    points = th.stack([g.read.universe.point({k}) for k in range(n)])
    inside = regions.contains(points)                      # (classes, cells)
    assert th.equal(inside, regions.mask[:, 0])


@pytest.mark.parametrize("reading", CELL_READINGS)
def test_all_readings_give_the_same_truth_values_of_the_normal_forms(cm, reading):
    """φ(η) = M: every reading of one finite model satisfies the same normal-form axioms."""
    reference = construct(cm, "canonical model", "element sets")
    g = construct(cm, "canonical model", reading)
    n, r = g.num_classes, len(g.roles)
    rows = {"gci0": list(itertools.product(range(n), repeat=2)),
            "gci1": list(itertools.product(range(n), repeat=3)),
            "gci2": [(a, k, b) for a in range(n) for k in range(r) for b in range(n)],
            "gci3": [(k, a, b) for k in range(r) for a in range(n) for b in range(n)],
            "gci1_bot": list(itertools.product(range(n), repeat=2)),
            "gci0_bot": [(a, a) for a in range(n)],
            "role_inclusion": list(itertools.product(range(r), repeat=2)),
            "role_chain": list(itertools.product(range(r), repeat=3))}
    for form, data in rows.items():
        data = th.tensor(data)
        assert th.equal(g(data, form), reference(data, form)), form


def test_least_model_geometry_is_the_least_model():
    from mowl.geometric.least_model import least_model
    lm = least_model(5, nb_rels=1, nb_individuals=2, gci0=[[0, 1], [1, 2]],
                     gci2=[[0, 0, 3]], gci0_bot=[[4, 4]], class_assertion=[[0, 0], [1, 3]])
    for reading in CELL_READINGS:
        g = RECIPES["least_model"].build(lm, reading=reading)
        expected = th.tensor([[0.0 if lm.holds(a, b) else 1.0 for b in range(5)]
                              for a in range(5)])
        assert th.equal(_inclusion(g), expected), reading
        member = g.member(th.tensor([0] * 5 + [1] * 5), g.classes(th.arange(5).repeat(2)))
        expected = [0.0 if k in lm.instances(c) else 1.0 for k in (0, 1) for c in range(5)]
        assert member.flatten().tolist() == expected, reading


def test_box_model_geometry_is_the_box_model():
    from mowl.geometric.normal_forms import NF1, NF2
    helly = _helly()
    g = _boxes_or_skip(helly)
    model = g.built
    names = g.names
    nf1 = th.tensor([[0.0 if model.satisfies(NF1(a, b)) else 1.0] for a in names
                     for b in names])
    assert th.equal(_inclusion(g).reshape(-1, 1), nf1)
    triples = list(itertools.product(range(len(names)), repeat=3))
    nf2 = th.tensor([[0.0 if model.satisfies(NF2(names[a], names[b], names[c])) else 1.0]
                     for a, b, c in triples])
    assert th.equal(g(th.tensor(triples), "gci1"), nf2)


# -- (b) swaps ----------------------------------------------------------------------------------
SWAPS = {
    "canonical_model": [{"reading": "half-axes"}, {"reading": "intervals"}],
    "least_model": [{"reading": "intervals"}, {"reading": "simplex"}],
    "interval_union": [{"reading": "half-axes"}, {"source": "least model"},
                       {"reading_options": {"half_width": 0.1}}],
    "convex_polyhedral": [{"reading": "element sets"}, {"source": "least model"}],
    "convex_regions": [{"reading": "intervals"}, {"reading": "half-axes"}],
    "alcone-exact": [{"reading": "element sets"}, {"reading": "simplex"},
                     {"source": "least model"}],
}


@pytest.mark.parametrize("key, swap", [(k, s) for k, swaps in SWAPS.items() for s in swaps],
                         ids=lambda v: v if isinstance(v, str) else "-".join(
                             f"{a}={b}" for a, b in v.items()))
def test_swapped_parts_train_and_evaluate(ranked, key, swap):
    """A swapped combination is built ("trained" for one epoch: nothing to do), and an
    evaluation regime ranks with it: finite numbers."""
    from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
    from mowl.models import ConstructedModel
    model = ConstructedModel(ranked, method=key, **swap)
    model.train(epochs=1)
    if "reading" in swap:
        assert model.parts["reading"] == swap["reading"]
    if "source" in swap:
        assert model.parts["source"] == swap["source"]
    result = NormalFormRankingRegime(normal_forms=("gci0",), combined=False).evaluate(
        model, ranked)
    values = [v for v in result["gci0"].values() if isinstance(v, float)]
    assert values and all(math.isfinite(v) for v in values), result


@pytest.mark.parametrize("key", ["interval_union", "canonical_model"])
def test_another_evaluation_regime_the_exact_model_check(ranked, key):
    from mowl.evaluation.exact_model import ExactModelRegime
    from mowl.models import ConstructedModel
    model = ConstructedModel(ranked, method="interval_union")
    if key == "canonical_model":
        # element sets are no regions of R^d: the exact-model check says it cannot read them
        with pytest.raises(NotImplementedError):
            ExactModelRegime().evaluate(ConstructedModel(ranked, method=key), ranked)
        return
    result = ExactModelRegime().evaluate(model, ranked)["all"]
    assert result["is_model"] == 1.0 and result["axiom_satisfaction"] == 1.0


def test_box_milp_with_two_regimes_and_refused_swaps():
    from mowl.evaluation.exact_model import ExactModelRegime
    from mowl.evaluation.inclusion_accuracy import InclusionAccuracyRegime
    from mowl.models import ConstructedModel
    from mowl.geometric.errors import MissingDependency
    dataset = PathDataset(os.path.join(FIXTURES, "helly_ok.owl"))
    try:
        model = ConstructedModel(dataset, method="box_milp")
    except MissingDependency:
        pytest.skip("no MILP solver for the box construction")
    model.train(epochs=1)
    assert ExactModelRegime().evaluate(model, dataset)["all"]["is_model"] == 1.0
    accuracy = InclusionAccuracyRegime(splits=("train",)).evaluate(model, dataset)
    assert accuracy["train"]["accuracy"] == 1.0
    # swaps that are not meaningful are refused with the reason
    with pytest.raises(ConstructorUnavailable, match="no domain elements"):
        ConstructedModel(dataset, method="box_milp", reading="half-axes")
    with pytest.raises(ConstructorUnavailable, match="Helly"):
        ConstructedModel(dataset, method="canonical_model", reading="boxes")


def test_temporal_cones_swaps():
    onto = _temporal()
    for reading in ("element sets", "intervals", "half-axes"):
        g = RECIPES["temporal_cones"].build(onto, reading=reading)
        one = {n: g.classes(th.tensor([g.names.index(n)])) for n in "ABC"}
        assert g.inside(one["A"], g.next(one["B"])).item() == 0.0
        assert g.inside(one["A"], one["B"]).item() == 1.0
        assert g.inside(one["C"], g.until(g.top(), one["A"])).item() == 0.0
        assert g.member(th.tensor([0]), one["A"]).item() == 0.0
    # faces of a simplex have no unions, so no until
    g = RECIPES["temporal_cones"].build(onto, reading="simplex")
    a = g.classes(th.tensor([0]))
    assert g.inside(a, g.next(a)).item() in (0.0, 1.0)
    with pytest.raises(ConstructorUnavailable):
        g.until(a, a)


def test_attributed_dl_swaps():
    from mowl.geometric.attributed import Specifier
    s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
    names = [("A", s), ("A", t), ("B", s), ("B", t)]
    costs = []
    for reading in ("simplex", "half-axes", "intervals", "element sets"):
        g = RECIPES["attributed_dl"].build(_attributed(), reading=reading, names=names)
        cost = _inclusion(g)
        assert cost[0, 2] == 0.0 and cost[1, 3] == 1.0
        ind = g.individual_names.index("a")
        assert g.member(th.tensor([ind]), g.classes(th.tensor([2]))).item() == 0.0
        costs.append(cost)
    assert all(th.equal(c, costs[0]) for c in costs)


def test_boolean_atoms_read_as_cones_and_as_sets():
    from mowl.geometric.cones import And, Atomic, ConceptAssertion, GCI, Not, Or
    A, B, C = Atomic("A"), Atomic("B"), Atomic("C")
    axioms = [GCI(A, B), GCI(And((B, C)), Not(A)), ConceptAssertion(Or((A, C)), "a")]
    for faithful in (False, True):
        cones = RECIPES["alcone-exact"].build(
            axioms, source="boolean atoms", source_options={"faithful": faithful})
        sets = construct(axioms, "boolean atoms", "element sets",
                         source_options={"faithful": faithful})
        assert th.equal(_inclusion(cones), _inclusion(sets))
        a = cones.classes(th.tensor([cones.names.index("A")]))
        b = cones.classes(th.tensor([cones.names.index("B")]))
        assert cones.inside(a, b).item() == 0.0
        assert cones.inside(cones.top(), cones.disjunction(a, cones.negation(a))).item() == 0.0
        member = cones.member(th.tensor([0, 0]), cones.disjunction(a, cones.classes(
            th.tensor([cones.names.index("C")]))).take(th.tensor([0, 0])))
        # with the faithful (doubled) reading a is in A or C, but in neither alone
        assert member.flatten().tolist() == [0.0, 0.0]
    with pytest.raises(ConstructorUnavailable):
        construct(axioms, "boolean atoms", "intervals", source_options={"faithful": True})


# -- (c) the parts agree with the catalogue cards ---------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_the_parts_agree_with_the_card(cm, key):
    from mowl.catalog import get_method
    from mowl.representation import REPRESENTATIONS
    card, rep = get_method(key), REPRESENTATIONS[key]
    assert card.status.value == "constructive"
    reading = READINGS[RECIPES[key].reading](**RECIPES[key].reading_options)
    data = {"temporal_cones": _temporal, "attributed_dl": _attributed,
            "box_milp": _helly}.get(key, lambda: cm)()
    if key == "box_milp":
        g = _boxes_or_skip(data)
    elif key == "least_model":
        from mowl.geometric.least_model import least_model
        g = RECIPES[key].build(least_model(3, nb_rels=1, nb_individuals=1, gci2=[[0, 0, 1]],
                                           class_assertion=[[0, 0]]))
    else:
        g = RECIPES[key].build(data)
    parts = g.parts
    assert rep.classes[0].value == reading.class_part == parts["classes"]
    if rep.relations[0].value == "none":
        assert parts["relations"] == "none"
    else:              # what a relation is in the reading (the test data may have no roles)
        assert rep.relations[0].value == reading.relation_part
        assert parts["relations"] in (reading.relation_part, "none")
    assert rep.individuals[0].value == reading.individual_part
    assert parts["individuals"] in (reading.individual_part, "none")
    assert rep.measure[0].value == parts["measure"] == "exact"
    # every constructor the card states as exact is exact in the combination (a reading may
    # be closed under more: the canonical model read as element sets has a complement)
    for name, construction in rep.constructors.items():
        if construction.value == "exact":
            assert reading.constructors[name] == "exact", name
            assert parts.get(name, "exact") in ("exact", "none"), name
    # the tasks of the card are answered by the geometry
    if "subsumption" in card.tasks:
        assert g(th.tensor([[0, 0]]), "gci0").item() == 0.0
    if "satisfiability" in card.tasks:
        assert g.nonempty(g.top()).item() == 0.0 and g.nonempty(g.bottom()).item() == 1.0


def test_finite_model_conversions(cm):
    from mowl.geometric.cones import FiniteInterpretation as ConesFI
    from mowl.geometric.convex_regions import FiniteInterpretation
    for model in (cm, FiniteInterpretation.from_canonical_model(cm),
                  ConesFI.from_canonical_model(cm)):
        fm = FiniteModel.of(model)
        assert fm.size == len(cm.names)
        for name in cm.concept_names:
            if name in fm.concepts:
                assert fm.extension(name) == cm.extension(name)
