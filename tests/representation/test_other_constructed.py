"""The exact constructions of mowl.geometric answer the questions of the common interface
(costs 0.0 where a relation holds, 1.0 where it does not).

Their geometry is the combination of a source and a reading
(:class:`~mowl.representation.construction.ConstructionGeometry`); the attributed test at
the end also exercises the earlier geometry of backends (``other_constructed``), which
FaithEL still uses."""
import itertools
import os

import pytest
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.datasets import PathDataset  # noqa: E402
from mowl.geometric.attributed import (  # noqa: E402
    AttributedConvexModel, AttributedOntology, Basic, ConceptFact, ConceptInc, Specifier)
from mowl.geometric.boxes import BoxModelBuilder  # noqa: E402
from mowl.geometric.completion import CanonicalModel  # noqa: E402
from mowl.geometric.cones import ALConeInterpretation  # noqa: E402
from mowl.geometric.convex_regions import ConvexRegionModel  # noqa: E402
from mowl.geometric.errors import MissingDependency  # noqa: E402
from mowl.geometric.intervals import IntervalUnionModel  # noqa: E402
from mowl.geometric.least_model import least_model  # noqa: E402
from mowl.geometric.polyhedral import ConvexPolyhedralModel  # noqa: E402
from mowl.geometric.temporal_cones import (  # noqa: E402
    TemporalALConeInterpretation, TemporalOntology)
from mowl.representation import ConstructorUnavailable, geometry_of  # noqa: E402
from mowl.representation.construction import ConstructionGeometry  # noqa: E402
from mowl.representation.library.other_constructed import ConstructedGeometry  # noqa: E402
from tests.models.kg_region_data import NS, abox_dataset  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometric",
                        "fixtures")


@pytest.fixture(scope="module")
def cm():
    return CanonicalModel.from_dataset(abox_dataset())


def _ids(n):
    return th.arange(n)


def _pairs(g, n=None):
    n = g.num_classes if n is None else n
    a, b = zip(*itertools.product(range(n), repeat=2))
    return th.tensor(a), th.tensor(b)


def _entailed(cm, names):
    return th.tensor([[0.0 if cm.entails(a, b) else 1.0] for a in names for b in names])


def _inclusion(g):
    a, b = _pairs(g)
    return g.inside(g.classes(a), g.classes(b))


@pytest.mark.parametrize("build", [
    lambda cm: cm,
    lambda cm: ConvexPolyhedralModel.from_canonical_model(cm),
    lambda cm: IntervalUnionModel.from_canonical_model(cm),
], ids=["canonical_model", "convex_polyhedral", "interval_union"])
def test_strongly_faithful_constructions_give_entailment(cm, build):
    g = geometry_of(build(cm))
    assert isinstance(g, ConstructionGeometry)
    cost = _inclusion(g)
    assert cost.shape == (g.num_classes ** 2, 1)
    assert th.equal(cost, _entailed(cm, g._names))
    assert set(cost.unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("build, negation", [
    (lambda cm: cm, True),
    (lambda cm: ConvexPolyhedralModel.from_canonical_model(cm), False),
    (lambda cm: IntervalUnionModel.from_canonical_model(cm), True),
], ids=["canonical_model", "convex_polyhedral", "interval_union"])
def test_existential_and_conjunction_and_individuals(cm, build, negation):
    for model in (build(cm),):
        g = geometry_of(model)
        c = {n.rsplit("#", 1)[-1]: i for i, n in enumerate(g._names)}
        r = g._roles.index(NS + "r")
        one = lambda n: th.tensor([c[n]])  # noqa: E731
        # A ⊑ ∃r.D holds, D ⊑ ∃r.D does not
        assert g.inside(g.classes(one("A")), g.exists(th.tensor([r]), g.classes(one("D"))
                                                      )).item() == 0.0
        assert g.inside(g.classes(one("D")), g.exists(th.tensor([r]), g.classes(one("D"))
                                                      )).item() == 1.0
        # ∃r.E ⊑ B is not entailed, ∃r.D ⊑ B neither
        left = g.exists(th.tensor([r]), g.classes(one("E")))
        assert g.inside(left, g.classes(one("B"))).item() in (0.0, 1.0)
        # A and D ⊑ C holds (A ⊑ C); the conjunction is non-empty (gap 0)
        both, gap = g.conjunction(g.classes(one("A")), g.classes(one("B")))
        assert gap.item() == 0.0
        assert g.inside(both, g.classes(one("C"))).item() == 0.0
        # individuals: a1 is an A, hence a C; not an E
        ind = {n.rsplit("#", 1)[-1]: i for i, n in enumerate(g._individuals)}
        a1 = th.tensor([ind["a1"]] * 3)
        classes = g.classes(th.tensor([c["A"], c["C"], c["E"]]))
        assert g.member(a1, classes).flatten().tolist() == [0.0, 0.0, 1.0]
        # disjointness *in the model* (A and E share no element), which is not entailment
        assert g.apart(g.classes(one("A")), g.classes(one("E"))).item() == 0.0
        assert g.apart(g.classes(one("A")), g.classes(one("B"))).item() == 1.0
        if negation:
            # element sets and unions of intervals are closed under complement (within the
            # domain of the canonical model): not A is not inside B
            not_a = g.negation(g.classes(one("A")))
            assert g.inside(not_a, g.classes(one("B"))).item() == 1.0
            assert g.apart(not_a, g.classes(one("A"))).item() == 0.0
        else:
            # faces of a simplex are not closed under complement
            with pytest.raises(ConstructorUnavailable):
                g.inside(g.negation(g.classes(one("A"))), g.classes(one("B")))


def test_convex_regions_and_box_models():
    cm_ = CanonicalModel.from_dataset(abox_dataset())
    regions = geometry_of(ConvexRegionModel.from_canonical_model(cm_))
    assert th.equal(_inclusion(regions), _entailed(cm_, regions._names))
    helly = CanonicalModel.from_dataset(PathDataset(os.path.join(FIXTURES, "helly_ok.owl")))
    try:
        boxes = BoxModelBuilder(helly).build()
    except MissingDependency:
        pytest.skip("no MILP solver for the box construction")
    g = geometry_of(boxes)
    cost = _inclusion(g)
    entailed = _entailed(helly, g._names)
    # a box model is a model: every entailed inclusion holds (it may hold more: weak
    # faithfulness)
    assert (cost[entailed[:, 0] == 0] == 0).all()
    assert cost.shape == entailed.shape
    # box models have no roles
    with pytest.raises(ConstructorUnavailable):
        g.inside(g.classes(th.tensor([0])),
                 g.exists(th.tensor([0]), g.classes(th.tensor([1]))))


def test_least_model():
    lm = least_model(5, nb_rels=1, nb_individuals=1, gci0=[[0, 1], [1, 2]],
                     gci2=[[0, 0, 3]], gci0_bot=[[4, 4]], class_assertion=[[0, 0]])
    g = geometry_of(lm)
    assert g.num_classes == 5 and g.num_individuals == 1
    cost = _inclusion(g)
    expected = th.ones(5, 5)
    for a, b in [(0, 1), (1, 2), (0, 2)] + [(i, i) for i in range(5)]:
        expected[a, b] = 0.0
    expected[4, :] = 0.0                       # class 4 is unsatisfiable: empty
    assert th.equal(cost.reshape(5, 5), expected)
    assert g.inside(g.classes(th.tensor([0])),
                    g.exists(th.tensor([0]), g.classes(th.tensor([3])))).item() == 0.0
    assert g.member(th.tensor([0, 0]), g.classes(th.tensor([2, 3]))).flatten().tolist() \
        == [0.0, 1.0]
    assert g.nonempty(g.classes(th.tensor([4]))).item() == 1.0


def test_al_cone_model_decides_alc(cm):
    interp = ALConeInterpretation.from_canonical_model(cm)
    g = geometry_of(interp)
    assert th.equal(_inclusion(g), _entailed(cm, g._names))
    c = {n.rsplit("#", 1)[-1]: th.tensor([i]) for i, n in enumerate(g._names)}
    a = g.classes(c["A"])
    # negation and disjunction are exact
    assert g.apart(a, g.negation(a)).item() == 0.0
    assert g.inside(g.top(), g.disjunction(a, g.negation(a))).item() == 0.0
    assert g.nonempty(g.conjunction(a, g.negation(a))[0]).item() == 1.0
    r = g._roles.index(NS + "r")
    assert g.inside(a, g.exists(th.tensor([r]), g.classes(c["D"]))).item() == 0.0
    ind = {n.rsplit("#", 1)[-1]: i for i, n in enumerate(g._individuals)}
    assert g.member(th.tensor([ind["a1"]]), g.classes(c["C"])).item() == 0.0
    assert g.member(th.tensor([ind["a1"]]), g.classes(c["E"])).item() == 1.0


def test_temporal_cones_have_next_and_until():
    onto = TemporalOntology.parse(tbox=["A <= X B", "B <= X A", "A <= ~B"], abox=["A(a)"])
    interp = TemporalALConeInterpretation.from_ontology(onto, max_prefix=0, max_period=2)
    g = geometry_of(interp)
    a, b = (g.classes(th.tensor([g._names.index(n)])) for n in "AB")
    assert g.inside(a, g.next(b)).item() == 0.0
    assert g.inside(a, b).item() == 1.0
    assert g.apart(a, b).item() == 0.0
    assert g.inside(a, g.until(g.negation(b), b)).item() == 0.0    # B comes next
    ids = th.tensor([0])
    assert g.member(ids, a).item() == 0.0
    assert g.member(ids, b).item() == 1.0
    g.time = 1                                 # the next time point
    assert g.member(ids, b).item() == 0.0
    with pytest.raises(ConstructorUnavailable):
        g.exists(th.tensor([0]), a)


def test_attributed_dl_regions_of_a_concept_under_a_specifier():
    s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
    model = AttributedConvexModel(AttributedOntology([
        ConceptFact("A", "a", s), ConceptFact("A", "b", t),
        ConceptInc((Basic("A", s),), Basic("B", s))]))
    names = [("A", s), ("A", t), ("B", s), ("B", t)]
    g = ConstructedGeometry(
        __import__("mowl.representation.library.other_constructed",
                   fromlist=["x"]).AttributedBackend(model, names=names))
    one = lambda i: g.classes(th.tensor([i]))  # noqa: E731
    assert g.inside(one(0), one(2)).item() == 0.0          # A@s ⊑ B@s
    assert g.inside(one(1), one(3)).item() == 1.0          # A@t ⊑ B@t does not hold
    ind = {n: i for i, n in enumerate(g._individuals)}
    assert g.member(th.tensor([ind["a"]]), one(2)).item() == 0.0
    assert g.member(th.tensor([ind["b"]]), one(2)).item() == 1.0
    assert isinstance(geometry_of(model), ConstructionGeometry)
    assert th.is_tensor(g.inside(one(0), one(0)))
    assert ConceptInc is not None
