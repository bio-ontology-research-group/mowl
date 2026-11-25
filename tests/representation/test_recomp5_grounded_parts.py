"""The general parts GroundedEL is now wired from, each on its own and in a combination
other than GroundedEL: placing a finite interpretation, re-seeding a union of boxes, the
degrees of encoded points, the exact-domain check, the loop of a sampled objective and the
canonical row order."""
import pytest
import torch as th
import torch.nn.functional as F

from mowl.datasets.rows import lexicographic_order, sort_rows
from mowl.nn.el.structural import Regions
from mowl.representation.library import grounded_parts as gp
from mowl.representation.library.finite_model_placement import (
    FiniteModelPlacement, close_labels, edge_triples, edges_by_relation, interpretation_labels,
    live_elements, separated_points, syntactic_edges)
from mowl.representation.library.grounded_evaluation import ExactDomainCheck
from mowl.representation.library.inductive_membership import (MembershipCrossEntropy,
                                                              SoftClassScores)
from mowl.representation.library.other_regions import MaskedUnionOfBoxes
from mowl.representation.library.union_reseeding import OwnPointReseeding, slot_point_counts
from mowl.training.regime import ComposedRegime, TrainingRegime
from mowl.training.sampled_objective import SampledObjectiveTraining, find_regime
from mowl.training.schedules import TemperatureAnnealing


def _linf(points):
    d = th.amax(th.abs(points.unsqueeze(1) - points.unsqueeze(0)), dim=-1)
    return d.masked_fill(th.eye(len(points), dtype=th.bool), float("inf")).min()


class _Parts(th.nn.Module):
    """A geometry of the three parts without GroundedELModule (another combination)."""

    def __init__(self, n_elements, n_classes, n_rels, dim, k, relations="translation",
                 agg="max"):
        super().__init__()
        self.nb_classes, self.nb_elements = n_classes, n_elements
        self.measure = gp.UnionDepth(agg=agg)
        self.elements = gp.ElementDomain(n_elements, n_classes, dim)
        self.class_model = gp.BoxUnionClasses(n_classes, k, dim, self.measure, self.elements)
        factory = gp.GROUNDED_PARTS["relations"][relations]
        self.roles = factory(n_rels, k, dim, self.measure, self.elements)
        self.n_components = k

    def positions(self, ids=None):
        return self.elements.positions(ids)

    def dead(self, ids=None):
        return self.elements.dead(ids)

    def class_regions(self, ids=None):
        return self.class_model.regions(ids)

    def depth(self, points, ids, crisp=False):
        return self.class_model.depth(points, ids, crisp=crisp)


# -- placing a finite interpretation ------------------------------------------------------
@pytest.mark.parametrize("differences", [False, True])
def test_separated_points_are_four_delta_apart(differences):
    p = separated_points(6, 3, 0.05, seed=1, distinct_differences=differences)
    assert p.dtype == th.float64 and p.shape == (6, 3)
    assert float(_linf(p)) >= 0.2 - 1e-12
    if differences:
        diff = (p.unsqueeze(0) - p.unsqueeze(1)).reshape(-1, 3)[~th.eye(6, dtype=th.bool).reshape(-1)]
        assert float(_linf(diff)) >= 0.2 - 1e-12
        assert float(th.amax(th.abs(diff), dim=-1).min()) >= 0.2 - 1e-12
    assert th.equal(p, separated_points(6, 3, 0.05, seed=1, distinct_differences=differences))


def test_reading_an_interpretation():
    classes = {"A": 0, "B": 1, "T": 2, "Bot": 3}
    elements = dict(classes, a=4)
    labels = interpretation_labels({"A": ["A", "a"], "a": ["B"]}, 5, classes, elements)
    assert labels[0, 0] and labels[4, 0] and labels[4, 1] and not labels[1, 1]
    close_labels(labels, 4, top_id=2, bot_id=3)
    assert labels[2, 2] and labels[0, 2] and labels[4, 2] and not labels[1, 2]
    assert not labels[:, 3].any() and not labels[3].any()
    assert th.equal(live_elements(labels, 4, bot_id=3), th.tensor([1, 0, 1, 0, 1]).bool())
    assert edge_triples({"r": [(0, 4)]}, {"r": 0}) == [(0, 4, 0)]
    assert edge_triples(th.tensor([[0, 4, 1]]), {}) == [(0, 4, 1)]
    with pytest.raises(ValueError):
        interpretation_labels(th.zeros(2, 2), 5, classes, elements)


def test_syntactic_edges_are_closed_under_inclusions_chains_and_inverses():
    labels = th.zeros(5, 3, dtype=th.bool)
    labels[0, 0] = labels[1, 1] = labels[2, 2] = True      # x_A in A, x_B in B, x_C in C
    tables = {"gci2": th.tensor([[0, 0, 1]]),              # A <= exists r.B
              "object_property_assertion": th.tensor([[0, 0, 1]]),
              "role_inclusion": th.tensor([[0, 1]]), "role_chain": th.tensor([[0, 1, 2]])}
    edges = syntactic_edges(labels, tables, 3, inverse_of=[-1, -1, -1])
    assert (0, 1, 0) in edges and (0, 1, 1) in edges and (3, 4, 1) in edges
    assert edges == sorted(edges)
    with_inverse = syntactic_edges(labels, {"gci2": th.tensor([[0, 0, 1]])}, 3,
                                   inverse_of=[1, 0, -1])
    assert (1, 0, 1) in with_inverse
    per = edges_by_relation([(0, 1, 1)], 2, 5, inverse_of=th.tensor([1, 0]))
    assert per[0] == [(1, 0)] and per[1] == [(0, 1)]


@pytest.mark.parametrize("relations", ["translation", "product"])
def test_placement_realises_the_interpretation_in_another_combination(relations):
    th.manual_seed(0)
    g = _Parts(5, 3, 1, 2, 5, relations=relations)
    labels = th.zeros(5, 3, dtype=th.bool)
    labels[0, 0] = labels[1, 1] = labels[1, 0] = labels[3, 1] = True
    FiniteModelPlacement(delta=0.1, seed=2).place(g.elements, g.class_model, g.roles, labels,
                                                  edges=[(1, 3, 0)])
    member = th.stack([g.depth(g.positions(), th.full((5,), c), crisp=True) > 0
                       for c in range(3)], dim=1)
    assert th.equal(member, labels)
    assert th.equal(g.elements.dead_logits > 0.5, th.tensor([0, 0, 1, 0, 1]).bool())
    p = g.positions()
    assert float(g.roles.depth(p[1:2], p[3:4], th.tensor([0]), crisp=True)) > 0
    assert float(g.roles.depth(p[3:4], p[1:2], th.tensor([0]), crisp=True)) <= 0
    with pytest.raises(ValueError):
        FiniteModelPlacement().place(g.elements, _Parts(5, 3, 1, 2, 2).class_model, g.roles,
                                     labels)


# -- re-seeding a union of boxes ----------------------------------------------------------
@pytest.mark.parametrize("small", [2048, 0])
def test_slot_point_counts_are_exact(small):
    th.manual_seed(0)
    cen, w = th.rand(4, 3, 2), th.rand(4, 3, 2) * 0.3
    active = th.rand(4, 3) > 0.3
    points = th.rand(50, 2)
    query = th.rand(4, 3) > 0.2
    counts = slot_point_counts(Regions(cen=cen, w=w, active=active), points, query,
                               chunk_size=7, small=small)
    inside = (gp.UnionDepth.box_depth(points.unsqueeze(1).unsqueeze(1), cen, w) > 0).sum(0)
    inside = inside.reshape(4, 3)
    if small:  # per class: active and queried slots
        expected = th.where(query & active, inside, th.zeros_like(inside))
    else:      # binned: every queried slot
        expected = th.where(query, inside, th.zeros_like(inside))
    assert th.equal(counts, expected)


def test_reseeding_puts_a_box_at_the_own_element_and_prunes_empty_ones():
    th.manual_seed(0)
    g = _Parts(4, 2, 0, 2, 2)
    with th.no_grad():
        g.elements.elem_pos.weight.copy_(th.tensor([[0., 0.], [5., 5.], [9., 9.], [-5., 3.]]))
        g.elements.dead_logits.zero_()
        g.class_model.centers.copy_(th.tensor([[[20., 20.], [30., 30.]], [[5., 5.], [40., 40.]]]))
    moves = OwnPointReseeding(size=0.5).apply(g, assertions=th.tensor([[0, 1]]))
    p = g.positions()
    assert float(g.depth(p[0:1], th.tensor([0]), crisp=True)) > 0       # own element of A
    assert float(g.depth(p[2:3], th.tensor([1]), crisp=True)) > 0       # stated B(a0)
    assert moves >= 3
    assert int(g.class_model.active.sum()) <= 3                         # empty boxes pruned
    assert OwnPointReseeding(size=0.5).apply(g, assertions=th.tensor([[0, 1]])) == 0


# -- degrees of encoded points ------------------------------------------------------------
@pytest.mark.parametrize("agg", ["max", "lse"])
def test_soft_scores_in_chunks_and_their_cross_entropy(agg):
    th.manual_seed(0)
    g = _Parts(4, 5, 0, 3, 2, agg=agg)
    points, ids = th.randn(6, 3), th.tensor([0, 2, 3, 4])
    tau = th.tensor(0.3)
    whole = SoftClassScores()(g, points, ids, tau)
    chunked = SoftClassScores(max_elements=6 * 2 * 3)(g, points, ids, tau)
    assert whole.shape == (6, 4) and th.allclose(whole, chunked)
    direct = th.sigmoid(th.stack([g.depth(points, th.full((6,), int(c)), crisp=True)
                                  for c in ids], 1) / tau)
    assert th.allclose(whole, direct)
    targets = (th.rand(6, 4) > 0.5).float()
    loss, scores = MembershipCrossEntropy()(g, points, ids, targets, tau)
    assert th.allclose(loss, F.binary_cross_entropy(direct, targets))
    assert SoftClassScores()(g, points[:0], ids, tau).shape == (0, 4)


# -- the exact-domain check ---------------------------------------------------------------
class _Reasoner:
    def __init__(self, ext):
        self.ext = ext

    def is_satisfiable(self, iri):
        return bool(self.ext[iri])

    def extension(self, iri):
        return self.ext[iri]


def _base(**kw):
    out = {"elem_viol": 0.0, "region_viol": 0.0, "n_empty_classes": 0, "n_dead_elements": 0,
           "n_axioms_violated_by_depth": {"1e-4": 0}}
    out.update(kw)
    return out


def test_exact_domain_check_on_a_placed_interpretation():
    th.manual_seed(0)
    g = _Parts(3, 3, 0, 2, 3)
    labels = th.eye(3, dtype=th.bool)
    labels[0, 1] = True                                                 # A <= B
    FiniteModelPlacement(delta=0.1).place(g.elements, g.class_model, g.roles, labels)
    gen = th.Generator().manual_seed(0)
    plain = ExactDomainCheck().report(g, _base(), {}, gen)
    assert plain["exact"] and plain["exist"] == 1.0 and plain["faithful_named"] == (None, None)
    reasoner = _Reasoner({"A": {"x"}, "B": {"x", "y"}, "C": {"z"}})
    index = {"A": 0, "B": 1, "C": 2}
    full = ExactDomainCheck().report(g, _base(), {}, th.Generator().manual_seed(0),
                                     reasoner=reasoner, class_index=index, named_index=index)
    assert full["exact"] and full["dead_ok"] and full["n_false_dead"] == 0
    assert full["faithful_named"] == (1.0, 1.0)
    violated = ExactDomainCheck().report(g, _base(region_viol=0.5), {"gci0": th.zeros(2, 2)},
                                         gen)
    assert not violated["exact"] and violated["region_viol_tol1e-4"] == 0.0


# -- the loop of a sampled objective (a toy method) ----------------------------------------
class _Toy:
    """A method that is not GroundedEL: one parameter pulled to 3 by a sampled square."""

    seed, learning_rate, model_filepath, dataset, eval_gci_name = 0, 0.1, None, None, None

    def __init__(self):
        self.module = th.nn.Linear(1, 1, bias=False)
        self.module.tau = 1.0
        self.projected = 0

    def compute_batch_losses(self, generator):
        noise = th.rand((), generator=generator) * 0.0
        loss = (self.module.weight.sum() - 3.0) ** 2 + noise
        return {"total": loss, "square": loss}

    def project_parameters(self):
        self.projected += 1

    def epoch_record(self, rec):
        rec["extra"] = 1.0


class _Stop(TrainingRegime):
    def should_stop(self, epoch):
        return epoch >= 4


def test_sampled_objective_loop_trains_another_model_with_regimes():
    toy = _Toy()
    annealing = TemperatureAnnealing(start=1.0, end=0.5)
    regime = ComposedRegime(annealing, _Stop())
    regime.attach(toy)
    history = SampledObjectiveTraining().run(toy, regime, 20)
    assert len(history) == 5 and toy.projected == 5
    assert history[-1]["square"] < history[0]["square"]
    assert history[0]["extra"] == 1.0 and history[0]["tau"] == 1.0
    assert find_regime(regime, TemperatureAnnealing) is annealing
    assert SampledObjectiveTraining().run(toy, regime, 0) == []


# -- small helpers ------------------------------------------------------------------------
def test_sort_rows_and_union():
    t = th.tensor([[2, 1], [0, 5], [2, 0], [0, 1]])
    assert th.equal(sort_rows(t.clone()), th.tensor([[0, 1], [0, 5], [2, 0], [2, 1]]))
    assert th.equal(t[lexicographic_order(t)], sort_rows(t.clone()))
    a = MaskedUnionOfBoxes(th.zeros(1, 2, 2), th.ones(1, 2, 2), th.ones(1, 2, dtype=th.bool))
    b = MaskedUnionOfBoxes(th.ones(1, 1, 2), th.ones(1, 1, 2), th.zeros(1, 1, dtype=th.bool))
    u = a.union(b)
    assert u.cen.shape == (1, 3, 2) and u.active.tolist() == [[True, True, False]]


# -- GroundedEL reads these parts ---------------------------------------------------------
def _grounded(**kwargs):
    from mowl.models import GroundedEL
    return GroundedEL.from_tables(
        {"A": 0, "B": 1, "C": 2}, {"r": 0}, gci0=th.tensor([[0, 1]]),
        gci2=th.tensor([[1, 0, 2]]), class_assertion=th.tensor([[0, 0]]),
        individual_index_dict={"a": 0}, embed_dim=2, n_components=2, epochs=2, seed=0,
        log_every=0, **kwargs)


class _CountingCheck(ExactDomainCheck):
    calls = 0

    def report(self, *args, **kwargs):
        type(self).calls += 1
        return super().report(*args, **kwargs)


def test_grounded_el_evaluates_and_trains_through_the_parts(monkeypatch):
    model = _grounded(exact_regime=_CountingCheck(max_named=2))
    out = model.evaluate_model(use_reasoner=False, n_eval=64)
    assert _CountingCheck.calls == 1 and set(out) >= {"exact", "exist", "dead_ok"}
    assert isinstance(_grounded(exact_regime="exact domain check").exact_regime,
                      ExactDomainCheck)
    seen = []
    original = SampledObjectiveTraining.run

    def spy(self, model, regime, epochs, **kw):
        seen.append(epochs)
        return original(self, model, regime, epochs, **kw)
    monkeypatch.setattr(SampledObjectiveTraining, "run", spy)
    history = model.train(epochs=2)
    assert seen == [2] and len(history) == 2


def test_grounded_el_initialises_and_reseeds_through_the_parts():
    labels = th.zeros(4, 3, dtype=th.bool)
    labels[0, 0] = labels[0, 1] = labels[1, 1] = labels[2, 2] = labels[3, 0] = True
    labels[3, 1] = True
    model = _grounded(init="elements", init_labels=labels)
    m = model.module
    member = th.stack([m.depth(m.positions(), th.full((4,), c), crisp=True) > 0
                       for c in range(3)], 1)
    assert th.equal(member, labels)
    with th.no_grad():
        m.class_active.zero_()
    assert model.reseed_idle(prune_unused=False) >= 3
    assert float(m.depth(m.positions()[:1], th.tensor([0]), crisp=True)) > 0


# -- the set algebra of unions of boxes ---------------------------------------------------
def test_bounded_box_unions_algebra():
    from mowl.representation.library.box_union_algebra import (EMPTY_DEPTH, Boxes,
                                                               BoundedBoxUnions)
    alg = BoundedBoxUnions(max_boxes=3)
    a = Boxes(th.tensor([[[0., 0.], [5., 5.]]]), th.ones(1, 2, 2), th.ones(1, 2, dtype=th.bool))
    b = Boxes(th.tensor([[[0.5, 0.5]]]), th.ones(1, 1, 2), th.ones(1, 1, dtype=th.bool))
    inter = alg.intersect(a, b)
    assert inter.active.tolist() == [[True, False]]
    assert th.allclose(inter.cen[0, 0], th.tensor([0.25, 0.25]))
    assert th.allclose(inter.w[0, 0], th.tensor([0.75, 0.75]))
    shifted = alg.minkowski(a, th.tensor([[[1., 0.]]]), th.zeros(1, 1, 2),
                            th.ones(1, 1, dtype=th.bool))
    assert th.allclose(shifted.cen, a.cen + th.tensor([1., 0.]))
    both = alg.union(a, b)
    assert both.cen.shape == (1, 3, 2)
    capped = BoundedBoxUnions(max_boxes=1).union(a, Boxes(th.zeros(1, 1, 2), 3 * th.ones(1, 1, 2),
                                                          th.ones(1, 1, dtype=th.bool)))
    assert capped.cen.shape == (1, 1, 2) and float(capped.w[0, 0, 0]) == 3.0
    p = th.tensor([[0., 0.], [9., 9.]]).unsqueeze(0).unsqueeze(2)
    d = alg.depth(p, a)
    assert float(d[0, 0]) == 1.0 and float(d[0, 1]) < 0
    assert float(alg.depth(p, alg.empty(1, 2, a.cen))[0, 0]) == EMPTY_DEPTH
    assert float(alg.overlap_depth(a, b)) > 0
    pts = alg.points(th.tensor([[1., 2.]]))
    assert pts.w.abs().sum() == 0 and pts.active.all()


def test_bounded_box_unions_on_the_regions_of_a_class_model():
    """Another combination: the algebra on the boxes of a BoxUnionClasses (named classes),
    with MaskedUnionOfBoxes-shaped results: the intersection of two classes is where a point
    is in both."""
    from mowl.representation.library.box_union_algebra import BoundedBoxUnions
    th.manual_seed(0)
    g = _Parts(2, 2, 0, 2, 3)
    regions = g.class_regions()
    alg = BoundedBoxUnions(max_boxes=9, box_type=Regions)
    a = Regions(cen=regions.cen[:1], w=regions.w[:1], active=regions.active[:1])
    b = Regions(cen=regions.cen[1:], w=regions.w[1:], active=regions.active[1:])
    pts = th.rand(200, 2) - 0.5
    both = alg.depth(pts.unsqueeze(0).unsqueeze(2), alg.intersect(a, b))[0] > 0
    in_a = g.depth(pts, th.zeros(200, dtype=th.long), crisp=True) > 0
    in_b = g.depth(pts, th.ones(200, dtype=th.long), crisp=True) > 0
    assert th.equal(both, in_a & in_b)


# -- the grounding of a domain and the encoder of data elements ----------------------------
def test_domain_index_sets_and_relation_fillers():
    from mowl.representation.library.grounded_objective import (domain_index_sets,
                                                                relation_fillers)
    live, ordinary, probes = domain_index_sets(4, 6, top_id=0, bot_id=3)
    assert live.tolist() == [0, 1, 2] and ordinary.tolist() == [1, 2]
    assert probes.tolist() == [0, 1, 2, 4, 5]
    axioms = {"gci2": th.tensor([[0, 1, 2], [1, 1, 2]]),
              "object_property_assertion": th.tensor([[0, 0, 1]])}
    fillers, lhs, mask = relation_fillers(axioms, 2, 6, 4)
    assert fillers[1].tolist() == [2] and lhs[1].tolist() == [0, 1]
    assert fillers[0].tolist() == [5] and lhs[0].tolist() == [4]
    assert mask.tolist() == [[False] * 5 + [True], [False, False, True, False, False, False]]


def test_element_domain_encodes_data_elements():
    domain = gp.ElementDomain(3, 2, 2, encoder=th.nn.Linear(4, 2))
    assert domain.encode(th.ones(5, 4)).shape == (5, 2)
    assert domain.encode(th.ones(4)).shape == (1, 2)
    with pytest.raises(ValueError):
        domain.encode(th.ones(2, 2, 2))


class _Validated(_Toy):
    """The toy method with validation data: the loop validates it every epoch."""

    eval_gci_name = "gci0"

    def __init__(self):
        super().__init__()
        self.module = _ToyModule()
        self.dataset = type("D", (), {"validation": True})()
        self.validation_datasets = {"gci0": th.zeros((2, 2), dtype=th.long)}


class _ToyModule(th.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)
        self.tau = 1.0

    def forward(self, rows, form):
        return (self.weight.sum() - 3.0).abs().expand(len(rows))


def test_sampled_objective_loop_leaves_the_module_in_eval_mode_after_validation():
    toy = _Validated()
    regime = ComposedRegime(TemperatureAnnealing(start=1.0, end=0.5))
    regime.attach(toy)
    SampledObjectiveTraining().run(toy, regime, 2)
    assert not toy.module.training


# -- swapping a part of GroundedEL ---------------------------------------------------------
def test_composed_el_builds_grounded_and_refuses_parts_of_other_families():
    """Round 7: ComposedEL builds the grounded geometry (its joint objective) and swaps its
    parts by the names of the general model; parts of other families are refused at build."""
    from mowl.datasets.builtin import FamilyDataset
    from mowl.models import ComposedEL
    from mowl.representation import ConstructorUnavailable
    data = FamilyDataset()
    model = ComposedEL(data, method="grounded", relations="translation", embed_dim=4)
    assert model.module.joint_objective is not None
    with pytest.raises(ConstructorUnavailable, match="element domain"):
        ComposedEL(data, method="grounded", classes="ball", embed_dim=4)


@pytest.mark.parametrize("swap", [dict(relation_model="translation"),
                                  dict(relation_model="box of translations", agg="lse"),
                                  dict(role_mode="product")])
def test_grounded_el_trains_with_a_swapped_part(swap):
    import math
    model = _grounded(**swap)
    history = model.train(epochs=2)
    assert len(history) == 2 and all(math.isfinite(r["total"]) for r in history)
    assert all(bool(th.isfinite(p).all()) for p in model.module.parameters())


def test_fast_paths_of_the_element_box_family_give_the_same_numbers():
    from mowl.evaluation.query_search import MaxMinBeam, WitnessChain, read_batch
    from mowl.nn import GroundedELModule
    from mowl.representation.library.element_box_family import ElementBoxFamily
    th.manual_seed(0)
    m = GroundedELModule(nb_elements=400, nb_classes=40, nb_rels=3, embed_dim=3,
                         n_components=2, n_role_components=2, role_mode="translation")
    node = read_batch(("e", ("r", "r", "r")), [(50, (0, 1, 2)), (60, (2, 2, 1))])
    targets = th.randint(0, 400, (2, 30))
    with th.no_grad():
        p = m.positions()
        out = {}
        for fast in (True, False):
            fam = ElementBoxFamily(m, p, fast=fast)
            out[fast] = (MaxMinBeam(k=8).scores(fam, node),
                         WitnessChain(k=8).scores(fam, fam, node, targets, m.tau))
    assert th.equal(out[True][0], out[False][0])
    assert th.equal(out[True][1], out[False][1])
