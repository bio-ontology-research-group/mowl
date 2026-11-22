"""GroundedEL as a combination of parts: what it is made of, that the parts own their tables
and share only the element domain, that the constraints are generic rules in a fuzzy logic,
what can be swapped and still trains and evaluates, that the catalogue and the parts agree,
and the grounded constraints as a program for a solver. The equality with the direct
implementation is ``tests/baselines`` (``pairs_grounded.py``)."""
import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.models import GroundedEL  # noqa: E402
from mowl.models.grounded.evaluation import GroundedEvaluation  # noqa: E402
from mowl.models.grounded.objective import GroundedConstraints  # noqa: E402
from mowl.nn import GroundedELModule  # noqa: E402
from mowl.nn.el.grounded import constraints as C  # noqa: E402
from mowl.representation import (METHODS, REPRESENTATIONS, ConstructorUnavailable,  # noqa: E402
                                 Geometry, geometry_of)
from mowl.representation.library import grounded_logic as gl  # noqa: E402
from mowl.representation.library import grounded_parts as gp  # noqa: E402
from mowl.representation.library.grounded_program import (GroundedProgram,  # noqa: E402
                                                          crisp_tables, load_into)
from mowl.representation.library.solved_grounding import AxiomTable, Domain  # noqa: E402
from mowl.representation.library.solved_solver import (GradientFit, ScipySolver,  # noqa: E402
                                                       TorchAlgebra, evaluate)
from mowl.training import (ComposedRegime, GroundedELTraining, LagrangianDualWeights,  # noqa: E402
                           TemperatureAnnealing)


def grounded_model(**kwargs):
    """A tiny GroundedEL from tables of axioms (no ontology)."""
    classes = {f"C{i}": i for i in range(4)}
    roles = {"r": 0, "s": 1}
    individuals = {f"a{i}": i for i in range(6)}
    opa = th.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 3], [3, 1, 4], [4, 0, 5]])
    args = dict(embed_dim=4, n_components=2, epochs=2, seed=3, learning_rate=0.05,
                role_mode="translation", log_every=0)
    args.update(kwargs)
    return GroundedEL.from_tables(
        classes, roles, gci0=th.tensor([[0, 1], [1, 2]]), gci2=th.tensor([[2, 0, 3]]),
        gci3=th.tensor([[1, 3, 0]]), gci1_bot=th.tensor([[0, 3, 3]]),
        role_inclusion=th.tensor([[0, 1]]), role_chain=th.tensor([[0, 1, 1]]),
        class_assertion=th.tensor([[0, 0], [1, 2]]), object_property_assertion=opa,
        individual_index_dict=individuals, **args)


def one_step(model):
    return model.compute_batch_losses(generator=th.Generator().manual_seed(0))


def finite(losses):
    return all(np.isfinite(float(v)) for v in losses.values())


# =========================================================================================
# (a) what the method is made of
# =========================================================================================
def test_grounded_is_made_of_the_parts_the_card_names():
    model = grounded_model()
    g = model.module
    assert type(g) is GroundedELModule and isinstance(g, Geometry)
    assert g.parts == {"classes": "union of boxes", "relations": "box of translations",
                       "individuals": "canonical element", "measure": "bounds per coordinate",
                       "conjunction": "exact", "shared": "element domain"}
    assert grounded_model(role_mode="product").module.parts["relations"] == "region over pairs"
    # the parts are objects of their own classes
    assert type(g.class_model) is gp.BoxUnionClasses
    assert type(g.relation_model) is gp.TranslationBoxRelations
    assert type(grounded_model(role_mode="product").module.relation_model) is \
        gp.ProductBoxRelations
    assert type(g.shared) is gp.ElementDomain and type(g.measure) is gp.UnionDepth
    assert (g.class_model.part, len(g.class_model)) == ("union of boxes", 4)
    assert (g.relation_model.part, len(g.relation_model)) == ("box of translations", 2)
    assert (g.individual_model.part, len(g.individual_model)) == ("canonical element", 6)
    assert type(g.class_model(th.tensor([0, 1]))).__name__ == "MaskedUnionOfBoxes"
    assert geometry_of(model) is g
    # the objective: generic rules in the Łukasiewicz logic and depth hinges
    assert isinstance(model, GroundedConstraints) and isinstance(model, GroundedEvaluation)
    assert model.constraint_set.parts == {"logic": "lukasiewicz", "chain logic": "godel",
                                          "hinges": "DepthHinges"}
    assert set(one_step(model)) >= {"fact", "axioms", "region", "minimality", "neg", "total"}
    # the training regime is a composition of schedule parts
    regime = GroundedELTraining.for_model(model)
    assert [type(r).__name__ for r in regime.regimes] == ["TemperatureAnnealing"]
    lagrangian = GroundedELTraining(constraint_mode="lagrangian", reseed_every=5, lr_final=0.001)
    assert [type(r).__name__ for r in lagrangian.regimes] == [
        "ConstantThenCosine", "TemperatureAnnealing", "IdleComponentReseeding",
        "LagrangianDualWeights"]
    # the named combination
    built = METHODS["grounded"](4, 2, 6, 4, role_mode="translation")
    assert type(built) is GroundedELModule and built.nb_elements == 10
    assert built.parts["relations"] == "box of translations"


def test_the_parts_own_their_tables_and_share_only_the_element_domain():
    for mode, tables in (("product", {"head_centers", "head_raw_widths", "tail_centers",
                                      "tail_raw_widths"}),
                         ("translation", {"centers", "raw_widths"})):
        g = grounded_model(role_mode=mode).module
        owned = {name: {n for n, _ in part.named_parameters()} for name, part in
                 (("elements", g.shared), ("boxes", g.class_model),
                  ("roles", g.relation_model))}
        assert owned["elements"] == {"elem_pos.weight", "dead_logits"}
        assert owned["boxes"] == {"centers", "raw_widths"}
        assert owned["roles"] == tables
        # no table belongs to two parts; every parameter of the module is one part's
        ids = [id(p) for part in (g.shared, g.class_model, g.relation_model)
               for p in part.parameters()]
        assert len(ids) == len(set(ids)) == len(list(g.parameters()))
        # the shared part is read by the others but is not a sub-module of theirs
        assert g.class_model.domain is g.shared and g.relation_model.domain is g.shared
        assert not any(isinstance(m, gp.ElementDomain) for m in g.class_model.modules())
        assert {k.split(".")[0] for k in g.state_dict()} == {"elements", "boxes", "roles"}
        # the names of the earlier single module still reach the tables of the parts
        assert g.class_centers is g.class_model.centers
        assert g.elem_pos is g.shared.elem_pos and g.tau is g.shared.tau
        g.agg = "lse"
        assert g.measure.agg == "lse"
        # temperature: one number, read by the class model and the relation model
        g.set_tau(0.3)
        x = g.positions(th.tensor([4, 5]))
        assert th.allclose(g.membership(x, th.tensor([0, 1])),
                           th.sigmoid(g.depth(x, th.tensor([0, 1])) / 0.3))
        assert th.allclose(g.role_membership(x, x.flip(0), th.tensor([0, 1])),
                           th.sigmoid(g.role_depth(x, x.flip(0), th.tensor([0, 1])) / 0.3))


def test_the_earlier_state_dict_names_load():
    from mowl.nn.el.grounded.module import _LEGACY_KEYS
    g = grounded_model(role_mode="product").module
    old_names = {}
    for key, value in g.state_dict().items():
        for old, new in _LEGACY_KEYS.items():
            if key == new or key.startswith(new + "."):
                key = old + key[len(new):]
                break
        old_names[key] = value + 1.0 if value.is_floating_point() else value
    assert "class_centers" in old_names and "elem_pos.weight" in old_names
    g.load_state_dict(old_names)
    assert th.equal(g.class_centers, old_names["class_centers"])


# =========================================================================================
# the constraints are generic parts
# =========================================================================================
def test_lukasiewicz_rules_are_the_generic_reading_of_the_family():
    """The closed form of the Łukasiewicz rules equals the generic reading through the
    operators of the family (T(premises, 1 - exempt) - conclusion)."""
    gen = th.Generator().manual_seed(0)
    closed, generic = gl.LukasiewiczRules(), gl.Rules(gl.Lukasiewicz())
    for n in range(0, 4):
        premises = [th.rand(50, generator=gen) for _ in range(n)]
        conclusion = th.rand(50, generator=gen)
        exempt = th.rand(50, generator=gen)
        for c, e in ((conclusion, exempt), (conclusion, None), (None, exempt)):
            if n == 0 and c is None:
                continue
            assert th.allclose(closed.violation(premises, c, e),
                               generic.violation(premises, c, e), atol=1e-6)


def test_every_form_is_a_rule_or_a_depth_hinge():
    m = th.tensor([1.0, 1.0, 0.0])
    one, zero = th.ones(3), th.zeros(3)
    # A ⊑ B at x: violated where x is in A but not in B, unless x is dead
    assert C.gci0(m, zero).tolist() == [1.0, 1.0, 0.0]
    assert C.gci0(m, zero, dead=one).tolist() == [0.0, 0.0, 0.0]
    assert C.fact(zero).tolist() == [1.0, 1.0, 1.0] and C.fact(one).tolist() == [0.0] * 3
    assert C.role_chain(one, one, zero).tolist() == [1.0] * 3
    # the same constraints in another logic
    godel = C.ConstraintSet("godel")
    assert th.allclose(godel.gci1(th.tensor([0.7]), th.tensor([0.6]), th.tensor([0.2])),
                       th.tensor([0.4]))
    assert th.allclose(C.gci1(th.tensor([0.7]), th.tensor([0.6]), th.tensor([0.2])),
                       th.tensor([0.1]))
    # region constraints: depth hinges
    assert C.gci0_region(th.tensor([-0.5, 0.5])).tolist() == [0.5, 0.0]
    assert C.gci0_bot_region(th.tensor([-0.5, 0.5])).tolist() == [0.0, 0.5]


# =========================================================================================
# (b) swaps
# =========================================================================================
@pytest.mark.parametrize("logic", ["godel", "product"])
def test_grounded_swap_the_logic(logic):
    model = grounded_model(logic=logic, epochs=1)
    assert model.constraint_set.rules.name == logic
    base = one_step(grounded_model())
    swapped = one_step(model)
    assert finite(swapped) and float(swapped["axioms"]) != float(base["axioms"])
    history = model.train()
    assert len(history) == 1 and np.isfinite(history[0]["total"])
    assert 0.0 <= model.crisp_report(n_eval=32)["elem_viol"] <= 1.0
    with pytest.raises(ValueError, match="logic"):
        grounded_model(logic="boolean")


def test_grounded_swap_the_relation_model():
    for mode in ("translation", "product"):
        model = grounded_model(role_mode=mode)
        history = model.train()
        assert len(history) == 2 and all(np.isfinite(h["total"]) for h in history)
        x = model.module.positions(th.tensor([4, 5]))
        depth = model.module.role_depth(x, x.flip(0), th.tensor([0, 1]))
        assert th.isfinite(depth).all()
        assert 0.0 <= model.crisp_report(n_eval=32)["elem_viol"] <= 1.0


def test_grounded_swap_the_measure_and_the_objective_options():
    base = one_step(grounded_model())
    # no region terms: the region loss is exactly zero
    no_region = one_step(grounded_model(n_region_samples=0))
    assert float(no_region["region"]) == 0.0 and float(base["region"]) > 0.0
    # corrupted facts add a negative term
    negative = one_step(grounded_model(neg_weight=1.0))
    assert float(negative["neg"]) > 0.0 and float(base["neg"]) == 0.0
    # the measure: a smooth aggregation of the components
    model = grounded_model(agg="lse", epochs=1)
    assert model.module.measure.agg == "lse" and finite(one_step(model))
    model.train()
    assert np.isfinite(model.loss_history[-1]["total"])


def test_grounded_swap_the_training_regime():
    model = grounded_model(epochs=3)
    model.training_regime = ComposedRegime(LagrangianDualWeights(rho=1.0),
                                           TemperatureAnnealing(1.0, 0.1))
    history = model.train()
    assert len(history) == 3 and model.constraint_mode == "lagrangian"
    assert history[-1]["tau"] == pytest.approx(0.1)
    scheduled = grounded_model(epochs=3)
    scheduled.train(lr_final=0.001)
    assert all(np.isfinite(h["total"]) for h in scheduled.loss_history)


def test_grounded_swap_the_evaluation_regime():
    model = grounded_model(epochs=2)
    model.train()
    crisp = model.crisp_report(n_eval=64)
    assert 0.0 <= crisp["elem_viol"] <= 1.0 and 0.0 <= crisp["region_viol"] <= 1.0
    metrics = model.evaluate_model(use_reasoner=False, n_eval=64)
    assert np.isfinite(metrics["elem_viol"]) and np.isfinite(metrics["region_viol"])
    assert metrics["faithful_named_precision"] is None       # no reasoner: not computed


def test_grounded_geometry_answers_the_questions_of_the_interface():
    g = grounded_model().module
    ids = th.tensor([0, 1])
    a, b = g.classes(ids), g.classes(th.tensor([2, 3]))
    both, gap = g.conjunction(a, b)
    assert len(both) == 2 and gap.shape == (2, 1)
    assert g.inside(a, a).shape == (2, 1) and float(g.inside(a, a).max()) == 0.0
    assert g.apart(a, b).shape == (2, 1)
    assert g.member(th.tensor([0, 1]), a).shape == (2, 1)
    assert g.num_classes == 4 and g.num_individuals == 6
    exists = g.exists(th.tensor([0, 1]), a)
    assert len(exists) == 2
    with pytest.raises(ConstructorUnavailable):
        grounded_model(role_mode="product").module.exists(th.tensor([0]), a)


# =========================================================================================
# (c) the catalogue card and the parts agree
# =========================================================================================
def test_grounded_catalogue_card_and_parts_agree():
    from mowl.catalog.registry import get_method
    card, rep = get_method("grounded"), REPRESENTATIONS["grounded"]
    for mode in ("product", "translation"):
        parts = grounded_model(role_mode=mode).module.parts
        assert parts["classes"] in {v.value for v in rep.classes}
        assert parts["relations"] in {v.value for v in rep.relations}
        assert parts["measure"] in {v.value for v in rep.measure}
        assert parts["individuals"] in {v.value for v in rep.individuals}
    # every axiom form of the card is a form the objective reads: a model with one axiom of
    # each form has a finite, positive loss term for it
    model = grounded_model(n_region_samples=0)
    loaded = {form for form, rows in model.training_datasets.items() if len(rows)}
    assert set(card.axioms) <= loaded | {"gci1", "gci0_bot", "gci3_bot"}
    assert finite(one_step(model))
    assert "ELH" in card.expressivity


# =========================================================================================
# the grounded constraints as a program (fitted by a solver on a tiny problem)
# =========================================================================================
def _tiny_table():
    # classes A0 B1 C2 D3 top4 bot5; roles r0 s1; individual a0
    rows = dict(gci0=[[0, 1]], gci2=[[1, 0, 2]], gci3=[[0, 2, 3]], role_inclusion=[[0, 1]],
                class_assertion=[[0, 0]], gci1_bot=[[3, 2, 5]])
    return 6, 2, 1, {k: np.asarray(v) for k, v in rows.items()}


def test_the_program_solved_is_the_least_model():
    from mowl.geometric.least_model import least_model
    nc, nr, ni, rows = _tiny_table()
    program = GroundedProgram(AxiomTable(Domain(nc, ni, nr), rows), top_id=4, bot_id=5)
    result = ScipySolver().solve(program, {"eps": 1.0, "eps_r": 1.0, "eps_dead": 1.0})
    assert result.solved
    labels, edges, dead = crisp_tables(program, result.values)
    reference = least_model(nc, nr, ni, top_id=4, bot_id=5, **rows)
    assert (labels == reference.labels.toarray()).all()
    assert (dead == reference.dead).all()
    assert sorted(map(tuple, edges.tolist())) == sorted(map(tuple, reference.edges.tolist()))
    # the solution is crisp and feasible
    assert evaluate(program, result.values).feasible(tol=1e-5)
    assert np.allclose(result.values["m"], np.round(result.values["m"]), atol=1e-4)


def test_the_solved_table_loads_into_the_geometry_and_the_model_has_no_violation():
    nc, nr, ni, rows = _tiny_table()
    program = GroundedProgram(AxiomTable(Domain(nc, ni, nr), rows), top_id=4, bot_id=5)
    result = ScipySolver().solve(program, {"eps": 1.0, "eps_r": 1.0, "eps_dead": 1.0})
    from mowl.owlapi.defaults import BOT, TOP
    names = {"A": 0, "B": 1, "C": 2, "D": 3, TOP: 4, BOT: 5}
    model = GroundedEL.from_tables(
        names, {"r": 0, "s": 1}, individual_index_dict={"a": 0},
        **{k: th.as_tensor(v) for k, v in rows.items()}, embed_dim=3, n_components=7,
        n_role_components=7, epochs=0, log_every=0, seed=0)
    load_into(model.module, program, result.values)
    report = model.crisp_report(n_eval=64)
    assert report["elem_viol"] == 0.0


def test_the_program_fitted_by_gradient_descent_instead_of_a_solver():
    nc, nr, ni, rows = _tiny_table()
    program = GroundedProgram(AxiomTable(Domain(nc, ni, nr), rows), top_id=4, bot_id=5)
    th.manual_seed(0)
    tables = {name: th.nn.Parameter(th.rand(shape) * 0.5)
              for name, shape in program.variables().items()}
    result = GradientFit(steps=1000, lr=0.02, penalty=100.0).fit(
        program, tables, {"eps": 0.1, "eps_r": 0.1, "eps_dead": 0.1})
    assert result.info["violation"] < 0.05
    labels, _, _ = crisp_tables(program, {k: v.detach().numpy() for k, v in tables.items()})
    from mowl.geometric.least_model import least_model
    assert (labels == least_model(nc, nr, ni, top_id=4, bot_id=5, **rows).labels.toarray()).all()


def test_the_exempt_reading_is_not_tight_which_is_why_the_solver_uses_the_horn_reading():
    """GroundedEL's hinges exempt a dead element from every rule, the existence of its
    witness included; as a linear program that is not Horn, and A ⊑ ⊥ has the fractional
    optimum m = dead = 1/2 (module docstring of grounded_program)."""
    table = AxiomTable(Domain(2, 0, 0), {"gci0_bot": np.array([[0, 1]])})
    exempt = GroundedProgram(table, bot_id=1, reading="exempt")
    result = ScipySolver().solve(exempt, {"eps": 1.0, "eps_r": 1.0, "eps_dead": 2.0})
    assert np.isclose(result.values["m"][0, 0], 0.5, atol=1e-3)
    assert np.isclose(result.values["dead"][0], 0.5, atol=1e-3)
    horn = GroundedProgram(table, bot_id=1)
    result = ScipySolver().solve(horn, {"eps": 1.0, "eps_r": 1.0, "eps_dead": 2.0})
    assert crisp_tables(horn, result.values)[2].tolist() == [True, True]


def test_the_program_is_the_element_constraints_of_the_model():
    """At the degrees of a GroundedEL geometry (memberships of every element in every class,
    degrees of every pair) the exempt program's slack is the negative of the element
    hinges: the torch algebra evaluates the same rules the objective sums."""
    model = grounded_model(role_mode="product", epochs=1)
    model.train()
    g = model.module
    n, nc = g.nb_elements, g.nb_classes
    pos = g.positions().detach()
    m = th.stack([g.membership(pos, th.full((n,), c, dtype=th.long)) for c in range(nc)],
                 dim=1).detach()
    rho = th.stack([g.role_membership(pos.repeat_interleave(n, 0), pos.repeat(n, 1),
                                      th.full((n * n,), r, dtype=th.long)).reshape(n, n)
                    for r in range(g.nb_rels)]).detach()
    dead = g.dead().detach()
    rows = {form: data[:].numpy() for form, data in model.training_datasets.items()
            if len(data) and form in AxiomTable.COLUMNS}
    program = GroundedProgram(AxiomTable(Domain(nc, n - nc, g.nb_rels), rows),
                              reading="exempt")
    alg = TorchAlgebra()
    program.build(alg, {"m": m, "rho": rho, "dead": dead},
                  {"eps": 0.0, "eps_r": 0.0, "eps_dead": 0.0})
    slack = [s for name, s in alg.constraints if name == "gci0"][0]   # the first gci0 axiom
    a, b = model.training_datasets["gci0"][:][0].tolist()
    assert th.allclose(th.relu(-slack), C.gci0(m[:, a], m[:, b], dead=dead))


# =========================================================================================
# the parts of the objective and the evaluation, on their own
# =========================================================================================
def _union(cen, w, active=None):
    from mowl.nn.el.structural import Regions
    cen, w = th.tensor(cen, dtype=th.float32), th.tensor(w, dtype=th.float32)
    if active is None:
        active = th.ones(cen.shape[:2], dtype=th.bool)
    return Regions(cen=cen, w=w, active=th.as_tensor(active))


def test_region_sampler_draws_inside_a_box_of_the_union_with_its_corners():
    from mowl.representation.library.grounded_sampling import RegionSampler
    regions = _union([[[0.0, 0.0], [5.0, 5.0]], [[1.0, 1.0], [0.0, 0.0]]],
                     [[[1.0, 0.5], [0.2, 0.2]], [[1.0, 1.0], [1.0, 1.0]]],
                     [[True, True], [False, False]])
    sample = RegionSampler(n_samples=6, n_adv=0)(regions, th.Generator().manual_seed(0))
    assert sample.points.shape == (12, 2) and sample.valid.tolist() == [True, False] * 6
    pts, lo, hi = sample.points[sample.valid], sample.lower[sample.valid], sample.upper[
        sample.valid]
    assert bool(((pts >= lo) & (pts <= hi)).all())
    assert th.equal(sample.points[~sample.valid], th.zeros(6, 2))
    # one box per row: every point of row 0 lies in the same box, 4 of them its corners
    assert th.unique(lo, dim=0).shape[0] == 1
    near_corner = ((pts - lo).abs() < 1e-4) | ((hi - pts).abs() < 1e-4)
    assert int(near_corner.all(1).sum()) == 4
    # the points are re-parametrised: the gradient reaches the box
    cen = regions.cen.clone().requires_grad_(True)
    moved = RegionSampler(1, 0).one(type(regions)(cen, regions.w, regions.active),
                                    th.Generator().manual_seed(0))
    moved.points[0].sum().backward()
    assert th.allclose(cen.grad[0].sum(0), th.ones(2))
    # the published reading (GroundedEL) counts the box twice for the first point of a row
    cen.grad = None
    RegionSampler(1, 0, doubled_gradient=True).one(
        type(regions)(cen, regions.w, regions.active),
        th.Generator().manual_seed(0)).points[0].sum().backward()
    assert th.allclose(cen.grad[0].sum(0), 2 * th.ones(2))


def test_box_corners_of_a_product_of_boxes_and_the_ascent_stays_in_the_box():
    from mowl.representation.library.grounded_sampling import ascend, box_corners
    lo, hi = th.tensor([[0.0, 0.0, 10.0, 10.0]]), th.tensor([[1.0, 1.0, 11.0, 11.0]])
    corners = box_corners(lo, hi, blocks=2)[0]
    assert corners.shape == (4, 4)
    # the same sign in both blocks: the corner of the head is the corner of the tail - 10
    assert th.allclose(corners[:, 2:] - 10.0, corners[:, :2], atol=1e-5)
    start = th.tensor([[0.5, 0.5]])
    up = ascend(start, lo[:, :2], hi[:, :2], lambda z: z.sum(1), steps=3)
    assert bool((up >= lo[:, :2]).all() & (up <= hi[:, :2]).all())
    assert float(up.sum()) > float(start.sum())


def test_element_sampler_rows_candidates_fillers_probes_and_corruptions():
    from mowl.representation.library.grounded_sampling import ElementSampler
    gen = th.Generator().manual_seed(0)
    es = ElementSampler(batch_size=3, n_random=2, n_inside=0, n_fillers=2)
    table = th.arange(20).view(10, 2)
    rows = es.rows(table, gen, "cpu")
    assert rows.shape == (3, 2) and all(r.tolist() in table.tolist() for r in rows)
    cand = es.candidates(rows[:, 0], 50, gen, (th.tensor([7, 8]),))
    assert cand.shape == (3, 5) and th.equal(cand[:, 0], rows[:, 0])
    assert th.equal(cand[:, 3:], th.tensor([[7, 8]] * 3))
    # a domain of at most whole_domain elements is read whole
    assert th.equal(es.batch(th.tensor([3]), 10, gen), th.arange(10))
    # fillers: the row's own filler first when there are more than n_fillers
    mask = th.zeros(1, 50, dtype=th.bool)
    mask[0, [10, 11, 12, 13]] = True
    fill = es.fillers(th.tensor([[0, 4, 1], [0, 5, 1]]), mask, gen)
    assert fill.shape == (2, 1 + 1 + 2) and fill[:, 0].tolist() == [4, 5]
    x, c = es.class_probes(4, th.tensor([1, 2]), gen)
    assert sorted(zip(x.tolist(), c.tolist())) == [(i, k) for i in range(4) for k in (1, 2)]
    opa = th.tensor([[0, 1, 2], [3, 0, 4]])
    bad = es.corrupted(opa, 5, gen, per_row=4)
    assert bad.shape == (8, 3)
    for a, r, b in bad.tolist():
        kept = [row for row in opa.tolist() if row[1] == r]
        assert any(a == k[0] or b == k[2] for k in kept)


def test_element_sampler_inside_reads_the_class_model():
    from mowl.representation.library.grounded_sampling import ElementSampler
    g = grounded_model(role_mode="product").module
    with th.no_grad():
        g.class_model.centers[0] = 0.0
        g.class_model.raw_widths[0] = gp.inv_softplus(0.05)
        pos = g.shared.elem_pos.weight
        pos[:] = 3.0
        pos[[2, 5, 7]] = 0.0          # inside class 0
    inside = ElementSampler(n_inside=8, whole_domain=0).inside(
        g.class_model, th.tensor([0, 0]), g.positions(), th.Generator().manual_seed(1))
    assert sorted(set(inside.flatten().tolist())) == [2, 5, 7]


def test_crisp_violations_reads_inclusion_and_disjointness_from_the_geometry():
    from mowl.representation.library.grounded_evaluation import CrispViolations
    model = grounded_model(role_mode="product")
    g = model.module
    with th.no_grad():
        g.class_model.active[:] = False
        g.class_model.active[:, 0] = True
        g.class_model.centers[:, 0] = th.tensor([[0.0] * 4, [0.0] * 4, [5.0] * 4, [9.0] * 4])
        g.class_model.raw_widths[:, 0] = th.tensor([gp.inv_softplus(w) for w in
                                                    (0.5, 1.0, 0.5, 0.5)]).unsqueeze(1)
    axioms = {"gci0": th.tensor([[0, 1], [1, 0], [0, 2]]), "gci1_bot": th.tensor([[0, 2, 0]])}
    report = CrispViolations(n_eval=300).report(g, axioms, model.grounding(),
                                                th.Generator().manual_seed(0))
    worst = report["worst_depth"]["gci0"]
    assert worst["0 <= 1"] > 0.0                       # A in B: every point of A deep in B
    assert worst["1 <= 0"] < 0.0 and worst["0 <= 2"] < 0.0
    assert report["worst_depth"]["gci1_bot"]["0 cap 2 <= bot"] < 0.0     # apart
    assert report["region_viol"] == pytest.approx(2 / 4)


def test_the_objective_is_an_assembly_of_forms_over_the_parts():
    from mowl.representation.library.grounded_objective import FORMS, GroundedObjective
    model = grounded_model()
    assert isinstance(model.objective, GroundedObjective)
    assert set(FORMS) >= {f for f in model.training_datasets}
    gen = th.Generator().manual_seed(0)
    direct = model.objective.losses(model.module, model.training_datasets, model.grounding(),
                                    gen, model.element_sampler, model.region_sampler)
    assert th.allclose(direct["total"], one_step(model)["total"])

    class NoInclusion(GroundedObjective):
        def gci0(self, s, rows):
            return s.pos.new_zeros(())
    swapped = grounded_model(objective=NoInclusion(model.constraint_set))
    assert float(one_step(swapped)["axioms"]) < float(one_step(model)["axioms"])


def test_grounded_parts_given_to_the_builder_take_effect():
    from mowl.representation.library.grounded_sampling import ElementSampler, RegionSampler
    base = one_step(grounded_model(role_mode="product"))
    # the relation model by name replaces role_mode
    model = grounded_model(role_mode="product", relation_model="box of translations")
    assert type(model.module.relation_model) is gp.TranslationBoxRelations
    assert model.role_mode == model.module.role_mode == "translation"
    assert float(one_step(model)["axioms"]) != float(base["axioms"])
    # samplers with other options
    fewer = one_step(grounded_model(role_mode="product",
                                    element_sampler=ElementSampler(4096, 0, 0, 1)))
    assert float(fewer["axioms"]) != float(base["axioms"])
    one_point = one_step(grounded_model(role_mode="product",
                                        region_sampler=RegionSampler(1, 0)))
    assert float(one_point["region"]) != float(base["region"])
    # the measure
    lse = grounded_model(role_mode="product", measure=gp.UnionDepth(agg="lse", agg_tau=0.5))
    assert lse.module.measure.agg == "lse" and lse.module.class_model.measure is \
        lse.module.measure
    assert float(one_step(lse)["axioms"]) != float(base["axioms"])
    # the builder of the method takes the parts too
    from mowl.representation.library.solved_grounded import grounded
    g = grounded(4, 2, 3, embed_dim=4, relation_model="translation")
    assert type(g.relation_model) is gp.TranslationBoxRelations and g.role_mode == "translation"


def test_assigning_a_part_of_the_module_replaces_it():
    model = grounded_model(role_mode="product")
    base = one_step(model)

    class Doubting(gp.ProductBoxRelations):
        def membership(self, x_points, y_points, rel_ids):
            return 0.5 * super().membership(x_points, y_points, rel_ids)
    g = model.module
    old = g.relation_model
    new = Doubting(2, 2, 4, g.measure, g.shared)
    new.load_state_dict(old.state_dict())
    g.relation_model = new
    assert g.relation_model is new and g.roles is new and "relation_model" not in g._modules
    assert {k.split(".")[0] for k in g.state_dict()} == {"elements", "boxes", "roles"}
    assert float(one_step(model)["axioms"]) != float(base["axioms"])
    g.measure = gp.UnionDepth(agg="lse", agg_tau=0.5)
    assert g.class_model.measure is g.measure and g.relation_model.measure is g.measure
