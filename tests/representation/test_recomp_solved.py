"""The solved methods as combinations of parts: ``boxlite``, ``faithel`` (``grounded``:
``test_recomp_grounded.py``). What each is made of, what can be swapped and still trains and
evaluates, and that the catalogue and the parts agree. The equality with the direct
implementations is ``tests/baselines`` (``pairs_solved.py``).

BoxLitE is solved by a conic solver (the optional package ``cvxpy``). Everything but that call
is tested here; the tests that need it skip with a reason.
"""
import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets.dllite import DLLiteKB  # noqa: E402
from mowl.models import BoxLitE, FaithEL  # noqa: E402
from mowl.representation import (METHODS, REPRESENTATIONS,  # noqa: E402
                                 ComposedGeometry, ConstructorUnavailable, Geometry,
                                 geometry_of)
from mowl.representation.library import solved_boxlite as sb  # noqa: E402
from mowl.representation.library import solved_faithel as sf  # noqa: E402
from mowl.representation.library.solved_grounding import (AxiomTable, Domain,  # noqa: E402
                                                          inside_bounds)
from mowl.representation.library.solved_solver import (  # noqa: E402
    CvxpySolver, GradientFit, NumpyAlgebra, ScipySolver, TorchAlgebra, evaluate)
from tests.models.test_canonical_mixture import (IND, NS,  # noqa: E402
                                                 _write_family_abox_dataset)

needs_cvxpy = pytest.mark.skipif(
    not CvxpySolver.available(),
    reason="the optional package cvxpy is not installed, so the conic solver is unavailable")


def values_of(embedding):
    return {k: getattr(embedding, k) for k in embedding._ARRAYS}


# =========================================================================================
# BoxLitE
# =========================================================================================
def tiny_kb():
    """A TBox with a negative inclusion, a role and an inclusion with an existential, and an
    ABox with a concept and a role assertion (dimension 4 is enough)."""
    return DLLiteKB.from_lists(
        inclusions=[("A", "B"), ("B", "∃r")], negative_inclusions=[("A", "D")],
        concept_assertions=[("A", "a")], role_assertions=[("r", "a", "b")])


def smaller_kb():
    return DLLiteKB.from_lists(inclusions=[("A", "B")], concept_assertions=[("A", "a")])


@pytest.fixture(scope="module")
def solved_tiny():
    """The tiny KB solved with SciPy (no cvxpy needed)."""
    kb = tiny_kb()
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic + 1)
    model.train(solver=ScipySolver(maxiter=120))
    return model


def test_boxlite_is_made_of_the_parts_the_card_names():
    model = BoxLitE.from_kb(tiny_kb(), embed_dim=6)
    g = model.module
    assert isinstance(g, ComposedGeometry) and isinstance(g, Geometry)
    assert type(g.class_model) is sb.BoundsBoxes
    assert type(g.relation_model) is sb.HeadTailBumpBoxes
    assert type(g.measure) is sb.SignedDistance
    assert type(g.objective) is sb.BoxLitEObjective
    assert g.individual_model.bumps is not None          # a position and a bump per individual
    assert g.parts == {"classes": "box", "relations": "head and tail boxes",
                       "individuals": "point with bump", "measure": "exact",
                       "conjunction": "exact", "negation": "exact"}
    # the builder is the named combination, and a part of it can be replaced
    built = METHODS["boxlite"](3, 1, 2, 6, variant="reference")
    assert type(built).__name__ == "BoxLitEGeometry" and built.objective.margin == 0.0
    replaced = METHODS["boxlite"](3, 1, 2, 6, objective=sb.BoxLitEObjective(0.2))
    assert replaced.objective.margin == 0.2
    assert geometry_of(model) is g
    # the fit is a part too: the program, its weights and the parts that add terms
    program = model.program
    assert program.weights == ("lambda1", "lambda2", "lambda3")
    assert [type(r).__name__ for r in program.regularizers] == [
        "NegativeConceptPenalty", "WidthPenalty", "WidthPenalty"]
    assert program.variables()["head_lower"] == (1, 6)


def test_boxlite_catalogue_card_and_parts_agree():
    from mowl.catalog.registry import get_method
    card = get_method("boxlite")
    rep = REPRESENTATIONS["boxlite"]
    parts = METHODS["boxlite"](3, 1, 2, 6).parts
    assert parts["classes"] in {v.value for v in rep.classes}
    assert parts["relations"] in {v.value for v in rep.relations}
    assert parts["measure"] in {v.value for v in rep.measure}
    assert parts["individuals"] in {v.value for v in rep.individuals}
    assert parts["conjunction"] == rep.conjunction.value
    assert parts["negation"] == rep.negation.value
    assert rep.parametrisation[0].value == "construction"       # solved, not trained
    # every axiom form the card states is one the geometry reads
    g = METHODS["boxlite"](3, 1, 2, 6)
    reads = {"concept_inclusion": "gci0", "role_inclusion": "role_inclusion",
             "concept_assertion": "class_assertion",
             "role_assertion": "object_property_assertion"}
    for axiom in card.axioms:
        if axiom == "negative_inclusion":
            assert callable(g.negative_inclusion_cost)
        else:
            assert reads[axiom] in g.supported_forms, axiom
    assert card.logic == "DL-Lite_H" and "instance-checking" in card.tasks


def test_the_program_is_the_objective_of_the_geometry_at_a_solution(solved_tiny):
    """With all weights zero the objective is the largest assertion loss: the program (NumPy
    algebra) and the geometry (tensors) are the same formulas. The solution satisfies every
    constraint."""
    model = solved_tiny
    e = model.embedding
    ev = evaluate(model.program, values_of(e), {})
    assert ev.feasible(1e-4), ev.slacks
    g = model.module
    kb = model.kb
    concept = th.tensor([[a, b] for b, a in kb.concept_assertions])
    role = th.tensor([[a, r, b] for r, a, b in kb.role_assertions])
    losses = th.cat([g.cost("class_assertion", concept).reshape(-1),
                     g.cost("object_property_assertion", role).reshape(-1)])
    assert ev.objective == pytest.approx(float(losses.max()), abs=1e-5)
    assert float(losses.max()) <= 1e-6                 # Theorem 4 (iv): a model of a satisfiable KB
    check = model.check_model(tol=1e-3)
    assert check.well_formed and check.box_consistent and check.is_model


def test_the_program_reports_the_constraint_a_bad_embedding_violates():
    kb = tiny_kb()
    program = sb.BoxLitEProgram(kb, kb.n_basic + 1)
    rng = np.random.default_rng(0)
    values = {name: rng.uniform(-2, 2, shape) for name, shape in program.variables().items()}
    ev = evaluate(program, values, {})
    assert not ev.feasible()
    assert {"width", "universe"} <= set(k for k, v in ev.slacks.items() if v < 0)


def test_regulariser_terms_are_the_same_formula_on_arrays_and_tensors(solved_tiny):
    model = solved_tiny
    g, e = model.module, model.embedding
    for reg in (sb.WidthPenalty("boxes", 1.0), sb.WidthPenalty("bump", 1.0),
                sb.NegativeConceptPenalty(sb.negative_concept_assertions(model.kb), 1.0)):
        numpy_term = reg.term(sb._Context(NumpyAlgebra(), values_of(e), e.s_omega, e.margin,
                                          "paper", e.dim))
        torch_term = reg.penalty(g)
        assert float(torch_term) == pytest.approx(float(numpy_term), rel=1e-5, abs=1e-6)


# -- swaps: another solver, another fit, another variant, an added regulariser ----------------
def test_boxlite_swap_the_solver_for_another_variant_of_the_programme():
    kb = smaller_kb()
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic, variant="reference")
    result = model.train(solver=ScipySolver(maxiter=60))
    assert result.solved and np.isfinite(result.value)
    assert evaluate(model.program, values_of(model.embedding), {}).feasible(1e-4)
    assert model.module.objective.margin == 0.0          # the reference code: closed boxes
    assert np.isfinite(model.score_concept_assertion(1, "a"))


def test_boxlite_swap_the_fit_for_gradient_descent_on_the_same_parts():
    kb = tiny_kb()
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic + 1)
    th.manual_seed(0)
    before = float(model.module.cost("class_assertion", th.tensor([[0, 0]])).sum())
    result = model.train(regime=GradientFit(steps=30, lr=0.05), lambdas=(0.1, 0.01, 0.01))
    assert result.status == "gradient" and np.isfinite(result.value)
    assert all(np.isfinite(v).all() for v in values_of(model.embedding).values())
    after = float(model.module.cost("class_assertion", th.tensor([[0, 0]])).sum())
    assert np.isfinite(before) and np.isfinite(after)
    scores = model.role_scores(0, subject=0)
    assert scores.shape == (len(kb.individuals),) and np.isfinite(scores).all()


def test_boxlite_swap_in_a_regulariser_on_the_geometry_and_a_part():
    kb = smaller_kb()
    d = kb.n_basic + 1
    geometry = METHODS["boxlite"](len(kb.concepts), len(kb.roles), len(kb.individuals), d,
                                  regularizers=(sb.WidthPenalty("boxes", 0.5),),
                                  classes=sb.BoundsBoxes(len(kb.concepts), d, scale=0.1))
    assert float(geometry.regularization_loss()) > 0
    loss = geometry.cost("class_assertion", th.tensor([[0, 0]])).sum() \
        + geometry.regularization_loss()
    loss.backward()
    assert geometry.class_model.lower.weight.grad is not None
    assert th.isfinite(geometry.class_model.lower.weight.grad).all()


def test_boxlite_negative_concept_loss_and_complement_box_are_the_papers():
    g = METHODS["boxlite"](2, 1, 2, 4)
    rows = th.tensor([[0, 1], [1, 0]])
    positive, negative = g.cost("class_assertion", rows), g.cost("class_assertion", rows, True)
    box = g.classes(th.tensor([1]))
    complement = g.negation(box)
    s = g.s_omega
    assert th.allclose(complement.lower, -s - box.lower) and th.allclose(complement.upper,
                                                                         s - box.upper)
    assert positive.shape == negative.shape == (2, 1) and not th.equal(positive, negative)
    with pytest.raises(NotImplementedError):          # only concept assertions have a negative
        g.cost("object_property_assertion", th.tensor([[0, 0, 1]]), True)
    with pytest.raises(ConstructorUnavailable):       # DL-Lite: unqualified existentials only
        g.exists(th.tensor([0]), g.classes(th.tensor([0])))


@needs_cvxpy
def test_boxlite_conic_solver_gives_a_verified_model():
    kb = tiny_kb()
    model = BoxLitE.from_kb(kb, embed_dim=sb.kb_faithful_dimension(kb))
    result = model.train()
    assert result.solved and result.value <= 1e-6
    assert model.check_model().is_model
    assert evaluate(model.program, values_of(model.embedding), {}).feasible(1e-5)
    # the module holds the solution
    assert np.allclose(model.module.snapshot().concept_lower, model.embedding.concept_lower)


# -- the general parts: algebra, grounding -----------------------------------------------------
def test_signed_distance_is_the_same_on_arrays_and_tensors():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(6, 5))
    y[0] = -np.abs(y[0])
    a, b = NumpyAlgebra().sdist_rows(y), TorchAlgebra().sdist_rows(th.tensor(y))
    assert np.allclose(a, b.numpy())
    assert a[0] == pytest.approx(y[0].max())
    assert a[1] == pytest.approx(np.linalg.norm(np.maximum(y[1], 0)))


def test_grounding_checks_the_ground_instances_against_the_domain():
    domain = Domain(nb_classes=3, nb_individuals=2, nb_relations=1)
    assert domain.size == 5 and domain.individual_element(th.tensor([1])).item() == 4
    table = AxiomTable(domain, {"gci0": np.array([[0, 1], [2, 0]]),
                                "class_assertion": np.array([[1, 2]])})
    assert table.forms() == ["gci0", "class_assertion"] and table.count()["gci0"] == 2
    with pytest.raises(ValueError, match="outside"):
        AxiomTable(domain, {"class_assertion": np.array([[2, 0]])})   # individual 2 of 2
    alg = NumpyAlgebra()
    inside_bounds(alg, (np.array([0.2]), np.array([0.5])), (np.array([0.0]), np.array([1.0])))
    assert min(s.min() for _, s in alg.constraints) >= 0


# =========================================================================================
# FaithEL
# =========================================================================================
@pytest.fixture(scope="module")
def faithel_dataset():
    base = os.environ.get("TMPDIR") or None
    return _write_family_abox_dataset(pathlib.Path(tempfile.mkdtemp(prefix="mowl_rs_", dir=base)))


def new_faithel(dataset, **options):
    options.setdefault("batch_size", 8)
    options.setdefault("learning_rate", 0.01)
    options.setdefault("seed", 0)
    return FaithEL(dataset, **options)


def test_faithel_is_made_of_the_parts_the_card_names(faithel_dataset):
    model = new_faithel(faithel_dataset)
    g = model.module
    assert isinstance(g, ComposedGeometry)
    assert type(g.tbox) is sf.FaithfulTBox                  # constructed, fixed by reasoning
    assert type(g.class_model) is sf.AnchoredClasses
    assert type(g.relation_model) is sf.AnchoredPairs
    assert type(g.measure) is sf.AnchorDistance
    assert type(g.objective) is sf.AnchoredObjective
    assert type(model.training) is sf.AlternatingMinibatches
    assert g.parts["classes"] == "convex region" and g.parts["relations"] == "region over pairs"
    assert g.parts["measure"] == "point-to-region distance" and g.parts["individuals"] == "point"
    assert (g.objective.phi, g.objective.gamma, g.objective.psi) == (1.0, 1.0, 0.7)
    assert geometry_of(model) is g
    # the builder alone (from sizes: no TBox) is the same combination
    built = METHODS["faithel"](4, 1, 3, 5)
    assert type(built.class_model) is sf.AnchoredClasses and built.tbox is None
    with pytest.raises(ConstructorUnavailable, match="constructed TBox"):
        built.cost("gci0", th.tensor([[0, 1]]))


def test_faithel_catalogue_card_and_parts_agree(faithel_dataset):
    from mowl.catalog.registry import get_method
    card = get_method("faithel")
    rep = REPRESENTATIONS["faithel"]
    parts = new_faithel(faithel_dataset).module.parts
    assert parts["classes"] in {v.value for v in rep.classes}
    assert parts["relations"] in {v.value for v in rep.relations}
    assert parts["measure"] in {v.value for v in rep.measure}
    assert parts["individuals"] in {v.value for v in rep.individuals}
    assert parts["conjunction"] == rep.conjunction.value
    assert parts["existential"] == rep.existential.value
    # the tasks of the card are the forms the model scores
    model = new_faithel(faithel_dataset)
    assert {"subsumption", "instance-checking", "role-assertion"} <= set(card.tasks)
    assert {"gci0", "class_assertion", "object_property_assertion"} <= set(model.scored_forms)
    assert {"gci0", "gci1", "gci2", "gci3", "class_assertion",
            "object_property_assertion"} <= model.module.supported_forms


def test_faithel_tbox_forms_are_the_constructed_models_and_the_abox_forms_are_learned(
        faithel_dataset):
    model = new_faithel(faithel_dataset, neg_sampling=False)
    g, ind, cls = model.module, model.individual_index_dict, model.class_index_dict
    father, parent, female = (cls[NS + n] for n in ("Father", "Parent", "Female"))
    gci0 = th.tensor([[father, parent], [parent, father]])
    assert g.cost("gci0", gci0).reshape(-1).tolist() == [0.0, 1.0]       # entailed, not entailed
    rows = th.tensor([[ind[IND + "bob"], father], [ind[IND + "bob"], female]])
    assert g.cost("class_assertion", rows).shape == (2, 1)
    assert th.allclose(g.class_assertion_costs(rows), model.axiom_costs("class_assertion", rows))
    g.cost("class_assertion", rows).sum().backward()
    assert g.individual_model.centres.weight.grad is not None
    assert g.class_model.moving.weight.grad is not None


def test_faithel_swap_the_training_regime(faithel_dataset):
    model = new_faithel(faithel_dataset, training=sf.FullBatch())
    model.train(2)
    assert len(model.loss_history) == 2 and np.isfinite(model.loss_history).all()
    alternating = new_faithel(faithel_dataset)
    alternating.train(2)
    assert np.isfinite(alternating.loss_history).all()
    assert model.loss_history != alternating.loss_history      # a different loop, other numbers
    d = (model.module.concepts.weight - model.module.concept_centroid).norm(dim=1)
    assert float(d.max()) <= model.radius + 1e-5                # still projected to the radius


def test_faithel_swap_the_measure_and_the_objective_options(faithel_dataset):
    model = new_faithel(faithel_dataset, centroid_score=False, neg_sampling=False, psi=0.0)
    assert model.module.measure.centroid_score is False
    rows = th.tensor([[0, 0], [1, 1]])
    anchored = model.axiom_costs("class_assertion", rows)
    both = new_faithel(faithel_dataset).axiom_costs("class_assertion", rows)
    assert th.isfinite(anchored).all() and (both >= anchored - 1e-6).all()   # the centroid term
    model.train(1)
    assert np.isfinite(model.loss_history).all()
    scores = model.axiom_costs("object_property_assertion", th.tensor([[0, 0, 1]]))
    assert th.isfinite(scores).all()


def test_faithel_swap_in_a_regulariser_and_a_radius(faithel_dataset):
    class Shrink:
        """A penalty on the individuals (any object with ``penalty(geometry)``)."""
        kind = "penalty"

        def penalty(self, geometry):
            return 0.1 * geometry.individual_model.centres.weight.pow(2).mean()

    model = new_faithel(faithel_dataset, radius=0.0)
    model.module.regularizers = (Shrink(),)
    assert float(model.module.regularization_loss()) > 0
    model.train(1)
    # radius 0: the moving points cannot leave the centroids
    assert th.allclose(model.module.concepts.weight, model.module.concept_centroid, atol=1e-6)
