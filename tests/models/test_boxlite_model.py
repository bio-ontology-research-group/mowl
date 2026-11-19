"""Tests of BoxLitE (Lourenço et al., KR 2026): the convex programme, the independent
model checker, satisfiability decisions, the paper's examples and scoring."""

import os
import tempfile

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

import mowl.datasets  # noqa: E402,F401  (import order: datasets before nn)
from mowl.datasets.dllite import DLLiteKB  # noqa: E402
from mowl.models.boxlite.checker import check_model  # noqa: E402
from mowl.models.boxlite.embedding import (BoxLitEEmbedding, box_distance,  # noqa: E402
                                           signed_distance_to_nonpositive_orthant)
from mowl.models.boxlite.model import BoxLitE, kb_faithful_dimension  # noqa: E402
from mowl.models.boxlite.program import (BoxLitEProgram,  # noqa: E402
                                         negative_concept_assertions)

# Satisfiable DL-Lite_H KBs with negative inclusions and role hierarchies with inverses.
KBS = {
    "hierarchy": dict(
        inclusions=[("A", "B"), ("B", "C"), ("∃r", "A"), ("∃r-", "D")],
        concept_assertions=[("C", "x")],
        role_assertions=[("r", "a", "b"), ("r", "b", "c")]),
    "negation": dict(
        inclusions=[("A", "B")], negative_inclusions=[("B", "D"), ("∃r", "D")],
        concept_assertions=[("A", "a"), ("D", "d")], role_assertions=[("r", "a", "d")]),
    "inverse_roles": dict(
        inclusions=[("∃s-", "P")], negative_inclusions=[("P", "∃t")],
        role_inclusions=[("r", "s-"), ("t", "t-"), ("u", "r")],
        concept_assertions=[("∃u", "a")],
        role_assertions=[("u", "a", "b"), ("t", "c", "d"), ("r", "b", "e")]),
    "family_figure2": dict(  # the TBox of Figure 2 with a small ABox
        negative_inclusions=[("∃hasFather-", "∃hasMother-")],
        role_inclusions=[("relative-", "relative"), ("hasSibling", "relative"),
                         ("hasChild", "relative"), ("hasParent", "relative"),
                         ("hasFather", "hasParent"), ("hasMother", "hasParent"),
                         ("spouse-", "spouse"), ("hasSibling-", "hasSibling"),
                         ("spouse", "relative")],
        role_assertions=[("hasFather", "ann", "bob"), ("hasMother", "ann", "cat"),
                         ("spouse", "bob", "cat"), ("hasSibling", "ann", "dan"),
                         ("hasChild", "cat", "dan")]),
}


def _kb(name):
    return DLLiteKB.from_lists(**KBS[name])


def _scratch():
    base = os.environ.get("TMPDIR") or os.path.expanduser("~/tmp")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="mowl_boxlite_", dir=base)


# ---------------------------------------------------------------------------------------
# Signed distance (Propositions 3 and 4)
# ---------------------------------------------------------------------------------------

def test_signed_distance_matches_proposition_4():
    """The NumPy formula of Proposition 3 equals the partial minimisation of
    Proposition 4 used in the programme, inside and outside the orthant."""
    rng = np.random.default_rng(0)
    for y in [rng.normal(size=6), -np.abs(rng.normal(size=6)), np.array([1., -2., 3., -4.])]:
        z = cp.Variable(len(y))
        prob = cp.Problem(cp.Minimize(cp.norm(z, 2) + cp.max(y - z)), [z >= y])
        prob.solve()
        assert prob.value == pytest.approx(
            float(signed_distance_to_nonpositive_orthant(y)), abs=1e-6)
    assert signed_distance_to_nonpositive_orthant(np.array([1., -2., 3., -4.])) == \
        pytest.approx(np.sqrt(10))


def test_box_distance_sign():
    lo, up = np.zeros(2), np.ones(2)
    assert box_distance(lo, up, np.array([0.5, 0.5]), 0.01) < 0
    assert box_distance(lo, up, np.array([1.5, 0.5]), 0.01) == pytest.approx(0.51)
    assert box_distance(lo, up, np.array([0.995, 0.5]), 0.01) > 0  # in [L, U], not shrunk


# ---------------------------------------------------------------------------------------
# The programme
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["paper", "reference"])
@pytest.mark.parametrize("name", sorted(KBS))
def test_programme_is_dcp_and_dpp(name, variant):
    kb = _kb(name)
    program = BoxLitEProgram(kb, kb.n_basic + 2, variant=variant)
    assert program.problem.is_dcp()
    assert program.problem.is_dpp()


def test_dimension_must_cover_box_consistency():
    kb = _kb("hierarchy")
    with pytest.raises(ValueError, match="box consistency"):
        BoxLitEProgram(kb, kb.n_basic - 1)


_NAMES = [pytest.param(n, marks=pytest.mark.slow) if n == "family_figure2" else n
          for n in sorted(KBS)]


@pytest.mark.parametrize("variant", ["paper", "reference"])
@pytest.mark.parametrize("name", _NAMES)
def test_solution_is_verified_model(name, variant):
    """Theorem 4 (iii)/(iv) and Proposition 2: with lambda = 0 and d as in Corollary 2,
    the solved embedding of a satisfiable KB is, by the independent checker, a
    well-formed, box-consistent model (hence weakly KB faithful)."""
    kb = _kb(name)
    model = BoxLitE.from_kb(kb, embed_dim=kb_faithful_dimension(kb), variant=variant)
    result = model.train()
    assert result.solved
    assert result.value <= 1e-6  # Theorem 4 (iv)
    check = model.check_model()
    assert check.well_formed, check.violations
    assert check.box_consistent, check.violations
    assert check.is_model, check.violations
    assert check.weakly_kb_faithful


@pytest.mark.parametrize("variant", ["paper", "reference"])
@pytest.mark.parametrize("lambdas", [(0.1, 0.01, 0.01), (100.0, 100.0, 100.0)])
@pytest.mark.parametrize("name", ["hierarchy", "negation", "inverse_roles"])
def test_tbox_holds_for_any_lambda(name, lambdas, variant):
    """Theorem 4 (ii): the TBox and box consistency are constraints, so they hold for any
    feasible point, whatever the regularisation weights (with lambda > 0 the ABox need
    not be satisfied: Theorem 4 (iii) assumes lambda_1 = 0)."""
    kb = _kb(name)
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic + 2, variant=variant)
    model.train(lambdas=lambdas)
    check = model.check_model()
    for cat in ("concept_inclusion", "negative_inclusion", "role_inclusion",
                "box_consistency", "width", "universe"):
        assert not check.violations.get(cat), (cat, check.violations)


def test_resolve_reuses_compiled_programme():
    kb = _kb("hierarchy")
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic)
    model.train(lambdas=(0.0, 0.0, 0.0))
    program = model.program
    model.train(lambdas=(0.3, 0.1, 0.1))
    assert model.program is program
    assert model.lambdas == (0.3, 0.1, 0.1)


# ---------------------------------------------------------------------------------------
# Satisfiability (Theorem 4 iii and iv)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(KBS))
def test_satisfiable_kbs_are_recognised(name):
    res = BoxLitE.decide_satisfiability(_kb(name))
    assert res.satisfiable is True
    assert res.optimum <= 1e-6
    assert res.check.is_model and res.check.box_consistent


UNSAT = {
    "disjoint_assertions": dict(negative_inclusions=[("A", "B")],
                                concept_assertions=[("A", "x"), ("B", "x")]),
    "bottom": dict(negative_inclusions=[("A", "A")], concept_assertions=[("A", "x")]),
    "via_role_hierarchy_and_inverse": dict(
        negative_inclusions=[("∃r", "A")], role_inclusions=[("s", "r-")],
        concept_assertions=[("A", "x")], role_assertions=[("s", "y", "x")]),
    "family_father_and_mother": dict(
        negative_inclusions=[("∃hasFather-", "∃hasMother-")],
        role_inclusions=[("hasFather", "hasParent"), ("hasMother", "hasParent")],
        role_assertions=[("hasFather", "a", "p"), ("hasMother", "b", "p")]),
}


@pytest.mark.parametrize("name", sorted(UNSAT))
def test_unsatisfiable_kbs_have_positive_optimum(name):
    kb = DLLiteKB.from_lists(**UNSAT[name])
    res = BoxLitE.decide_satisfiability(kb)
    assert res.satisfiable is False
    assert res.optimum > 1e-3  # violations are of the order of epsilon
    assert not res.check.is_model
    assert res.check.box_consistent  # constraints still hold


# ---------------------------------------------------------------------------------------
# The paper's examples
# ---------------------------------------------------------------------------------------

def test_figure1_example_in_two_dimensions():
    """Figure 1: a 2-D box interpretation satisfying {R(a,b), R(b,d), R(b,c), R(c,d),
    R(c,c), ∃R(a), ∃R(b), ∃R(c)} over {a, b, c, d, e}. d = 2 = |N_C| + 2|N_R| is the
    minimum for box consistency; the programme finds such a model."""
    kb = DLLiteKB.from_lists(
        individuals=["a", "b", "c", "d", "e"],
        concept_assertions=[("∃R", "a"), ("∃R", "b"), ("∃R", "c")],
        role_assertions=[("R", "a", "b"), ("R", "b", "d"), ("R", "b", "c"),
                         ("R", "c", "d"), ("R", "c", "c")])
    model = BoxLitE.from_kb(kb, embed_dim=2)
    result = model.train()
    assert result.value <= 1e-6
    check = model.check_model()
    assert check.is_model and check.box_consistent, check.violations


def test_existential_box_contains_subjects():
    """Section 3: if R(a, b) holds then pos(a) ∈ η(∃R) (Head(R) enlarged by Bump(R))."""
    kb = _kb("hierarchy")
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic)
    model.train()
    e = model.embedding
    for r, a, _ in kb.role_assertions:
        assert e.concept_loss(kb.exists_index(r), [a])[0] <= 1e-6


# ---------------------------------------------------------------------------------------
# Independent checker and Theorem 2
# ---------------------------------------------------------------------------------------

def _embedding(concept_lower, concept_upper, pos, s=1.0, margin=0.01):
    d = concept_lower.shape[1]
    z = np.zeros((0, d))
    return BoxLitEEmbedding(pos=pos, bump=np.zeros_like(pos), concept_lower=concept_lower,
                            concept_upper=concept_upper, head_lower=z, head_upper=z,
                            tail_lower=z, tail_upper=z, bump_lower=z, bump_upper=z,
                            s_omega=s, margin=margin)


def test_checker_detects_violations():
    kb = DLLiteKB.from_lists(inclusions=[("A", "B")], concept_assertions=[("A", "x")])
    # dims: i_A = 0, i_B = 1; box consistency needs L + U <= -1 in the own dimension.
    good = _embedding(np.array([[-0.8, -0.8], [-0.9, -0.9]]),
                      np.array([[-0.3, -0.4], [0.0, -0.2]]), np.array([[-0.5, -0.6]]))
    assert check_model(good, kb).weakly_kb_faithful
    not_included = _embedding(good.concept_lower, np.array([[-0.3, -0.4], [-0.5, -0.2]]),
                              good.pos)
    chk = check_model(not_included, kb)
    assert not chk.is_model and chk.violations["concept_inclusion"] == ["A ⊑ B"]
    outside = _embedding(good.concept_lower, good.concept_upper, np.array([[0.5, 0.0]]))
    assert check_model(outside, kb).violations["concept_assertion"] == ["A(x)"]
    inconsistent = _embedding(np.array([[-0.3, -0.3], [-0.9, -0.9]]),
                              np.array([[0.3, 0.3], [0.0, -0.2]]), np.array([[0.0, 0.0]]))
    assert not check_model(inconsistent, kb).box_consistent


def test_theorem2_complement_properties():
    """Theorem 2 (ii) and (iii) for complement boxes: involution and anti-monotonicity."""
    rng = np.random.default_rng(1)
    e = _embedding(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)))
    for _ in range(50):
        lc = rng.uniform(-1, 0, 3)
        uc = lc + rng.uniform(0, 1, 3)
        ld = lc - rng.uniform(0, 0.5, 3)
        ud = uc + rng.uniform(0, 0.5, 3)
        l2, u2 = e.complement_bounds(*e.complement_bounds(lc, uc))
        assert np.allclose(l2, lc) and np.allclose(u2, uc)
        nlc, nuc = e.complement_bounds(lc, uc)
        nld, nud = e.complement_bounds(ld, ud)
        assert (nlc <= nld).all() and (nud <= nuc).all()  # η(¬D) ⊆ η(¬C)


# ---------------------------------------------------------------------------------------
# Negatives, scoring, persistence
# ---------------------------------------------------------------------------------------

def test_negative_concept_assertions():
    kb = DLLiteKB.from_lists(roles=["r", "s"], concept_assertions=[("A", "a")],
                             role_assertions=[("r", "a", "b")])
    names = {(kb.basic_concept_name(b), kb.individuals[a])
             for b, a in negative_concept_assertions(kb, "paper")}
    assert ("A", "b") in names and ("A", "a") not in names
    assert ("∃r", "a") not in names and ("∃r", "b") in names
    assert ("∃r⁻", "b") not in names and ("∃r⁻", "a") in names
    assert {("∃s", "a"), ("∃s⁻", "b")} <= names
    ref = {(kb.basic_concept_name(b), kb.individuals[a])
           for b, a in negative_concept_assertions(kb, "reference")}
    assert not any(n == "∃r⁻" for n, _ in ref)  # reference code: no ∃R⁻ for ABox roles
    assert {("∃s", "a"), ("∃s⁻", "a")} <= ref


def test_scores_rank_training_assertions_first():
    kb = DLLiteKB.from_lists(role_assertions=[("r", "a", "b"), ("r", "c", "d")],
                             individuals=["a", "b", "c", "d", "e"])
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic + 2, lambda1=1.0, lambda2=0.1,
                            lambda3=0.1)
    model.train()
    scores = model.role_scores("r", subject="a")
    assert scores.shape == (5,)
    assert int(np.argmax(scores)) == kb.individuals.index("b")
    assert model.score_role_assertion("r", "a", "b") == pytest.approx(scores[1])
    assert model.score_role_assertion("r", "a", "b") > model.score_role_assertion("r", "a", "e")
    assert model.score_concept_assertion(kb.exists_index(0), "a") > \
        model.score_concept_assertion(kb.exists_index(0), "e")


def test_save_and_load_roundtrip():
    kb = _kb("hierarchy")
    model = BoxLitE.from_kb(kb, embed_dim=kb.n_basic,
                            model_filepath=os.path.join(_scratch(), "boxlite.npz"))
    model.train()
    before = model.role_scores("r", subject="a")
    model.save()
    model.embedding = None
    model.load()
    assert np.allclose(model.role_scores("r", subject="a"), before)


# ---------------------------------------------------------------------------------------
# Datasets, evaluation regime and benchmark plumbing
# ---------------------------------------------------------------------------------------

_NS = "http://mowl.borg/boxlite#"


def _toy_dataset():
    from mowl.datasets import PathDataset

    def ont(name, lines):
        path = os.path.join(directory, f"{name}.owl")
        with open(path, "w") as f:
            f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
                    f"Ontology(<http://mowl.borg/boxlite/{name}>\n" + "\n".join(lines) + "\n)\n")
        return path

    def opa(r, a, b):
        return f"ObjectPropertyAssertion(<{_NS}{r}> <{_NS}{a}> <{_NS}{b}>)"

    directory = _scratch()
    train = [f"SubObjectPropertyOf(<{_NS}hasFather> <{_NS}hasParent>)",
             f"SubObjectPropertyOf(<{_NS}hasParent> ObjectInverseOf(<{_NS}hasChild>))",
             opa("hasFather", "a", "b"), opa("hasFather", "c", "d"),
             opa("hasParent", "e", "f"), opa("hasChild", "f", "e")]
    valid = [opa("hasParent", "a", "b")]
    test = [opa("hasParent", "c", "d"), opa("hasChild", "b", "a"), opa("hasChild", "d", "c")]
    return PathDataset(ont("train", train), ont("valid", valid), ont("test", test))


def test_model_from_dataset_and_ranking_regime():
    from mowl.evaluation.assertion_ranking import RoleAssertionRankingRegime
    ds = _toy_dataset()
    model = BoxLitE(ds, embed_dim=8, lambda1=0.1, lambda2=0.01, lambda3=0.01)
    assert model.kb.summary()["role_assertions"] == 4
    model.train()
    assert model.check_model().box_consistent
    res = RoleAssertionRankingRegime(split="test").evaluate(model, ds)["all"]
    assert set(res) == {"mrr", "hits@1", "hits@3", "hits@5", "hits@10"}
    assert 0 < res["mrr"] <= 1


def test_runner_with_solver_family_and_lambda_selection():
    from mowl.benchmarks.protocols.boxlite import (SolverModelFamily, run_family_benchmark,
                                                   select_lambdas)
    from mowl.benchmarks.runner import _FAMILIES, register_family, run_benchmark
    from mowl.benchmarks.spec import Benchmark, Task
    from mowl.catalog.card import Reference
    bench = Benchmark(
        key="boxlite-toy", name="toy", task=Task.LINK_PREDICTION, description="toy",
        dataset="mowl.datasets.PathDataset",
        protocol="mowl.benchmarks.protocols.boxlite.BoxLitEFamilyProtocol",
        metrics=("mrr", "hits@1", "hits@3", "hits@5", "hits@10"),
        references=(Reference("toy", "toy"),))
    ds = _toy_dataset()
    family = SolverModelFamily()
    register_family(family)
    try:
        result = run_benchmark(bench, method="boxlite", dataset=ds, workdir=_scratch(),
                               model_kwargs={"embed_dim": 8, "lambda1": 0.1})
    finally:
        _FAMILIES.remove(family)
    assert 0 < result.metrics["all"]["mrr"] <= 1
    assert result.protocol_info["training"]["status"] in ("optimal", "optimal_inaccurate")

    direct = run_family_benchmark(bench, dataset=ds, model_kwargs={"embed_dim": 8},
                                  grid=[(0.0, 0.0, 0.0), (0.1, 0.01, 0.01)])
    assert direct.protocol_info["training"]["configurations"] == 2
    model = BoxLitE(ds, embed_dim=8)
    lambdas, metrics, results = select_lambdas(model, ds, grid=[(0.0, 0.0, 0.0)])
    assert lambdas == (0.0, 0.0, 0.0) and len(results) == 1


@pytest.mark.slow
def test_family_dataset_matches_table1_and_reference_tbox():
    """Downloads the pinned release (checksummed)."""
    from mowl.datasets.builtin.boxlite_family import BoxLitEFamilyDataset
    from mowl.datasets.dllite import extract_dllite
    ds = BoxLitEFamilyDataset(1, data_root=_scratch())
    kb = extract_dllite(ds.ontology)
    assert len(kb.role_assertions) == 300 and len(kb.individuals) == 155
    assert len(kb.role_inclusions) == 10 and not kb.skipped
    assert len(extract_dllite(ds.validation, signature=kb).role_assertions) == 110
    assert len(extract_dllite(ds.testing, signature=kb).role_assertions) == 440
