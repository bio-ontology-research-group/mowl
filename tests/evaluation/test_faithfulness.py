"""Tests of mowl.evaluation.faithfulness (Mashkova & Hoehndorf 2026) and
mowl.evaluation.properties (Bourgaux et al. 2024)."""

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

import mowl
mowl.init_jvm("2g")

import mowl.datasets  # noqa: E402,F401
from mowl.datasets import Dataset  # noqa: E402
from mowl.evaluation.faithfulness import (  # noqa: E402
    CONTRADICTORY, ENTAILED, UNDETERMINED, CallableScorer, FaithfulnessHierarchy,
    FaithfulnessTestSet, LogicalStatusOracle, LossModelness, RelativeModelCounter,
    _VarMap, admissibility_at_k, count_models, discrimination_auc, evaluate_faithfulness,
    faithfulness_violation_rate, generate_test_sets, herbrand_clauses, is_admissible,
    is_discriminative, is_monotonic_faithful, is_probabilistic_faithful, loss_modelness,
    oracle_scorer, pair_fvr, ranking_metrics, spearman_rho)
from mowl.evaluation.properties import kb_embedding_properties  # noqa: E402
from mowl.geometric.normal_forms import (BOTTOM, NF1, NF2, NF3, NF4,  # noqa: E402
                                         nominal_name)
from mowl.owlapi import OWLAPIAdapter  # noqa: E402

NS = "http://faithfulness.test/"
PhD, Student, Prof, Person, TA = (
    NS + n for n in ("PhD", "Student", "Professor", "Person", "TA"))
Course, Staff, Employee = NS + "Course", NS + "Staff", NS + "Employee"
teaches, enrolled = NS + "teaches", NS + "enrolledIn"


def university():
    """Appendix D of Mashkova & Hoehndorf (2026), extended with roles."""
    return [
        NF1(PhD, Student), NF2(Student, Prof, BOTTOM),
        NF1(Student, Person), NF1(Prof, Person), NF1(TA, Person),
        NF3(Student, enrolled, Course), NF3(Prof, teaches, Course),
        NF4(teaches, Course, Staff), NF2(Staff, Student, BOTTOM),
        NF2(Student, TA, Employee),
    ]


@pytest.fixture(scope="module")
def oracle():
    return LogicalStatusOracle(university(), witnesses="paper")


# ---------------------------------------------------------------- relative model counts

def test_worked_example_counts():
    """Appendix A: O = {A ⊑ B}, |Δ| = 2: #(O) = 576, #(O + ∃r.A ⊑ C) = 425."""
    var = _VarMap()
    dom = ["a1", "a2"]
    base = herbrand_clauses([NF1("A", "B")], dom, var)
    for n in "ABC":
        for a in dom:
            var(("c", n, a))
    for a in dom:
        for b in dom:
            var(("r", "r", a, b))
    assert count_models(base, var.ids.values(), backend="python") == 576
    plus = base + herbrand_clauses([NF4("r", "A", "C")], dom, var)
    assert count_models(plus, var.ids.values(), backend="python") == 425
    counter = RelativeModelCounter([NF1("A", "B")], domain_size=2, backend="python")
    assert counter.relative_count("gci3", ("r", "A", "C")) == Fraction(425, 576)
    # Proposition 1: entailed -> 1, Herbrand contradiction -> 0.
    assert counter.relative_count("gci0", ("A", "B")) == 1
    assert RelativeModelCounter([NF1(nominal_name("x"), "A"), NF2("A", "B", BOTTOM)],
                                domain_size=2, backend="python"
                                ).relative_count("gci0", (nominal_name("x"), "B")) == 0


def test_python_counter_matches_brute_force():
    rng = random.Random(3)
    for _ in range(20):
        n = 7
        clauses = [tuple(rng.choice((1, -1)) * v for v in rng.sample(range(1, n + 1), 3))
                   for _ in range(rng.randint(1, 9))]
        brute = sum(all(any((x > 0) == bits[abs(x) - 1] for x in cl) for cl in clauses)
                    for bits in itertools.product((False, True), repeat=n))
        assert count_models(clauses, range(1, n + 1), backend="python") == brute


def test_existential_tseitin_does_not_change_counts():
    # A ⊑ ∃r.B over one element: models of {A(e) -> r(e,e) & B(e)} on 3 atoms = 5.
    counter = RelativeModelCounter([], domain_size=1, backend="python")
    assert counter.relative_count("gci2", ("A", "r", "B")) == Fraction(5, 8)


# ---------------------------------------------------------------- logical status

def test_appendix_d_statuses(oracle):
    assert oracle.status("gci0", (PhD, Student)) == ENTAILED
    assert oracle.status("gci0", (PhD, Prof)) == CONTRADICTORY
    assert oracle.status("gci0", (TA, Student)) == UNDETERMINED
    # Without a non-emptiness assumption a TBox has no contradictions (Definition 1).
    strict = LogicalStatusOracle(university(), witnesses="none")
    assert strict.status("gci0", (PhD, Prof)) == UNDETERMINED


def test_abox_contradiction_without_witnesses():
    bob = nominal_name(NS + "bob")
    o = LogicalStatusOracle(university() + [NF1(bob, Prof)], witnesses="none")
    assert o.status("gci0", (Prof, Student)) == CONTRADICTORY
    assert o.status("gci0", (TA, Student)) == UNDETERMINED


def test_statuses_per_normal_form(oracle):
    names = LogicalStatusOracle(university(), witnesses="names")
    # GCI1: C ⊓ D ⊑ E
    assert oracle.status("gci1", (PhD, TA, Student)) == ENTAILED
    assert oracle.status("gci1", (Student, TA, Prof)) == CONTRADICTORY  # attested LHS
    assert names.status("gci1", (Student, TA, Prof)) == UNDETERMINED
    assert oracle.status("gci1", (Person, Course, TA)) == UNDETERMINED
    # GCI2: C ⊑ ∃r.D (TA ⊑ ∃teaches.Course would make the attested TA ⊓ Student empty)
    assert oracle.status("gci2", (PhD, enrolled, Course)) == ENTAILED
    assert oracle.status("gci2", (Student, teaches, Course)) == CONTRADICTORY
    assert oracle.status("gci2", (Course, teaches, Course)) == UNDETERMINED
    # GCI3: ∃r.C ⊑ D
    assert oracle.status("gci3", (teaches, Course, Staff)) == ENTAILED
    assert oracle.status("gci3", (teaches, Course, Student)) == CONTRADICTORY
    assert oracle.status("gci3", (teaches, Course, TA)) == UNDETERMINED
    # GCI1-bot: C ⊓ D ⊑ ⊥
    assert oracle.status("gci1_bot", (Student, Prof)) == ENTAILED
    assert oracle.status("gci1_bot", (PhD, Student)) == CONTRADICTORY
    assert oracle.status("gci1_bot", (TA, Student)) == CONTRADICTORY  # attested TA ⊓ Student
    assert names.status("gci1_bot", (TA, Student)) == UNDETERMINED
    assert oracle.status("gci1_bot", (Course, TA)) == UNDETERMINED


def test_generate_test_sets(oracle):
    sets = generate_test_sets(oracle, n_entailed=5, n_contradictory=5, n_undetermined=5,
                              seed=1)
    assert set(sets) == {"gci0", "gci1", "gci2", "gci3", "gci1_bot"}
    for nf, ts in sets.items():
        for label, axs in ((ENTAILED, ts.entailed), (CONTRADICTORY, ts.contradictory),
                           (UNDETERMINED, ts.undetermined)):
            assert len(axs) <= 5
            assert all(s == label for s in oracle.classify(nf, axs))
        assert not set(ts.entailed) & oracle.asserted.get(nf, set())
    assert sets["gci0"].entailed and sets["gci0"].contradictory and sets["gci0"].undetermined


# ---------------------------------------------------------------- metrics

def test_metric_definitions_hand_computed():
    assert discrimination_auc([3, 2], [1, 2]) == pytest.approx(0.875)
    assert discrimination_auc([], [1]) is None
    assert faithfulness_violation_rate([0, 2], [1, 3], [2, 4]) == pytest.approx(0.5)
    assert pair_fvr([0, 2], [1, 3]) == pytest.approx(0.25)
    assert pair_fvr([1], [1]) == 1.0
    assert pair_fvr([1], [1], strict=False) == 0.0
    assert spearman_rho([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1, 1, 1], [1, 2, 3]) is None
    assert discrimination_auc([1.0, np.nan], [0.0]) == 1.0  # nan = unscorable, dropped


def test_criteria_non_reversal_examples():
    """Appendix B constructions."""
    neg, und, pos = [0.0], [2.0], [1.0]  # f(a-) < f(a+) < f(a?)
    assert is_admissible(neg, und, pos) and not is_monotonic_faithful(neg, und, pos)
    neg, und, pos = [1.0], [0.0], [2.0]  # f(a?) < f(a-) < f(a+)
    assert is_discriminative(pos, neg) and not is_admissible(neg, und, pos)
    assert is_monotonic_faithful([0.0], [1.0], [2.0])
    assert is_probabilistic_faithful([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
    assert not is_probabilistic_faithful([0.5, 0.1, 0.9], [0.2, 0.4, 0.8])
    assert is_probabilistic_faithful([0.3, 0.1], [0.5, 0.5])  # silent on equal P


def test_ranking_and_admissibility():
    cands = ["a", "b", "c"]
    scores = {("q", "a"): 3.0, ("q", "b"): 2.0, ("q", "c"): 1.0}

    def scorer(nf, axs):
        return np.array([scores.get(tuple(a), np.nan) for a in axs])
    res = ranking_metrics(scorer, "gci0", [("q", "b")], cands, ks=(1, 2))
    assert res["mrr"] == pytest.approx(0.5) and res["hits@1"] == 0 and res["hits@2"] == 1
    assert admissibility_at_k(scorer, "gci0", [("q", "b")], cands, [("q", "a")], k=2) == 0.5


# ---------------------------------------------------------------- oracle / inverted scorer

def _sets_with_p(oracle):
    sets = generate_test_sets(oracle, n_entailed=6, n_contradictory=6, n_undetermined=6,
                              seed=0)
    counter = RelativeModelCounter(oracle.axioms, domain_size=2, backend="python")
    for nf, ts in sets.items():
        ts.p_values = {a: float(counter.relative_count(nf, a)) for a in ts.undetermined}
    return sets


def test_perfect_and_inverted_scorers(oracle):
    sets = _sets_with_p(oracle)
    p_all = {a: p for ts in sets.values() for a, p in ts.p_values.items()}
    perfect = oracle_scorer(oracle, p_all)
    res = evaluate_faithfulness(perfect, sets, oracle.classes, k=1, oracle=oracle)
    for nf, r in res.items():
        ts = sets[nf]
        if ts.entailed and ts.contradictory:
            assert r["auc_pm"] == 1.0 and r["discriminative"]
            assert r["adm@1"] == 1.0
        if ts.entailed and ts.contradictory and ts.undetermined:
            assert r["fvr"] == 0.0 and r["pair_fvr"] == 0.0
            assert r["admissible"] and r["monotonic_faithful"]
            assert r["probabilistic_faithful"]
        if len(set(ts.p_values.values())) > 1:
            assert r["rho_p"] == pytest.approx(1.0)

    def inverted(nf, axs):
        return -perfect(nf, axs)
    res = evaluate_faithfulness(inverted, sets, oracle.classes, k=1)
    for nf, r in res.items():
        ts = sets[nf]
        if ts.entailed and ts.contradictory:
            assert r["auc_pm"] == 0.0 and not r["discriminative"]
        if ts.entailed and ts.contradictory and ts.undetermined:
            assert r["fvr"] == 1.0 and r["pair_fvr"] == 1.0
            assert not r["admissible"] and not r["monotonic_faithful"]
            assert not r["probabilistic_faithful"]
        if len(set(ts.p_values.values())) > 1:
            assert r["rho_p"] == pytest.approx(-1.0)


def test_callable_scorer_forms():
    one = CallableScorer(lambda axs: [len(a[0]) for a in axs])
    two = CallableScorer(lambda nf, axs: [1.0 if nf == "gci0" else 0.0] * len(axs))
    loss = CallableScorer(lambda axs: [2.0] * len(axs), higher_is_better=False)
    per = CallableScorer({"gci0": lambda axs: [5.0] * len(axs)})
    assert list(one("gci0", [("ab", "c")])) == [2.0]
    assert list(two("gci1", [("a", "b", "c")])) == [0.0]
    assert list(loss("gci0", [("a", "b")])) == [-2.0]
    assert np.isnan(per("gci2", [("a", "r", "b")])).all()


# ---------------------------------------------------------------- regimes on a dataset

def _university_dataset():
    adapter = OWLAPIAdapter()
    ont = adapter.create_ontology(NS + "university")
    c = {n: adapter.create_class(n) for n in (PhD, Student, Prof, Person, TA, Course,
                                              Staff, Employee)}
    t, e = adapter.create_object_property(teaches), adapter.create_object_property(enrolled)
    nothing = adapter.data_factory.getOWLNothing()
    some = adapter.create_object_some_values_from
    axioms = [
        adapter.create_subclass_of(c[PhD], c[Student]),
        adapter.create_subclass_of(adapter.create_object_intersection_of(c[Student], c[Prof]),
                                   nothing),
        adapter.create_subclass_of(c[Student], c[Person]),
        adapter.create_subclass_of(c[Prof], c[Person]),
        adapter.create_subclass_of(c[TA], c[Person]),
        adapter.create_subclass_of(c[Student], some(e, c[Course])),
        adapter.create_subclass_of(c[Prof], some(t, c[Course])),
        adapter.create_subclass_of(some(t, c[Course]), c[Staff]),
    ]
    for ax in axioms:
        adapter.owl_manager.addAxiom(ont, ax)
    return Dataset(ont)


def test_faithfulness_regime_on_dataset():
    ds = _university_dataset()
    regime = FaithfulnessHierarchy(n_per_class=4, count_models=True, k=2)
    oracle = regime.oracle(ds)
    assert set(oracle.classes) == {PhD, Student, Prof, Person, TA, Course, Staff}
    res = regime.evaluate(oracle_scorer(oracle), ds)
    assert set(res) <= set(regime.subsets) and "gci0" in res
    for r in res.values():
        assert set(regime.metrics) <= set(r)
    assert res["gci0"]["auc_pm"] == 1.0


def test_properties_with_oracle_satisfaction(oracle):
    sets = generate_test_sets(oracle, n_entailed=5, n_contradictory=5, n_undetermined=5)
    cands = {nf: ts.entailed + ts.contradictory + ts.undetermined for nf, ts in sets.items()}

    def entailed_only(ax):
        from mowl.evaluation.faithfulness import from_normal_form_axiom
        parsed = from_normal_form_axiom(ax)
        return parsed is None or oracle.is_entailed(*parsed)
    rep = kb_embedding_properties(entailed_only, oracle, cands)
    assert rep["model"]["is_model"]
    for nf in cands:
        assert rep[nf]["entailment_closed"] and rep[nf]["strongly_faithful"]
        assert rep[nf]["weakly_faithful"]
    rep = kb_embedding_properties(lambda ax: True, oracle, cands)
    assert not rep["gci0"]["weakly_faithful"] and not rep["gci0"]["strongly_faithful"]


# ---------------------------------------------------------------- EL models

@pytest.fixture(scope="module")
def trained_elbe():
    from mowl.models.elbe.model import ELBE
    model = ELBE(_university_dataset(), embed_dim=4, margin=0.1)
    model.train(epochs=2)
    return model


def test_el_model_scorer_and_modelness(trained_elbe):
    model = trained_elbe
    oracle = LogicalStatusOracle.from_dataset(model.dataset)
    sets = generate_test_sets(oracle, n_entailed=3, n_contradictory=3, n_undetermined=3)
    res = evaluate_faithfulness(model, sets, oracle.classes)
    assert res["gci0"]["n_entailed"] == len(sets["gci0"].entailed)
    from mowl.evaluation.faithfulness import ELModelScorer
    scores = ELModelScorer(model)("gci0", [(PhD, Student), (NS + "Unknown", Student)])
    assert np.isfinite(scores[0]) and np.isnan(scores[1])

    old = model.module.margin
    rep = loss_modelness(model)
    assert model.module.margin == old and rep["margin_overridden"]
    assert 0.0 <= rep["all"]["fraction"] <= 1.0
    assert rep["gci0"]["n"] == len(model.training_datasets["gci0"].data)
    out = LossModelness().evaluate(model, model.dataset)
    assert out["all"]["n"] == rep["all"]["n"]


def test_test_set_p_values_ignore_conjunct_order():
    ts = FaithfulnessTestSet("gci1", [], [], [("b", "a", "c")], {("a", "b", "c"): 0.3})
    assert ts.p_value(("b", "a", "c")) == 0.3
