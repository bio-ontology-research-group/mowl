"""DF-ALC (Wu et al., 2022): Gödel semantics, losses, revising a grounding, scoring."""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models.dfalc.model import DFALCModel
from mowl.nn.alc.dfalc.module import hierarchical_loss, rule_based_loss
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/dfalc#"
T = th.tensor


def _ontology(with_individuals=True):
    """A ⊑ B, B ⊓ C ⊑ ⊥, A ⊑ ∃r.B, ∃r.C ⊑ D; A(x), r(x, y), B(y) [optional]."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/dfalc/o")
    A, B, C, D = (ad.create_class(NS + n) for n in "ABCD")
    r = ad.create_object_property(NS + "r")
    axs = HashSet()
    axs.add(ad.create_subclass_of(A, B))
    axs.add(ad.create_disjoint_classes(B, C))
    axs.add(ad.create_subclass_of(A, ad.create_object_some_values_from(r, B)))
    axs.add(ad.create_subclass_of(ad.create_object_some_values_from(r, C), D))
    if with_individuals:
        x, y = ad.create_individual(NS + "x"), ad.create_individual(NS + "y")
        axs.add(ad.create_class_assertion(A, x))
        axs.add(ad.create_object_property_assertion(r, x, y))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def test_goedel_operators_follow_equations_1_to_5():
    ds = _ontology()
    m = DFALCModel(ds)
    mod = m.module
    c = T([[0.2, 0.9]])
    r = T([[[0.1, 0.8], [0.6, 0.3]]])
    assert th.allclose(mod._logical_and(c, T([[0.5, 0.5]])), T([[0.2, 0.5]]))
    assert th.allclose(mod._logical_or(c, T([[0.5, 0.5]])), T([[0.5, 0.9]]))
    assert th.allclose(mod._logical_not(c), T([[0.8, 0.1]]))
    # (∃r.C)(a) = max_b min(r(a,b), C(b)); (∀r.C)(a) = min_b max(1 - r(a,b), C(b))
    assert th.allclose(mod._logical_exist(r, c), T([[0.8, 0.3]]))
    assert th.allclose(mod._logical_forall(r, c), T([[0.9, 0.4]]))


def test_hierarchical_loss_is_zero_iff_pointwise_inclusion():
    c = T([[0.2, 0.5]])
    assert hierarchical_loss(c, T([[0.2, 0.6]])).item() == 0
    assert hierarchical_loss(c, T([[0.1, 0.6]])).item() == pytest.approx(0.05)
    # rule-based loss: zero when included, pushes only the superclass (detached G)
    d = T([[0.1, 0.6]], requires_grad=True)
    cc = T([[0.2, 0.5]], requires_grad=True)
    rule_based_loss(cc, d).sum().backward()
    assert cc.grad is None and d.grad[0, 0] < 0
    assert rule_based_loss(c, T([[0.2, 0.6]])).item() == 0


def test_domain_is_individuals_and_needs_a_domain():
    m = DFALCModel(_ontology(), anon_domain=2)
    assert m.ndomain == 4 and len(m.domain) == 4
    with pytest.raises(ValueError):
        DFALCModel(_ontology(with_individuals=False))
    assert DFALCModel(_ontology(with_individuals=False), anon_domain=3).ndomain == 3


@pytest.mark.parametrize("loss_type", ["hierarchical", "rule"])
def test_training_revises_a_grounding_into_a_model(loss_type):
    th.manual_seed(0)
    m = DFALCModel(_ontology(), anon_domain=2, loss_type=loss_type, learning_rate=0.1)
    before = m.total_loss()
    m.train(epochs=300)
    after = m.total_loss()
    assert after < before and after < 0.05
    g = m.concept_grounding()
    x = m.individual_index_dict[NS + "x"]
    assert g[NS + "A"][x] > 0.9                       # assertion A(x) = 1
    if loss_type == "hierarchical":                   # A ⊑ B pointwise (up to the loss)
        assert (g[NS + "A"] <= g[NS + "B"] + 0.1).all()
    assert m.score_instance(NS + "x", NS + "A") > 0.9


def test_perceptual_grounding_is_kept_where_consistent():
    """Start from a grounding that is already a model: loss is ~0 and training keeps it."""
    m = DFALCModel(_ontology(), anon_domain=0)
    cid = m.class_index_dict
    # domain order: individuals of the dataset
    n = m.ndomain
    ones, zeros = np.ones(n), np.zeros(n)
    rmat = np.zeros((n, n))
    x, y = m.individual_index_dict[NS + "x"], m.individual_index_dict[NS + "y"]
    rmat[x, y] = 1.0
    b = np.zeros(n); b[x] = 1; b[y] = 1
    a = np.zeros(n); a[x] = 1
    m.set_grounding(concepts={NS + "A": a, NS + "B": b, NS + "C": zeros, NS + "D": zeros},
                    roles={NS + "r": rmat})
    assert m.total_loss() < 0.02
    assert m.violations(tolerance=0.05) == []
    # the grounding is readable back
    assert m.concept_grounding()[NS + "A"][x] == pytest.approx(1.0, abs=0.01)
    assert m.role_grounding()[NS + "r"][x, y] == pytest.approx(1.0, abs=0.01)


def test_unsatisfied_grounding_is_reported_and_costs_follow_inclusion():
    m = DFALCModel(_ontology())
    n = m.ndomain
    x = m.individual_index_dict[NS + "x"]
    a = np.zeros(n); a[x] = 1.0
    m.set_grounding(concepts={NS + "A": a, NS + "B": np.zeros(n)})   # A ⊑ B violated
    assert m.violations(tolerance=0.1)
    assert m.score_subsumption(NS + "A", NS + "B") < -0.2
    assert m.score_subsumption(NS + "B", NS + "A") > -0.01
    rows = T([[m.class_index_dict[NS + "A"], m.class_index_dict[NS + "B"]],
              [m.class_index_dict[NS + "B"], m.class_index_dict[NS + "A"]]])
    costs = m.axiom_costs("gci0", rows)
    assert costs[0] > costs[1]
    # existential forms score too
    r = m.object_property_index_dict[NS + "r"]
    c2 = T([[m.class_index_dict[NS + "A"], r, m.class_index_dict[NS + "B"]]])
    assert m.axiom_costs("gci2", c2).shape == (1,)
    c3 = T([[r, m.class_index_dict[NS + "B"], m.class_index_dict[NS + "A"]]])
    assert m.axiom_costs("gci3", c3).shape == (1,)


def test_declared_scoring():
    assert DFALCModel.axiom_scoring == "method"
