"""BoxSEL (Zhu et al., 2024): volume-ratio losses (Eqs. 4-5), probabilities (Theorem 2), the
paper's Berkeley example (Figure 1, Table 1) and the probabilistic modus ponens."""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models.boxsel.model import BoxSEL, Conditional
from mowl.baselines.nn.el.boxsel import losses as S
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/boxsel#"
T = th.tensor
APP, DEPT, ADM = NS + "Applicant", NS + "DeptA", NS + "Admitted"


def _dataset():
    """Admitted ⊑ Applicant, DeptA ⊑ Applicant, ∃applies.Applicant ⊑ Person (for ∃)."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/boxsel/o")
    c = {n: ad.create_class(NS + n) for n in ["Applicant", "DeptA", "Admitted", "Person"]}
    applies = ad.create_object_property(NS + "applies")
    axs = HashSet()
    axs.add(ad.create_subclass_of(c["Admitted"], c["Applicant"]))
    axs.add(ad.create_subclass_of(c["DeptA"], c["Applicant"]))
    axs.add(ad.create_subclass_of(ad.create_object_some_values_from(applies, c["Applicant"]),
                                  c["Person"]))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def box(lo, hi):
    return T([lo], dtype=th.float), T([hi], dtype=th.float)


def test_volume_ratio_and_conditional_loss_follow_equations_3_to_5():
    c = box([0., 0.], [4., 10.])          # Vol 40
    d = box([0., 0.], [4., 2.])           # Vol(D ∩ C) = 8 -> p = 0.2
    assert S.conditional_probability(c, d).item() == pytest.approx(0.2)
    # satisfied for p in [l, u]: zero loss (temperature large: softplus ~ relu)
    t = 1e4
    assert S.conditional_loss(c, d, 0.2, 0.25, t).item() == pytest.approx(0.0, abs=1e-2)
    assert S.conditional_loss(c, d, 0.1, 0.2, t).item() == pytest.approx(0.0, abs=1e-2)
    # lower bound violated: l Vol(C) - Vol(D∩C) = 0.5*40 - 8 = 12
    assert S.conditional_loss(c, d, 0.5, 0.9, t).item() == pytest.approx(12.0, rel=1e-3)
    # upper bound violated: Vol(D∩C) - u Vol(C) = 8 - 0.1*40 = 4
    assert S.conditional_loss(c, d, 0.0, 0.1, t).item() == pytest.approx(4.0, rel=1e-3)
    # a premise without volume gives an undefined proportion
    assert th.isnan(S.conditional_probability(box([0., 0.], [0., 1.]), d)).all()


def test_conditionals_are_compiled_with_and_exists_and_top_is_refused():
    m = BoxSEL(_dataset(), embed_dim=3, conditionals=[
        (DEPT, APP, 0.2), (ADM, ("and", APP, DEPT), 0.8, 0.8),
        (NS + "Person", ("some", NS + "applies", APP), 0.5, 0.6),
        (("some", NS + "applies", DEPT), APP, 0.1, 0.3)])
    groups = m._compiled_groups()
    assert len(groups) == 4                       # PNF1, PNF2, PNF3, PNF4 patterns
    assert m.training_regime.additional_loss(0).ndim == 0   # the regime ConditionalStatements
    for c in m.conditionals:
        p = m.probability(c.conclusion, c.premise)
        assert 0.0 <= p <= 1.0
    with pytest.raises(ValueError):
        BoxSEL(_dataset(), embed_dim=3, conditionals=[(DEPT, "http://www.w3.org/2002/07/owl#Thing", 0.5)]
               ).training_regime.additional_loss(0)
    with pytest.raises(ValueError):
        BoxSEL(_dataset(), embed_dim=3, conditionals=[(DEPT, APP, 0.7, 0.2)])
    with pytest.raises(KeyError):
        BoxSEL(_dataset(), embed_dim=3, conditionals=[(DEPT, NS + "Nope", 0.5)]
               ).training_regime.additional_loss(0)


def test_existential_region_is_the_preimage_of_the_filler_box():
    """∃r.D = (T^r)^{-1}(Box(D)): a point a is in it iff T^r(a) is in D."""
    m = BoxSEL(_dataset(), embed_dim=2)
    mod = m.module
    r = m.object_property_index_dict[NS + "applies"]
    d = m.class_index_dict[APP]
    with th.no_grad():
        mod.relation_embedding.weight[r] = T([1.0, -1.0])
        mod.scaling_embedding.weight[r] = T([2.0, 0.5])
    lo, hi = S.expression_box(("some", ("atom",)), T([[r, d]]), 0, mod.min_embedding,
                              mod.delta_embedding, mod.relation_embedding,
                              mod.scaling_embedding)[:2]
    dlo = mod.min_embedding.weight[d]
    dhi = dlo + th.exp(mod.delta_embedding.weight[d])
    for corner, bound in ((lo, dlo), (hi, dhi)):
        image = corner[0] * T([2.0, 0.5]) + T([1.0, -1.0])      # T^r of a corner of ∃r.D
        assert th.allclose(image, bound, atol=1e-4)


def test_probabilistic_modus_ponens_bounds_hold_for_exact_embeddings():
    """Proposition 2 / Section 4: for the two embeddings of Figure 1 the volumes of Table 1
    give p(Admitted ⊓ DeptA | Applicant) in [0.16, 0.2] = [l1 l2, u1 u2]."""
    left = {"app": 40, "dept": 8, "adm": 17, "adm_dept": 6.4}
    right = {"app": 60, "dept": 15, "adm": 50, "adm_dept": 12}
    ps = [v["adm_dept"] / v["app"] for v in (left, right)]
    assert (min(ps), max(ps)) == (pytest.approx(0.16), pytest.approx(0.2))
    l1, u1, l2, u2 = 0.2, 0.25, 0.8, 0.8
    assert l1 * l2 == pytest.approx(0.16) and u1 * u2 == pytest.approx(0.2)
    # the premises hold in both embeddings
    for v in (left, right):
        assert 0.2 <= v["dept"] / v["app"] <= 0.25
        assert v["adm_dept"] / v["dept"] == pytest.approx(0.8)


def test_training_embeds_the_berkeley_example_and_zero_loss_satisfies_conditionals():
    th.manual_seed(0)
    conds = [Conditional(ADM, DEPT, 0.8, 0.8), Conditional(DEPT, APP, 0.2, 0.25),
             Conditional(ADM, APP, 0.2, 0.9)]
    m = BoxSEL(_dataset(), conditionals=conds, embed_dim=4, learning_rate=0.05, beta=1.0,
               temperature=10.0, temperature_final=1000.0)
    before = m.mean_absolute_error()
    m.train(epochs=600)
    after = m.mean_absolute_error()
    assert after < before
    # deterministic axioms of the TBox (BoxEL losses): DeptA ⊑ Applicant
    assert m.probability(APP, DEPT) > 0.95
    # the conditionals are satisfied within a small tolerance after training
    for c, e in zip(m.conditionals, m.conditional_errors()):
        assert e < 0.06, (c, e)
    # two embeddings give an interval estimate that contains each point estimate
    lo, hi = BoxSEL.probability_interval([m, m], DEPT, APP)
    assert lo == hi and 0.14 <= lo <= 0.31


def test_declared_scoring_is_that_of_boxel():
    m = BoxSEL(_dataset(), embed_dim=2)
    assert m.axiom_scoring == "module"
    rows = T([[m.class_index_dict[ADM], m.class_index_dict[APP]]])
    assert m.module(rows, "gci0").shape == (1, 1)


def test_volume_regulariser_is_eq_10_of_the_paper():
    """L_vol = sum_i [beta^n - Vol(C_i) - eps]_+: zero for a box that fills [0, beta]^n."""
    m = BoxSEL(_dataset(), embed_dim=2, beta=2.0)
    n_classes = m.module.min_embedding.weight.shape[0]
    with th.no_grad():
        m.module.delta_embedding.weight.fill_(float(np.log(2.0)))       # side 2 = beta
    assert S.volume_loss(m.module.min_embedding, m.module.delta_embedding, 2.0).item() \
        == pytest.approx(0.0, abs=1e-6)
    with th.no_grad():
        m.module.delta_embedding.weight.fill_(0.0)                      # side 1: Vol 1 < 4
    assert S.volume_loss(m.module.min_embedding, m.module.delta_embedding, 2.0).item() \
        == pytest.approx(3.0, abs=1e-5)
    assert n_classes > 0
    m.module.volume_weight = 1.0
    assert m.module.regularization_loss().item() >= 3.0 - 1e-5
