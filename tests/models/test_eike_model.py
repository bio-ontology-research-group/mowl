"""EIKE (Wang et al., 2024): scores of Section 4.2, ranking loss of Section 4.3, training."""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models.eike.model import EIKE, name_to_text
from mowl.nn.el.eike.module import EIKEModule
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/eike#"
T = th.tensor


def _dataset():
    """Cat ⊑ Animal ⊑ Thing2, Dog ⊑ Animal; Cat(tom), Dog(rex), likes(tom, rex)."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/eike/o")
    cls = {n: ad.create_class(NS + n) for n in ["Cat", "Dog", "Animal", "Thing2", "Plant"]}
    likes = ad.create_object_property(NS + "likes")
    axs = HashSet()
    for a, b in [("Cat", "Animal"), ("Dog", "Animal"), ("Animal", "Thing2")]:
        axs.add(ad.create_subclass_of(cls[a], cls[b]))
    axs.add(ad.create_subclass_of(cls["Plant"], cls["Thing2"]))
    tom, rex = ad.create_individual(NS + "tom"), ad.create_individual(NS + "rex")
    axs.add(ad.create_class_assertion(cls["Cat"], tom))
    axs.add(ad.create_class_assertion(cls["Dog"], rex))
    axs.add(ad.create_object_property_assertion(likes, tom, rex))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def _module(**kw):
    m = EIKEModule(3, 1, 2, embed_dim=2, **kw)
    with th.no_grad():
        m.class_center.weight.copy_(T([[0., 0.], [0.1, 0.], [5., 5.]]))
        # semi-axes b = 2, radius r = 1 for every concept
        m.class_axes_raw.weight.fill_(float(np.log(np.expm1(2.0))))
        m.class_radius_raw.weight.fill_(float(np.log(np.expm1(1.0))))
        m.class_intensional.weight.copy_(T([[1., 0.], [2., 0.], [0., 1.]]))
        m.ind_embed.weight.copy_(T([[1.0, 0.0], [0.0, 3.0]]))
    return m


def test_instance_score_is_zero_inside_the_ellipsoid_and_positive_outside():
    m = _module(alpha=0.0)
    inside = m.class_assertion_loss(T([[0, 0]]))       # |(1,0)/2|^2 = 0.25 <= 1
    outside = m.class_assertion_loss(T([[1, 0]]))      # |(0,3)/2|^2 = 2.25 > 1
    assert inside.item() == pytest.approx(0.0, abs=1e-5)
    assert outside.item() == pytest.approx(1.25, abs=1e-4)
    # the intensional part: 1 - cos(M i, c); the identity map for EYE
    m2 = _module(alpha=1.0)
    f = m2.class_assertion_loss(T([[0, 2]]))           # i = (1, 0), c_int = (0, 1): cos 0
    assert f.item() == pytest.approx(1.0 + 0.0 + (((T([1., 0.]) - T([5., 5.])) / 2) ** 2).sum().item() - 1.0, abs=1e-3)


def test_subclass_score_extensional_and_intensional_parts():
    m = _module(alpha=0.0)
    # same axes, r_i = r_j: the printed formula is the squared normalised centre distance
    assert m.gci0_loss(T([[0, 0]])).item() == pytest.approx(0.0, abs=1e-5)
    assert m.gci0_loss(T([[0, 1]])).item() == pytest.approx((0.1 / 2) ** 2, abs=1e-5)
    m2 = _module(alpha=1.0)
    # intensional: cat(0, 1) = 1; norms: |(1,0)| - |(2,0)| = -1  => 0 + (-1)... ex 0.0025
    f01 = m2.gci0_loss(T([[0, 1]])).item()
    f10 = m2.gci0_loss(T([[1, 0]])).item()
    assert f01 == pytest.approx(0.0025 + (1 - 1.0) + (1.0 - 2.0), abs=1e-4)
    # the paper: the superclass has the longer vector, so the sub -> super order scores lower
    assert f01 < f10


def test_relational_triple_is_transe():
    m = _module()
    with th.no_grad():
        m.rel_embed.weight.copy_(T([[-1.0, 3.0]]))
    # tom (1,0) + r = (0, 3) = rex
    assert m.object_property_assertion_loss(T([[0, 0, 1]])).item() == pytest.approx(0.0, abs=1e-6)
    assert m.object_property_assertion_loss(T([[1, 0, 0]])).item() > 1.0


def test_negative_scores_carry_the_margin_and_the_ranking_loss_is_a_hinge():
    m = _module(margins=(0.5, 2.0, 0.3), alpha=0.0)
    pos = m.gci0_loss(T([[0, 1]]))
    neg = m.gci0_loss(T([[0, 1]]), neg=True)
    assert (pos - neg).item() == pytest.approx(0.5, abs=1e-6)     # gamma_sub
    assert (m.class_assertion_loss(T([[0, 0]])) - m.class_assertion_loss(T([[0, 0]]), neg=True)
            ).item() == pytest.approx(2.0, abs=1e-6)               # gamma_ins
    model = EIKE(_dataset(), embed_dim=4)
    # [γ + f(pos) - f(neg)]_+ : f(pos) = 0.5, f(neg) = 2 with γ = 1  ->  0 ; f(neg) = 1 -> 0.5
    loss = model.compute_loss(T([0.5]), T([2.0 - 1.0]))
    assert loss.item() == 0.0
    assert model.compute_loss(T([0.5]), T([1.0 - 1.0])).item() == pytest.approx(0.5)
    # two negatives per positive: blocks follow the corrupted columns
    assert model.compute_loss(T([0.5, 0.5]), T([0.0, 0.0, 9.0, 9.0])).item() == \
        pytest.approx(0.25)


def test_intensional_vectors_are_an_optional_input():
    ds = _dataset()
    vec = {NS + "Cat": np.ones(4), NS + "Dog": 2 * np.ones(4)}
    model = EIKE(ds, embed_dim=4, intensional_vectors=vec)
    w = model.module.class_intensional.weight
    assert th.allclose(w[model.class_index_dict[NS + "Cat"]], th.ones(4))
    assert th.allclose(w[model.class_index_dict[NS + "Dog"]], 2 * th.ones(4))
    other = w[model.class_index_dict[NS + "Plant"]]
    assert not th.allclose(other, th.ones(4)) and not th.isnan(other).any()
    with pytest.raises(ValueError):
        EIKE(ds, embed_dim=5, intensional_vectors=vec)
    # no pre-trained model is needed: a callable encoder is the caller's business
    texts = model.class_texts()
    assert texts[NS + "Cat"] == "Cat"
    enc = EIKE.encode_class_texts(texts, lambda xs: np.array([[len(x)] * 4 for x in xs]))
    assert enc[NS + "Animal"][0] == len("Animal")


def test_name_to_text():
    assert name_to_text("<http://yago/wikicat_Danish_male_film_actors>") == \
        "wikicat Danish male film actors"
    assert name_to_text("http://ex#Heart_Valve") == "Heart Valve"


def test_training_uses_only_eike_forms_and_learns_the_hierarchy():
    th.manual_seed(0)
    np.random.seed(0)
    model = EIKE(_dataset(), embed_dim=8, alpha=0.5, learning_rate=0.05)
    assert set(model.training_datasets) <= {"gci0", "class_assertion",
                                            "object_property_assertion"}
    assert {"gci0", "class_assertion", "object_property_assertion"} <= \
        set(model.training_datasets)
    cfg = model.get_negative_sampling_config()
    assert cfg["gci0"]["corrupt_column"] == [0, 1]
    ci, ii = model.class_index_dict, model.individual_index_dict
    pos = T([[ci[NS + "Cat"], ci[NS + "Animal"]]])
    neg = T([[ci[NS + "Cat"], ci[NS + "Plant"]]])
    gap0 = (model.module.gci0_loss(neg) - model.module.gci0_loss(pos)).item()
    model.train(epochs=300)
    gap = (model.module.gci0_loss(neg) - model.module.gci0_loss(pos)).item()
    assert gap > gap0 and gap > 0
    inst = T([[ii[NS + "tom"], ci[NS + "Cat"]]])
    wrong = T([[ii[NS + "tom"], ci[NS + "Plant"]]])
    assert model.module.class_assertion_loss(inst).item() < \
        model.module.class_assertion_loss(wrong).item()


def test_axiom_scores_for_the_joint_benchmarks():
    from mowl.evaluation.model_scoring import axiom_scoring_of, resolve_model_scorer
    model = EIKE(_dataset(), embed_dim=4)
    assert axiom_scoring_of(model) == "module"
    assert model.scored_forms == ("gci0", "class_assertion", "object_property_assertion")
    scorer, _ = resolve_model_scorer(model)
    rows = T([[model.class_index_dict[NS + "Cat"], model.class_index_dict[NS + "Animal"]]])
    assert scorer(model, "gci0", rows).shape == (1,)
