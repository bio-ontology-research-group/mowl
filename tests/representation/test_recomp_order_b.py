"""Hyperbolic cones, shadow cones, disk embeddings, HiT and OnT are combinations of parts.

(a) the named combination can be read back from the model: which class model, measure,
relation model and objective make up each method; (b) each method trains for an epoch and is
evaluated with a part swapped (another measure, another objective, another manifold or
encoder, a regulariser, another training regime, another evaluation regime) and every number is
finite; (c) the catalogue card and the parts agree. That the combinations give the numbers of the
direct implementations is ``tests/baselines`` (``pairs_order_b.py``).
"""
import math

import pytest
import torch as th

from mowl.datasets.builtin import FamilyDataset
from mowl.models import (ComposedEL, DiskEmbeddings, HiT, HyperbolicCones, OnT,
                         ShadowCones)
from mowl.nn.lm.encoders import BagOfWordsEncoder
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.representation import (METHODS, ComposedGeometry, Geometry, Objective, geometry_of)
from mowl.representation.library import hierarchy_methods as HM
from mowl.representation.library.hierarchy_classes import (
    ConePoints, DiskClasses, PointClasses, TextPoints)
from mowl.representation.library.hierarchy_geometry import TextGeometry
from mowl.representation.library.hierarchy_measures import (
    ConeAngle, DistanceDepth, Protrusion, ShadowCone)
from mowl.representation.library.hierarchy_regions import CurvedBall, Point
from mowl.representation.library.hierarchy_relations import TextRotation
from mowl.training import NormPenalty, PerNormalFormMiniBatches, SumLoss

D = 6
LEARNING = dict(learning_rate=0.05)
TEXT = dict(batch_size=8, warmup_steps=0, learning_rate=0.05, seed=0, centri_weight=0.5)


@pytest.fixture(scope="module")
def family():
    return FamilyDataset()


@pytest.fixture(scope="module")
def tiny():
    """Training, validation and test ontologies (the evaluation regimes need validation axioms)."""
    from tests.benchmarks.tiny import tiny_dataset
    return tiny_dataset()


def encoder(dim=8):
    return BagOfWordsEncoder(dim=dim, num_buckets=64)


def order_models(dataset):
    return {
        "hyperbolic_cones": HyperbolicCones(dataset, embed_dim=D, num_neg=2, seed=0),
        "shadow_cones": ShadowCones(dataset, embed_dim=D, num_neg=2, seed=0),
        "disk_embeddings": DiskEmbeddings(dataset, embed_dim=D, num_neg=2, seed=0)}


def text_models(dataset):
    return {"hit": HiT(dataset, encoder=encoder(), num_negatives=2, **TEXT),
            "ont": OnT(dataset, encoder=encoder(), **TEXT)}


def all_pairs_rows(model):
    n = len(model.named_class_index_dict)
    rows = th.cartesian_prod(th.arange(n), th.arange(n))
    return rows[rows[:, 0] != rows[:, 1]]


def finite_scores(model):
    """The energies of every pair of named classes, and the loss of every training form."""
    rows = all_pairs_rows(model)
    with th.no_grad():
        energies = model.module(rows, "gci0")
        negatives = model.module(rows, "gci0", neg=True)
    assert energies.shape[0] == len(rows)
    assert bool(th.isfinite(energies).all()) and bool(th.isfinite(negatives).all())
    return energies


def finite_text_scores(model):
    n = len(model.class_index_dict)
    rows = th.cartesian_prod(th.arange(n), th.arange(n))
    assert bool(th.isfinite(model.axiom_costs("gci0", rows)).all())
    scores = model.score_candidates("gci0", rows[:6], 1, th.arange(n))
    assert bool(th.isfinite(scores).all())


def train_one_epoch(model):
    model.train(epochs=1) if not isinstance(model, HiT) else model.train(epochs=1,
                                                                         validate=False)


# -- (a) the combination can be read back -----------------------------------------------
def test_the_named_combination_of_every_method_is_registered_and_readable(family):
    for key in ("hyperbolic_cones", "shadow_cones", "disk_embeddings", "hit", "ont"):
        assert key in METHODS
    cones = HM.hyperbolic_cones(10, embed_dim=3)
    assert isinstance(cones, ComposedGeometry)
    assert (type(cones.class_model), type(cones.measure)) == (ConePoints, ConeAngle)
    assert cones.relation_model is None


@pytest.mark.parametrize("key,model", list(order_models(FamilyDataset()).items()))
def test_the_order_models_are_made_of_these_parts(key, model):
    geometry = model.module
    assert isinstance(geometry, Geometry) and isinstance(geometry, ComposedGeometry)
    assert geometry.relation_model is None
    assert geometry.parts["relations"] == "none"
    assert geometry.supported_forms == frozenset({"gci0"})
    assert isinstance(geometry.objective, Objective)
    assert geometry.objective.negatives == "gci0"
    expected = {"hyperbolic_cones": (ConePoints, "geodesic cone", ConeAngle, "angle"),
                "shadow_cones": (PointClasses, "point", ShadowCone, "angle"),
                "disk_embeddings": (DiskClasses, "disk", Protrusion, "protrusion")}[key]
    assert type(geometry.class_model) is expected[0]
    assert geometry.parts["classes"] == expected[1]
    assert type(geometry.measure) is expected[2]
    assert geometry.parts["measure"] == expected[3]
    # the module a user gets from mowl.nn is the same combination
    assert type(geometry).__name__ == {"hyperbolic_cones": "HyperbolicConesModule",
                                       "shadow_cones": "ShadowConesModule",
                                       "disk_embeddings": "DiskEmbeddingsModule"}[key]
    assert geometry_of(model) is geometry


def test_the_options_of_the_named_classes_select_parts(family):
    for formulation, space in (("umbral-half-space", UpperHalfSpace),
                               ("penumbral-half-space", UpperHalfSpace),
                               ("umbral-ball", PoincareBall),
                               ("penumbral-ball", PoincareBall)):
        g = ShadowCones(family, embed_dim=D, formulation=formulation).module
        assert g.measure.formulation == formulation
        assert isinstance(g.class_model.manifold, space)
        assert g.measure.signed and g.measure.negative_form == "hinge"
    energy = ShadowCones(family, embed_dim=D, negative="energy").module
    assert energy.measure.negative_form == "energy"
    for geometry, space in (("hyperbolic", Hyperboloid), ("spherical", Sphere),
                            ("euclidean", Euclidean)):
        g = DiskEmbeddings(family, embed_dim=D, geometry=geometry, margin=0.2).module
        assert type(g.class_model.manifold) is space
        assert g.objective.margin == pytest.approx(0.2)
    cones = HyperbolicCones(family, embed_dim=D, K=0.2, margin=0.3).module
    assert cones.class_model.K == pytest.approx(0.2) and cones.objective.margin == 0.3
    assert cones.measure.apex == "formula"      # the published formula, as an option of the method


def test_hit_and_ont_are_made_of_these_parts(family):
    hit, ont = text_models(family)["hit"].module, text_models(family)["ont"].module
    for g in (hit, ont):
        assert isinstance(g, TextGeometry) and isinstance(g, ComposedGeometry)
        assert type(g.class_model) is TextPoints
        assert type(g.class_model.manifold) is CurvedBall
        assert type(g.measure) is DistanceDepth
        assert g.parts["measure"] == "distance and depth"
        assert g.supported_forms == frozenset({"gci0"})
    assert hit.relation_model is None and hit.parts["relations"] == "none"
    assert type(ont.relation_model) is TextRotation and ont.parts["relations"] == "rotation"
    # the encoder is a part of the class model: its parameters are the model's parameters
    names = {id(p) for p in hit.class_model.encoder.parameters()}
    assert names and names <= {id(p) for p in hit.parameters()}
    assert OnT(family, encoder=encoder(), use_roles=False, **TEXT).module.relation_model is None


# -- (b) swaps -------------------------------------------------------------------------
def swap_model(model, geometry):
    """Put another combination of parts into a named model (its builder's arguments)."""
    model.module = geometry.to(model.device)
    return model


@pytest.mark.parametrize("measure", ["angle (inside)", "distance and depth"])
def test_cones_with_another_measure(family, measure):
    model = HyperbolicCones(family, embed_dim=D, num_neg=2, seed=0, **LEARNING)
    n = len(model.class_index_dict)
    chosen = ConeAngle(apex="inside") if measure.startswith("angle") else DistanceDepth(0.5)
    swap_model(model, HM.hyperbolic_cones(n, embed_dim=D, measure=chosen, margin=0.1, seed=0))
    assert model.module.measure is chosen
    train_one_epoch(model)
    finite_scores(model)


def test_cones_with_a_regulariser_and_a_training_regime(family):
    th.manual_seed(0)
    model = ComposedEL(family, method="hyperbolic_cones", embed_dim=D, margin=0.1,
                       regularizers=(NormPenalty(parameters="class_embed", target=0.5),),
                       learning_rate=0.05)
    assert len(model.module.regularizers) == 1
    assert float(model.module.regularization_loss()) >= 0.0
    model.training_regime = PerNormalFormMiniBatches(batch_size=3)
    model.train(epochs=1)
    assert set(model.training_datasets) == {"gci0"}
    for form, data in model.training_datasets.items():
        assert bool(th.isfinite(model.module(data[:], form)).all())


def test_shadow_cones_with_another_measure_and_objective(family):
    model = ShadowCones(family, embed_dim=D, num_neg=2, seed=0, **LEARNING)
    n = len(model.class_index_dict)
    for measure, objective in ((ShadowCone("umbral-half-space", 0.05, signed=False),
                                Objective.hierarchy(0.1)),
                               (DistanceDepth(0.0), Objective.hierarchy())):
        th.manual_seed(0)
        swap_model(model, HM.shadow_cones(n, embed_dim=D, measure=measure, objective=objective,
                                          seed=0))
        assert isinstance(model.module.class_model.manifold, UpperHalfSpace)
        train_one_epoch(model)
        finite_scores(model)


def test_shadow_cones_on_the_ball_with_the_softmax_loss_regime(family):
    th.manual_seed(0)
    from mowl.training import SoftmaxEnergyContrastiveLoss
    model = ShadowCones(family, embed_dim=D, formulation="umbral-ball", num_neg=2, seed=0,
                        **LEARNING)
    model.training_regime = SoftmaxEnergyContrastiveLoss(gamma2=0.01, normal_forms=("gci0",))
    train_one_epoch(model)
    finite_scores(model)


def test_shadow_cones_with_a_regulariser(family):
    th.manual_seed(0)
    model = ComposedEL(family, method="shadow_cones", embed_dim=D, margin=0.05,
                       formulation="penumbral-half-space", init_scale=0.3,
                       regularizers=(NormPenalty(parameters="class_embed", target=1.0),),
                       learning_rate=0.01)
    model.train(epochs=1)
    assert bool(th.isfinite(model.module.regularization_loss()))


@pytest.mark.parametrize("geometry", ["hyperbolic", "spherical", "euclidean"])
def test_disks_in_every_space_with_another_objective(family, geometry):
    th.manual_seed(0)
    model = DiskEmbeddings(family, embed_dim=D, geometry=geometry, num_neg=2, seed=0,
                           **LEARNING)
    n = len(model.class_index_dict)
    swap_model(model, HM.disk_embeddings(n, embed_dim=D, geometry=geometry, seed=0,
                                         objective=Objective.hierarchy(0.5)))
    assert model.module.objective.margin == 0.5
    train_one_epoch(model)
    finite_scores(model)


def test_disks_with_the_cone_angle_measure_on_points_and_a_sum_loss(family):
    """A measure of another method on the class model of this one: the protrusion of disks
    replaced by the signed distance of the shadow cone cannot read disks, so the pair of parts
    is refused with a message that says which part does not fit."""
    model = DiskEmbeddings(family, embed_dim=D, seed=0)
    n = len(model.class_index_dict)
    with pytest.raises(TypeError, match="compares"):
        swap_model(model, HM.disk_embeddings(n, embed_dim=D, measure=ConeAngle()))
    model = DiskEmbeddings(family, embed_dim=D, num_neg=2, seed=0, **LEARNING)
    model.training_regime = SumLoss(neg_weight=0.5)
    train_one_epoch(model)
    finite_scores(model)


def test_the_order_models_are_evaluated_with_another_regime(tiny):
    from mowl.evaluation import InferredSubsumptionRankingRegime, NormalFormRankingRegime
    for key, model in order_models(tiny).items():
        th.manual_seed(0)
        train_one_epoch(model)
        for regime in (NormalFormRankingRegime(normal_forms=("gci0",), split="valid"),
                       InferredSubsumptionRankingRegime(split="valid")):
            result = regime.evaluate(model, tiny)
            assert result
            for subset in result.values():
                assert all(math.isfinite(v) for v in subset.values()), (key, subset)


def test_hit_with_another_encoder_manifold_and_measure(family):
    th.manual_seed(0)
    model = HiT(family, encoder=encoder(), num_negatives=2, **TEXT)
    train_one_epoch(model)
    finite_text_scores(model)
    # another encoder (another size: the encoder is a part of the class model)
    other = HiT(family, encoder=encoder(dim=12), num_negatives=2, **TEXT)
    assert other.module.class_model.embed_dim == 12
    train_one_epoch(other)
    finite_text_scores(other)
    # another curvature of the ball and another weight of the depth in the measure
    other = HiT(family, encoder=encoder(), curvature=0.5, num_negatives=2, **TEXT)
    assert other.module.class_model.manifold.c == 0.5
    train_one_epoch(other)
    finite_text_scores(other)
    # another measure: distance only
    swapped = swap_model(HiT(family, encoder=encoder(), num_negatives=2, **TEXT),
                         HM.hit(encoder=encoder(), measure=DistanceDepth(0.0)))
    assert swapped.module.measure.centri_weight == 0.0
    train_one_epoch(swapped)
    finite_text_scores(swapped)


def test_hit_with_a_regulariser_and_another_objective(family):
    from mowl.training.regularization import Regularizer

    class EncoderNorm(Regularizer):
        kind = "penalty"

        def __init__(self):
            super().__init__(parameters="class_embed", weight=1e-3)

        def penalty(self, model):
            return self.weight * sum((p ** 2).sum() for p in model.class_model.parameters())

    model = HiT(family, encoder=encoder(), num_negatives=2, **TEXT)
    swap_model(model, HM.hit(encoder=model.module.class_model.encoder,
                             regularizers=(EncoderNorm(),), objective=Objective.hierarchy()))
    assert float(model.module.regularization_loss()) > 0.0
    train_one_epoch(model)
    finite_text_scores(model)


def test_ont_swaps(family):
    th.manual_seed(0)
    # the ablation without the relation model, and roles read as the rotated filler
    for options in ({"use_roles": False}, {"existential_mode": "role", "role_scaling": True},
                    {"role_pairing": "adjacent"}):
        model = OnT(family, encoder=encoder(), **TEXT, **options)
        train_one_epoch(model)
        finite_text_scores(model)
    # another measure on the same combination
    model = OnT(family, encoder=encoder(), **TEXT)
    swap_model(model, HM.ont(encoder=model.module.class_model.encoder,
                             role_model=model.module.role_model,
                             role_texts=None, measure=DistanceDepth(0.0)))
    train_one_epoch(model)
    finite_text_scores(model)


def test_hit_and_ont_with_another_training_and_evaluation_regime(tiny):
    from mowl.evaluation import OnTRankingRegime
    from mowl.training.lm_reference import HiTCodeTraining, OnTCodeTraining
    th.manual_seed(0)
    for key, model in text_models(tiny).items():
        # the named design of the code (an LM model implements it itself and consults no other
        # hooks), and a ranking regime that reads the measure's scores
        model.training_regime = {"hit": HiTCodeTraining, "ont": OnTCodeTraining}[key]()
        train_one_epoch(model)
        result = OnTRankingRegime(normal_forms=("gci0",), split="valid",
                                  centri_weight=0.5).evaluate(model, tiny)
        assert result
        for subset in result.values():
            assert all(math.isfinite(v) for v in subset.values()), (key, subset)


# -- (c) the catalogue agrees with the parts --------------------------------------------
def _geometry_of_key(key):
    return {"hyperbolic_cones": lambda: HM.hyperbolic_cones(8, embed_dim=3),
            "shadow_cones": lambda: HM.shadow_cones(8, embed_dim=3),
            "disk_embeddings": lambda: HM.disk_embeddings(8, embed_dim=3),
            "hit": lambda: HM.hit(encoder=encoder()),
            "ont": lambda: HM.ont(encoder=encoder(), role_model=_role_model())}[key]()


@pytest.mark.parametrize("key", ["hyperbolic_cones", "shadow_cones", "disk_embeddings", "hit",
                                 "ont"])
def test_the_catalogue_card_and_the_parts_agree(key):
    from mowl.catalog.registry import get_method
    from mowl.representation import REPRESENTATIONS
    card, representation = get_method(key), REPRESENTATIONS[key]
    geometry = _geometry_of_key(key)
    for part in geometry.parts.values():
        assert representation.has(part), (key, part)
    # the card states the axiom forms the model trains on and the tasks it scores; the geometry
    # reads the subclass order between named classes, which is the task of the card
    assert "gci0" in card.axioms and "gci0" in geometry.supported_forms
    assert card.tasks == ("subsumption",)
    # the card's relation model is the geometry's: none, except for the rotation of OnT
    assert (geometry.parts["relations"] == "rotation") == (key == "ont")
    # the classes the card names are the module's and the model's
    assert card.implementation.startswith("mowl.models.") and card.module.startswith("mowl.nn.")


def test_the_card_of_ont_states_the_forms_the_model_scores(family):
    from mowl.catalog.registry import get_method
    model = OnT(family, encoder=encoder(), **TEXT)
    assert set(model.scored_forms) <= set(get_method("ont").axioms)
    assert set(HiT(family, encoder=encoder(), **TEXT).scored_forms) <= set(
        get_method("hit").axioms)


def _role_model():
    from mowl.representation.library.hierarchy_relations import RoleTransformation
    return RoleTransformation(8)
