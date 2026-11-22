"""The order and box embeddings of hierarchies are combinations of parts (group
``recomp-order-a``: order_embeddings, cate, box_lattice, gumbel_box, bcbox).

Each method is read back as the combination it names (a); swapping one part (another
measure, class model, objective option, regulariser, training regime, evaluation regime)
still trains for an epoch and evaluates to finite numbers (b); the catalogue card and the
parts agree (c). The comparison with the direct implementations is in
``tests/baselines/pairs_order_a.py``.
"""
import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.models as models  # noqa: E402
from mowl.catalog.registry import get_method  # noqa: E402
from mowl.evaluation.filtered_ranking import cate_ranking  # noqa: E402
from mowl.evaluation.normal_form_ranking import SubsumeeRankingRegime  # noqa: E402
from mowl.models import ComposedEL  # noqa: E402
from mowl.nn import (BinaryCodeBoxModule, BoxLatticeModule, CatEModule,  # noqa: E402
                     GumbelBoxModule, OrderEmbeddingsModule)
from mowl.representation import (ComposedGeometry, Geometry, Objective,  # noqa: E402
                                 REPRESENTATIONS)
from mowl.representation import methods as M  # noqa: E402
from mowl.representation.library import hierarchy_classes as HC  # noqa: E402
from mowl.representation.library import hierarchy_measures as HMe  # noqa: E402
from mowl.representation.library.hierarchy_geometry import LatticeGeometry  # noqa: E402
from mowl.representation.library.hierarchy_methods import (  # noqa: E402
    CodePenalty, MaxCornerPenalty)
from mowl.training import TransitiveClosurePositives  # noqa: E402
from mowl.training.regularization import NormPenalty  # noqa: E402
from tests.benchmarks.tiny import scratch_dir, tiny_dataset  # noqa: E402

KEYS = ("order_embeddings", "cate", "box_lattice", "gumbel_box", "bcbox")


@pytest.fixture(scope="module")
def dataset():
    return tiny_dataset()


def build(key, dataset, **options):
    th.manual_seed(0)
    np.random.seed(0)
    path = scratch_dir() + "/m.pt"
    cls = {"order_embeddings": models.OrderEmbeddings, "cate": models.CatE,
           "box_lattice": models.BoxLattice, "gumbel_box": models.GumbelBox,
           "bcbox": models.BinaryCodeBox}[key]
    options.setdefault("embed_dim", 4)
    options.setdefault("learning_rate", 0.05)
    return cls(dataset, model_filepath=path, **options)


def train_one_epoch(key, model):
    if key in ("cate", "bcbox"):
        history = model.train(epochs=1)
        assert len(history) == 1 and np.isfinite(history[0])
    else:
        model.train(epochs=1, validate_every=10 ** 6)


def all_pairs_rows(n):
    sub, sup = th.meshgrid(th.arange(n), th.arange(n), indexing="ij")
    return th.stack([sub.reshape(-1), sup.reshape(-1)], dim=1)


def scores_of(key, model):
    """What the evaluation regimes rank, for every pair of the first classes."""
    with th.no_grad():
        model.module.eval()
        if key == "bcbox":
            return model.energy_matrix()[:6, :6]
        n = 6
        rows = all_pairs_rows(n)
        return model.module(rows, "gci0").reshape(-1)


def swap(model, classes=None, measure=None, objective=None, regularizers=()):
    """Replace parts of the combination of a model (they are attributes of its geometry)."""
    g = model.module
    if classes is not None:
        g.class_model = classes
        g.class_embed = getattr(classes, "centres", None)
    if measure is not None:
        g.measure = measure
    if objective is not None:
        g.objective = objective
    g.regularizers = tuple(g.regularizers) + tuple(regularizers)
    return model


# -- (a) the named combination, read back ------------------------------------------------------
EXPECTED = {
    "order_embeddings": dict(module=OrderEmbeddingsModule, classes=HC.OrderPoints,
                             measure=HMe.OrderViolation,
                             parts={"classes": "order cone", "relations": "none",
                                    "individuals": "none", "measure": "order violation",
                                    "conjunction": "none"}),
    "cate": dict(module=CatEModule, classes=HC.OrderPoints, measure=HMe.OrderViolation,
                 parts={"classes": "order cone", "relations": "none",
                        "individuals": "region", "measure": "order violation",
                        "conjunction": "own embedding"}),
    "box_lattice": dict(module=BoxLatticeModule, classes=HC.UnitCubeBoxes,
                        measure=HMe.LogVolumeRatio,
                        parts={"classes": "box", "relations": "none", "individuals": "none",
                               "measure": "volume ratio", "conjunction": "exact"}),
    "gumbel_box": dict(module=GumbelBoxModule, classes=HC.UnitCubeBoxes,
                       measure=HMe.LogVolumeRatio,
                       parts={"classes": "soft box", "relations": "none",
                              "individuals": "none", "measure": "smoothed volume ratio",
                              "conjunction": "exact"}),
    "bcbox": dict(module=BinaryCodeBoxModule, classes=HC.CodedBoxes,
                  measure=HMe.LogVolumeRatio,
                  parts={"classes": "soft box", "relations": "none", "individuals": "none",
                         "measure": "smoothed volume ratio", "conjunction": "exact"}),
}


@pytest.mark.parametrize("key", KEYS)
def test_the_model_is_the_named_combination(key, dataset):
    expected = EXPECTED[key]
    model = build(key, dataset)
    module = model.module
    assert isinstance(module, expected["module"])
    assert isinstance(module, Geometry) and isinstance(module, ComposedGeometry)
    assert module.parts == expected["parts"]
    assert type(module.class_model) is expected["classes"]
    assert type(module.measure) is expected["measure"]
    assert module.relation_model is None
    # the builder that defines the method is the one place that names the combination
    assert key in M.METHODS and M.METHOD_KINDS[key] == "axioms"
    from_builder = M.METHODS[key](10, 0, 0, embed_dim=4)
    assert from_builder.parts == module.parts
    assert type(from_builder.class_model) is type(module.class_model)
    assert type(from_builder.measure) is type(module.measure)
    assert from_builder.objective.gci0_negative == module.objective.gci0_negative == "negative"
    assert module.supported_forms == frozenset({"gci0"})
    assert isinstance(module, LatticeGeometry) == (key == "cate")


def test_the_options_of_the_named_classes_are_parts():
    cases = {
        "box_lattice": (dict(max_regularization=0.005), MaxCornerPenalty),
        "gumbel_box": (dict(max_regularization=0.005), MaxCornerPenalty),
        "bcbox": (dict(code_weight=0.5), CodePenalty),
    }
    for key, (options, regularizer) in cases.items():
        g = M.METHODS[key](6, 0, 0, embed_dim=3, **options)
        assert [type(r) for r in g.regularizers] == [regularizer], key
        assert g.regularization_loss() > 0
    assert M.METHODS["order_embeddings"](6, margin=0.3).objective.margin == 0.3
    hard = M.METHODS["box_lattice"](6, 0, 0, embed_dim=3, zero_at_inclusion=False)
    assert hard.measure.zero_at_inclusion is False
    assert M.METHODS["gumbel_box"](6, 0, 0, embed_dim=3, log_guard="epsilon").measure.tail \
        == "floor"
    assert M.METHODS["bcbox"](6, 0, 0, embed_dim=3, variant="gbc").class_model.variant == "gbc"


@pytest.mark.parametrize("key", ["order_embeddings", "box_lattice", "gumbel_box", "bcbox"])
def test_composed_el_builds_the_method_by_name(key, dataset):
    model = ComposedEL(dataset, method=key, embed_dim=4, learning_rate=0.05,
                       model_filepath=scratch_dir() + "/c.pt")
    assert model.module.parts == EXPECTED[key]["parts"]
    model.train(epochs=1, validate_every=10 ** 6)
    assert set(model.training_datasets) == {"gci0"}
    assert th.isfinite(model.module(model.training_datasets["gci0"][:], "gci0")).all()


# -- (b) swaps ---------------------------------------------------------------------------------
def check(key, model):
    train_one_epoch(key, model)
    scores = scores_of(key, model)
    assert th.isfinite(scores[th.isfinite(scores)]).all() and th.isfinite(scores).sum() > 0
    return scores


def ranking(model, dataset, regime):
    out = regime.evaluate(model, dataset)["all"]
    assert all(np.isfinite(v) for v in out.values() if isinstance(v, (int, float))), out
    return out


def test_order_embeddings_swaps(dataset):
    # another measure: the L1 norm of the violation, in the product order of CatE
    model = swap(build("order_embeddings", dataset), measure=HMe.OrderViolation("reversed", 1))
    check("order_embeddings", model)
    # another class model (free vectors) with the measure of CatE and a smaller margin
    model = swap(build("order_embeddings", dataset),
                 classes=HC.OrderPoints(len(build("order_embeddings", dataset)
                                                .class_index_dict), 4, False, "xavier"),
                 measure=HMe.OrderViolation("product", 2), objective=Objective.hierarchy(0.5))
    check("order_embeddings", model)
    # an added regulariser
    model = swap(build("order_embeddings", dataset), regularizers=[NormPenalty(weight=0.1)])
    check("order_embeddings", model)
    assert model.module.regularization_loss() > 0
    # another training regime
    model = build("order_embeddings", dataset)
    model.training_regime = TransitiveClosurePositives()
    check("order_embeddings", model)
    # another evaluation regime
    ranking(model, dataset, cate_ranking("gci0", candidates="classes"))
    ranking(model, dataset, SubsumeeRankingRegime())


def test_box_lattice_swaps(dataset):
    # the measure of the Gumbel boxes on the same boxes
    model = swap(build("box_lattice", dataset),
                 measure=HMe.LogVolumeRatio(0.05, 0.05, gumbel_shift=True, tail="linear"))
    check("box_lattice", model)
    # the objective without frozen sizes in negative pairs, with the max-corner regulariser
    model = swap(build("box_lattice", dataset),
                 objective=Objective.hierarchy(freeze_size_negatives=False),
                 regularizers=[MaxCornerPenalty(0.005)])
    check("box_lattice", model)
    assert model.module.regularization_loss() > 0
    # the class model of BCBox (boxes with codes) under the hard measure is refused as a
    # mismatch only if the regions do not fit; boxes with codes are boxes, so it fits
    model = swap(build("box_lattice", dataset),
                 classes=HC.CodedBoxes(len(build("box_lattice", dataset).class_index_dict),
                                       4, "gbox"),
                 measure=HMe.LogVolumeRatio(0.01, 1.0, gumbel_shift=False, tail="linear"))
    check("box_lattice", model)
    model = build("box_lattice", dataset)
    model.training_regime = TransitiveClosurePositives()
    check("box_lattice", model)
    ranking(model, dataset, cate_ranking("gci0", candidates="classes"))


def test_gumbel_box_swaps(dataset):
    # the hard measure of the box lattice on the same boxes
    model = swap(build("gumbel_box", dataset), measure=HMe.LogVolumeRatio())
    check("gumbel_box", model)
    # the earlier guards of the logarithms, as an option of the measure
    model = swap(build("gumbel_box", dataset),
                 measure=HMe.LogVolumeRatio(0.01, 0.01, gumbel_shift=True, tail="floor",
                                            energy_floor=1e-9))
    check("gumbel_box", model)
    # the corner penalty and a training regime
    model = swap(build("gumbel_box", dataset), regularizers=[MaxCornerPenalty(0.01)])
    model.training_regime = TransitiveClosurePositives()
    check("gumbel_box", model)
    assert model.module.regularization_loss() > 0
    ranking(model, dataset, SubsumeeRankingRegime())
    ranking(model, dataset, cate_ranking("gci0", candidates="classes"))


def test_bcbox_swaps(dataset):
    # another variant of the class model (global codes), with the code penalty
    model = build("bcbox", dataset, variant="gbc", num_codes=3, code_reg=0.1)
    assert isinstance(model.module.regularizers[0], CodePenalty)
    check("bcbox", model)
    # the measure of the Gumbel box module (volume temperature as its scale, no codes read)
    model = swap(build("bcbox", dataset, variant="gbox"),
                 measure=HMe.LogVolumeRatio(0.05, 0.05, gumbel_shift=True, tail="floor",
                                            energy_floor=1e-9, dtype=None))
    check("bcbox", model)
    # another objective option: plain sum of the negatives instead of self-adversarial weights
    model = build("bcbox", dataset, self_adversarial=False)
    check("bcbox", model)
    # another evaluation: the best F1 of the reconstruction, from the same energies
    from mowl.models.bcbox.model import best_f1
    energies = model.energy_matrix()
    adjacency = th.zeros_like(energies, dtype=th.bool)
    adjacency[model.edge_index[:, 1], model.edge_index[:, 0]] = True
    f1, _ = best_f1(energies, adjacency)
    assert 0.0 <= f1 <= 1.0


def test_cate_swaps(dataset):
    # another measure: the L1 order violation
    model = swap(build("cate", dataset), measure=HMe.OrderViolation("product", 1))
    check("cate", model)
    # another objective option: the hinge on the mean of the corruptions (the reference code)
    model = build("cate", dataset, negative_aggregation="mean", filter_negatives=False)
    check("cate", model)
    # another class model: non-negative vectors
    model = build("cate", dataset)
    swap(model, classes=HC.OrderPoints(len(model.lattice), 4, True, "uniform"))
    check("cate", model)
    # an added regulariser
    model = swap(build("cate", dataset), regularizers=[NormPenalty(weight=0.1)])
    model.module.class_embed = model.module.class_model.centres
    check("cate", model)
    assert model.module.regularization_loss() > 0
    # another training regime: the design of the reference code
    from mowl.training import CateCodeTraining
    model = build("cate", dataset)
    model.training_regime = CateCodeTraining()
    check("cate", model)
    # another evaluation regime
    ranking(model, dataset, cate_ranking("gci0", candidates="classes"))
    ranking(model, dataset, cate_ranking("gci0", candidates="classes", closure=True))


def test_the_parts_are_exchangeable_between_the_methods(dataset):
    """A part of one method in another: the order violation reads the vectors of CatE, a
    box measure reads the boxes of BCBox."""
    cate = build("cate", dataset)
    oe = build("order_embeddings", dataset)
    assert type(cate.module.measure) is type(oe.module.measure)
    assert cate.module.measure.direction == "product" and oe.module.measure.direction == "reversed"
    gumbel, bc = build("gumbel_box", dataset), build("bcbox", dataset)
    assert type(gumbel.module.measure) is type(bc.module.measure)
    assert gumbel.module.measure.smoothed and bc.module.measure.smoothed


# -- (c) the catalogue card and the parts agree --------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_the_card_and_the_parts_agree(key, dataset):
    card, rep = get_method(key), REPRESENTATIONS[key]
    module = build(key, dataset).module
    parts = module.parts
    assert card.implementation.endswith(
        {"order_embeddings": "OrderEmbeddings", "cate": "CatE", "box_lattice": "BoxLattice",
         "gumbel_box": "GumbelBox", "bcbox": "BinaryCodeBox"}[key])
    assert card.module.endswith(type(module).__name__)
    assert parts["classes"] in {c.value for c in rep.classes}
    assert parts["measure"] in {m.value for m in rep.measure}
    assert [r.value for r in rep.relations] == ["none"] and parts["relations"] == "none"
    assert "subsumption" in card.tasks
    if key == "cate":
        # the lattice carries the logic: compound classes are elements of their own
        assert parts["conjunction"] == rep.conjunction.value == "own embedding"
        assert card.logic == "ALC" and "SubClassOf" in card.axioms
    else:
        assert card.expressivity == "taxonomy"
        assert tuple(card.axioms) == ("gci0",) == tuple(sorted(module.supported_forms))
        assert rep.conjunction.value == "none"      # the method does not read conjunctions
        assert "gci1" not in module.supported_forms
