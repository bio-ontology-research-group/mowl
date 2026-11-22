"""The parts that the order-family, DF-ALC, Al-cone and E2R classes used to compute with
themselves: each is tested on its own, and used in a combination other than the method it came
from. That the methods give the numbers of their direct implementations is ``tests/baselines``."""
import math

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.evaluation.model_scoring import pair_costs  # noqa: E402
from mowl.evaluation.score_ranking import rank_by_score  # noqa: E402
from mowl.evaluation.subspace_violation import discretised_violations  # noqa: E402
from mowl.models import (ComposedALC, ComposedSubspace, DiskEmbeddings, E2R,  # noqa: E402
                         OrderEmbeddings, ShadowCones)
from mowl.representation.library.hierarchy_measures import (EnergyMeasure,  # noqa: E402
                                                            OrderViolation)
from mowl.representation.library.logic_fuzzy import (HalfAxisClasses, TruthMatrix,  # noqa: E402
                                                     TruthTableClasses, empty_cone_penalty)
from mowl.training import SumLoss  # noqa: E402
from mowl.training.taxonomy import (named_pair_rows, pair_codes,  # noqa: E402
                                    redraw_known_pairs)
from tests.baselines.pairs_logic import e2r_dataset  # noqa: E402
from tests.models.order_tree import iri, tree_dataset  # noqa: E402


# -- taxonomy rows and negatives ------------------------------------------------------------
def test_named_pair_rows_keeps_named_rows_sorted_and_unique():
    rows = th.tensor([[3, 1], [0, 5], [3, 1], [2, 2], [0, 1], [1, 0]])
    out = named_pair_rows(rows, 4)
    assert out.tolist() == [[0, 1], [1, 0], [2, 2], [3, 1]]
    assert named_pair_rows(rows, 4, distinct=True).tolist() == [[0, 1], [1, 0], [3, 1]]
    assert named_pair_rows(rows, 4, distinct=True, exclude_first=0).tolist() == [[1, 0], [3, 1]]


def test_pair_codes_are_the_sorted_unique_codes():
    rows = th.tensor([[1, 2], [0, 3], [1, 2]])
    assert pair_codes(rows, 5).tolist() == [3, 7]


def test_redraw_known_pairs_removes_told_and_reflexive_corruptions():
    np.random.seed(0)
    told = th.tensor([[0, 1], [1, 2], [2, 3]])
    known = pair_codes(told, 4)
    # two blocks of three corruptions: block 0 corrupted column 0, block 1 column 1
    negatives = th.tensor([[1, 2], [0, 1], [2, 2], [0, 1], [1, 2], [3, 3]])
    out = redraw_known_pairs(negatives.clone(), 3, [0, 1], known, 4, np.arange(4), 50)
    codes = (out[:, 0] * 4 + out[:, 1]).tolist()
    assert not set(codes) & set(known.tolist())
    assert (out[:, 0] != out[:, 1]).all()
    # a row that was fine is untouched
    assert redraw_known_pairs(th.tensor([[3, 0]]), 1, [1], known, 4, np.arange(4)).tolist() \
        == [[3, 0]]


# -- losses, measures, scoring ---------------------------------------------------------------
def test_sum_loss_positive_floor_takes_the_hinge_of_the_positives():
    pos, neg = th.tensor([-1.0, 0.3]), th.tensor([0.4, 0.0])
    plain = SumLoss().combine("gci0", pos, neg)
    floored = SumLoss(positive_floor=True).combine("gci0", pos, neg)
    assert plain.item() == pytest.approx(-0.35 + 0.2)
    assert floored.item() == pytest.approx(0.15 + 0.2)


def test_energy_measure_negative_of_is_the_hinge_or_the_energy():
    energy = th.tensor([-0.5, 0.2, 2.0])
    assert OrderViolation().negative_of(energy, 1.0).tolist() == pytest.approx([1.5, 0.8, 0.0])
    raw = OrderViolation(negative_form="energy")
    assert raw.negative_of(energy, 1.0).tolist() == pytest.approx(energy.tolist())
    assert issubclass(OrderViolation, EnergyMeasure)


def test_pair_costs_scores_iris_and_indices_without_gradient():
    model = OrderEmbeddings(tree_dataset(), embed_dim=4)
    names = [iri("A1"), iri("A")]
    by_iri = pair_costs(model, names, [iri("A"), iri("R")])
    index = model.class_index_dict
    by_index = pair_costs(model, [index[n] for n in names], [index[iri("A")], index[iri("R")]])
    assert by_iri.shape == (2,) and not by_iri.requires_grad
    assert th.allclose(by_iri, by_index)
    assert np.allclose(model.subsumption_costs(names, [iri("A"), iri("R")]), by_iri.numpy())


def test_rank_by_score_is_stable_and_ascending():
    assert rank_by_score(["a", "b", "c", "d"], [0.5, 0.1, 0.5, 0.0]) == ["d", "b", "a", "c"]


def test_empty_cone_penalty_is_zero_when_a_half_axis_reaches_the_margin():
    logits = th.tensor([[0.0, 0.3, -1.0], [0.0, 0.0, 0.0]])
    assert empty_cone_penalty(logits[:1], 0.2).item() == 0.0
    assert empty_cone_penalty(logits, 0.2).item() == pytest.approx(0.1)
    # the same function the Al-cone regulariser uses, on another class model
    classes = HalfAxisClasses(3, 2)
    assert math.isfinite(empty_cone_penalty(classes.centres(th.arange(3)), 0.1).item())


# -- fuzzy tables ----------------------------------------------------------------------------
def test_set_degrees_starts_a_truth_table_and_a_truth_matrix():
    classes = TruthTableClasses(3, 4)
    classes.set_degrees(1, [0.0, 0.5, 1.0, 0.25], eps=1e-3, name="C")
    degrees = th.sigmoid(classes.centres.weight[1]).tolist()
    assert degrees == pytest.approx([1e-3, 0.5, 1 - 1e-3, 0.25], abs=1e-6)
    with pytest.raises(ValueError, match="C"):
        classes.set_degrees(1, [0.1, 0.2], name="C")
    relations = TruthMatrix(2, 4)
    relations.set_degrees(0, np.full((4, 4), 0.9))
    assert relations.matrix(th.tensor([0]))[0].min().item() == pytest.approx(0.9, abs=1e-5)
    with pytest.raises(ValueError):
        relations.set_degrees(0, np.zeros((3, 3)))


# -- subspace violations ---------------------------------------------------------------------
def test_discretised_violations_decides_with_set_operations():
    d = 2   # coordinates: 0, 1 real; 2, 3 imaginary
    concepts = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])
    relations = np.array([[1, 0, 0, 1]])
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    rows = [("concept_inclusion", th.tensor([[0, 1], [1, 0]])),
            ("concept_disjointness", th.tensor([[0, 2], [0, 1]])),
            ("concept_negation", th.tensor([[2, 0]])),
            ("class_assertion", th.tensor([[0, 0], [1, 0]])),
            ("role_assertion", th.tensor([[0, 0, 1]])),
            ("range", th.tensor([[0, 2], [0, 0]]))]
    bad = discretised_violations(concepts, relations, vectors, rows, d)
    assert ("concept_inclusion", (1, 0)) in bad and ("concept_inclusion", (0, 1)) not in bad
    assert ("concept_disjointness", (0, 1)) in bad and ("concept_disjointness", (0, 2)) not in bad
    assert ("class_assertion", (1, 0)) in bad and ("class_assertion", (0, 0)) not in bad
    assert ("concept_negation", (2, 0)) not in bad
    # the pair vector (x_0, x_1 real halves) = (1, 0, 0, 1) lies in the span of axes 0 and 3
    assert ("role_assertion", (0, 0, 1)) not in bad
    # the range of relation 0 (second-half axis 3 = real axis 1) meets concept 2, not concept 0
    assert ("range", (0, 2)) in bad and ("range", (0, 0)) not in bad


# -- the method classes are the combinations --------------------------------------------------
def test_disk_embeddings_loss_is_the_sum_loss_regime():
    model = DiskEmbeddings(tree_dataset(), embed_dim=3, margin=0.5, seed=0)
    regime = model.training_regime
    assert isinstance(regime, SumLoss) and regime.positive_floor
    pos, neg = th.tensor([-1.0, 0.3]), th.tensor([0.4, 0.0])
    assert model.compute_loss(pos, neg).item() == pytest.approx(0.15 + 0.2)


def test_shadow_cones_and_disk_embeddings_share_the_taxonomy_rows():
    for cls in (ShadowCones, DiskEmbeddings):
        model = cls(tree_dataset(), embed_dim=3, seed=0)
        rows = model.training_datasets["gci0"].data
        assert len(rows) == 15 and (rows[:, 0] != rows[:, 1]).all()
        negatives = model.generate_negatives("gci0", model.training_datasets["gci0"])
        told = {tuple(r) for r in rows.tolist()}
        assert not told & {tuple(r) for r in negatives.tolist()}


def test_manifolds_are_general_parts_with_the_old_import_path_kept():
    from mowl.nn.order import manifolds as old
    from mowl.representation.library import manifolds as new
    assert old.PoincareBall is new.PoincareBall and old.RiemannianSGD is new.RiemannianSGD


def test_taxonomy_base_is_renamed_and_the_old_path_kept():
    from mowl.models.order_parts_a import TaxonomyEmbeddingModel as old
    from mowl.models.taxonomy_base import TaxonomyEmbeddingModel as new
    assert old is new


# -- E2R through the general model -------------------------------------------------------------
def test_e2r_is_the_combination_of_the_general_subspace_model():
    assert issubclass(E2R, ComposedSubspace)
    model = ComposedSubspace(e2r_dataset(), method="e2r", embed_dim=4, seed=1)
    named = E2R(e2r_dataset(), embed_dim=4, seed=1)
    assert sorted(model.module.parts) == sorted(named.module.parts)
    for m in (model, named):
        m.train(epochs=1)
        assert math.isfinite(m.total_loss())
        assert [type(r).__name__ for r in m.module.regularizers][:3] \
            == ["SubspaceStructure", "Orthogonality", "Separation"]


def test_subspace_model_swaps_a_measure_and_adds_a_regulariser():
    from mowl.representation.library.logic_subspace import ResidualMeasure as Residual
    from mowl.training import NormPenalty
    penalty = NormPenalty("individual_model.centres", 1.0, 2, None, 0.01)
    model = ComposedSubspace(e2r_dataset(), method="e2r", embed_dim=4,
                             measure=Residual(unit_length=True), regularizers=(penalty,))
    assert model.module.measure.unit_length
    model.train(epochs=1)
    assert math.isfinite(model.total_loss())
    names = model.rank_members(next(iter(model.class_index_dict)))
    assert sorted(names) == sorted(model.individual_index_dict)


def test_subspace_model_refuses_a_method_it_cannot_train():
    with pytest.raises((TypeError, ValueError)):
        ComposedSubspace(e2r_dataset(), method="dfalc", embed_dim=4)
    with pytest.raises(TypeError, match="ComposedSubspace"):
        ComposedALC(e2r_dataset(), method="e2r", embed_dim=4)
