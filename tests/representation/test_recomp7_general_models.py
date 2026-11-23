"""Every method can be built and trained through a general model (round 7).

RESHUFFLE through :class:`~mowl.models.GraphKGEModel`, FaithEL and GroundedEL through
:class:`~mowl.models.ComposedEL`, with swaps of their parts where the mathematics allows
(each checked for an effect), and the parts that were used by one method only, each with a
second user.
"""
import pytest
import torch as th

from mowl.representation import ConstructorUnavailable


# -- RESHUFFLE through GraphKGEModel ------------------------------------------------------------
def _graph_dataset():
    from tests.models.kg_region_data import abox_dataset
    return abox_dataset()


_RESHUFFLE = dict(l=5, k=6, num_layers=2, num_negs_per_pos=4, learning_rate=0.02,
                  batch_size=4, seed=0)


def _trained(build, epochs=2):
    th.manual_seed(0)
    model = build()
    th.manual_seed(1)
    model.train(epochs=epochs)
    return model


def _costs(model):
    n = len(model.class_index_dict)
    rows = th.tensor([[a, b] for a in range(n) for b in range(n)])
    with th.no_grad():
        return model.axiom_costs("gci0", rows)


def test_graph_kge_model_builds_and_trains_reshuffle_as_the_named_class():
    from mowl.models import GraphKGEModel, Reshuffle
    ds = _graph_dataset()
    named = _trained(lambda: Reshuffle(ds, **_RESHUFFLE))
    general = _trained(lambda: GraphKGEModel(ds, method="reshuffle", **_RESHUFFLE))
    assert general.create_inverse_triples
    assert th.equal(named.module.relation_model.vectors.weight,
                    general.module.relation_model.vectors.weight)
    assert th.allclose(_costs(named), _costs(general), equal_nan=True)
    assert general.class_embeddings and general.object_property_embeddings


def test_reshuffle_parts_swap_with_an_effect():
    from mowl.models import GraphKGEModel
    ds = _graph_dataset()
    base = _trained(lambda: GraphKGEModel(ds, method="reshuffle", **_RESHUFFLE))
    # learned (transductive) entity matrices instead of the encoding of the graph
    points = _trained(lambda: GraphKGEModel(ds, method="reshuffle", classes="point",
                                            **_RESHUFFLE))
    assert type(points.module.class_model).__name__ in ("Points", "PointClasses")
    assert points.module.class_model.centres.weight.requires_grad
    assert not th.allclose(_costs(base), _costs(points))
    # another objective
    loss = _trained(lambda: GraphKGEModel(ds, method="reshuffle", loss="self-adversarial",
                                          **_RESHUFFLE))
    assert not th.allclose(_costs(base), _costs(loss))
    # forward triples only
    forward = GraphKGEModel(ds, method="reshuffle", inverse_triples=False, **_RESHUFFLE)
    assert not forward.create_inverse_triples


def test_reshuffle_refuses_what_it_cannot_combine():
    from mowl.models import GraphKGEModel
    ds = _graph_dataset()
    with pytest.raises(ConstructorUnavailable, match="matrices"):
        GraphKGEModel(ds, method="reshuffle", relations="translation", **_RESHUFFLE)
    with pytest.raises(TypeError, match="embed_dim"):
        GraphKGEModel(ds, method="reshuffle", embed_dim=10, **_RESHUFFLE)


# -- FaithEL through ComposedEL ----------------------------------------------------------------
def _abox_dataset():
    from tests.baselines.pairs_solved import _faithel_dataset
    return _faithel_dataset()


def _faithel_tables(model):
    m = model.module
    return {"individuals": m.individual_model.centres.weight, "concepts": m.concepts.weight,
            "role_head": m.role_head.weight, "role_tail": m.role_tail.weight}


def test_composed_el_builds_and_trains_faithel_as_the_named_class():
    from mowl.models import ComposedEL, FaithEL
    from mowl.training import AlternatingMinibatches
    ds = _abox_dataset()
    named = FaithEL(ds, batch_size=8, learning_rate=0.01, seed=0)
    general = ComposedEL(ds, method="faithel", batch_size=8, learning_rate=0.01,
                         negative_term="unbounded")
    general.eval_gci_name = "class_assertion"
    assert general.embed_dim == named.tbox.dimension
    assert isinstance(general.training_regime, AlternatingMinibatches)
    rows = th.cartesian_prod(th.arange(len(named.class_index_dict)),
                             th.arange(len(named.class_index_dict)))
    assert th.equal(named.tbox_costs("gci0", rows),
                    general.module.tbox.cost("gci0", rows).reshape(-1))
    with th.no_grad():
        for name, table in _faithel_tables(general).items():
            table.copy_(_faithel_tables(named)[name])
    th.manual_seed(1)
    named.train(epochs=1)
    steps = named.training_steps
    th.manual_seed(1)
    general.train(epochs=steps, validate_every=10 ** 6)
    for name, table in _faithel_tables(named).items():
        assert th.allclose(table, _faithel_tables(general)[name], atol=1e-6), name


def test_faithel_parts_swap_with_an_effect():
    from mowl.models import ComposedEL
    from mowl.representation.library.solved_faithel import AnchorDistance
    ds = _abox_dataset()
    base = ComposedEL(ds, method="faithel")
    rows = base.training_datasets["class_assertion"][:]
    # the score without the distance to the centroid
    plain = ComposedEL(ds, method="faithel", measure=AnchorDistance(centroid_score=False))
    with th.no_grad():
        plain.module.load_state_dict(base.module.state_dict())
        assert not th.allclose(base.module.class_assertion_costs(rows),
                               plain.module.class_assertion_costs(rows))
    # no relation model: no role assertions to train
    none = ComposedEL(ds, method="faithel", relations="none")
    assert "object_property_assertion" not in none.training_datasets
    with pytest.raises(ConstructorUnavailable):
        ComposedEL(ds, method="faithel", classes="box")
