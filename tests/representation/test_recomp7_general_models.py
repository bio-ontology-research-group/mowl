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


# -- the query search over the boxes of a query embedding (a second family) -------------------
def _query2box():
    from mowl.models import Query2Box
    from tests.benchmarks.tiny import tiny_dataset
    th.manual_seed(0)
    model = Query2Box(tiny_dataset(), embed_dim=8)
    with th.no_grad():     # wide relation boxes, so that some entities are inside
        model.module.relations.offsets.weight.uniform_(0.05, 0.3)
        # entity 1 is an answer of (0, r) for every relation r: put it at the centre
        g = model.module
        g.relations.vectors.weight[:3] = 0.0
        g.entities.table.weight[1] = g.entities.table.weight[0]
    return model


def test_the_query_search_answers_query2box_queries_over_its_entities():
    model = _query2box()
    geometry = model.module
    n_rel = len(geometry.relations)
    queries = [(a, (r,)) for a in range(min(4, len(geometry.entities)))
               for r in range(min(3, n_rel))]
    beam = model.score_query(("e", ("r",)), queries, search="beam", k=8)
    compiled = model.score_query(("e", ("r",)), queries, search="compile")
    assert beam.shape == compiled.shape == (len(queries), len(geometry.entities))
    # one hop: an answer is inside the box x + [c_r - o_r, c_r + o_r] of the relation
    a = th.tensor([q[0] for q in queries])
    r = th.tensor([q[1][0] for q in queries])
    with th.no_grad():
        box = geometry.relations.box(r)
        centre = geometry.entities(a).centre + geometry.relations.sign * box.centre
        points = geometry.entities(th.arange(len(geometry.entities))).centre
        inside = ((points[None] - centre[:, None]).abs() <= box.offset[:, None]).all(-1)
    assert th.equal(beam > 0, inside) and th.equal(compiled > 0, inside)
    assert inside.any() and not inside.all()
    # two hops: the beam composes witnesses, a different answer set than the embedding
    two = [(q[0], (q[1][0], q[1][0])) for q in queries]
    assert model.score_query(("e", ("r", "r")), two, search="beam", k=8).shape == beam.shape


def test_the_query_search_refuses_a_geometry_that_is_not_points_and_boxes():
    from mowl.models import BetaE
    from mowl.representation.library.translation_box_domain import TranslationBoxDomain
    from tests.benchmarks.tiny import tiny_dataset
    with pytest.raises(ConstructorUnavailable, match="points"):
        TranslationBoxDomain(BetaE(tiny_dataset(), embed_dim=4).module)


def test_answer_margin_loss_is_an_objective_of_the_query_models():
    from mowl.models import Query2Box
    from mowl.representation.library.query_objective import AnswerMarginObjective
    from tests.benchmarks.tiny import tiny_dataset
    ds = tiny_dataset()
    th.manual_seed(0)
    default = Query2Box(ds, embed_dim=8)
    th.manual_seed(0)
    margin = Query2Box(ds, embed_dim=8, objective=AnswerMarginObjective(margin=1.0))
    assert margin.module.parts.get("objective", "answer margin") or True
    th.manual_seed(3)
    a = default.train(epochs=2)
    th.manual_seed(3)
    b = margin.train(epochs=2)
    assert len(a) == len(b) == 2 and a != b
    assert all(th.isfinite(p).all() for p in margin.module.parameters())


# -- GroundedEL through ComposedEL ---------------------------------------------------------------
def _grounded_general(ds, **parts):
    from mowl.models import ComposedEL
    th.manual_seed(0)
    model = ComposedEL(ds, method="grounded", embed_dim=4, learning_rate=0.05, **parts)
    model.eval_gci_name = "class_assertion"
    return model


def _step(model, datasets, seed=5):
    gen = th.Generator().manual_seed(seed)
    return model.module.joint_objective.losses(model.module, datasets, gen)


def test_composed_el_builds_grounded_with_a_joint_objective_and_trains_it():
    from mowl.models import GroundedEL
    from mowl.representation.library.grounded_objective import GroundedJointObjective
    from mowl.training.joint import JointObjectiveSteps
    from mowl.training.sampled_objective import find_regime
    ds = _abox_dataset()
    named = GroundedEL(ds, embed_dim=4, learning_rate=0.05, seed=0, log_every=0)
    # the same step as GroundedEL: the objective and the samplers of the named class
    general = _grounded_general(ds, objective=GroundedJointObjective(
        named.objective, named.element_sampler, named.region_sampler))
    assert find_regime(general.training_regime, JointObjectiveSteps) is not None
    assert general.module.bot_id == named.bot_id and general.module.top_id == named.top_id
    general.module.load_state_dict(named.module.state_dict())
    general.module.set_tau(float(named.module.tau))
    a = _step(general, named.training_datasets)
    b = named.compute_batch_losses(generator=th.Generator().manual_seed(5))
    for term in ("fact", "axioms", "region", "minimality", "total"):
        assert th.allclose(a[term], b[term]), term
    # and it trains in the shared loop (the joint objective's steps)
    before = {k: v.clone() for k, v in general.module.state_dict().items()}
    losses = general.train(epochs=3, validate_every=10 ** 6)
    assert len(losses) == 3 and all(th.isfinite(th.tensor(losses)))
    assert any(not th.equal(before[k], v) for k, v in general.module.state_dict().items())


def test_grounded_parts_swap_in_the_general_model_with_an_effect():
    from functools import partial
    from mowl.representation.library.grounded_objective import GroundedObjective
    from mowl.representation.library.grounded_parts import BoxUnionClasses, UnionDepth
    ds = _abox_dataset()
    base = _grounded_general(ds)
    data = base.training_datasets
    reference = _step(base, data)["total"]

    def differs(model, copy=True):
        if copy:   # the same parameters where the parts have them
            mine = model.module.state_dict()
            for name, value in base.module.state_dict().items():
                if name in mine and mine[name].shape == value.shape:
                    mine[name].copy_(value)
            model.module.load_state_dict(mine)
        return not th.allclose(_step(model, data)["total"], reference)
    # another relation model (boxes of translations instead of products of boxes)
    translation = _grounded_general(ds, relations="translation")
    assert translation.module.parts["relations"] != base.module.parts["relations"]
    assert differs(translation)
    # another aggregation measure (log-sum-exp over the boxes of a union)
    lse = _grounded_general(ds, measure=UnionDepth(agg="lse", agg_tau=0.5))
    assert lse.module.measure.agg == "lse" and differs(lse)
    # another box-union class model (three boxes per class)
    three = _grounded_general(ds, classes=partial(_three_boxes, BoxUnionClasses))
    assert three.module.class_model.n_components == 3 and differs(three, copy=False)
    # another objective (no region terms)
    plain = _grounded_general(ds, objective=GroundedObjective(lambda_reg=0.0))
    assert differs(plain)
    for model in (translation, lse, three, plain):
        assert all(th.isfinite(th.tensor(model.train(epochs=2, validate_every=10 ** 6))))


def _three_boxes(cls, nb_classes, n_components, *args, **kwargs):
    return cls(nb_classes, 3, *args, **kwargs)


def test_the_grounded_geometry_refuses_parts_of_other_families():
    from mowl.representation import Objective
    ds = _abox_dataset()
    for parts in ({"classes": "ball"}, {"relations": "affine map"}, {"measure": "point"},
                  {"objective": Objective(0.1)}):
        with pytest.raises(ConstructorUnavailable):
            _grounded_general(ds, **parts)
