"""The parts of the ALC training and the swaps of the ALC methods (audit 2, items 3 and 8).

* The objective part that combines the axiom patterns into a loss
  (:class:`~mowl.representation.library.logic_alc.AxiomKindWeighting`), the sampler of the
  anonymous elements (:class:`~mowl.representation.library.logic_alc.AnonymousElements`), the
  schedule of Al-cone as a training regime, each on its own and in a combination other than
  the method it came from.
* The ALC models train in the shared loop of :mod:`mowl.training.loop`.
* A swap on ``dfalc``, ``falcon`` or ``alcone`` either trains (and changes the training) or
  is refused when the model is built, with the reason; none fails during training.
"""
import math

import pytest
import torch as th

from mowl.models import ALConeModel, ComposedALC, DFALCModel, FALCONModel
from mowl.representation import CLASS_MODELS, MEASURES, RELATION_MODELS
from mowl.representation.library import logic_fuzzy as F
from mowl.representation.library.logic_alc import (ABOX_EC, ABOX_EE, TBOX,
                                                   AnnealedRelaxation, AnonymousElements,
                                                   AxiomKindWeighting)
from mowl.representation.regions import ConstructorUnavailable
from mowl.training import LinearDecay
from mowl.training.loop import TrainingLoop
from mowl.training.negatives import EitherSideCorruption
from tests.baselines.pairs_logic import alc_dataset

D = 4

_DATA = {}


def data():
    if "alc" not in _DATA:
        _DATA["alc"] = alc_dataset()
    return _DATA["alc"]


def alc(method, **parts):
    th.manual_seed(0)
    options = {"anon_e": 2} if method == "falcon" else {}
    return ComposedALC(data(), method=method, embed_dim=D, **options, **parts)


def finite(t):
    return bool(th.isfinite(th.as_tensor(t)).all())


def trains(model, epochs=2):
    """Train, check the parameters move and stay finite and that subsumptions are scored."""
    before = [p.detach().clone() for p in model.module.parameters()]
    history = model.train(epochs=epochs)
    assert len(history) == epochs and all(math.isfinite(x) for x in history)
    after = list(model.module.parameters())
    assert all(finite(p) for p in after)
    assert any(not th.equal(a, b) for a, b in zip(before, after))
    n = len(model.class_index_dict)
    assert finite(model.axiom_costs("gci0", th.cartesian_prod(th.arange(n), th.arange(n))))
    return history


def first_loss(model):
    """The loss of the first training step (no update)."""
    model._load_datasets()
    th.manual_seed(3)
    context = model._entity_context(model.sample_anonymous_entities())
    with th.no_grad():
        return float(model._epoch_loss(model.training_dataloaders, context))


# -- the objective part: AxiomKindWeighting ---------------------------------------------------
class _Geometry:
    """``geometry(axiom, rows, context)``: the rows as costs."""

    def __call__(self, axiom, rows, context=None):
        return rows.float()


def test_the_weighting_classifies_axioms_by_kind():
    from mowl.owlapi import OWLAPIAdapter
    ad = OWLAPIAdapter()
    a, b = ad.create_class("http://x#A"), ad.create_class("http://x#B")
    x, y = ad.create_individual("http://x#x"), ad.create_individual("http://x#y")
    r = ad.create_object_property("http://x#r")
    kind = AxiomKindWeighting.kind
    assert kind(ad.create_subclass_of(a, b)) == TBOX
    assert kind(ad.create_equivalent_classes(a, b)) == TBOX
    assert kind(ad.create_disjoint_classes(a, b)) == TBOX
    assert kind(ad.create_class_assertion(a, x)) == ABOX_EC
    assert kind(ad.create_object_property_assertion(r, x, y)) == ABOX_EE
    assert kind(ad.data_factory.getOWLSubObjectPropertyOfAxiom(r, r)) is None


def test_the_weighting_is_the_weighted_mean_of_batch_costs_per_kind():
    from mowl.owlapi import OWLAPIAdapter
    ad = OWLAPIAdapter()
    a, b = ad.create_class("http://x#A"), ad.create_class("http://x#B")
    x = ad.create_individual("http://x#x")
    tbox1, tbox2 = ad.create_subclass_of(a, b), ad.create_disjoint_classes(a, b)
    ec = ad.create_class_assertion(a, x)
    batches = {tbox1: [(th.tensor([1.0, 3.0]),), (th.tensor([5.0]),)],
               tbox2: [(th.tensor([7.0]),)], ec: [(th.tensor([2.0, 4.0]),)]}
    w = AxiomKindWeighting(0.5, 0.25)
    assert w.role_assertions == pytest.approx(0.25)
    # TBox batches: 2, 5, 7 -> mean 14/3; class assertions: 3
    expected = 0.5 * 14 / 3 + 0.25 * 3
    assert float(w.loss(_Geometry(), batches)) == pytest.approx(expected)
    counts = w.batch_counts(batches)
    assert counts == {TBOX: 3, ABOX_EC: 1, ABOX_EE: 0}
    parts = sum(float(w.combine(ax, w.batch_costs(_Geometry(), ax, bs), counts))
                for ax, bs in batches.items())
    assert parts == pytest.approx(expected)
    assert w.loss(_Geometry(), {}) is None


def test_the_weighting_refuses_weights_that_do_not_add_up():
    with pytest.raises(ValueError):
        AxiomKindWeighting(0.8, 0.5)
    with pytest.raises(ValueError):
        AxiomKindWeighting(1.0, 1.0, -1.0)
    assert AxiomKindWeighting(1.0, 1.0, 1.0).weights == {TBOX: 1.0, ABOX_EC: 1.0,
                                                         ABOX_EE: 1.0}


@pytest.mark.parametrize("method", ["dfalc", "falcon", "alcone"])
def test_the_named_methods_combine_their_losses_with_the_weighting_part(method):
    model = {"dfalc": lambda: DFALCModel(data(), anon_domain=1),
             "falcon": lambda: FALCONModel(data(), embed_dim=D, anon_e=2),
             "alcone": lambda: ALConeModel(data(), dim=2)}[method]()
    weighting = model.module.objective.weighting
    assert isinstance(weighting, AxiomKindWeighting)
    assert model.weighting is weighting
    assert (weighting.tbox, weighting.class_assertions) == (model.alpha, model.beta)


def test_a_weighting_swapped_through_the_objective_changes_the_loss():
    own = alc("dfalc")
    tbox_only = alc("dfalc", objective_options={"weighting": AxiomKindWeighting(1.0, 0.0)})
    assert tbox_only.module.objective.weighting.tbox == 1.0
    assert (tbox_only.alpha, tbox_only.beta) == (1.0, 0.0)
    assert first_loss(own) != pytest.approx(first_loss(tbox_only))
    trains(tbox_only)


def test_the_weighting_in_a_combination_of_parts_without_a_method():
    """The weighting part with parts of DF-ALC and the measure of FALCON: the loss is the
    mean cost of the terminological axioms when only they are weighted."""
    weighting = AxiomKindWeighting(1.0, 0.0)
    objective = F.FuzzyObjective(role_assertion="image", weighting=weighting)
    th.manual_seed(0)
    model = ComposedALC(data(), classes="fuzzy set (product)", relations="fuzzy relation",
                        measure="conflict", objective=objective, embed_dim=D)
    assert model.weighting is weighting
    model._load_datasets()
    with th.no_grad():
        costs = [float(model.module(ax, d.tensors[0]).mean())
                 for ax, d in model._training_datasets.items()
                 if AxiomKindWeighting.kind(ax) == TBOX]
        loss = float(model._epoch_loss(model.training_dataloaders, None))
    assert loss == pytest.approx(sum(costs) / len(costs), rel=1e-5)
    trains(model)


def test_weights_given_twice_are_refused():
    objective = F.FuzzyObjective(role_assertion="image", weighting=AxiomKindWeighting(1, 0))
    with pytest.raises(TypeError, match="given twice"):
        ComposedALC(data(), classes="fuzzy set", relations="fuzzy relation",
                    objective=objective, alpha=0.5, embed_dim=D)


# -- the sampler of anonymous elements -------------------------------------------------------
def test_anonymous_elements_perturb_named_ones_and_draw_fresh_ones():
    named = th.zeros(3, 5)
    sampler = AnonymousElements(4, noise=0.1)
    th.manual_seed(0)
    sample = sampler.sample(named, 5)
    assert sample.shape == (4, 5)
    assert float(sample[:2].abs().max()) < 1.0          # small perturbations of zeros
    context = sampler.context(th.ones(3, 5), 2, sample)
    assert context.shape == (6, 5) and th.equal(context[:2], th.ones(2, 5))
    th.manual_seed(1)
    state = th.random.get_rng_state()
    a = sampler.with_seed(lambda: sampler.sample(named, 5))
    assert th.equal(th.random.get_rng_state(), state)
    assert th.equal(a, sampler.with_seed(lambda: sampler.sample(named, 5)))
    assert AnonymousElements(1).sample(th.zeros(0, 5), 5).shape == (1, 5)


def test_a_model_samples_its_anonymous_entities_with_the_sampler():
    model = FALCONModel(data(), embed_dim=D, anon_e=3)
    assert model.anonymous_elements.count == 3
    th.manual_seed(0)
    sample = model.sample_anonymous_entities()
    assert sample.shape == (3, D)
    assert model._entity_context(sample).shape == (len(data().individuals) + 3, D)


# -- the shared training loop ----------------------------------------------------------------
@pytest.mark.parametrize("cls", [ComposedALC, FALCONModel, DFALCModel, ALConeModel])
def test_the_alc_models_have_no_loop_of_their_own(cls):
    assert issubclass(cls, TrainingLoop)
    assert cls.run_training is TrainingLoop.run_training
    assert not hasattr(cls, "_train") and not hasattr(cls, "_train_swapped")


def test_a_schedule_regime_is_consulted_and_a_loss_regime_is_refused():
    model = FALCONModel(data(), embed_dim=D, anon_e=2, learning_rate=0.01)
    decay = LinearDecay()
    model.training_regime = decay
    model.train(epochs=3)
    assert model.training_regime is decay
    assert len(decay.history) == 3 and decay.history[-1] < decay.history[0] == 0.01
    with pytest.raises(TypeError, match="negatives"):
        model.training_regime = EitherSideCorruption()


def test_the_annealing_of_alcone_is_its_training_regime():
    model = ALConeModel(data(), dim=2)
    regime = model.training_regime
    assert regime is model.module.schedule and isinstance(regime, AnnealedRelaxation)
    seen = []
    model.module.schedule.noise = 0.0
    original = regime.on_epoch_start

    def record(epoch, epochs):
        original(epoch, epochs)
        seen.append(model.module.temperature)
    regime.on_epoch_start = record
    model.train(epochs=4, temperature=0.2)
    assert seen == pytest.approx([0.2, 0.15, 0.1, 0.05])
    assert model.module.temperature == 0.0 and model.temperature == 0.3


def test_the_annealing_with_other_parts_or_another_loop():
    """The annealed relaxation of Al-cone with the objective of DF-ALC, and in the loop of
    ComposedEL; a geometry without a relaxation refuses it when the model is built."""
    schedule = AnnealedRelaxation(temperature=0.2, noise=0.0)
    model = alc("alcone", anneal=None, schedule=schedule, objective=F.FuzzyObjective.dfalc())
    assert model.training_regime is schedule
    trains(model)
    assert model.module.temperature == 0.0
    from mowl.models import ComposedEL
    el = ComposedEL(data(), method="alcone", embed_dim=2)
    assert el.training_regime is el.module.schedule
    with pytest.raises(TypeError, match="temperature"):
        ComposedALC(data(), classes="half-axis cone", relations="incidence matrix",
                    measure="half-axis hinge", embed_dim=2, schedule=schedule,
                    objective=F.FuzzyObjective(0.1, validity=True))
    with pytest.raises(TypeError, match="temperature"):
        ComposedALC(data(), method="dfalc", embed_dim=D).training_regime = schedule


# -- swaps: refused at build with a reason, or they train ------------------------------------
REFUSED = [
    # (method, part, value, words of the reason)
    ("falcon", "classes", "fuzzy set (product)", "open domain"),
    ("falcon", "relations", "translation", "open domain"),
    ("falcon", "individuals", "fuzzy set", "not read"),
    ("falcon", "measure", "half-axis hinge", "member_at"),
    ("falcon", "objective_options", {"rule": True}, "rule=True"),
    ("falcon", "objective_options", {"margin": 0.07}, "no margin"),
    ("dfalc", "measure", "half-axis hinge", "logits"),
    ("dfalc", "classes", "half-axis cone", "logits"),
    ("dfalc", "relations", "incidence matrix", "elements"),
    ("dfalc", "objective_options", {"abox": "sampled"}, "abox='sampled'"),
    ("dfalc", "objective_options", {"validity": True}, "validity=True"),
    ("alcone", "classes", "fuzzy set", "degrees"),
    ("alcone", "measure", "mean violation", "validity=True"),
    ("alcone", "measure", "conflict", "validity=True"),
    ("alcone", "relations", "angle rotation", "rotated"),
    ("alcone", "objective_options", {"rule": True}, "rule=True"),
]


@pytest.mark.parametrize("method,kind,value,reason", REFUSED,
                         ids=[f"{m}-{k}-{v}" for m, k, v, _ in REFUSED])
def test_a_swap_that_cannot_train_is_refused_when_the_model_is_built(method, kind, value,
                                                                     reason):
    with pytest.raises(ConstructorUnavailable) as error:
        alc(method, **{kind: value})
    message = str(error.value)
    assert repr(method) in message and kind in message and reason in message, message


WORKS = [
    ("falcon", "measure", "mean violation"),
    ("falcon", "objective_options", {"abox": "semantic"}),
    ("falcon", "objective", F.FuzzyObjective(role_assertion="image")),
    ("dfalc", "classes", "fuzzy set (product)"),
    ("dfalc", "measure", "conflict"),
    ("dfalc", "individuals", "fuzzy set"),
    ("dfalc", "objective_options", {"rule": True}),
    ("alcone", "objective", F.FuzzyObjective.dfalc()),
    ("alcone", "objective_options", {"weighting": AxiomKindWeighting(0.6, 0.2)}),
]


@pytest.mark.parametrize("method,kind,value", WORKS,
                         ids=[f"{m}-{k}-{i}" for i, (m, k, _) in enumerate(WORKS)])
def test_a_meaningful_swap_trains_and_changes_the_loss(method, kind, value):
    if kind == "objective":
        value = value.replace()
    own, swapped = alc(method), alc(method, **{kind: value})
    assert first_loss(own) != pytest.approx(first_loss(swapped), rel=1e-6)
    trains(swapped)


def test_falcon_with_the_measure_of_dfalc_reads_degrees_with_its_costs():
    m = F.MeanViolation()
    degree = th.tensor([0.0, 0.25, 1.0])
    assert th.allclose(m.member_at(degree), 1 - degree)
    assert th.allclose(m.apart_at(degree), degree)


@pytest.mark.parametrize("method", ["dfalc", "falcon", "alcone"])
def test_every_single_part_swap_trains_or_is_refused_at_build(method):
    """No swap of a class model, relation model, individual model or measure by name is
    accepted and then fails in training."""
    names = [("classes", n) for n in CLASS_MODELS] + \
        [("relations", n) for n in list(RELATION_MODELS) + ["none"]] + \
        [("individuals", n) for n in CLASS_MODELS] + [("measure", n) for n in MEASURES]
    trained = 0
    for kind, name in names:
        try:
            model = alc(method, **{kind: name})
        except ConstructorUnavailable:
            continue
        model.train(epochs=1)
        assert all(finite(p) for p in model.module.parameters()), (kind, name)
        trained += 1
    assert trained >= 2


def test_a_combination_of_parts_without_a_method_is_checked_at_build():
    with pytest.raises(ConstructorUnavailable, match="The combination"):
        ComposedALC(data(), classes="half-axis cone", relations="fuzzy relation",
                    embed_dim=D)


def test_the_check_does_not_change_the_training_run():
    """The check at build draws on the random state and puts it back, and leaves no
    gradient behind: the run that follows is the one it would have been."""
    model = alc("falcon", measure="mean violation")
    th.manual_seed(0)
    state = th.random.get_rng_state()
    model.check_combination()
    assert th.equal(th.random.get_rng_state(), state)
    assert all(p.grad is None for p in model.module.parameters())


# -- the parts that hold the arithmetic of the DF-ALC and Al-cone modules ---------------------
def test_the_hierarchical_loss_is_the_inside_of_the_mean_violation():
    from mowl.nn.alc.dfalc.module import hierarchical_loss
    c, d = th.tensor([[0.2, 0.5]]), th.tensor([[0.1, 0.6]])
    assert hierarchical_loss is F.MeanViolation.inside_degrees
    region = F.FuzzySet
    inside = F.MeanViolation().inside(region(c, F.Goedel()), region(d, F.Goedel()))
    assert th.allclose(hierarchical_loss(c, d), inside.squeeze(-1))


def test_half_axis_parts_discretise_cones_roles_and_points():
    classes = F.HalfAxisClasses(2, 2)
    with th.no_grad():
        classes.centres.weight.copy_(th.tensor([[1.0, -1.0, -1.0, 2.0],
                                                [0.5, 0.7, -0.1, -0.2]]))
    assert classes.crisp().tolist() == [[True, False, False, True],
                                        [True, True, False, False]]
    # coordinate 0: positive half-axis; coordinate 1: negative one; then both > 0, larger
    # negative; then neither
    assert classes.points().tolist() == [[1.0, -1.0], [-1.0, 0.0]]
    relations = F.IncidenceMatrices(1, 1)
    with th.no_grad():
        relations.vectors.weight.copy_(th.tensor([[1.0, -1.0, 2.0, -2.0]]))
    stored = relations.stored(th.tensor([0]))
    assert th.equal(stored[0], th.tensor([[1.0, -1.0], [2.0, -2.0]]))
    assert th.equal(relations.matrix(th.tensor([0]))[0], stored[0].T)
    assert relations.crisp().tolist() == [[[True, False], [True, False]]]


def test_the_tables_of_dfalc_and_alcone_are_parts_drawn_in_the_old_order():
    th.manual_seed(0)
    c, r = F.truth_tables(3, 2, 4, init_std=1.0)
    th.manual_seed(0)
    c2 = th.nn.Embedding(3, 4)
    r2 = th.nn.Embedding(2, 16)
    for t in (c2, r2):
        th.nn.init.normal_(t.weight, std=1.0)
    assert th.equal(c.weight, c2.weight) and th.equal(r.weight, r2.weight)
    c, r, e = F.half_axis_tables(3, 0, 2, 2)
    assert (c.weight.shape, r.weight.shape, e.weight.shape) == ((3, 4), (1, 16), (2, 4))


def test_domain_elements_follow_a_domain_with_a_context():
    table = th.nn.Embedding(2, 3)
    domain = F.Domain(table)
    individuals = F.DomainElements(2, 2, elements=domain)
    assert individuals(th.tensor([1])).values.tolist() == [[0.0, 1.0]]
    domain.context = th.zeros(4, 3)
    assert individuals(th.tensor([1])).values.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    domain.context = None


def test_the_falcon_evaluator_scores_with_the_measure_of_the_geometry():
    """The energy of the FALCON evaluator, ``-log(1 - max_e (C ⊓ ¬D)(e))`` on the named and
    fresh anonymous entities, is the ``inside`` of the conflict measure of the geometry."""
    from mowl.models.falcon.evaluate import FALCONSubsumptionEvaluator
    model = FALCONModel(data(), embed_dim=D, anon_e=2)
    evaluator = FALCONSubsumptionEvaluator.__new__(FALCONSubsumptionEvaluator)
    evaluator.n_anon = 3
    m = model.module
    cid = model.class_index_dict
    ns = "http://mowl.borg/logic#"
    batch = th.tensor([[cid[ns + "A"], cid[ns + "B"]], [cid[ns + "C"], cid[ns + "D"]]])
    th.manual_seed(0)
    with th.no_grad():
        scores = evaluator.get_scores(m, batch)
    th.manual_seed(0)
    anon = th.empty(3, D)
    th.nn.init.xavier_uniform_(anon)
    with th.no_grad():
        e = th.cat([m.e_embedding.weight, anon])
        c = m._get_c_fs_batch(m.c_embedding(batch[:, 0]), e)
        d = m._get_c_fs_batch(m.c_embedding(batch[:, 1]), e)
        expected = -th.log(1 - m._logical_and(c, m._logical_not(d)).max(dim=-1)[0] + 1e-10)
    assert th.allclose(scores, expected, atol=1e-6)
