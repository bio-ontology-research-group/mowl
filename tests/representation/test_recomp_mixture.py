"""The canonical mixture as a combination of parts: what it is made of, what can be swapped and
still trains and evaluates, and that the catalogue and the parts agree. The equality with the
direct implementation is ``tests/baselines`` (``pairs_mixture.py``)."""
import itertools
import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset  # noqa: E402
from mowl.models import CanonicalMixture  # noqa: E402
from mowl.nn import CanonicalMixtureModule  # noqa: E402
from mowl.representation import (METHODS, REPRESENTATIONS, ComposedGeometry,  # noqa: E402
                                 Geometry, Objective, geometry_of)
from mowl.representation.construction import SOURCES, construct  # noqa: E402
from mowl.representation.library import mixture_individuals as mi  # noqa: E402
from mowl.representation.library import mixture_measures as mm  # noqa: E402
from mowl.representation.library import mixture_methods as mx  # noqa: E402
from mowl.representation.library import mixture_tables as mt  # noqa: E402
from mowl.representation.library import mixture_training as mtr  # noqa: E402
from mowl.training.regularization import PowerPenalty  # noqa: E402
from tests.models.test_canonical_mixture import (IND, NS,  # noqa: E402
                                                 _write_family_abox_dataset)


def _folder():
    base = os.environ.get("TMPDIR") or None
    return pathlib.Path(tempfile.mkdtemp(prefix="mowl_rm_", dir=base))


@pytest.fixture(scope="module")
def dataset():
    return _write_family_abox_dataset(_folder(), include_unseen_val_ind=True)


@pytest.fixture(scope="module")
def dataset_with_test():
    """The family ABox with its validation file as test file, and the training ontology (the
    TBox axioms) as a second test set for the subsumption ranking."""
    ds = _write_family_abox_dataset(_folder())
    return ds, PathDataset(ds.ontology_path, validation_path=ds.ontology_path,
                           testing_path=ds.ontology_path)


def new(dataset, **options):
    options.setdefault("rank", 4)
    options.setdefault("batch_size", 2)
    options.setdefault("use_cache", False)
    th.manual_seed(0)
    return CanonicalMixture(dataset, **options)


def rebuilt(model, **parts):
    """The model with its module rebuilt by the named combination with parts replaced."""
    ca = model.training_datasets["class_assertion"][:]
    model.module = METHODS["canonical_mixture"](
        len(model.class_index_dict), len(model.object_property_index_dict),
        len(model.individual_index_dict), embed_dim=model.rank, structure=model.structure,
        training_class_assertions=ca, **parts)
    return model


def guarantees_hold(model):
    """Entailed class assertions score 0.0, contradicted ones inf (whatever was trained)."""
    c, i = model.class_index_dict, model.individual_index_dict
    entailed = th.tensor([[i[IND + "bob"], c[NS + "Father"]], [i[IND + "bob"], c[NS + "Person"]],
                          [i[IND + "carl"], c[NS + "Parent"]]])
    contradicted = th.tensor([[i[IND + "bob"], c[NS + "Female"]],
                              [i[IND + "alice"], c[NS + "Male"]]])
    m = model.module
    return bool((m.class_assertion_score(entailed) == 0.0).all()) and \
        bool(th.isinf(m.class_assertion_score(contradicted)).all())


def train_one(model, **options):
    model.train(epochs=1, **options)
    assert len(model.loss_history) == 1 and np.isfinite(model.loss_history).all()
    scores = model.module.rank_classes(th.arange(len(model.individual_index_dict)))
    assert not th.isnan(scores).any()
    assert guarantees_hold(model)


# -- (a) what the method is made of ---------------------------------------------------------
def test_canonical_mixture_is_made_of_the_parts_the_card_names(dataset):
    model = new(dataset, entropy_reg=0.01)
    g = model.module
    assert isinstance(g, CanonicalMixtureModule) and isinstance(g, mx.MixtureGeometry)
    assert isinstance(g, ComposedGeometry) and isinstance(g, Geometry)
    assert geometry_of(model) is g
    # constructed: the canonical model, read sparsely
    assert type(model.source) is mt.CanonicalCompletionSource
    assert type(model.reading) is mt.MembershipReading
    assert type(g.class_model) is mt.MembershipClasses
    assert type(g.relation_model) is mt.CanonicalRoles
    # trained: the individuals
    assert type(g.individual_model) is mi.MixtureIndividuals
    assert type(g.measure) is mm.ProbabilityMass and g.measure.loss == "bce"
    assert type(g.objective) is Objective
    assert g.objective.neg_forms == frozenset({"class_assertion", "object_property_assertion"})
    assert [type(r) for r in g.regularizers] == [mm.EntropyPenalty]
    assert g.regularizers[0].weight == 0.01
    assert type(model.training) is mtr.IndividualBatches
    assert type(model.training.class_negatives) is mtr.NonEntailedClasses
    assert type(model.training.role_negatives) is mtr.UniformTails
    assert g.parts == {"classes": "fixed by reasoning", "relations": "fixed by reasoning",
                       "individuals": "distribution", "measure": "exact",
                       "conjunction": "exact", "existential": "exact",
                       "source": "canonical completion",
                       "reading": "membership"}
    # only the individual model has parameters
    assert {n for n, _ in g.named_parameters()} == {"individual_model.U", "individual_model.V",
                                                    "individual_model.b"}
    # without the entropy weight there is no regulariser
    assert new(dataset).module.regularizers == ()
    # the builder is the named combination, also from sizes (one element per class)
    built = METHODS["canonical_mixture"](5, 2, 3, 8)
    assert type(built) is mx.MixtureGeometry and built.n_elements == 8
    assert type(built.individual_model) is mi.MixtureIndividuals and built.rank == 8
    assert built.parts["individuals"] == "distribution"


def test_the_constructed_part_is_a_construction(dataset):
    """The canonical model comes from the construction interface: the source builds it, the
    reading restricts it; read with the probability-mass measure it answers TBox questions
    exactly, as the class model of the method does."""
    model = new(dataset)
    names = [n for n, _ in sorted(model.class_index_dict.items(), key=lambda kv: kv[1])]
    constructed = construct(model.dataset.ontology, "canonical completion", "membership",
                            names=names, roles=sorted(model.object_property_index_dict),
                            individuals=[], measure=mm.ProbabilityMass())
    assert constructed.reading.key == "membership" and constructed.source.key == \
        "canonical completion"
    cm = SOURCES["canonical completion"]().build(model.dataset.ontology)
    pairs = th.tensor(list(itertools.product(range(len(names)), repeat=2)))
    cost = constructed.cost("gci0", pairs).reshape(-1)
    mine = model.module.cost("gci0", pairs).reshape(-1)
    assert th.equal(cost, mine)
    from mowl.geometric.normal_forms import TOP
    for (a, b), value in zip(pairs.tolist(), cost.tolist()):
        if names[a] == TOP:
            # the top class has no element of its own in the support: over the support it
            # may lie inside a class that covers every element (here: Person)
            continue
        assert value == (0.0 if cm.entails(names[a], names[b]) else 1.0), (names[a], names[b])
    # the same reading of a finite model built by the "canonical model" source
    finite = SOURCES["canonical model"]().build(model.dataset.ontology)
    read = mt.MembershipReading().structure(finite, names, [], [])
    assert read.element_names == model.reading.structure(cm, names, [], []).element_names


def test_the_tbox_forms_are_read_exactly_by_the_element_sets(dataset):
    model = new(dataset)
    g, c = model.module, model.class_index_dict
    r = model.object_property_index_dict[NS + "hasChild"]
    father, parent, male, female, person = (c[NS + n] for n in
                                            ("Father", "Parent", "Male", "Female", "Person"))
    assert g.cost("gci0", th.tensor([[father, parent], [parent, father]])).reshape(-1).tolist() \
        == [0.0, 1.0]
    assert g.cost("gci1_bot", th.tensor([[male, female, 0]])).item() == 0.0     # disjoint
    assert g.cost("gci1", th.tensor([[father, male, person]])).item() == 0.0
    # exists hasChild.Person <= Parent holds over the support
    assert g.cost("gci3", th.tensor([[r, person, parent]])).item() == 0.0
    with pytest.raises(NotImplementedError):
        g.cost("gci0", th.tensor([[father, parent]]), neg=True)


def test_the_dense_path_of_full_supports_gives_the_same_numbers(dataset):
    """Individuals whose support is every element take a dense row; the gathered path gives
    the same scores and gradients."""
    model = new(dataset)
    g = model.module
    assert bool(g.is_full_support.any())                         # eve: only in validation
    k, n = len(model.individual_index_dict), len(model.class_index_dict)
    rows = th.tensor([[a, cl] for a in range(k) for cl in range(n)])
    roles = th.tensor([[a, 0, b] for a in range(k) for b in range(k)])
    with th.no_grad():
        g.b.normal_(0.0, 1.0)

    def run():
        g.zero_grad()
        out = (g.class_assertion_score(rows), g.object_property_assertion_score(roles),
               g.log_p(th.arange(k)))
        loss = g(rows, "class_assertion").sum() + g(roles, "object_property_assertion").sum()
        loss.backward()
        return out, g.V.grad.clone()

    dense, grad_dense = run()
    g.individual_model.dense_full_support = False
    gathered, grad_gathered = run()
    for x, y in zip(dense, gathered):
        assert th.allclose(x, y, atol=1e-5, equal_nan=True)
    assert th.allclose(grad_dense, grad_gathered, atol=1e-5)


def test_role_scores_are_the_dense_formula(dataset):
    model = new(dataset)
    g = model.module
    k = len(model.individual_index_dict)
    rows = th.tensor([[a, 0, b] for a in range(k) for b in range(k)])
    probs = g.individual_model.probs(th.arange(k))
    e = g.relation_model.matrix(0).to_dense()
    expected = th.stack([probs[a] @ (e @ probs[b]) for a, _, b in rows.tolist()])
    assert th.allclose(th.exp(-g.object_property_assertion_score(rows)), expected, atol=1e-6)


# -- (b) swaps ------------------------------------------------------------------------------
def test_swap_the_training_regime(dataset):
    model = new(dataset, training=mtr.AssertionBatches(batch_size=2))
    train_one(model)
    other = new(dataset)
    train_one(other)
    assert not th.allclose(model.module.U, other.module.U)       # another loop, other numbers


def test_swap_the_measure_reading_and_an_objective_option(dataset):
    model = rebuilt(new(dataset), measure=mm.ProbabilityMass(loss="log", eps=1e-6),
                    objective=mx.mixture_objective(weights={"object_property_assertion": 0.5}))
    assert model.module.measure.loss == "log"
    rows = model.training_datasets["object_property_assertion"][:]
    full = rebuilt(new(dataset), measure=mm.ProbabilityMass(loss="log", eps=1e-6))
    assert th.allclose(model.module(rows, "object_property_assertion"),
                       0.5 * full.module(rows, "object_property_assertion"))
    train_one(model)


def test_swap_in_regularisers(dataset):
    """The entropy penalty of the method and a penalty of the general library on the logits'
    individual factors, together."""
    model = rebuilt(new(dataset), regularizers=(
        mm.EntropyPenalty(0.1), PowerPenalty(parameters="individual_model.U", weight=0.5)))
    assert float(model.module.regularization_loss()) > 0
    train_one(model)
    spread = rebuilt(new(dataset), regularizers=(mm.EntropyPenalty(-0.1),))   # favours spread
    train_one(spread)


def test_swap_the_negative_sampler(dataset):
    class AnyClass:
        """Unfiltered negatives: any class (also an entailed one)."""

        def sample(self, geometry, rows):
            return th.stack([rows[:, 0], th.randint(0, len(geometry.class_model),
                                                    (len(rows),))], dim=1)

    model = new(dataset, training=mtr.IndividualBatches(class_negatives=AnyClass()))
    train_one(model)


def test_swap_the_evaluation_regime(dataset_with_test):
    from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
    from mowl.evaluation.triple_classification import TripleClassificationRegime
    train, both = dataset_with_test
    model = new(train)
    model.train(epochs=1)
    result = TripleClassificationRegime(form="class_assertion", negatives="sampled").evaluate(
        model, both)
    assert all(np.isfinite(v) for v in result["all"].values())
    ranking = NormalFormRankingRegime(normal_forms=("gci0",), split="test").evaluate(model, both)
    # the TBox is the canonical model: every stated subsumption ranks among the entailed ones
    assert ranking["gci0"]["mrr"] > 0 and np.isfinite(ranking["gci0"]["mr"])


# -- (c) the catalogue and the parts --------------------------------------------------------
def test_catalogue_card_and_parts_agree(dataset):
    from mowl.catalog.registry import get_method
    card = get_method("canonical_mixture")
    rep = REPRESENTATIONS["canonical_mixture"]
    model = new(dataset)
    parts = model.module.parts
    assert parts["classes"] in {v.value for v in rep.classes}
    assert parts["relations"] in {v.value for v in rep.relations}
    assert parts["measure"] in {v.value for v in rep.measure}
    assert parts["individuals"] in {v.value for v in rep.individuals}
    assert parts["conjunction"] == rep.conjunction.value
    assert parts["existential"] == rep.existential.value
    # the card's axioms are the forms the regime trains, with a negative loss each
    assert set(card.axioms) == {f for f, _ in mtr._FORMS} == model.module.neg_capable_gcis
    # the tasks: subsumption, satisfiability and disjointness by the constructed model,
    # instance checking and role assertions by the trained individuals
    assert {"subsumption", "satisfiability", "disjointness", "instance-checking",
            "role-assertion"} <= set(card.tasks)
    assert {"gci0", "gci0_bot", "gci1_bot", "class_assertion",
            "object_property_assertion"} <= model.module.supported_forms
    assert card.implementation == "mowl.models.canonical_mixture.model.CanonicalMixture"
    assert card.load() is CanonicalMixture
    assert "EL++" in card.logic and "EL++" in model.source.logic
