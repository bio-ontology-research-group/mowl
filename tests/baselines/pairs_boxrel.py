"""ELBE, TransBox, BoxEL, BoxSEL and ELK-Em: the class users get (built from parts) against the
direct implementation (``mowl.baselines``).

Every pair copies the parameters of the direct module into the combination through
``load_state_dict`` (the checkpoints of the direct module load into the combination), so the
copy is itself a test of that.
"""
import numpy as np
import torch as th

from tests.baselines.harness import Pair

EMBED = 8
FORMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")


NS = "http://test/boxrel/"


def tiny_ontology():
    """A small EL++ ontology with every normal form, a role inclusion, a role chain, owl:Thing,
    owl:Nothing, individuals with class and role assertions (the ontology of the ELK-Em
    tests, a little smaller)."""
    from jpype import java
    from org.semanticweb.owlapi.model import IRI
    from mowl.datasets import Dataset
    from mowl.owlapi import OWLAPIAdapter
    adapter = OWLAPIAdapter()
    f = adapter.data_factory
    c = {n: f.getOWLClass(IRI.create(NS + n)) for n in "ABCDEFGHKLM"}
    r = {n: f.getOWLObjectProperty(IRI.create(NS + n)) for n in "rtsp"}
    ont = adapter.owl_manager.createOntology(IRI.create(NS + "abox"))

    def sub(a, b):
        ont.addAxiom(f.getOWLSubClassOfAxiom(a, b))

    ex = f.getOWLObjectSomeValuesFrom
    sub(c["A"], c["B"])
    sub(c["B"], c["C"])
    sub(c["A"], ex(r["r"], c["D"]))
    sub(c["D"], ex(r["t"], c["E"]))
    chain = java.util.ArrayList()
    chain.add(r["r"])
    chain.add(r["t"])
    ont.addAxiom(f.getOWLSubPropertyChainOfAxiom(chain, r["s"]))
    sub(ex(r["s"], c["E"]), c["F"])
    ont.addAxiom(f.getOWLSubObjectPropertyOfAxiom(r["r"], r["p"]))
    sub(ex(r["p"], c["D"]), c["G"])
    sub(f.getOWLObjectIntersectionOf(c["B"], c["G"]), c["H"])
    sub(f.getOWLObjectIntersectionOf(c["C"], c["K"]), f.getOWLNothing())
    sub(ex(r["t"], c["K"]), f.getOWLNothing())
    sub(c["L"], f.getOWLNothing())
    sub(c["M"], c["L"])
    a, b = (f.getOWLNamedIndividual(IRI.create(NS + n)) for n in ("a", "b"))
    ont.addAxiom(f.getOWLClassAssertionAxiom(c["A"], a))
    ont.addAxiom(f.getOWLClassAssertionAxiom(c["K"], b))
    ont.addAxiom(f.getOWLObjectPropertyAssertionAxiom(r["t"], a, b))
    return Dataset(ont)


def ordered(build):
    """The rows of the training axioms in a fixed order. The order in which the ontology
    reader yields the axioms changes from one model to the next (it follows the hashes of Java
    objects), and the negatives of a row are drawn by row, so two models that start from the
    same parameters and the same seed would still see different negatives."""
    def make(dataset):
        model = build(dataset)
        for data in model.training_datasets.values():
            if len(data):
                rows = data.data.cpu().numpy()
                order = np.lexsort(rows.T[::-1])
                data._data = data.data[th.as_tensor(order, device=data.data.device)]
        return model
    return make


def copy_state(old, new):
    """The checkpoint of the direct module loads into the part-built module, strictly."""
    new.module.load_state_dict(old.module.state_dict(), strict=True)
    new.module.train(old.module.training)


def scores_of(forms=FORMS, rows=40):
    """What an evaluation regime ranks: the module's score of the first rows of every
    training axiom form, in evaluation mode and without gradients."""
    def scores(model):
        out = {}
        was_training = model.module.training
        model.module.eval()
        with th.no_grad():
            for form, data in model.training_datasets.items():
                if len(data) and form in forms:
                    out[form] = model.module(data[:][:rows], form)
        model.module.train(was_training)
        return out
    return scores


def sgd_train(model, epochs):
    """Train with plain gradient descent. The two implementations compute the same gradients
    up to float rounding (1e-7), but BoxEL's loss has parameters whose true gradient is zero
    (a saturated clamp) and whose computed gradient is a rounding residue of 1e-8; Adam divides
    by the size of the gradient, so such a residue moves the parameter by a full learning-rate
    step in one implementation and not in the other. Gradient descent keeps the residues
    at 1e-7 and so shows whether the two follow the same path."""
    model.get_optimizer = lambda: th.optim.SGD(model.module.parameters(), lr=0.05)
    model.train(epochs=epochs)


def weights(names):
    """``model -> {name: tensor}`` of the tables the two modules share a name for."""
    def get(model):
        out = {}
        for name in names:
            table = getattr(model.module, name)
            out[name] = table.weight if hasattr(table, "weight") else table
        return out
    return get


# -- ELBE --------------------------------------------------------------------------------------
def _elbe(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.elbe.model import ELBE
        else:
            from mowl.models import ELBE
        return ELBE(dataset, embed_dim=EMBED, learning_rate=0.01, batch_size=64, **options)
    return ordered(build)


ELBE_TABLES = ("class_embed", "class_offset", "rel_embed")

# -- TransBox ----------------------------------------------------------------------------------
TRANSBOX_TABLES = ("class_center_embedding", "class_offset_embedding",
                   "relation_center_embedding", "relation_offset_embedding")


def _transbox(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.transbox.model import TransBox
        else:
            from mowl.models import TransBox
        return TransBox(dataset, embed_dim=EMBED, learning_rate=0.01, batch_size=64, **options)
    return ordered(build)


# -- BoxEL and BoxSEL --------------------------------------------------------------------------
BOXEL_TABLES = ("min_embedding", "delta_embedding", "relation_embedding", "scaling_embedding")


def _boxel(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.boxel.model import BoxEL
        else:
            from mowl.models import BoxEL
        return BoxEL(dataset, embed_dim=EMBED, learning_rate=0.01, batch_size=64, **options)
    return ordered(build)


CONDITIONALS = [("http://Mother", "http://Female", 0.6), ("http://Father", "http://Male", 0.4)]


def _boxsel(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.boxsel.model import BoxSEL
        else:
            from mowl.models import BoxSEL
        return BoxSEL(dataset, conditionals=CONDITIONALS, embed_dim=EMBED, learning_rate=0.01,
                      batch_size=64, **options)
    return ordered(build)


def _boxsel_abox(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.boxsel.model import BoxSEL
        else:
            from mowl.models import BoxSEL
        conditionals = [(NS + "B", NS + "A", 0.7), (NS + "D", NS + "A", 0.0, 0.3)]
        return BoxSEL(dataset, conditionals=conditionals, embed_dim=EMBED, learning_rate=0.01,
                      batch_size=64, **options)
    return ordered(build)


# -- ELK-Em ------------------------------------------------------------------------------------
ELKEM_TABLES = ("class_center", "class_offset", "range_center", "range_offset", "error_offset",
                "slope", "shift")


def _elkem(old, **options):
    def build(dataset):
        if old:
            from mowl.baselines.models.elkem.model import ELKEm
        else:
            from mowl.models import ELKEm
        return ELKEm(dataset, embed_dim=EMBED, learning_rate=0.01, batch_size=64,
                     lr_decay_epoch=None, **options)
    return ordered(build)


def _elkem_losses(model):
    """The loss of every training form (the module is in training mode, margin as set)."""
    return {form: model.module(data, form) for form, data in model._training_tensors().items()}


def _elkem_scores(model):
    out = {}
    was_training = model.module.training
    model.module.eval()
    with th.no_grad():
        for form, data in model._training_tensors().items():
            out[form] = model.module(data[:40], form)
            out[form + "/score"] = model.module.score(data[:40], form)
    model.module.train(was_training)
    return out


ALL_FORMS = FORMS + ("class_assertion", "object_property_assertion", "role_inclusion",
                     "role_chain")


def _pair(key, make, tables, scores=None, losses=None, **extra):
    options = extra.pop("options", {})
    kwargs = dict(key=key, new=make(False, **options), old=make(True, **options),
                  copy=copy_state, scores=scores or scores_of(),
                  parameters=weights(tables))
    if losses is not None:
        kwargs["losses"] = losses
    kwargs.update(extra)
    return Pair(**kwargs)


PAIRS = [
    _pair("elbe", _elbe, ELBE_TABLES),
    _pair("elbe/inclusion-joint", _elbe, ELBE_TABLES,
          options=dict(gci3_form="inclusion", init="joint", margin=0.05)),
    _pair("transbox", _transbox, TRANSBOX_TABLES),
    _pair("transbox/reference", _transbox, TRANSBOX_TABLES,
          options=dict(formulation="reference", margin=0.0)),
    _pair("boxel", _boxel, BOXEL_TABLES),
    _pair("boxel/signed-scale", _boxel, BOXEL_TABLES,
          options=dict(positive_scale=False, role_chain_composition="as_released")),
    _pair("boxsel", _boxsel, BOXEL_TABLES),
    _pair("elkem", _elkem, ELKEM_TABLES, scores=_elkem_scores, losses=_elkem_losses),
    _pair("elkem/reference", _elkem, ELKEM_TABLES, scores=_elkem_scores,
          losses=_elkem_losses, options=dict(variant="reference")),
    # individuals, assertions, role inclusions and role chains (the EL++ forms)
    _pair("elbe/assertions", _elbe, ELBE_TABLES, scores=scores_of(ALL_FORMS),
          dataset=tiny_ontology),
    _pair("transbox/assertions-roles", _transbox, TRANSBOX_TABLES,
          scores=scores_of(ALL_FORMS), dataset=tiny_ontology,
          options=dict(load_role_axioms=True)),
    _pair("boxel/assertions", _boxel, BOXEL_TABLES, scores=scores_of(ALL_FORMS),
          dataset=tiny_ontology, train=sgd_train),
    _pair("boxsel/assertions", _boxsel_abox, BOXEL_TABLES, scores=scores_of(ALL_FORMS),
          dataset=tiny_ontology, train=sgd_train),
    _pair("elkem/assertions-roles", _elkem, ELKEM_TABLES, scores=_elkem_scores,
          losses=_elkem_losses, dataset=tiny_ontology, options=dict(load_role_axioms=True)),
]

EXEMPT = {}
