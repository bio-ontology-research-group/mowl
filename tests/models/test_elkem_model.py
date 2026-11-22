"""Tests for the ELK-Em model [singh2026]: ⊤/⊥ handling, negative sampling, a soundness
check of Theorem 6 / Corollary 7 on a zero-loss embedding, end-to-end training and the
lexical regulariser."""

from mowl import init_jvm
init_jvm("1g")

import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np
import pytest
import torch as th
from jpype import java
from org.semanticweb.owlapi.model import IRI

from mowl.datasets import Dataset
from mowl.geometric.completion import CanonicalModel
from mowl.models.elkem.model import ELKEm, BOT_IRI, TOP_IRI
from mowl.owlapi import OWLAPIAdapter

NS = "http://test/elkem/"


def build_ontology(with_abox=False):
    """A small EL++ ontology with every normal form, ⊤, ⊥, a role inclusion and a role
    chain. ELK derives A ⊑ F (via the chain), A ⊑ G (via the role inclusion), A ⊑ H and
    M ⊑ ⊥."""
    adapter = OWLAPIAdapter()
    f = adapter.data_factory
    c = {n: f.getOWLClass(IRI.create(NS + n)) for n in "ABCDEFGHKLMNP"}
    r = {n: f.getOWLObjectProperty(IRI.create(NS + n)) for n in "rtspqu"}
    ont = adapter.owl_manager.createOntology(IRI.create(NS + ("abox" if with_abox else "tbox")))

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
    sub(ex(r["q"], c["K"]), f.getOWLNothing())
    sub(c["L"], f.getOWLNothing())
    sub(c["M"], c["L"])
    sub(ex(r["u"], f.getOWLThing()), c["P"])
    sub(c["N"], f.getOWLThing())
    if with_abox:
        a, b = (f.getOWLNamedIndividual(IRI.create(NS + n)) for n in ("a", "b"))
        ont.addAxiom(f.getOWLClassAssertionAxiom(c["A"], a))
        ont.addAxiom(f.getOWLClassAssertionAxiom(c["K"], b))
        ont.addAxiom(f.getOWLObjectPropertyAssertionAxiom(r["t"], a, b))
    return ont


# ------------------------------------------------------- loss-independent semantic checks

def box(model, idx):
    lower, upper = model.module.class_box(th.tensor([idx]))
    return lower[0].double(), upper[0].double()


def role(model, idx):
    return tuple(v[0].double() for v in model.module.role(th.tensor([idx])))


def empty(lower, upper):
    return bool((lower > upper).any())


def in_box(x, lower, upper):
    return bool(((lower <= x) & (x <= upper)).all())


def contained(b1, b2):
    return empty(*b1) or bool(((b2[0] <= b1[0]) & (b1[1] <= b2[1])).all())


def disjoint(b1, b2):
    return bool((th.maximum(b1[0], b2[0]) > th.minimum(b1[1], b2[1])).any())


def in_role(x, y, rl):
    rlo, rhi, el, eu, m, c = rl
    return in_box(y, rlo, rhi) and in_box(x, m * y + c + el, m * y + c + eu)


def has_witness(x, rl, cbox):
    """∃y ∈ C with (x, y) ∈ r^E."""
    rlo, rhi, el, eu, m, c = rl
    lo = th.maximum(th.maximum(cbox[0], rlo), (x - c - eu) / m)
    hi = th.minimum(th.minimum(cbox[1], rhi), (x - c - el) / m)
    return bool((lo <= hi).all())


def samples(lower, upper, k, gen):
    if empty(lower, upper):
        return []
    lower = lower.clamp(min=-50)
    upper = upper.clamp(max=50)
    u = th.rand(k, lower.shape[0], generator=gen, dtype=th.float64)
    return [lower, upper] + list(lower + u * (upper - lower))


def role_pairs(rl, k, gen):
    rlo, rhi, el, eu, m, c = rl
    return [(x, y) for y in samples(rlo, rhi, k, gen)
            for x in samples(m * y + c + el, m * y + c + eu, 3, gen)]


def chain_pairs(r1, r2, k, gen):
    out = []
    for z in samples(r2[0], r2[1], k, gen):
        lo = th.maximum(r2[4] * z + r2[5] + r2[2], r1[0])
        hi = th.minimum(r2[4] * z + r2[5] + r2[3], r1[1])
        for y in samples(lo, hi, 3, gen):
            for x in samples(r1[4] * y + r1[5] + r1[2], r1[4] * y + r1[5] + r1[3], 2, gen):
                out.append((x, z))
    return out


def check_axiom(test, model, form, row, gen, k=40):
    """Check ``row`` of normal form ``form`` against the relational semantics by point
    sampling. The loss code is not used."""
    if form == "gci0":
        test.assertTrue(contained(box(model, row[0]), box(model, row[1])), (form, row))
        for x in samples(*box(model, row[0]), k, gen):
            test.assertTrue(in_box(x, *box(model, row[1])))
    elif form == "gci0_bot":
        test.assertTrue(empty(*box(model, row[0])), (form, row))
    elif form == "gci1":
        b1, b2, d = box(model, row[0]), box(model, row[1]), box(model, row[2])
        inter = (th.maximum(b1[0], b2[0]), th.minimum(b1[1], b2[1]))
        test.assertTrue(contained(inter, d), (form, row))
        for x in samples(*inter, k, gen):
            test.assertTrue(in_box(x, *d))
    elif form == "gci1_bot":
        test.assertTrue(disjoint(box(model, row[0]), box(model, row[1])), (form, row))
    elif form == "gci2":
        rl, d = role(model, row[1]), box(model, row[2])
        for x in samples(*box(model, row[0]), k, gen):
            test.assertTrue(has_witness(x, rl, d), (form, row))
    elif form == "gci3":
        rl, cb, d = role(model, row[0]), box(model, row[1]), box(model, row[2])
        inter = (th.maximum(cb[0], rl[0]), th.minimum(cb[1], rl[1]))
        for y in samples(*inter, k, gen):
            for x in samples(rl[4] * y + rl[5] + rl[2], rl[4] * y + rl[5] + rl[3], 3, gen):
                test.assertTrue(in_box(x, *d), (form, row))
    elif form == "gci3_bot":
        rl, cb = role(model, row[0]), box(model, row[1])
        test.assertTrue(empty(*cb) or disjoint((rl[0], rl[1]), cb), (form, row))
    elif form == "role_inclusion":
        s = role(model, row[1])
        for x, y in role_pairs(role(model, row[0]), k, gen):
            test.assertTrue(in_role(x, y, s), (form, row))
    elif form == "role_chain":
        s = role(model, row[2])
        pairs = chain_pairs(role(model, row[0]), role(model, row[1]), k, gen)
        for x, z in pairs:
            test.assertTrue(in_role(x, z, s), (form, row))
    elif form == "class_assertion":
        point = model.module.ind_embedding.weight[row[0]].detach().double()
        test.assertTrue(in_box(point, *box(model, row[1])), (form, row))
    elif form == "object_property_assertion":
        w = model.module.ind_embedding.weight.detach().double()
        test.assertTrue(in_role(w[row[0]], w[row[2]], role(model, row[1])), (form, row))
    else:
        raise AssertionError(f"unexpected normal form {form}")


class TestFixedClasses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.model = ELKEm(Dataset(build_ontology()), embed_dim=4, load_role_axioms=True,
                          model_filepath=os.path.join(cls.tmp, "m.pt"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_top_and_bottom(self):
        cid = self.model.class_index_dict
        bot = {cid[BOT_IRI], cid[NS + "L"], cid[NS + "M"]}
        self.assertEqual(set(self.model._bot_ids), bot)
        self.assertEqual(self.model._top_ids, [cid[TOP_IRI]])
        for i in bot:
            self.assertTrue(empty(*box(self.model, i)))
        lower, upper = box(self.model, cid[TOP_IRI])
        self.assertTrue(bool((lower < -1e5).all() and (upper > 1e5).all()))
        # every other class box is non-empty
        for iri, i in cid.items():
            if i not in bot:
                self.assertFalse(empty(*box(self.model, i)), iri)

    def test_dropped_and_rerouted_axioms(self):
        tensors = self.model._training_tensors()
        cid = self.model.class_index_dict
        # M ⊑ L has an empty subclass and is not trained
        self.assertNotIn([cid[NS + "M"], cid[NS + "L"]], tensors["gci0"].tolist())
        self.assertIn("role_inclusion", tensors)
        self.assertIn("role_chain", tensors)

    def test_negative_sampling(self):
        cid = self.model.class_index_dict
        excluded = set(self.model._bot_ids) | set(self.model._top_ids)
        pos = self.model._training_tensors()["gci0"]
        neg = self.model.generate_negatives("gci0", pos)
        self.assertEqual(neg.shape, (self.model.k_neg * len(pos), 2))
        rep = pos.repeat_interleave(self.model.k_neg, dim=0)
        self.assertTrue(th.equal(neg[:, 0], rep[:, 0]))
        self.assertTrue(bool((neg[:, 1] != rep[:, 1]).all()))
        self.assertTrue(bool((neg[:, 1] != rep[:, 0]).all()))
        self.assertFalse(set(neg[:, 1].tolist()) & excluded)

        pos = self.model._training_tensors()["gci2"]
        neg = self.model.generate_negatives("gci2", pos)
        rep = pos.repeat_interleave(self.model.k_neg, dim=0)
        self.assertTrue(th.equal(neg[:, 1], rep[:, 1]))
        changed_c = neg[:, 0] != rep[:, 0]
        changed_d = neg[:, 2] != rep[:, 2]
        self.assertTrue(bool((changed_c ^ changed_d).all()))   # exactly one side corrupted
        self.assertTrue(set(cid.values()) >= set(neg[:, [0, 2]].flatten().tolist()))

    def test_contrastive_only_without_assertions(self):
        self.assertTrue(self.model.contrastive_active())


def total_margin_loss(model):
    model.module.train()
    with th.no_grad():
        return sum(model.module(data, form).sum().item()
                   for form, data in model._training_tensors().items())


@pytest.mark.slow
class TestSoundness(TestCase):
    """Theorem 6 and Corollary 7 of [singh2026] on a trained zero-loss embedding.

    Training to zero loss takes about 400 epochs (10 s), hence the slow mark.
    """

    @classmethod
    def setUpClass(cls):
        th.manual_seed(0)
        np.random.seed(0)
        cls.tmp = tempfile.mkdtemp()
        cls.ontology = build_ontology(with_abox=True)
        cls.model = ELKEm(Dataset(cls.ontology), embed_dim=8, batch_size=64,
                          load_role_axioms=True, reg_factor=0.0, use_contrastive=False,
                          abox_neg_weight=0.0, model_filepath=os.path.join(cls.tmp, "m.pt"))
        for _ in range(30):
            cls.model.train(50)
            if total_margin_loss(cls.model) == 0.0:
                break
        cls.model.module.train()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_zero_loss_reached(self):
        """Precondition of Theorem 6: every axiom has zero loss (with margin)."""
        with th.no_grad():
            for form, data in self.model._training_tensors().items():
                self.assertEqual(self.model.module(data, form).max().item(), 0.0, form)

    def test_theorem_6_every_axiom_holds(self):
        """Every training axiom (including those not trained because a concept is fixed
        empty) holds in the embedding, checked by sampling the relational semantics."""
        gen = th.Generator().manual_seed(0)
        forms = set()
        for form, dataset in self.model.training_datasets.items():
            for row in dataset[:].tolist():
                check_axiom(self, self.model, form, row, gen)
                forms.add(form)
        self.assertTrue({"gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                         "role_inclusion", "role_chain", "class_assertion",
                         "object_property_assertion"} <= forms)

    def test_corollary_7_derived_axioms_hold(self):
        """Every atomic subsumption and every C ⊑ ∃r.D edge derived by the ELK calculus
        (mowl.geometric.completion) holds as a box containment / by sampling."""
        cm = CanonicalModel.from_ontology(self.ontology)
        comp = cm.completion
        cid = self.model.class_index_dict
        pid = self.model.object_property_index_dict
        derived = [(a, b) for a, b in comp.all_subsumptions() if a in cid and b in cid]
        names = {(a.rsplit("/", 1)[-1], b.rsplit("/", 1)[-1]) for a, b in derived}
        for expected in [("A", "F"), ("A", "G"), ("A", "H"), ("A", "C"),
                         ("M", "owl#Nothing")]:
            self.assertIn(expected, names)
        for a, b in derived:
            self.assertTrue(contained(box(self.model, cid[a]), box(self.model, cid[b])),
                            (a, b))
        gen = th.Generator().manual_seed(1)
        n_edges = 0
        for a, edges in comp.edges.items():
            for r, d in edges:
                if a in cid and d in cid and r in pid:
                    n_edges += 1
                    check_axiom(self, self.model, "gci2", [cid[a], pid[r], cid[d]], gen)
        self.assertGreater(n_edges, 3)


class TestTraining(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_end_to_end_with_validation(self):
        th.manual_seed(0)
        ont = build_ontology()
        model = ELKEm(Dataset(ont, validation=ont), embed_dim=8, batch_size=4,
                      k_neg=2, lr_decay_epoch=20, load_role_axioms=True,
                      model_filepath=os.path.join(self.tmp, "e2e.pt"))
        th.manual_seed(1)
        before = model.training_loss(model._training_tensors()).item()
        model.train(40, validate_every=10)
        th.manual_seed(1)
        after = model.training_loss(model._training_tensors()).item()
        self.assertLess(after, before)
        self.assertIsNotNone(model.best_validation_median_rank)
        self.assertTrue(os.path.exists(model.model_filepath))
        model.load_best_model()
        classes, roles, inds = model.get_embeddings()
        self.assertEqual(len(next(iter(classes.values()))), 16)
        self.assertEqual(len(next(iter(roles.values()))), 48)
        # scoring of an OWL axiom: exp(-ℓ0) in (0, 1]
        f = OWLAPIAdapter().data_factory
        axiom = f.getOWLSubClassOfAxiom(f.getOWLClass(IRI.create(NS + "A")),
                                        f.getOWLClass(IRI.create(NS + "B")))
        s = model.score(axiom).item()
        self.assertTrue(0 < s <= 1)

    @pytest.mark.slow
    def test_trains_on_family(self):
        from tests.datasetFactory import FamilyDataset
        model = ELKEm(FamilyDataset(), embed_dim=10, model_filepath=os.path.join(
            self.tmp, "family.pt"))
        model.train(20)
        for p in model.module.parameters():
            self.assertTrue(bool(th.isfinite(p).all()))

    def test_lexical_regulariser_decreases(self):
        th.manual_seed(0)
        ont = build_ontology(with_abox=True)
        ds = Dataset(ont)
        n = len(ds.individuals)
        features = np.random.default_rng(0).normal(size=(n, 6))
        model = ELKEm(ds, embed_dim=8, lexical_features=features, lexical_weight=10.0,
                      lexical_batch_size=n, model_filepath=os.path.join(self.tmp, "lex.pt"))
        self.assertFalse(model.contrastive_active())
        start = model.lexical_regularization_loss().item()
        model.train(100)
        self.assertLess(model.lexical_regularization_loss().item(), start)
        # dictionary input
        iris = list(model.individual_index_dict)
        model2 = ELKEm(ds, embed_dim=8, lexical_features={iris[0]: features[0]},
                       model_filepath=os.path.join(self.tmp, "lex2.pt"))
        self.assertEqual(model2._lexical_idx.tolist(), [model.individual_index_dict[iris[0]]])

    def test_assertion_negatives_are_filtered(self):
        ont = build_ontology(with_abox=True)
        model = ELKEm(Dataset(ont), embed_dim=4, abox_k_neg=5,
                      model_filepath=os.path.join(self.tmp, "abox.pt"))
        pos = model._training_tensors()["class_assertion"]
        observed = set(map(tuple, pos.tolist()))
        for _ in range(5):
            neg = model.generate_negatives("class_assertion", pos)
            self.assertEqual(len(neg), 5 * len(pos))
            self.assertFalse(set(map(tuple, neg.tolist())) & observed)


def test_model_passes_the_variant_to_its_module():
    ont = build_ontology()
    assert ELKEm(Dataset(ont), embed_dim=4).module.variant == "paper"
    assert ELKEm(Dataset(ont), embed_dim=4, variant="reference").module.variant == "reference"
    with pytest.raises(ValueError):
        ELKEm(Dataset(ont), embed_dim=4, variant="other")
