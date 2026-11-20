"""The loss functions of the DELE reference code (``dele-code-losses``) and the
``dele-code-training`` composition: what they change, and that the modules' default
losses are untouched."""

from unittest import TestCase

import numpy as np
import torch as th

import mowl.nn.el.elem.losses as EL
import mowl.nn.el.elbe.losses as LB
from mowl.models import BoxSquaredEL, ELBE, ELEmbeddings
from mowl.nn import ELBEModule, ELEmModule
from mowl.training import AllNormalFormNegatives, DELECodeLosses, DELECodeTraining
from tests.datasetFactory import FamilyDataset


class TestReferenceLossFunctions(TestCase):
    """The *_reference / *_dele functions against their geometric zero conditions."""

    N, R, ROWS = 10, 3, 4000

    def _elem(self):
        g = th.Generator().manual_seed(2)
        m = ELEmModule(self.N, self.R, 0, embed_dim=2, margin=0.05)
        for p in m.parameters():
            p.data = (th.rand(p.shape, generator=g) * 2 - 1) * 2
        m.class_rad.weight.data = th.rand_like(m.class_rad.weight.data) * 0.6
        return m

    def _elbe(self):
        g = th.Generator().manual_seed(3)
        m = ELBEModule(self.N, self.R, embed_dim=2, margin=0.05)
        for p in m.parameters():
            p.data = (th.rand(p.shape, generator=g) * 2 - 1) * 2
        m.class_offset.weight.data = th.rand_like(m.class_offset.weight.data) * 0.6
        return m

    @staticmethod
    def _rows(columns, n, r):
        g = th.Generator().manual_seed(4)
        cols = [th.randint(0, n if x == "c" else r, (4000,), generator=g) for x in columns]
        return th.stack(cols, dim=1)

    def test_elem_gci3_reference(self):
        # zero iff the translated ball of C lies inside the ball of D (up to margin)
        m = self._elem()
        d = self._rows("rcc", self.N, self.R)
        loss = EL.gci3_loss_reference(d, m.class_embed, m.class_rad, m.rel_embed,
                                      m.margin).flatten().detach().numpy()
        r = m.rel_embed(d[:, 0])
        c = m.class_embed(d[:, 1])
        e = m.class_embed(d[:, 2])
        rc = m.class_rad(d[:, 1]).abs().flatten()
        rd = m.class_rad(d[:, 2]).abs().flatten()
        inside = ((c - r - e).norm(dim=1) + rc - rd - m.margin) <= 1e-7
        np.testing.assert_array_equal(loss <= 1e-6, inside.numpy())

    def test_elbe_gci0_dele(self):
        # slack margin: the box of C may stick out of the box of D by the margin
        m = self._elbe()
        d = self._rows("cc", self.N, self.R)
        loss = LB.gci0_loss_dele(d, m.class_embed, m.class_offset,
                                 m.margin).flatten().detach().numpy()
        c = m.class_embed(d).detach().numpy()
        o = m.class_offset.weight.abs().detach().numpy()[d]
        zero = ((np.abs(c[:, 0] - c[:, 1]) + o[:, 0] - o[:, 1] - m.margin) <= 1e-7).all(axis=1)
        np.testing.assert_array_equal(loss <= 1e-6, zero)

    def test_elbe_gci2_neg_dele(self):
        # zero iff the translated box of C and the box of D are separated by the margin
        m = self._elbe()
        d = self._rows("crc", self.N, self.R)
        loss = LB.gci2_loss_dele(d, m.class_embed, m.class_offset, m.rel_embed,
                                 m.margin, neg=True).flatten().detach().numpy()
        r = m.rel_embed(d[:, 1])
        c = m.class_embed(d[:, 0])
        e = m.class_embed(d[:, 2])
        oc = m.class_offset(d[:, 0]).abs()
        od = m.class_offset(d[:, 2]).abs()
        apart = ((c + r - e).abs() - oc - od - m.margin >= -1e-7).all(dim=1)
        np.testing.assert_array_equal(loss <= 1e-6, apart.numpy())

    def test_elbe_gci3_dele(self):
        # corrected: zero iff the translated box of C lies inside the box of D
        m = self._elbe()
        d = self._rows("rcc", self.N, self.R)
        loss = LB.gci3_loss_dele(d, m.class_embed, m.class_offset, m.rel_embed,
                                 m.margin).flatten().detach().numpy()
        r = m.rel_embed(d[:, 0]).detach().numpy()
        c = m.class_embed(d[:, 1]).detach().numpy()
        e = m.class_embed(d[:, 2]).detach().numpy()
        o = m.class_offset.weight.abs().detach().numpy()
        zero = ((np.abs(c - r - e) + o[d[:, 1]] - o[d[:, 2]] - m.margin)
                <= 1e-7).all(axis=1)
        np.testing.assert_array_equal(loss <= 1e-6, zero)


class TestDELECodeLosses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def test_switches_elem_gci3(self):
        model = ELEmbeddings(self.dataset, embed_dim=4, batch_size=64)
        g = th.Generator().manual_seed(5)
        model.module.class_rad.weight.data = th.rand_like(
            model.module.class_rad.weight.data) * 0.5
        before = model.module(th.tensor([[0, 1, 2]]), "gci3").detach().clone()
        model.training_regime = DELECodeLosses()
        after = model.module(th.tensor([[0, 1, 2]]), "gci3").detach()
        expected = EL.gci3_loss_reference(
            th.tensor([[0, 1, 2]]), model.module.class_embed, model.module.class_rad,
            model.module.rel_embed, model.module.margin)
        self.assertTrue(th.equal(after, expected))
        # the corrected loss adds the filler radius; they differ unless the radius is 0
        self.assertFalse(th.equal(before, after))

    def test_switches_elbe(self):
        model = ELBE(self.dataset, embed_dim=4, batch_size=64)
        self.assertEqual(model.module.margin_semantics, "strict")
        model.training_regime = DELECodeLosses()
        self.assertEqual(model.module.margin_semantics, "slack")
        d = th.tensor([[0, 1]])
        expected = LB.gci0_loss_dele(d, model.module.class_embed, model.module.class_offset,
                                     model.module.margin)
        self.assertTrue(th.equal(model.module(d, "gci0"), expected))
        # gci2 negatives switch to the separation form of the reference code
        d = th.tensor([[0, 0, 1]])
        neg = model.module(d, "gci2", neg=True)
        self.assertTrue(th.equal(
            neg, LB.gci2_score_neg_dele(
                model.module.class_embed(d[:, 0]), th.abs(model.module.class_offset(d[:, 0])),
                model.module.class_embed(d[:, 2]), th.abs(model.module.class_offset(d[:, 2])),
                model.module.rel_embed(d[:, 1]), model.module.margin)))

    def test_box2el_unchanged(self):
        model = BoxSquaredEL(self.dataset, embed_dim=4, batch_size=64)
        d = th.tensor([[0, 1], [2, 3]])
        before = model.module(d, "gci0").detach().clone()
        model.training_regime = DELECodeLosses()
        self.assertTrue(th.equal(model.module(d, "gci0"), before))

    def test_rejects_other_modules(self):
        from mowl.models import BoxEL
        model = BoxEL(self.dataset, embed_dim=4)
        with self.assertRaises(TypeError):
            model.training_regime = DELECodeLosses()


class TestDELECodeTraining(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _train(self, cls_, **kwargs):
        th.manual_seed(0)
        np.random.seed(0)
        model = cls_(self.dataset, embed_dim=4, batch_size=64)
        model.eval_gci_name = "gci0"
        model.training_regime = DELECodeTraining(batch_size=32, **kwargs)
        # FamilyDataset has no validation split; feed the plateau a constant metric
        plateau = model.training_regime.regimes[-1]
        values = iter([1.0] * 100)
        plateau.metric = lambda m: next(values)
        model.train(epochs=4)
        for p in model.module.parameters():
            self.assertTrue(th.isfinite(p).all())
        return model

    def test_composition_l(self):
        model = self._train(ELEmbeddings, negatives="all", epsilon=0.01)
        parts = model.training_regime.regimes
        self.assertTrue(model.module.all_negative_losses)
        self.assertEqual(parts[0].key, "dele-code-losses")
        keys = [p.key for p in parts]
        self.assertIn("per-normal-form-minibatch", keys)
        self.assertIn("dele-all-normal-form-negatives", keys)
        self.assertNotIn("dele-closure-filtered-negatives", keys)

    def test_composition_baseline(self):
        model = self._train(ELBE, negatives="gci2")
        keys = [p.key for p in model.training_regime.regimes]
        self.assertNotIn("dele-all-normal-form-negatives", keys)
        self.assertFalse(model.module.all_negative_losses)

    def test_composition_ln_and_box2el(self):
        model = self._train(BoxSquaredEL, negatives="all", filter_negatives=True,
                            epsilon=0.01)
        keys = [p.key for p in model.training_regime.regimes]
        self.assertIn("dele-closure-filtered-negatives", keys)
        self.assertEqual(len(keys), 5)

    def test_kind_prefixes(self):
        model = ELEmbeddings(self.dataset, embed_dim=4, batch_size=64)
        model.training_regime = AllNormalFormNegatives(kind_prefixes=("Father", "Mother"),
                                                      override_model_sampler=True)
        named = model.named_class_index_dict
        match = {i for iri, i in named.items() if "Father" in iri or "Mother" in iri}
        ds = model.training_datasets["gci0"]
        negs = model.training_regime.negatives("gci0", ds)
        if negs is None or len(negs) == 0:
            self.skipTest("no gci0 axioms")
        for orig, new in zip(ds[:][:, 1].tolist(), negs[:, 1].tolist()):
            self.assertEqual(orig in match, new in match,
                            "a corrupted class changed its kind")
        # the corrupting class is drawn from the kind pool only
        self.assertTrue(set(negs[:, 1].tolist()) <= set(named.values()))
