"""The readings of the direct FALCON module (mowl.baselines.nn.alc.falcon): quantifiers over all
pairs (as in the original code), a constant owl:Nothing, and the earlier readings."""
from unittest import TestCase

import torch as th

from mowl.baselines.models.falcon.model import FALCONModel
from mowl.baselines.nn.alc.falcon.module import FALCONModule
from tests.datasetFactory import FamilyDataset


class TestFALCONReadings(TestCase):
    """Quantifiers over all pairs (as in the original code) and constant owl:Nothing; the
    earlier readings stay available."""

    @classmethod
    def setUpClass(cls):
        from mowl.owlapi import OWLAPIAdapter
        ad = OWLAPIAdapter()
        cls.ad = ad
        cls.a = ad.create_class("http://x#A")
        cls.r = ad.create_object_property("http://x#r")
        cls.nothing = ad.create_class("http://www.w3.org/2002/07/owl#Nothing")

    def _module(self, **kw):
        th.manual_seed(0)
        return FALCONModule(3, 5, 1, {}, {}, embed_dim=6, **kw)

    def test_exists_differs_between_entities(self):
        """Over all pairs the degree of ∃r.A depends on the entity; with the self pair it
        is one value for all entities."""
        expr = self.ad.create_object_some_values_from(self.r, self.a)
        x = th.tensor([[0, 0]])
        e_emb = th.randn(5, 6)
        pair = self._module()
        fs, _ = pair.forward_fs(expr, x, e_emb)
        self.assertEqual(fs.shape, (1, 5))
        self.assertGreater(float(fs.detach().std()), 1e-6)
        selfp = self._module(role_pairs='self')
        fs, _ = selfp.forward_fs(expr, x, e_emb)
        self.assertLess(float(fs.detach().std()), 1e-7)

    def test_exists_forall_values(self):
        """Hand check of the pair table for the product t-norm."""
        mod = self._module()
        e_emb = th.randn(5, 6)
        x = th.tensor([[0, 0]])
        r = mod.r_embedding(x[:, 0])
        table = th.stack([mod._mem((e_emb[i] + r[0]).expand_as(e_emb), e_emb).squeeze(-1)
                          for i in range(5)])
        c = mod._get_c_fs_batch(mod.c_embedding(x[:, 1]), e_emb)[0]
        ex, _ = mod.forward_fs(self.ad.create_object_some_values_from(self.r, self.a),
                               x, e_emb)
        fa, _ = mod.forward_fs(self.ad.create_object_all_values_from(self.r, self.a),
                               x, e_emb)
        self.assertTrue(th.allclose(ex[0], (table * c).max(dim=-1)[0], atol=1e-6))
        self.assertTrue(th.allclose(fa[0], ((1 - table) + c - (1 - table) * c).min(dim=-1)[0],
                                    atol=1e-6))

    def test_self_pair_reproduces_old_numbers(self):
        """role_pairs='self' is the earlier formula: max_y mu(e_y + r, e_y) * C(y)."""
        mod = self._module(role_pairs='self')
        e_emb = th.randn(5, 6)
        x = th.tensor([[0, 0]])
        r = mod.r_embedding(x[:, 0])
        diag = mod._mem(e_emb + r, e_emb).squeeze(-1)
        c = mod._get_c_fs_batch(mod.c_embedding(x[:, 1]), e_emb)[0]
        ex, _ = mod.forward_fs(self.ad.create_object_some_values_from(self.r, self.a),
                               x, e_emb)
        self.assertTrue(th.allclose(ex[0], (diag * c).max().expand(5), atol=1e-6))

    def test_minmax_and_lukasiewicz_batches(self):
        expr = self.ad.create_object_all_values_from(self.r, self.a)
        for t_norm in ('minmax', 'Łukasiewicz'):
            mod = self._module(t_norm=t_norm)
            fs, _ = mod.forward_fs(expr, th.tensor([[0, 0], [0, 1], [0, 2]]), th.randn(5, 6))
            self.assertEqual(fs.shape, (3, 5))

    def test_nothing_is_empty(self):
        x = th.tensor([[2]])
        e_emb = th.randn(5, 6)
        mod = self._module(bottom_index=2)
        fs, _ = mod.forward_fs(self.nothing, x, e_emb)
        self.assertTrue(th.all(fs == 0))
        learned = self._module(bottom_index=2, nothing='learned')
        fs, _ = learned.forward_fs(self.nothing, x, e_emb)
        self.assertTrue(th.all(fs > 0))

    def test_thing_option(self):
        x = th.tensor([[1]])
        e_emb = th.randn(5, 6)
        fs, _ = self._module(top_index=1).forward_fs(self.a, x, e_emb)
        self.assertTrue(th.all((fs > 0) & (fs < 1)))
        fs, _ = self._module(top_index=1, thing='all').forward_fs(self.a, x, e_emb)
        self.assertTrue(th.all(fs == 1))

    def test_model_options_reach_module(self):
        model = FALCONModel(FamilyDataset(), embed_dim=6, role_pairs='self',
                            thing='all', nothing='learned')
        self.assertEqual(model.module.role_pairs, 'self')
        self.assertEqual(model.module.thing, 'all')
        self.assertEqual(model.module.nothing_reading, 'learned')
        self.assertIsNone(model.module.bottom_index)
        with self.assertRaises(ValueError):
            FALCONModule(3, 5, 1, {}, {}, role_pairs='diagonal')
