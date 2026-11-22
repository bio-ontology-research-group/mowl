"""The direct implementation of OnT: its own loss assembly on top of the loop of the direct
HiT. Kept for the comparison in ``tests/baselines``; the verbalised axioms, the triplets and
the scoring are in the frozen base class (:mod:`mowl.baselines.models.ont.base`)."""

import logging

import torch as th

from mowl.baselines.models.hit.model import HiT, DEFAULT_CENTRI_GRID  # noqa: F401
from mowl.baselines.models.ont.base import OnTBase

logger = logging.getLogger(__name__)


class OnT(OnTBase, HiT):
    """OnT with the loss written out (see :class:`mowl.models.OnT` for the parameters)."""

    def init_module(self):
        from mowl.baselines.nn.lm.ont import OnTModule
        return OnTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight,
                         use_roles=self.use_roles,
                         conjunction_weight=self.conjunction_weight,
                         existential_weight=self.existential_weight,
                         role_pairing=self.role_pairing, role_scaling=self.role_scaling)

    # ------------------------------------------------------------------ training
    def _sample(self, items, n):
        if len(items) <= n:
            return list(items)
        return self.rng.sample(items, n)

    def batch_loss(self, batch, negatives):
        """Mean of :math:`L_\\prec` over the batch plus, if roles are used,
        ``existential_weight`` times the mean :math:`L_r` and ``conjunction_weight``
        times the mean :math:`L_\\sqcap` of random samples of the size of the batch."""
        m = self.module
        child = m.embed([t[0] for t in batch])
        parent = m.embed([t[1] for t in batch])
        hier = m.hierarchy_loss(child, parent, negatives).mean()
        parts = {"hierarchy": float(hier.detach())}
        loss = hier
        n = len(batch)
        if self.use_roles and self.existentials:
            ex = self._sample(self.existentials, n)
            k = len(ex)
            neg1 = negatives[th.randperm(n, device=negatives.device)[:k]]
            neg2 = negatives[th.randperm(n, device=negatives.device)[:k]]
            l_r = m.existential_loss([e[0] for e in ex], [e[1] for e in ex],
                                     [e[2] for e in ex], neg1, neg2).mean()
            loss = loss + self.existential_weight * l_r
            parts["existential"] = float(l_r.detach())
        if self.use_roles and self.conjunctions:
            cj = self._sample(self.conjunctions, n)
            conj_texts, part_texts, weights = [], [], []
            for text, ops in cj:
                for o in ops:
                    conj_texts.append(text)
                    part_texts.append(o)
                    weights.append(1.0 / len(ops))
            idx = th.randint(0, n, (len(conj_texts),), device=negatives.device)
            rows = m.hierarchy_loss(m.embed(conj_texts), m.embed(part_texts), negatives[idx])
            w = th.tensor(weights, device=rows.device, dtype=rows.dtype)
            l_c = (w * rows).sum() / len(cj)
            loss = loss + self.conjunction_weight * l_c
            parts["conjunction"] = float(l_c.detach())
        return loss, parts

