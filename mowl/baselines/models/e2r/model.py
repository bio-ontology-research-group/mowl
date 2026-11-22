r"""The direct implementation of E2R: its own loss assembly and Adam loop in the model class.
Kept for the comparison in ``tests/baselines``; the extraction of the axioms and the scoring are
in the frozen base class (:mod:`mowl.baselines.models.e2r.base`)."""
import math

import torch as th

from mowl.baselines.nn.alc.e2r import losses as L
from mowl.baselines.nn.alc.e2r.module import AXIOM_KINDS, E2RModule
from mowl.baselines.models.e2r.base import E2RBase, _THING, _NOTHING  # noqa: F401


class E2R(E2RBase):
    """E2R with the loss and the loop written out (see :class:`mowl.models.E2R` for the
    parameters)."""

    _algebra = L

    def init_module(self):
        self.module = E2RModule(
            len(self._iid), len(self._cid), len(self._rid), dim=self.embed_dim,
            top_index=self._cid.get(_THING), bottom_index=self._cid.get(_NOTHING),
            role_unit_length=self.role_unit_length).to(self.device)

    # -- training ------------------------------------------------------------
    def _batch(self, rows):
        n = rows.shape[0]
        if n <= self.batch_size:
            return rows
        return rows[th.randint(n, (self.batch_size,), device=rows.device)]

    def _negatives(self, batch, kind):
        k = self.num_negatives
        n_ind = len(self._iid)
        neg = batch.repeat(k, 1)
        rand = th.randint(n_ind, (neg.shape[0],), device=neg.device)
        if kind == "class_assertion":
            neg[:, 0] = rand
            known = self._known_ca
        else:
            col = th.randint(2, (neg.shape[0],), device=neg.device) * 2  # 0 or 2
            neg[th.arange(neg.shape[0]), col] = rand
            known = self._known_ra
        keep = [tuple(r) not in known for r in neg.tolist()]
        return neg[th.tensor(keep, dtype=th.bool, device=neg.device)]

    def compute_loss(self, full=False):
        """The E2R objective (eq. (10) with regularisers and negatives) on one random
        batch per axiom kind (``full=True``: on all rows)."""
        m = self.module
        loss = m.structure_loss(self._class_idx)
        for key, rows in self.training_rows.items():
            kind = self._kind(key)
            batch = rows if full else self._batch(rows)
            loss = loss + m(kind, batch).mean()
            use_neg = min(self.num_negatives, self.negative_weight, len(self._iid) - 1) > 0
            if kind in ("class_assertion", "role_assertion") and use_neg:
                neg = self._negatives(batch, kind)
                if len(neg) > 0:
                    loss = loss + self.negative_weight * m(kind, neg, neg=True).mean()
        if self.reg_orth > 0:
            for pairs, rel in ((self.sibling_pairs, False), (self.role_sibling_pairs, True)):
                if len(pairs) > 0:
                    loss = loss + self.reg_orth * m.orthogonality_loss(pairs, rel).mean()
        if self.reg_sep > 0:
            for pairs, rel in ((self.separation_pairs, False),
                               (self.role_separation_pairs, True)):
                if len(pairs) > 0:
                    loss = loss + self.reg_sep * m.separation_loss(
                        pairs, self.sep_gap, rel).mean()
        return loss

    def train(self, epochs=100, verbose=False):
        """Adam on the E2R objective; indicators are clipped to :math:`[0, 1]` after each
        step. An epoch has :math:`\\max_k \\lceil N_k / \\text{batch size} \\rceil` steps.

        :param epochs: Number of epochs.
        :param verbose: Print the loss of every epoch.
        :returns: List of the mean loss of each epoch.
        """
        self.module.train()
        opt = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        sizes = [len(r) for r in self.training_rows.values()] or [1]
        steps = max(1, max(math.ceil(n / self.batch_size) for n in sizes))
        history = []
        for epoch in range(epochs):
            total = 0.0
            for _ in range(steps):
                loss = self.compute_loss()
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.module.project_()
                total += loss.item()
            history.append(total / steps)
            if verbose:
                print(f"epoch {epoch}: {history[-1]:.6f}")
        if self._model_filepath is not None:
            th.save(self.module.state_dict(), self._model_filepath)
        return history

