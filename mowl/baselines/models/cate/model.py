"""The direct implementation of CatE: its own loss, minibatches, cyclic learning rate and
early stopping in the model class. Kept for the comparison in ``tests/baselines``; the lattice,
the validation queries, the scoring and the evaluation are shared
(:mod:`mowl.baselines.models.cate.base`)."""

import torch as th

from mowl.baselines.nn.graph.cate.losses import order_margin_loss
from mowl.baselines.nn.graph.cate.module import CatEModule
from mowl.baselines.models.cate.base import CatEBase, UnseenConceptError  # noqa: F401

__all__ = ["CatE", "UnseenConceptError"]


class CatE(CatEBase):
    """CatE with the loss and the loop written out (see :class:`mowl.models.CatE` for the
    parameters)."""

    @property
    def training_regime(self):
        """A :class:`~mowl.training.TrainingRegime`, or ``None`` to train with the
        model's own behaviour.

        Assigning a regime calls its ``attach`` method with this model; the
        'cate-code-training' regime switches the model to the training design of the
        CatE reference code. CatE's own :meth:`train` loop consults no regime hooks, so a
        regime that changes negatives, losses or the optimiser is refused.
        """
        return getattr(self, "_training_regime", None)

    @training_regime.setter
    def training_regime(self, regime):
        if regime is not None:
            from mowl.baselines.training.regime import TrainingRegime, check_compatible
            if not isinstance(regime, TrainingRegime):
                raise TypeError("training_regime must be a mowl.training.TrainingRegime "
                                "or None.")
            check_compatible(regime, self)
            regime.attach(self)
        self._training_regime = regime

    # ----------------------------------------------------------------- module
    def init_module(self):
        """Create the :class:`~mowl.nn.graph.cate.CatEModule`."""
        lat = self.lattice
        module = CatEModule(len(lat), self.embed_dim, p=self.p)
        if lat.existential_to_id:
            keys = list(lat.existential_to_id.items())
            module.set_existentials([k[0] for k, _ in keys], [k[1] for k, _ in keys],
                                    [v for _, v in keys])
        self._module = module.to(self.device)

    # --------------------------------------------------------------- training
    def _edge_keys(self):
        if self._keys is None or self._keys.device != th.device(self.device):
            e = self.lattice.edges.to(self.device)
            self._keys = th.sort(e[:, 0] * len(self.lattice) + e[:, 1])[0]
        return self._keys

    def sample_negatives(self, pos):
        """Replace the superconcept of every morphism ``num_negs`` times by an element
        drawn uniformly from the lattice.

        :return: ``(neg_dst, mask)``: corrupted superconcepts of shape ``(b, num_negs)``
            and, if ``filter_negatives``, a mask that is ``False`` where the corruption
            is itself a morphism of the lattice or reflexive (otherwise ``None``).
        """
        n = len(self.lattice)
        neg_dst = th.randint(0, n, (len(pos), self.num_negs), device=pos.device)
        if not self.filter_negatives:
            return neg_dst, None
        keys = pos[:, :1] * n + neg_dst
        mask = ~th.isin(keys, self._edge_keys()) & (neg_dst != pos[:, :1])
        return neg_dst, mask

    def compute_loss(self, pos):
        """Loss of Eq. 7 for a batch of morphisms ``pos`` (shape ``(b, 2)``), plus the
        interaction loss of the reference code if :attr:`interaction_loss` is set."""
        module = self.module
        neg_dst, mask = self.sample_negatives(pos)
        pos_scores = module.order_violation(pos[:, 0], pos[:, 1])
        neg_scores = module.order_violation(
            pos[:, 0].unsqueeze(1).expand(-1, self.num_negs), neg_dst)
        loss = order_margin_loss(pos_scores, neg_scores, self.margin,
                                 negative_aggregation=self.negative_aggregation,
                                 neg_mask=mask)
        if self.interaction_loss:
            loss = loss + self._interaction_loss(pos[:, 0], pos_scores)
        return loss

    def _interaction_loss(self, heads, pos_scores):
        """The PPI term of the reference training loop: mean order violation of the
        protein-headed morphisms of the batch plus a hinge on one corruption per row
        whose tail is drawn uniformly among the elements :math:`\\exists r.p`."""
        head_ids, ex_ids = self._interaction_ids()
        sel = th.isin(heads, head_ids)
        if not sel.any():
            return pos_scores.sum() * 0.0
        neg = ex_ids[th.randint(0, len(ex_ids), (int(sel.sum()),), device=ex_ids.device)]
        s_pos = pos_scores[sel]
        s_neg = self.module.order_violation(heads[sel], neg)
        return s_pos.mean() + th.relu(self.margin - s_neg).mean()

    def train(self, epochs=100, validate_every=None, verbose=False):
        """Train on all morphisms of the lattice.

        :param epochs: Number of passes over the morphisms.
        :type epochs: int
        :param validate_every: If given and the dataset has a validation ontology, compute
            the validation MRR every that many epochs, keep the best parameters and stop
            early after ``early_stopping_patience`` validations without improvement.
        :type validate_every: int, optional
        :param verbose: Print the loss of every epoch.
        :type verbose: bool
        :return: The mean training loss of every epoch.
        :rtype: list of float
        """
        module = self.module
        edges = self.lattice.edges.to(self.device)
        optimizer = th.optim.Adam(module.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        scheduler = None
        if self.cyclic_lr:
            scheduler = th.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.learning_rate / 10, max_lr=self.learning_rate,
                step_size_up=20, cycle_momentum=False)
        validate = validate_every is not None and self.dataset.validation is not None \
            and len(self.validation_queries()) > 0
        best, bad, history = -1.0, 0, []
        self.validation_history = []
        for epoch in range(epochs):
            module.train()
            perm = th.randperm(len(edges), device=self.device)
            total, batches = 0.0, 0
            for start in range(0, len(edges), self.batch_size):
                pos = edges[perm[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss = self.compute_loss(pos)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total += loss.item()
                batches += 1
            history.append(total / max(batches, 1))
            if verbose:
                print(f"Epoch {epoch + 1}: loss {history[-1]:.6f}")
            if validate and ((epoch + 1) % validate_every == 0 or epoch + 1 == epochs):
                mrr = self.validation_mrr()
                self.validation_history.append((epoch + 1, mrr))
                if mrr > best:
                    best, bad = mrr, 0
                    th.save(module.state_dict(), self.model_filepath)
                else:
                    bad += 1
                    if self.early_stopping_patience is not None and \
                            bad >= self.early_stopping_patience:
                        break
        if validate:
            self.load_best_model()
        else:
            th.save(module.state_dict(), self.model_filepath)
        module.eval()
        return history

