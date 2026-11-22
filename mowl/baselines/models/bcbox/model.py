"""The direct implementation of binary code box embeddings: its own loss and Adam loop in the
model class. Kept for the comparison in ``tests/baselines``; the graph, the scoring and the
evaluation are shared (:mod:`mowl.models.bcbox.common`)."""

import torch as th

from mowl.baselines.nn.graph.bcbox.losses import bce_negative_sampling_loss
from mowl.baselines.nn.graph.bcbox.module import BinaryCodeBoxModule
from mowl.models.bcbox.common import BinaryCodeBoxBase


class BinaryCodeBox(BinaryCodeBoxBase):
    """Binary code box embeddings with the loss and the loop written out (see
    :class:`mowl.models.BinaryCodeBox` for the parameters)."""

    # ----------------------------------------------------------------- module
    def init_module(self):
        """Create the :class:`~mowl.nn.graph.bcbox.BinaryCodeBoxModule`."""
        self._module = BinaryCodeBoxModule(
            self.num_nodes, self.embed_dim, variant=self.variant, num_codes=self.num_codes,
            code_dims=self.code_dims, intersection_temp=self.intersection_temp,
            volume_temp=self.volume_temp, code_init=self.code_init).to(self.device)

    def sample_negatives(self, pos):
        """Corrupt the source or the target of every positive edge ``num_negs`` times.

        :param pos: Positive edges (paper direction), shape ``(b, 2)``.
        :type pos: :class:`torch.Tensor`
        :returns: ``(neg, mask)`` with ``neg`` of shape ``(b, num_negs, 2)`` and a boolean
            ``mask`` that is ``False`` where the corruption is a training edge or a self-loop.
        :rtype: tuple
        """
        b, n = pos.shape[0], self.num_nodes
        neg = pos.unsqueeze(1).repeat(1, self.num_negs, 1)
        column = th.randint(0, 2, (b, self.num_negs), device=pos.device)
        random_nodes = th.randint(0, n, (b, self.num_negs), device=pos.device)
        neg[..., 0] = th.where(column == 0, random_nodes, neg[..., 0])
        neg[..., 1] = th.where(column == 1, random_nodes, neg[..., 1])
        keys = neg[..., 0] * n + neg[..., 1]
        known = self.edge_index.to(pos.device)
        known_keys = known[:, 0] * n + known[:, 1]
        mask = ~th.isin(keys, known_keys) & (neg[..., 0] != neg[..., 1])
        return neg, mask

    def compute_loss(self, pos):
        """Loss of Eq. (10) plus the code lasso for one minibatch of positive edges."""
        neg, mask = self.sample_negatives(pos)
        pos_energy = self.module(pos)
        neg_energy = self.module(neg)
        loss = bce_negative_sampling_loss(
            pos_energy, neg_energy, neg_mask=mask, self_adversarial=self.self_adversarial,
            adversarial_temperature=self.adversarial_temperature).mean()
        if self.code_reg > 0:
            loss = loss + self.code_reg * self.module.code_regularization()
        return loss

    def train(self, epochs=100, verbose=False):
        """Train with Adam on minibatches of positive edges.

        :param epochs: Number of passes over the edges.
        :type epochs: int
        :param verbose: Print the loss of every epoch.
        :type verbose: bool
        :returns: The mean training loss of every epoch.
        :rtype: list of float
        """
        module = self.module
        edges = self.edge_index.to(self.device)
        optimizer = th.optim.Adam(module.parameters(), lr=self.learning_rate)
        history = []
        module.train()
        for epoch in range(epochs):
            perm = th.randperm(len(edges), device=self.device)
            total, batches = 0.0, 0
            for start in range(0, len(edges), self.batch_size):
                pos = edges[perm[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss = self.compute_loss(pos)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            history.append(total / batches)
            if verbose:
                print(f"Epoch {epoch + 1}: loss {history[-1]:.4f}")
        module.eval()
        th.save(module.state_dict(), self.model_filepath)
        return history

