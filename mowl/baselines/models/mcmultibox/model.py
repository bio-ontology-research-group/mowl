"""The direct implementation of the volume-based MultiBoxEL model class (the module of
:mod:`mowl.baselines.nn.el.mcmultibox` and its own training loop). The options, the trainable
datasets and the evaluation are those of :class:`mowl.models.VolumeMultiBoxEL`
(:class:`~mowl.models.mcmultibox.common.VolumeMultiBoxELOptions`)."""
import logging

import torch as th

from mowl.baselines.nn.el.mcmultibox.module import VolumeMultiBoxELModule
from mowl.models.mcmultibox.common import VolumeMultiBoxELOptions

logger = logging.getLogger(__name__)


class VolumeMultiBoxEL(VolumeMultiBoxELOptions):
    """The volume-based MultiBoxEL with the direct module (see
    :class:`mowl.models.VolumeMultiBoxEL`)."""

    #: :meth:`train` is a loop of its own with a fixed objective; it takes no training regime.
    training_hooks = frozenset()

    module_class = VolumeMultiBoxELModule

    @staticmethod
    def _canonical_order(data):
        """Sort rows lexicographically: the axiom order produced by the normaliser is not
        stable between runs, and sorting makes training reproducible for a fixed seed."""
        for col in reversed(range(data.shape[1])):
            data = data[th.sort(data[:, col], stable=True).indices]
        return data

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train with ``num_batches`` optimiser steps per epoch.

        :param epochs: Number of epochs (thesis: 5).
        :param validate_every: Validate every this many epochs on ``eval_gci_name`` of the
            validation set if one exists; the best state is kept in ``model_filepath``.
        :param epoch_callback: Optional ``callable(epoch, model)`` called after each epoch.
        """
        datasets = self._trainable_datasets()
        if not datasets:
            raise ValueError("No axioms with a loss defined by the thesis were found.")
        optimizer = self.get_optimizer()
        gen = th.Generator().manual_seed(self.seed)
        best_valid = float("inf")
        self.training_losses = []

        for epoch in range(epochs):
            self.module.train()
            full = {name: self._canonical_order(ds[:]) for name, ds in datasets.items()}
            splits = {name: th.randperm(len(ds), generator=gen).chunk(self.num_batches)
                      for name, ds in datasets.items()}
            epoch_loss = 0.0
            for b in range(self.num_batches):
                loss = th.tensor(0.0, device=self.device)
                for name in datasets:
                    if b >= len(splits[name]):
                        continue
                    data = full[name][splits[name][b].to(full[name].device)]
                    loss = loss + th.mean(self.module(data, name))
                if not loss.requires_grad:
                    # Fewer axioms than batches: this batch is empty for every axiom type.
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() / self.num_batches
            self.training_losses.append(epoch_loss)
            msg = f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}"

            valid_loss = None
            if (epoch + 1) % validate_every == 0 and self.dataset.validation is not None \
                    and self._eval_gci_name is not None:
                with th.no_grad():
                    self.module.eval()
                    data = self.validation_datasets[self._eval_gci_name][:]
                    if len(data) > 0:
                        valid_loss = th.mean(self.module(data, self._eval_gci_name)).item()
                        msg += f", valid loss {valid_loss:.4f}"
            if valid_loss is None:
                th.save(self.module.state_dict(), self.model_filepath)
            elif valid_loss < best_valid:
                best_valid = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
            logger.info(msg)
            if epoch_callback is not None:
                epoch_callback(epoch, self)
