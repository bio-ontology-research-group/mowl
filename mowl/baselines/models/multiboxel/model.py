"""The direct implementation of the MultiBoxEL model class (the module of
:mod:`mowl.baselines.nn.el.multiboxel` and its own extra training loss and loop). The options,
the data handling and loading are those of :class:`mowl.models.MultiBoxEL`
(:class:`~mowl.models.multiboxel.common.MultiBoxELOptions`)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th

from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule
from mowl.baselines.nn.el.multiboxel.pointloss import volume_penalty
from mowl.baselines.geometric.core.normal_forms import BOTTOM, TOP
from mowl.models.multiboxel.common import MultiBoxELOptions
from mowl.baselines.nn.el.structural import build_structural_tables


class MultiBoxEL(MultiBoxELOptions):
    """MultiBoxEL with the direct module (see :class:`mowl.models.MultiBoxEL`)."""

    module_class = MultiBoxELModule

    def additional_training_loss(self, epoch: int) -> Optional[th.Tensor]:
        """Compute per-epoch extra training loss for point or structural objectives.

        Guarantees:

        - Returns ``None`` when ``objective == "axiom"``.
        - Under ``objective == "structural"``, returns
          ``self.module.structural_extra_loss()``.
        - Under ``objective in ("point", "mixed", "hybrid")``, samples at most
          ``points_per_epoch`` pairs using seed ``points_seed + epoch`` and
          returns ``point_membership_loss + point_role_loss + volume_reg * volume_penalty(offsets)``
          where ``TOP`` (``owl:Thing``) and ``BOTTOM`` (``owl:Nothing``) are
          excluded from the volume penalty.

        Non-guarantees:

        - Does not include region-level GCI losses (handled by the main epoch
          loop).

        Parameters
        ----------
        epoch : int
            Zero-based training epoch index used to seed per-epoch sampling.

        Returns
        ----------
        torch.Tensor or None
            Scalar additional training loss tensor, or ``None`` under
            ``objective="axiom"``.
        """
        if self.objective == "axiom":
            return None
        if self.objective == "structural":
            return self.module.structural_extra_loss()
        ps = self._ensure_point_supervision()
        epoch_rng = np.random.default_rng(self.points_seed + int(epoch))
        batch = ps.sample(self.points_per_epoch, rng=epoch_rng)
        loss = self.module.point_membership_loss(batch)
        loss = loss + self.module.point_role_loss(batch)
        if self.volume_reg > 0.0 and len(self.named_class_index_dict) > 0:
            non_top_indices = [
                int(idx)
                for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM)
            ]
            if non_top_indices:
                cls_idx = th.tensor(
                    non_top_indices, device=self.device, dtype=th.long
                )
                _, offsets = self.module.class_boxes(cls_idx)
                loss = loss + self.volume_reg * volume_penalty(offsets)
        return loss

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Build structural tables when ``objective='structural'``, then run the training loop.

        Guarantees:

        - When ``objective="structural"``, compiles a
          :class:`~mowl.baselines.nn.el.structural.DefinitionTable` and
          :class:`~mowl.baselines.nn.el.structural.ExistenceTargets` via
          :func:`~mowl.baselines.nn.el.structural.build_structural_tables`, caches them on
          the model so :meth:`init_module`, :meth:`load_best_model`, and
          :meth:`from_pretrained` preserve them, and attaches them via
          :meth:`~mowl.nn.MultiBoxELModule.set_structural_tables` before
          delegating to ``super().train(...)``.
        - Raises :class:`ValueError` if any required training GCI tensor has
          fewer columns than expected by its normal form.
        - When ``objective != "structural"``, delegates directly to
          ``super().train(...)``.

        Non-guarantees:

        - Gradient-based training is non-convex and is not guaranteed to reach a
          global minimum from arbitrary random initialisations.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        validate_every : int, default 1
            Validation frequency in epochs.
        epoch_callback : callable, optional
            Optional callback ``epoch_callback(epoch, model)`` invoked at the end
            of each 0-based epoch.

        Returns
        -------
        None
            Trains the model in place and returns ``None``.
        """
        if self.objective == "structural":
            self._structural_tables = build_structural_tables(
                self,
                k=self.n_components,
                max_components=self.max_components,
                enabled=self.structural_definitions,
                cm=self.canonical_model,
            )
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

        return super().train(
            epochs, validate_every=validate_every, epoch_callback=epoch_callback
        )
