"""The direct implementation of the MultiELBE model class (the module of
:mod:`mowl.baselines.nn.el.multielbe` and its own extra training loss). The options and the
point supervision are those of :class:`mowl.models.MultiELBE`
(:class:`~mowl.baselines.models.multielbe.base.MultiELBEOptions`)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th

from mowl.baselines.nn.el.multielbe.module import MultiELBEModule
from mowl.baselines.nn.el.multiboxel.pointloss import volume_penalty
from mowl.baselines.geometric.core.normal_forms import BOTTOM, TOP
from mowl.baselines.models.multielbe.base import MultiELBEOptions


class MultiELBE(MultiELBEOptions):
    """MultiELBE with the direct module (see :class:`mowl.models.MultiELBE`)."""

    module_class = MultiELBEModule

    @property
    def _mse_criterion(self):
        return th.nn.MSELoss()

    def additional_training_loss(self, epoch: int) -> Optional[th.Tensor]:
        """Compute sampled canonical-element point and role losses plus volume regulariser.

        Guarantees:

        - Returns ``None`` when ``objective == "axiom"``.
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
        -------
        torch.Tensor or None
            Scalar additional training loss tensor, or ``None`` under
            ``objective="axiom"``.
        """
        if self.objective == "axiom":
            return None
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

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for MultiELBE (matching ELBE).

        Positive samples should have scores close to 0.
        Negative samples should have scores close to 1.
        """
        pos_mean = th.mean(pos_scores)
        loss = self._mse_criterion(
            pos_mean, th.zeros_like(pos_mean, requires_grad=False)
        )

        if neg_scores is not None:
            neg_mean = th.mean(neg_scores)
            loss += self._mse_criterion(
                neg_mean, th.ones_like(neg_mean, requires_grad=False)
            )

        return loss
