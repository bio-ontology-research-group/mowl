"""MultiELBE neural module."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn

from mowl.nn import ELModule
from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.nn.el.multiboxel.module import MultiBoxELModule
import mowl.nn.el.multielbe.losses as L
from mowl.nn.el.softagg import diversity_penalty



class MultiELBEModule(ELModule):
    """Multi-component ELBE neural module.

    Every ontology class :math:`C` is represented as a union of
    :math:`k = \\text{n\\_components}` axis-parallel boxes in :math:`\\mathbb{R}^d`,
    stored in ``class_embed`` and ``class_offset`` of shape
    ``(nb_ont_classes, k * d)`` and viewed as ``(B, k, d)``. Roles are single
    translation vectors in ``rel_embed`` of shape ``(nb_rels, d)``, and optional
    individuals are single boxes in ``ind_embed`` and ``ind_offset`` of shape
    ``(nb_inds, d)``, parameterised identically to
    :class:`~mowl.nn.el.elbe.module.ELBEModule`.

    Guarantees:
    - **Exact k = 1 reduction:** When ``n_components == 1``,
      ``component_aggregation == "hard"``, and ``component_reg == 0.0`` with
      weights copied from an :class:`~mowl.nn.el.elbe.module.ELBEModule`, every
      GCI loss (positive and negative) agrees with
      :class:`~mowl.nn.el.elbe.module.ELBEModule` to within floating-point
      tolerance ``1e-6``.
    - **Effective component count:** :meth:`effective_components` counts, for
      each class on CPU via corner arithmetic, the components whose half-widths
      exceed ``eps`` in every coordinate and whose box is not contained in
      another component of the same class (with exact ties broken by keeping
      the lowest-index component).

    Non-guarantees:
    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).

    Parameters
    ----------
    nb_ont_classes : int
        Number of ontology classes.
    nb_rels : int
        Number of object properties (roles).
    nb_inds : Optional[int], default None
        Number of individuals (ABox).
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d`.
    margin : float, default 0.1
        ELBE inclusion margin (positive margin tightens).
    n_components : int, default 1
        Number of box components :math:`k \\ge 1` per class.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau_c - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau_c = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored on the module.
    diversity_reg : float, default 0.0
        Diversity regularisation weight stored on the module.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    def __init__(
        self,
        nb_ont_classes: int,
        nb_rels: int,
        nb_inds: Optional[int] = None,
        embed_dim: int = 50,
        margin: float = 0.1,
        n_components: int = 1,
        component_reg: float = 0.0,
        component_aggregation: str = "hard",
        tau: float = 1.0,
        diversity_reg: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components must be an int >= 1, got {n_components!r}")
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError(f"embed_dim must be an int >= 1, got {embed_dim!r}")
        if component_aggregation not in ("hard", "soft"):
            raise ValueError(
                f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}"
            )
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or float(tau) <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau!r}")
        if (
            isinstance(diversity_reg, bool)
            or not isinstance(diversity_reg, (int, float))
            or float(diversity_reg) < 0.0
        ):
            raise ValueError(f"diversity_reg must be >= 0, got {diversity_reg!r}")

        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds

        self.embed_dim = embed_dim
        self.margin = margin
        self.n_components = n_components
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.tau = float(tau)
        self.diversity_reg = float(diversity_reg)

        self.class_embed = self.init_class_embeddings(

            nb_ont_classes, n_components, embed_dim
        )
        self.class_offset = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )

        self.rel_embed = self.init_embeddings(nb_rels, embed_dim)

        if self.nb_inds is not None:
            self.ind_embed = self.init_embeddings(nb_inds, embed_dim)
            self.ind_offset = self.init_embeddings(nb_inds, embed_dim)
        else:
            self.ind_embed = None
            self.ind_offset = None

    init_embeddings = BoxSquaredELModule.init_embeddings
    init_class_embeddings = MultiBoxELModule.init_class_embeddings

    @property
    def class_center(self) -> nn.Embedding:
        """Alias for ``class_embed`` used by :class:`~mowl.geometric.init.ExactBoxInit`."""
        return self.class_embed

    def class_boxes(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return ``(centers, offsets)`` of shape ``(B, k, d)`` for class indices ``idx``."""
        b_size = idx.shape[0]
        cen = self.class_embed(idx).view(b_size, self.n_components, self.embed_dim)
        off = th.abs(self.class_offset(idx)).view(
            b_size, self.n_components, self.embed_dim
        )
        return cen, off

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci0_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci0_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci0_bot_loss(
            data,
            self.class_offset,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci1_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci1_bot_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_score(self, data: th.Tensor) -> th.Tensor:
        return L.gci2_score(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci2_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_bot_loss(
            data,
            self.class_offset,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.ind_embed is None or self.ind_offset is None or self.nb_inds == 0:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.class_assertion_loss(
            data,
            self.ind_embed,
            self.ind_offset,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        if self.ind_embed is None or self.ind_offset is None or self.nb_inds == 0:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_embed,
            self.ind_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def regularization_loss(self) -> th.Tensor:
        """Compute optional component and diversity penalties (zero when weights are ``0.0`` or ``k == 1``)."""
        loss = th.zeros(
            (),
            device=self.class_embed.weight.device,
            dtype=self.class_embed.weight.dtype,
        )
        if self.component_reg > 0.0 and self.n_components > 1:
            tau_c = 0.1
            centers = self.class_embed.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            idx_i, idx_j = th.triu_indices(
                self.n_components,
                self.n_components,
                offset=1,
                device=centers.device,
            )
            diffs = centers[:, idx_i, :] - centers[:, idx_j, :]
            dists = th.linalg.norm(diffs, dim=-1)
            comp_penalty = th.relu(tau_c - dists).mean(dim=1).mean(dim=0)
            loss = loss + self.component_reg * comp_penalty
        if self.diversity_reg > 0.0 and self.n_components > 1 and self.nb_ont_classes > 0:
            centers = self.class_embed.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            offsets = th.abs(self.class_offset.weight).view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            loss = loss + self.diversity_reg * diversity_penalty(centers, offsets)
        return loss

    def effective_components(self, eps: float = 1e-3) -> th.Tensor:
        """Count non-degenerate, non-nested components per class on CPU using corner arithmetic."""
        k = self.n_components
        d = self.embed_dim
        n = self.nb_ont_classes
        cen_np = (
            self.class_embed.weight.detach()
            .cpu()
            .view(n, k, d)
            .numpy()
        )
        off_np = (
            th.abs(self.class_offset.weight.detach())
            .cpu()
            .view(n, k, d)
            .numpy()
        )
        lo_np = cen_np - off_np
        hi_np = cen_np + off_np
        valid_np = np.all(off_np > eps, axis=-1)

        counts = np.zeros(n, dtype=np.int64)
        for c in range(n):
            v = valid_np[c]
            if not np.any(v):
                continue
            lo_c = lo_np[c]
            hi_c = hi_np[c]
            eff = 0
            for i in range(k):
                if not v[i]:
                    continue
                contained = False
                for j in range(k):
                    if i == j or not v[j]:
                        continue
                    if np.all(lo_c[j] <= lo_c[i]) and np.all(hi_c[i] <= hi_c[j]):
                        equal_boxes = np.all(lo_c[i] <= lo_c[j]) and np.all(
                            hi_c[j] <= hi_c[i]
                        )
                        if not equal_boxes or j < i:
                            contained = True
                            break
                if not contained:
                    eff += 1
            counts[c] = eff
        return th.from_numpy(counts)
