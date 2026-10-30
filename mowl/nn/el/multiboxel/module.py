"""MultiBoxEL neural module."""
from __future__ import annotations

from typing import Optional, Tuple

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
import mowl.nn.el.boxsquaredel.losses as B2L
from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.nn.el.multibox_utils import effective_components as _effective_components
import mowl.nn.el.multiboxel.losses as L
from mowl.nn.el.softagg import diversity_penalty



class MultiBoxELModule(ELModule):
    """Multi-component Box\\ :math:`^2`\\ EL neural module.

    Every ontology class :math:`C` is represented as a union of
    :math:`k = \\text{n\\_components}` axis-parallel boxes in :math:`\\mathbb{R}^d`,
    stored in ``class_center`` and ``class_offset`` of shape
    ``(nb_ont_classes, k * d)`` and viewed as ``(B, k, d)``. Each class has a
    single translation bump vector in ``bump_classes`` of shape
    ``(nb_ont_classes, d)`` shared by its components. Roles and optional
    individuals are parameterised identically to
    :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`.

    Guarantees:
    - **Exact k = 1 reduction:** When ``n_components == 1`` and ``component_reg == 0.0``
      with weights copied from a :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`,
      every GCI loss (positive and negative) and :meth:`regularization_loss`
      agree with :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule` to
      within floating-point tolerance ``1e-6``.
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
    n_components : int, default 2
        Number of box components :math:`k \\ge 1` per class.
    gamma : float, default 0.0
        Inclusion margin :math:`\\gamma`.
    delta : float, default 2.0
        Negative-sampling distance target :math:`\\delta` for ``gci2`` and
        ``object_property_assertion``.
    reg_factor : float, default 0.05
        Weight for the Box²EL bump :math:`L_2` regulariser.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored on the module.
    diversity_reg : float, default 0.0
        Weight for the pairwise component volume overlap regulariser
        :func:`~mowl.nn.el.softagg.diversity_penalty`.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    def __init__(
        self,
        nb_ont_classes: int,
        nb_rels: int,
        nb_inds: Optional[int] = None,
        embed_dim: int = 50,
        n_components: int = 2,
        gamma: float = 0.0,
        delta: float = 2.0,
        reg_factor: float = 0.05,
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
        self.n_components = n_components

        self.class_center = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )
        self.class_offset = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )

        self.head_center = self.init_embeddings(nb_rels, embed_dim)
        self.head_offset = self.init_embeddings(nb_rels, embed_dim)
        self.tail_center = self.init_embeddings(nb_rels, embed_dim)
        self.tail_offset = self.init_embeddings(nb_rels, embed_dim)

        self.bump_classes = self.init_embeddings(nb_ont_classes, embed_dim)

        if self.nb_inds is not None and self.nb_inds > 0:
            self.bump_individuals = self.init_embeddings(nb_inds, embed_dim)
            self.ind_center = self.init_embeddings(nb_inds, embed_dim)
            self.ind_offset = self.init_embeddings(nb_inds, embed_dim)
        else:
            self.bump_individuals = None
            self.ind_center = None
            self.ind_offset = None

        self.gamma = gamma
        self.delta = delta
        self.reg_factor = reg_factor
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.tau = float(tau)
        self.diversity_reg = float(diversity_reg)


    # Single-component tables (roles, bumps, individuals) use the Box²EL initialiser.
    init_embeddings = BoxSquaredELModule.init_embeddings

    def init_class_embeddings(
        self,
        num_classes: int,
        n_components: int,
        embed_dim: int,
        min: float = -1.0,
        max: float = 1.0,
    ) -> nn.Embedding:
        """Initialise a multi-component class embedding table normalised per component."""
        embeddings = nn.Embedding(num_classes, n_components * embed_dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if num_classes > 0:
            viewed = embeddings.weight.data.view(num_classes, n_components, embed_dim)
            norms = th.linalg.norm(viewed, dim=-1, keepdim=True)
            viewed.div_(norms)
        embeddings.n_components = n_components
        return embeddings

    def class_boxes(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return ``(centers, offsets)`` of shape ``(B, k, d)`` for class indices ``idx``.

        Guarantees:
        - ``offsets`` is non-negative (computed via ``torch.abs``).

        Non-guarantees:
        - Does not sort or deduplicate components.
        """
        b_size = idx.shape[0]
        cen = self.class_center(idx).view(b_size, self.n_components, self.embed_dim)
        off = th.abs(self.class_offset(idx)).view(
            b_size, self.n_components, self.embed_dim
        )
        return cen, off

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci0_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
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
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci1_bot_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci2_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_classes,
            self.gamma,
            self.delta,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_classes,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_bot_loss(
            data,
            self.head_offset,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.ind_center is None or self.ind_offset is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.class_assertion_loss(
            data,
            self.ind_center,
            self.ind_offset,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        if (
            self.ind_center is None
            or self.ind_offset is None
            or self.bump_individuals is None
        ):
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_center,
            self.ind_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_individuals,
            self.gamma,
            self.delta,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def regularization_loss(self) -> th.Tensor:
        """Compute Box²EL bump regularisation plus optional component and diversity penalties.

        Guarantees:
        - When ``component_reg == 0.0`` and ``diversity_reg == 0.0`` (or
          ``n_components == 1``), returns the exact same tensor value as
          :meth:`BoxSquaredELModule.regularization_loss`.

        Non-guarantees:
        - When ``component_reg == 0.0`` and ``diversity_reg == 0.0``, unused
          components are permitted to collapse or nest inside other components.
        """
        loss = B2L.reg_loss(self.bump_classes, self.reg_factor)
        if self.bump_individuals is not None:
            loss = loss + B2L.reg_loss(self.bump_individuals, self.reg_factor)
        if self.component_reg > 0.0 and self.n_components > 1:
            tau = 0.1
            centers = self.class_center.weight.view(
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
            comp_penalty = th.relu(tau - dists).mean(dim=1).mean(dim=0)
            loss = loss + self.component_reg * comp_penalty
        if self.diversity_reg > 0.0 and self.n_components > 1 and self.nb_ont_classes > 0:
            centers = self.class_center.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            offsets = th.abs(self.class_offset.weight).view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            loss = loss + self.diversity_reg * diversity_penalty(centers, offsets)
        return loss

    def effective_components(self, eps: float = 1e-3) -> th.Tensor:
        """Count non-degenerate, non-nested components per class on CPU using corner arithmetic.

        A component :math:`i` of class :math:`C` is counted if all of its
        half-widths satisfy :math:`|o_{C,i,m}| > \\text{eps}` and its box is not
        contained in any other valid component :math:`j \\neq i` of :math:`C`
        (checked on corners :math:`\\text{lo}_j \\le \\text{lo}_i` and
        :math:`\\text{hi}_i \\le \\text{hi}_j`; if two components have identical
        corners, only the lowest-index component is kept).

        Guarantees:
        - Returns a CPU ``torch.Tensor`` of shape ``(nb_ont_classes,)`` with
          integer counts in ``0 .. n_components``.

        Non-guarantees:
        - Does not check whether a component is covered by the *union* of two or
          more other components when it is not contained in any single one.

        Parameters
        ----------
        eps : float, default 1e-3
            Minimum half-width threshold for a component to be considered
            non-degenerate.

        Returns
        -------
        torch.Tensor
            1-D integer tensor of length ``nb_ont_classes`` on CPU.
        """
        centers = self.class_center.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        offsets = self.class_offset.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        return _effective_components(centers, offsets, eps=eps)
