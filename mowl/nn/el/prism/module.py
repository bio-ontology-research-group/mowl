"""PrismModule neural module for :math:`\\mathcal{EL}` prism embeddings."""
from __future__ import annotations

from typing import Optional, Tuple

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
import mowl.nn.el.prism.losses as L


class PrismModule(ELModule):
    """Prism (:math:`\\text{triangle} \\times \\text{box}`) neural module following Federico, Leemhuis & Kutz (DL 2026).

    Every ontology class :math:`C` is represented as a prism
    :math:`P_C = T_C \\times B_C \\subset \\mathbb{R}^2 \\times \\mathbb{R}^{d-2}`
    (§4.1 Definition 4 and Proposition 5 of Federico et al., DL 2026), with
    :math:`2d - 1` scalar parameters per class:

    - ``tri_center`` of shape ``(nb_ont_classes, 2)`` and ``tri_inradius`` of
      shape ``(nb_ont_classes, 1)`` (used as :math:`|\\rho| \\ge 0`), defining the
      three triangle support values :math:`h_k = n_k \\cdot c + |\\rho|`.
    - ``box_center`` and ``box_offset`` of shape ``(nb_ont_classes, d - 2)``
      (offset used as :math:`|o| \\ge 0`).
    - ``rel_embed`` of shape ``(nb_rels, d)``, where the first two coordinates
      translate the triangle and the remaining :math:`d - 2` coordinates
      translate the box.
    - Optional ``ind_embed`` of shape ``(nb_inds, d)`` representing individuals
      as points (degenerate prisms with zero inradius and zero box offsets).

    Guarantees:
    - Raises :class:`ValueError` if ``embed_dim < 3``.
    - :meth:`prism_params` returns ``(h, box_center, |box_offset|)`` of shapes
      ``(B, 3)``, ``(B, d - 2)``, ``(B, d - 2)`` with non-negative ``box_offset``.
    - Every GCI loss returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL
      (whereas ELBE returns ``(B, 1)``).
    - :meth:`gci1_bot_loss` aggregates box coordinates with ``min`` (semantically
      disjointness-if-any-separates; ELBE uses the Euclidean norm).

    Non-guarantees:
    - :meth:`gci0_bot_loss` and :meth:`gci3_bot_loss` have a floor of ``margin``
      at ``margin > 0`` since :math:`\\sum_k h_k = 3 |\\rho| \\ge 0`.
    - Touching triangles (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) = 0`) count as
      disjoint at ``margin = 0``.
    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).
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
    ) -> None:
        super().__init__()
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 3:
            raise ValueError(f"embed_dim must be an int >= 3, got {embed_dim!r}")

        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.margin = margin

        box_dim = embed_dim - 2
        self.tri_center = self._init_normalized_embedding(nb_ont_classes, 2)
        self.tri_inradius = nn.Embedding(nb_ont_classes, 1)
        nn.init.uniform_(self.tri_inradius.weight, a=0.1, b=1.0)

        self.box_center = self._init_normalized_embedding(nb_ont_classes, box_dim)
        self.box_offset = self._init_normalized_embedding(nb_ont_classes, box_dim)
        self.rel_embed = self._init_normalized_embedding(nb_rels, embed_dim)

        if self.nb_inds is not None and self.nb_inds > 0:
            self.ind_embed = self._init_normalized_embedding(self.nb_inds, embed_dim)
        else:
            self.ind_embed = None

    @staticmethod
    def _init_normalized_embedding(num_entities: int, dim: int) -> nn.Embedding:
        emb = nn.Embedding(num_entities, dim)
        nn.init.uniform_(emb.weight, a=-1.0, b=1.0)
        if num_entities > 0:
            norms = th.linalg.norm(emb.weight.data, dim=1, keepdim=True).clamp_min(1e-12)
            emb.weight.data.div_(norms)
        return emb

    def prism_params(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(h, box_center, box_offset)`` for class indices ``idx``.

        Guarantees:
        - ``h`` has shape ``(B, 3)`` with ``h.sum(dim=-1) == 3 * |tri_inradius(idx)|``.
        - ``box_center`` and ``box_offset`` have shape ``(B, d - 2)``, with
          ``box_offset >= 0``.

        Non-guarantees:
        - Does not detach the returned tensors from the autograd graph.
        """
        tc = self.tri_center(idx)
        tr = th.abs(self.tri_inradius(idx))
        h = L.supports(tc, tr)
        bc = self.box_center(idx)
        bo = th.abs(self.box_offset(idx))
        return h, bc, bo

    def regularization_loss(self) -> th.Tensor:
        """Return zero regularisation loss (no bump or component regulariser)."""
        return th.tensor(0.0, device=self.tri_center.weight.device)

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci0_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def gci0_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci0_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.margin,
            neg=neg,
        )

    def gci1_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci1_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci1_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci2_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        return L.gci3_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.margin,
            neg=neg,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.class_assertion_loss(
            data,
            self.ind_embed,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_embed,
            self.rel_embed,
            self.margin,
            neg=neg,
        )
