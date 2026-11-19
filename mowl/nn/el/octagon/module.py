r"""Neural module for octagon-product :math:`\mathcal{EL}^{++}` embeddings."""
from __future__ import annotations

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
from mowl.nn.el.octagon import geometry as G
from mowl.nn.el.octagon import losses as L


class OctagonELModule(ELModule):
    r"""Octagon-product :math:`\mathcal{EL}^{++}` module.

    An adaptation of Charpenay & Schockaert, IJCAI 2024.

    **What is from the paper.** The region: axis-aligned 2-D octagons
    :math:`\{(x, y) : x^- \le x \le x^+,\ y^- \le y \le y^+,\ u^- \le y - x \le u^+,\
    v^- \le x + y \le v^+\}` (Eq. 6), their closed-form normalisation and emptiness test
    (Proposition 1) and their closure under intersection (Section 3). In the paper the
    octagons are *relations* of a knowledge graph embedding (one octagon per coordinate,
    over head and tail coordinate).

    **What is ours (mOWL).** Octagons as *concept* regions: a concept name :math:`C` is a
    product :math:`O_C^{(1)} \times \dots \times O_C^{(d/2)}` of 2-D octagons over the
    consecutive coordinate pairs :math:`(x_{2i}, x_{2i+1})` of :math:`\mathbb{R}^d`. Roles
    are translations :math:`t_R \in \mathbb{R}^d` (as in ELBE and the prism model),
    interpreted as :math:`R^{\mathcal{I}} = \{(x, x + t_R)\}`, so that
    :math:`\exists R.C = C - t_R`. Individuals are points. All losses are ours and exact
    (see :mod:`mowl.nn.el.octagon.losses`):

    - ``gci0`` :math:`C \sqsubseteq D`: :math:`C \subseteq D`.
    - ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`C_1 \cap C_2` non-empty and
      contained in :math:`D` (the non-emptiness requirement is stronger than the axiom,
      as in ELBE and the prism model).
    - ``gci1_bot`` :math:`C_1 \sqcap C_2 \sqsubseteq \bot`: smallest width of
      :math:`C_1 \cap C_2` at most :math:`-` ``disjoint_margin`` (so disjoint when
      ``disjoint_margin > 0``).
    - ``gci2`` :math:`C \sqsubseteq \exists R.D`: :math:`C + t_R \subseteq D`.
    - ``gci3`` :math:`\exists R.C \sqsubseteq D`: :math:`C - t_R \subseteq D`. This is the
      exact condition under translation semantics; ELBE and the prism model only ask
      :math:`C - t_R` to *overlap* :math:`D`, which is not sound.
    - ``gci0_bot`` / ``gci3_bot`` (:math:`C \sqsubseteq \bot`,
      :math:`\exists R.C \sqsubseteq \bot`): concept names are non-empty by
      construction, so the loss flattens :math:`C` and is zero iff :math:`C` has empty
      interior.
    - ``class_assertion`` :math:`C(a)` (columns ``[a, C]``): :math:`p_a \in C`;
      ``object_property_assertion`` :math:`R(a, b)`: :math:`p_a + t_R = p_b`.

    Parameters per concept name: a centre :math:`c \in \mathbb{R}^d` (``class_embed``) and
    eight non-negative radii per coordinate pair (``class_radius``, used as
    :math:`|r|`), with bounds :math:`h_k = n_k \cdot c + |r_k|`; the centre always lies in
    the octagon, so concept names are non-empty. That is :math:`5d` numbers per concept.

    Role axioms are not supported (``role_axiom_capable = False``): with translations,
    :math:`R \sqsubseteq S` would force :math:`t_R = t_S`. The paper's closed-form
    composition (Proposition 2) applies to octagon *relations*, which this module does
    not use.

    :param nb_ont_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals, or ``None``.
    :param embed_dim: Even dimension :math:`d \ge 2`.
    :param margin: Margin of the inclusion and membership losses (positive values require
        strict inclusion, as in ELBE). Defaults to ``0.0``.
    :param disjoint_margin: Margin of ``gci1_bot``. ``None`` uses ``margin``. A positive
        value is needed for the zero loss to imply disjointness rather than only
        interior-disjointness. Defaults to ``None``.
    :param neg_margin: Margin of the negative ``gci2`` loss
        (:func:`~mowl.nn.el.octagon.losses.non_inclusion_loss`). Defaults to ``0.1``.
    :param init_radius: Radii are initialised uniformly in ``[0, init_radius]``.
    :param min_width: If positive, :meth:`regularization_loss` adds
        :math:`\mathrm{mean}_C \sum \mathrm{relu}(w_{\min} - \mathrm{width})` over all
        concept names, factors and facet axes. It keeps concepts from collapsing to
        (near-)points, which otherwise satisfies many axioms degenerately. It conflicts
        with ``gci0_bot``/``gci3_bot``, which flatten their concept. Defaults to ``0.0``.
    """

    neg_capable_gcis = frozenset({"gci2"})
    role_axiom_capable = False

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.0,
                 disjoint_margin=None, neg_margin=0.1, init_radius=0.5, min_width=0.0):
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) \
                or embed_dim < 2 or embed_dim % 2:
            raise ValueError(f"embed_dim must be an even int >= 2, got {embed_dim!r}")
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.nb_pairs = embed_dim // 2
        self.margin = float(margin)
        self.disjoint_margin = self.margin if disjoint_margin is None \
            else float(disjoint_margin)
        self.neg_margin = float(neg_margin)
        self.min_width = float(min_width)

        self.class_embed = nn.Embedding(nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1.0, b=1.0)
        self.class_radius = nn.Embedding(nb_ont_classes, 4 * embed_dim)
        nn.init.uniform_(self.class_radius.weight, a=0.0, b=init_radius)
        self.rel_embed = nn.Embedding(max(nb_rels, 1), embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-0.1, b=0.1)
        if nb_inds:
            self.ind_embed = nn.Embedding(nb_inds, embed_dim)
            nn.init.uniform_(self.ind_embed.weight, a=-1.0, b=1.0)
        else:
            self.ind_embed = None

    # ----------------------------------------------------------------------------------
    # Geometry accessors
    # ----------------------------------------------------------------------------------

    def _pairs(self, x: th.Tensor) -> th.Tensor:
        """Reshape ``(B, d)`` to ``(B, P, 2)``."""
        return x.reshape(x.shape[0], self.nb_pairs, 2)

    def class_bounds(self, idx: th.Tensor) -> th.Tensor:
        """Octagon bounds of the concept names ``idx``, shape ``(B, P, 8)``."""
        center = self._pairs(self.class_embed(idx))
        radius = th.abs(self.class_radius(idx)).reshape(idx.shape[0], self.nb_pairs, 8)
        return G.from_center_radius(center, radius)

    def rel_translation(self, idx: th.Tensor) -> th.Tensor:
        """Translations of the roles ``idx``, shape ``(B, P, 2)``."""
        return self._pairs(self.rel_embed(idx))

    def set_class_bounds(self, idx: int, h: th.Tensor) -> None:
        """Overwrite the parameters of class ``idx`` so that its bounds are ``h``.

        Uses the witness point of each (non-empty) factor as centre and the radii
        :math:`h_k - n_k \\cdot c \\ge 0`. Intended for tests and exact initialisation.

        :param h: Bounds, shape ``(P, 8)``; every factor must be non-empty.
        """
        h = h.to(self.class_embed.weight)
        if bool((G.emptiness_slack(h) < -1e-9).any()):
            raise ValueError("Every factor must be non-empty.")
        t = G.tighten(h)
        center = G.witness_point(t)
        radius = h - center @ G.directions_like(h).T
        with th.no_grad():
            self.class_embed.weight[idx] = center.reshape(-1)
            self.class_radius.weight[idx] = th.clamp_min(radius, 0.0).reshape(-1)

    # ----------------------------------------------------------------------------------
    # Normal-form losses
    # ----------------------------------------------------------------------------------

    def gci0_loss(self, data, neg=False):
        return L.inclusion_loss(self.class_bounds(data[:, 0]), self.class_bounds(data[:, 1]),
                                self.margin)

    def gci0_bot_loss(self, data, neg=False):
        return L.interior_empty_loss(self.class_bounds(data[:, 0]))

    def gci1_loss(self, data, neg=False):
        h_inter = th.minimum(self.class_bounds(data[:, 0]), self.class_bounds(data[:, 1]))
        h_d = self.class_bounds(data[:, 2])
        return L.inclusion_loss(h_inter, h_d, self.margin) + L.nonempty_loss(h_inter)

    def gci1_bot_loss(self, data, neg=False):
        return L.disjoint_loss(self.class_bounds(data[:, 0]), self.class_bounds(data[:, 1]),
                               self.disjoint_margin)

    def gci2_loss(self, data, neg=False):
        h_c = G.translate(self.class_bounds(data[:, 0]), self.rel_translation(data[:, 1]))
        h_d = self.class_bounds(data[:, 2])
        if neg:
            return L.non_inclusion_loss(h_c, h_d, self.neg_margin)
        return L.inclusion_loss(h_c, h_d, self.margin)

    def gci3_loss(self, data, neg=False):
        h_c = G.translate(self.class_bounds(data[:, 1]), -self.rel_translation(data[:, 0]))
        return L.inclusion_loss(h_c, self.class_bounds(data[:, 2]), self.margin)

    def gci3_bot_loss(self, data, neg=False):
        col = 1 if data.shape[1] > 1 else 0
        return L.interior_empty_loss(self.class_bounds(data[:, col]))

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function.")
        # Column order of mOWL's ClassAssertionDataset: [individual, class].
        points = self._pairs(self.ind_embed(data[:, 0]))
        return L.point_inclusion_loss(points, self.class_bounds(data[:, 1]), self.margin)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function.")
        return L.translation_loss(self.ind_embed(data[:, 0]), self.rel_embed(data[:, 1]),
                                  self.ind_embed(data[:, 2]))

    def regularization_loss(self):
        device = self.class_embed.weight.device
        if self.min_width <= 0.0 or self.nb_ont_classes == 0:
            return th.zeros((), device=device)
        h = self.class_bounds(th.arange(self.nb_ont_classes, device=device))
        short = th.relu(self.min_width - G.widths(h))
        return short.reshape(self.nb_ont_classes, -1).sum(dim=-1).mean()

    def all_class_bounds(self) -> th.Tensor:
        """Bounds of all classes, detached, shape ``(nb_ont_classes, P, 8)``."""
        with th.no_grad():
            idx = th.arange(self.nb_ont_classes, device=self.class_embed.weight.device)
            return self.class_bounds(idx).detach()


__all__ = ["OctagonELModule"]
