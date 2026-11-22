from contextlib import contextmanager

import torch as th

import mowl.baselines.nn.el.elem.losses as L
import mowl.baselines.nn.el.regd.dissimilarity as D
from mowl.baselines.nn.el.elbe.module import ELBEModule
from mowl.baselines.nn.el.elem.module import ELEmModule


class _RegDMixin:
    """Behaviour shared by :class:`RegDELBEModule` and :class:`RegDELEmModule`.

    A subclass provides ``_region(index)`` (centre and size of the regions of the given
    classes), ``_energy``, ``_boundary`` and the set ``regd_gcis`` of the normal forms whose
    inclusion energy RegD replaces.
    """

    #: Normal forms with a RegD contrastive loss (the others use the host model's loss).
    contrastive_gcis = frozenset({"gci0", "gci2"})
    neg_capable_gcis = frozenset({"gci0", "gci2", "object_property_assertion"})

    def _init_regd(self, lamda, p, gamma1, gamma2, eps, depth_on_containment,
                   flip_gci2=False):
        if lamda < 0:
            raise ValueError("lamda must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.lamda = lamda
        self.p = p
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.depth_on_containment = depth_on_containment
        self.flip_gci2 = bool(flip_gci2)
        self.energy_mode = False
        self.last_gci_name = None

    @contextmanager
    def scoring(self):
        """Within this context, the loss of a RegD normal form returns the raw energy
        :math:`E = d_{bd} + \\lambda d_{dep}` (which can be negative) instead of
        :math:`\\max\\{E, \\gamma_1\\}`. Use it to score or rank axioms."""
        previous = self.energy_mode
        self.energy_mode = True
        try:
            yield self
        finally:
            self.energy_mode = previous

    def forward(self, gci, gci_name, neg=False):
        self.last_gci_name = gci_name
        return super().forward(gci, gci_name, neg=neg)

    def energy(self, parent_center, parent_size, child_center, child_size):
        """RegD energy of child regions inside parent regions, shape ``(n, 1)``."""
        return self._energy(parent_center, parent_size, child_center, child_size,
                            lamda=self.lamda, p=self.p,
                            depth_on_containment=self.depth_on_containment).unsqueeze(-1)

    def boundary(self, parent_center, parent_size, child_center, child_size):
        """Boundary dissimilarity of child regions in parent regions, shape ``(n, 1)``."""
        return self._boundary(parent_center, parent_size, child_center,
                              child_size).unsqueeze(-1)

    def _finish(self, energy, boundary, neg, extra=None):
        """Turn an energy (and its boundary dissimilarity) into the value a loss returns."""
        if neg:
            return D.negative_hinge(boundary, self.gamma2)
        if self.energy_mode:
            return energy if extra is None else energy + extra
        loss = th.clamp(energy, min=self.gamma1)
        return loss if extra is None else loss + extra

    def _gci0_regions(self, data):
        child = self._region(data[:, 0])
        parent = self._region(data[:, 1])
        return parent, child

    def _gci2_regions(self, data):
        if self.flip_gci2:
            # The authors' prediction-task code (nf3Loss_ours) constrains the *reverse*
            # inclusion: box(D) + R inside box(C) (with the sign of R flipped relative to
            # the evaluation, which scores box(D) - R inside the candidates).
            parent = self._region(data[:, 0])
            child_center, child_size = self._region(data[:, 2])
            child_center = child_center + self.rel_embed(data[:, 1])
            return parent, (child_center, child_size)
        child_center, child_size = self._region(data[:, 0])
        parent = self._region(data[:, 2])
        child_center = child_center + self.rel_embed(data[:, 1])
        return parent, (child_center, child_size)

    def _gci3_regions(self, data):
        child_center, child_size = self._region(data[:, 1])
        parent = self._region(data[:, 2])
        child_center = child_center - self.rel_embed(data[:, 0])
        return parent, (child_center, child_size)

    def _regd_loss(self, parent, child, neg, extra=None):
        energy = self.energy(*parent, *child)
        boundary = self.boundary(*parent, *child) if neg else None
        return self._finish(energy, boundary, neg, extra)

    def gci0_loss(self, data, neg=False):
        """:math:`C \\sqsubseteq D`: the region of :math:`C` inside the region of :math:`D`."""
        parent, child = self._gci0_regions(data)
        return self._regd_loss(parent, child, neg)

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        """:math:`C \\sqsubseteq \\exists R.D`: the region of :math:`C` translated by
        :math:`R` inside the region of :math:`D`."""
        parent, child = self._gci2_regions(data)
        return self._regd_loss(parent, child, neg)

    def gci3_loss(self, data, neg=False):
        """:math:`\\exists R.C \\sqsubseteq D`: the region of :math:`C` translated by
        :math:`-R` inside the region of :math:`D`."""
        parent, child = self._gci3_regions(data)
        return self._regd_loss(parent, child, False)

    def gci2_score(self, data):
        with self.scoring():
            return self.gci2_loss(data)


class RegDELBEModule(_RegDMixin, ELBEModule):
    r"""ELBE ([peng2020]_) with the inclusion energy replaced by RegD ([yang2025regd]_).

    Concepts are boxes (centre ``class_embed``, offset ``|class_offset|``) and roles are
    translations, as in :class:`~mowl.nn.ELBEModule`. RegD replaces the inclusion energy
    of the normal forms

    * ``gci0`` :math:`C \sqsubseteq D`: :math:`E(\mathrm{box}(D), \mathrm{box}(C))`,
    * ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`E(\mathrm{box}(D), \mathrm{box}(C_1)
      \cap \mathrm{box}(C_2))`, plus ELBE's term that keeps the intersection non-empty,
    * ``gci2`` :math:`C \sqsubseteq \exists R.D`:
      :math:`E(\mathrm{box}(D), \mathrm{box}(C) + v_R)`,
    * ``gci3`` :math:`\exists R.C \sqsubseteq D`:
      :math:`E(\mathrm{box}(D), \mathrm{box}(C) - v_R)`,

    where :math:`E = d_{bd} + \lambda d_{dep}` (:mod:`mowl.baselines.nn.el.regd.dissimilarity`).
    ``gci0`` and ``gci2`` are trained with the contrastive loss of Section 3.3 (see
    :class:`~mowl.baselines.models.regd.RegDELBE`); the other forms use :math:`\max\{E, \gamma_1\}`.
    The bottom forms (``gci0_bot``, ``gci1_bot``, ``gci3_bot``) and the assertions keep ELBE's
    losses.

    :param lamda: Weight of the depth dissimilarity (:math:`\lambda`). Defaults to 1.
    :param p: Exponent of the :math:`L_p` norm of the depth dissimilarity. Defaults to 1, which
        the paper finds best for boxes.
    :param gamma1: Margin of the positive term. Defaults to 0.
    :param gamma2: Margin of the negative term. Defaults to 0.
    :param eps: Lower bound of the offsets, which are divided by in the depth dissimilarity.
    :param depth_on_containment: See :func:`~mowl.baselines.nn.el.regd.dissimilarity.box_energy`.
    :param flip_gci2: Train the *reverse* inclusion for ``gci2`` (box(D) + R inside
        box(C)), as the authors' prediction-task code (``nf3Loss_ours``) does. Defaults
        to ``False`` (the normal form's own direction).
    """

    regd_gcis = frozenset({"gci0", "gci1", "gci2", "gci3"})
    _energy = staticmethod(D.box_energy)
    _boundary = staticmethod(D.box_boundary_dissimilarity)

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 lamda=1.0, p=1, gamma1=0.0, gamma2=0.0, eps=1e-4,
                 depth_on_containment=True, flip_gci2=False):
        super().__init__(nb_ont_classes, nb_rels, nb_inds=nb_inds, embed_dim=embed_dim,
                         margin=margin)
        self._init_regd(lamda, p, gamma1, gamma2, eps, depth_on_containment,
                        flip_gci2=flip_gci2)

    def _region(self, index):
        center = self.class_embed(index)
        offset = th.abs(self.class_offset(index)).clamp_min(self.eps)
        return center, offset

    def gci1_loss(self, data, neg=False):
        center_c, offset_c = self._region(data[:, 0])
        center_d, offset_d = self._region(data[:, 1])
        parent = self._region(data[:, 2])

        start = th.maximum(center_c - offset_c, center_d - offset_d)
        end = th.minimum(center_c + offset_c, center_d + offset_d)
        child = ((start + end) / 2, (th.abs(start - end) / 2).clamp_min(self.eps))
        non_empty = th.linalg.norm(th.relu(start - end), dim=1, keepdim=True)
        return self._regd_loss(parent, child, False, extra=non_empty)


class RegDELEmModule(_RegDMixin, ELEmModule):
    r"""ELEm ([kulmanov2019]_) with the inclusion energy replaced by RegD ([yang2025regd]_).

    Concepts are balls (centre ``class_embed``, radius ``|class_rad|``) and roles are
    translations, as in :class:`~mowl.nn.ELEmModule`. RegD replaces the inclusion energy of
    the normal forms ``gci0``, ``gci2`` and ``gci3`` by :math:`E = d_{bd} + \lambda d_{dep}`
    of the (translated) child ball inside the parent ball, as in :class:`RegDELBEModule`.
    ``gci1`` (the intersection of two balls is not a ball), the bottom forms and the
    assertions keep ELEm's losses, and so does the centre-norm regularisation.

    :param lamda: Weight of the depth dissimilarity (:math:`\lambda`). Defaults to 1.
    :param p: Exponent of the :math:`L_p` norm of the depth dissimilarity. Defaults to 2, which
        the paper finds best for balls.
    :param gamma1: Margin of the positive term. Defaults to 0.
    :param gamma2: Margin of the negative term. Defaults to 0.
    :param eps: Lower bound of the radii, which are divided by in the depth dissimilarity.
    :param depth_on_containment: See :func:`~mowl.baselines.nn.el.regd.dissimilarity.ball_energy`.
    :param flip_gci2: See :class:`RegDELBEModule`.
    :param radius: ``"abs"`` (default, as mOWL's ELEm: the radius is ``|class_rad|``) or
        ``"softplus"`` (the radius is ``softplus(class_rad)``, as the training code of
        the RegD paper's ``Elem``; with it, the ELEm normal forms ``gci1`` and
        ``gci1_bot`` use the softplus radii too). Note that the RegD code *evaluates*
        with ``|class_rad|`` whatever the training used; the evaluation regime
        ``regd-sampled-f1`` does the same.
    :param consistent_radius: With ``radius="softplus"``: ``True`` (default) makes every
        normal form that reads a class radius use the softplus radius (``gci0_bot``,
        ``gci3_bot``, class assertions, and ``gci1`` / ``gci1_bot`` including their
        negatives), so a class has one radius. ``False`` is the earlier behaviour, in which
        only the positive ``gci0``-``gci3`` and ``gci1_bot`` used softplus and the other
        forms still read ``|class_rad|`` (one class, two radii). The RegD code uses softplus
        in every form. Individuals keep their own ``|ind_rad|`` radius. No effect with
        ``radius="abs"``.
    :param gci1_radius_term: See :class:`~mowl.nn.ELEmModule`. The RegD code's ``Elem``
        has the term commented out (``False``).
    """

    regd_gcis = frozenset({"gci0", "gci2", "gci3"})
    _energy = staticmethod(D.ball_energy)
    _boundary = staticmethod(D.ball_boundary_dissimilarity)

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 reg_norm=1, lamda=1.0, p=2, gamma1=0.0, gamma2=0.0, eps=1e-4,
                 depth_on_containment=True, flip_gci2=False, radius="abs",
                 gci1_radius_term=True, consistent_radius=True):
        super().__init__(nb_ont_classes, nb_rels, nb_inds, embed_dim=embed_dim, margin=margin,
                         reg_norm=reg_norm, gci1_radius_term=gci1_radius_term)
        self._init_regd(lamda, p, gamma1, gamma2, eps, depth_on_containment,
                        flip_gci2=flip_gci2)
        if radius not in ("abs", "softplus"):
            raise ValueError("radius must be 'abs' or 'softplus'.")
        self.radius = radius
        self.consistent_radius = bool(consistent_radius)

    def _radius(self, index):
        raw = self.class_rad(index).squeeze(-1)
        if self.radius == "softplus":
            return th.nn.functional.softplus(raw)
        return th.abs(raw)

    def _region(self, index):
        center = self.class_embed(index)
        return center, self._radius(index).clamp_min(self.eps)

    def _soft_radius(self, index):
        """Radius of the classes ``index`` with shape ``(n, 1)``, the callable the ELEm
        losses expect in place of ``class_rad``."""
        return self._radius(index).unsqueeze(-1)

    @property
    def _all_forms_softplus(self):
        return self.radius == "softplus" and self.consistent_radius

    def gci1_loss(self, data, neg=False):
        if self.radius == "abs":
            return super().gci1_loss(data, neg=neg)
        if self.consistent_radius:
            if self._dele_neg(neg):
                loss = L.gci1_loss_neg(data, self.class_embed, self._soft_radius, self.margin)
            else:
                loss = L.gci1_loss(data, self.class_embed, self._soft_radius, self.margin,
                                   radius_term=self.gci1_radius_term)
            return self._penalised(loss, data, (0, 1, 2))
        if neg:
            return super().gci1_loss(data, neg=neg)
        # ELEm's C1 ⊓ C2 ⊑ D with softplus radii (Elem.nf2Loss of the RegD code).
        c, d, e = (self.class_embed(data[:, i]) for i in range(3))
        rc, rd = self._radius(data[:, 0]).unsqueeze(-1), self._radius(data[:, 1]).unsqueeze(-1)
        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        m = self.margin
        loss = th.relu(dst - rc - rd - m) + th.relu(dst2 - rc - m) + th.relu(dst3 - rd - m)
        if self.gci1_radius_term:
            re = self._radius(data[:, 2]).unsqueeze(-1)
            loss = loss + th.relu(th.minimum(rc, rd) - re - m)
        return loss

    def gci1_bot_loss(self, data, neg=False):
        if self.radius == "abs":
            return super().gci1_bot_loss(data, neg=neg)
        if self.consistent_radius:
            if self._dele_neg(neg):
                loss = L.gci1_bot_loss_neg(data, self.class_embed, self._soft_radius,
                                           self.margin)
            else:
                loss = L.gci1_bot_loss(data, self.class_embed, self._soft_radius, self.margin)
            return self._penalised(loss, data, (0, 1))
        if neg:
            return super().gci1_bot_loss(data, neg=neg)
        # ELEm's disjointness with softplus radii (Elem.disJointLoss of the RegD code).
        c, d = self.class_embed(data[:, 0]), self.class_embed(data[:, 1])
        rc, rd = self._radius(data[:, 0]).unsqueeze(-1), self._radius(data[:, 1]).unsqueeze(-1)
        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        return th.relu(rc + rd - dst + self.margin)

    def gci0_bot_loss(self, data, neg=False):
        if not self._all_forms_softplus:
            return super().gci0_bot_loss(data, neg=neg)
        if self._dele_neg(neg):
            return L.gci0_bot_loss_neg(data, self._soft_radius, self.epsilon)
        return L.gci0_bot_loss(data, self._soft_radius)

    def gci3_bot_loss(self, data, neg=False):
        if not self._all_forms_softplus:
            return super().gci3_bot_loss(data, neg=neg)
        if self._dele_neg(neg):
            return L.gci3_bot_loss_neg(data, self._soft_radius, self.epsilon)
        return L.gci3_bot_loss(data, self._soft_radius)

    def class_assertion_loss(self, data, neg=False):
        if not self._all_forms_softplus:
            return super().class_assertion_loss(data, neg=neg)
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this "
                             "loss function.")
        return L.class_assertion_loss(data, self.ind_embed, self.ind_rad, self.class_embed,
                                      self._soft_radius, self.margin, neg=neg)
