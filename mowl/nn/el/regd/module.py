"""RegD on ELBE and on ELEmbeddings as combinations of parts.

:class:`RegDELBEModule` and :class:`RegDELEmModule` are the geometries that the methods
``"regd"`` and ``"regd_elem"`` of :mod:`mowl.representation.library.ballfam_parts` build
(the ``boundary and depth`` measure, the RegD objective with ELBE's or ELEmbeddings' reading
of the other forms), wrapped in the interface of the earlier modules. The direct,
hand-written modules are kept for comparison in :mod:`mowl.baselines.nn.el.regd`.
"""
from contextlib import contextmanager

from mowl.nn.el.elem.module import ELEmModule, _PartBuiltModule
from mowl.representation.library import ballfam_energy as D
from mowl.representation.methods import METHODS

__all__ = ["RegDELBEModule", "RegDELEmModule"]


class _RegDInterface:
    """What the two RegD modules share: the options of the RegD measure and objective under
    the names of the earlier modules, and the scoring context (raw energies)."""

    #: Normal forms with a RegD contrastive loss (the others use the host model's loss).
    contrastive_gcis = frozenset({"gci0", "gci2"})
    neg_capable_gcis = frozenset({"gci0", "gci2", "object_property_assertion"})

    # -- the options of the measure ------------------------------------------------------

    @property
    def lamda(self):
        """Weight :math:`\\lambda` of the depth dissimilarity."""
        return self.measure.lamda

    @lamda.setter
    def lamda(self, value):
        self.measure.lamda = value

    @property
    def p(self):
        """Exponent of the :math:`L_p` norm of the depth dissimilarity."""
        return self.measure.p

    @p.setter
    def p(self, value):
        self.measure.p = value

    @property
    def gamma1(self):
        """Margin :math:`\\gamma_1` of the positive term."""
        return self.measure.gamma1

    @gamma1.setter
    def gamma1(self, value):
        self.measure.gamma1 = value

    @property
    def gamma2(self):
        """Margin :math:`\\gamma_2` of the negative term."""
        return self.measure.gamma2

    @gamma2.setter
    def gamma2(self, value):
        self.measure.gamma2 = value

    @property
    def eps(self):
        """Lower bound of the sizes of the regions."""
        return self.measure.eps

    @eps.setter
    def eps(self, value):
        self.measure.eps = value

    @property
    def depth_on_containment(self):
        """Whether the depth term is also added to contained pairs."""
        return self.measure.depth_on_containment

    @depth_on_containment.setter
    def depth_on_containment(self, value):
        self.measure.depth_on_containment = value

    @property
    def flip_gci2(self):
        """Whether the reverse inclusion is trained for ``gci2`` (the objective's
        ``flip_gci2``)."""
        return self.objective.flip_gci2

    @flip_gci2.setter
    def flip_gci2(self, value):
        self.objective.flip_gci2 = bool(value)

    # -- scoring -------------------------------------------------------------------------

    @property
    def energy_mode(self):
        """Whether the loss of a RegD normal form returns the raw energy (the measure's
        ``raw``) instead of :math:`\\max\\{E, \\gamma_1\\}`."""
        return self.measure.raw

    @energy_mode.setter
    def energy_mode(self, value):
        self.measure.raw = bool(value)

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

    @property
    def last_gci_name(self):
        """The normal form of the last call of the module."""
        return getattr(self, "last_form", None)

    def _region(self, index):
        """Centre and size (at least ``eps``) of the regions of the classes ``index``."""
        region = self.measure.regularise(self.classes(index))
        if hasattr(region, "offset"):
            return region.centre, region.offset
        return region.centre, region.radius.squeeze(-1)

    def energy(self, parent_center, parent_size, child_center, child_size):
        """RegD energy of child regions inside parent regions, shape ``(n, 1)``."""
        return D.ENERGY[self._shape](parent_center, parent_size, child_center, child_size,
                                     lamda=self.lamda, p=self.p,
                                     depth_on_containment=self.depth_on_containment
                                     ).unsqueeze(-1)

    def boundary(self, parent_center, parent_size, child_center, child_size):
        """Boundary dissimilarity of child regions in parent regions, shape ``(n, 1)``."""
        return D.BOUNDARY[self._shape](parent_center, parent_size, child_center,
                                       child_size).unsqueeze(-1)


class RegDELBEModule(_RegDInterface, _PartBuiltModule):
    r"""ELBE ([peng2020]_) with the inclusion energy replaced by RegD ([yang2025regd]_).

    Concepts are boxes and roles are translations, as in ELBE. RegD replaces the inclusion
    energy of the normal forms

    * ``gci0`` :math:`C \sqsubseteq D`: :math:`E(\mathrm{box}(D), \mathrm{box}(C))`,
    * ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`E(\mathrm{box}(D), \mathrm{box}(C_1)
      \cap \mathrm{box}(C_2))`, plus ELBE's term that keeps the intersection non-empty,
    * ``gci2`` :math:`C \sqsubseteq \exists R.D`:
      :math:`E(\mathrm{box}(D), \mathrm{box}(C) + v_R)`,
    * ``gci3`` :math:`\exists R.C \sqsubseteq D`:
      :math:`E(\mathrm{box}(D), \mathrm{box}(C) - v_R)`,

    where :math:`E = d_{bd} + \lambda d_{dep}`
    (:class:`~mowl.representation.library.ballfam_parts.BoundaryDepth`). The module is the
    geometry of the method ``"regd"``: box classes, translations, the ``boundary and depth``
    measure and :class:`~mowl.representation.library.ballfam_parts.RegDObjective` over ELBE's
    objective. ``gci0`` and ``gci2`` are trained with the contrastive loss of Section 3.3
    (see :class:`~mowl.models.RegDELBE`); the other forms use :math:`\max\{E, \gamma_1\}`.
    The bottom forms (``gci0_bot``, ``gci1_bot``, ``gci3_bot``) and the assertions keep ELBE's
    losses.

    :param lamda: Weight of the depth dissimilarity (:math:`\lambda`). Defaults to 1.
    :param p: Exponent of the :math:`L_p` norm of the depth dissimilarity. Defaults to 1, which
        the paper finds best for boxes.
    :param gamma1: Margin of the positive term. Defaults to 0.
    :param gamma2: Margin of the negative term. Defaults to 0.
    :param eps: Lower bound of the offsets, which are divided by in the depth dissimilarity.
    :param depth_on_containment: See
        :class:`~mowl.representation.library.ballfam_parts.BoundaryDepth`.
    :param flip_gci2: Train the *reverse* inclusion for ``gci2`` (box(D) + R inside
        box(C)), as the authors' prediction-task code (``nf3Loss_ours``) does. Defaults
        to ``False`` (the normal form's own direction).
    """

    regd_gcis = frozenset({"gci0", "gci1", "gci2", "gci3"})
    margin_semantics = "strict"
    _shape = "box"  # key into ``ballfam_energy.ENERGY`` / ``BOUNDARY``

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 lamda=1.0, p=1, gamma1=0.0, gamma2=0.0, eps=1e-4,
                 depth_on_containment=True, flip_gci2=False):
        geometry = METHODS["regd"](nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                                   margin=margin, lamda=lamda, p=p, gamma1=gamma1,
                                   gamma2=gamma2, eps=eps,
                                   depth_on_containment=depth_on_containment,
                                   flip_gci2=flip_gci2)
        self._assemble(geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim)
        self.class_offset = self.class_model.offsets
        self.ind_offset = getattr(self.individual_model, "offsets", None)

    # ELBE's bottom losses were flat vectors before the losses became columns
    # (``per_axiom_column = False`` gives that shape back)
    def gci0_bot_loss(self, data, neg=False):
        out = super().gci0_bot_loss(data, neg)
        return out if self.per_axiom_column else out.reshape(-1)

    def gci3_bot_loss(self, data, neg=False):
        out = super().gci3_bot_loss(data, neg)
        return out if self.per_axiom_column else out.reshape(-1)


class RegDELEmModule(_RegDInterface, ELEmModule):
    r"""ELEm ([kulmanov2019]_) with the inclusion energy replaced by RegD ([yang2025regd]_).

    Concepts are balls and roles are translations, as in :class:`~mowl.nn.ELEmModule`. RegD
    replaces the inclusion energy of the normal forms ``gci0``, ``gci2`` and ``gci3`` by
    :math:`E = d_{bd} + \lambda d_{dep}` of the (translated) child ball inside the parent
    ball, as in :class:`RegDELBEModule`. ``gci1`` (the intersection of two balls is not a
    ball), the bottom forms and the assertions keep ELEm's losses, and so does the
    centre-norm regularisation. The module is the geometry of the method ``"regd_elem"``.

    :param lamda: Weight of the depth dissimilarity (:math:`\lambda`). Defaults to 1.
    :param p: Exponent of the :math:`L_p` norm of the depth dissimilarity. Defaults to 2, which
        the paper finds best for balls.
    :param gamma1: Margin of the positive term. Defaults to 0.
    :param gamma2: Margin of the negative term. Defaults to 0.
    :param eps: Lower bound of the radii, which are divided by in the depth dissimilarity.
    :param depth_on_containment: See
        :class:`~mowl.representation.library.ballfam_parts.BoundaryDepth`.
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
    _shape = "ball"  # key into ``ballfam_energy.ENERGY`` / ``BOUNDARY``

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 reg_norm=1, lamda=1.0, p=2, gamma1=0.0, gamma2=0.0, eps=1e-4,
                 depth_on_containment=True, flip_gci2=False, radius="abs",
                 gci1_radius_term=True, consistent_radius=True):
        if radius not in ("abs", "softplus"):
            raise ValueError("radius must be 'abs' or 'softplus'.")
        geometry = METHODS["regd_elem"](
            nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=margin,
            reg_norm=reg_norm, gci1_radius_term=gci1_radius_term, lamda=lamda, p=p,
            gamma1=gamma1, gamma2=gamma2, eps=eps, depth_on_containment=depth_on_containment,
            flip_gci2=flip_gci2, radius=radius, consistent_radius=consistent_radius)
        self._assemble(geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim, reg_norm)

    @property
    def radius(self):
        """``"abs"`` or ``"softplus"``: how the radius is obtained from the stored
        parameter (the class model's ``radius_form``)."""
        return self.class_model.radius_form

    @radius.setter
    def radius(self, value):
        if value not in ("abs", "softplus"):
            raise ValueError("radius must be 'abs' or 'softplus'.")
        self.class_model.radius_form = value

    @property
    def consistent_radius(self):
        """See the class parameter (the RegD objective's ``consistent_radius``)."""
        return self.objective.consistent_radius

    @consistent_radius.setter
    def consistent_radius(self, value):
        self.objective.consistent_radius = bool(value)
