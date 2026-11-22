"""EmEL++ and EmEL(var) as combinations of parts.

:class:`EmELPPModule` and :class:`EmELVarModule` are the geometries that the methods
``"emelpp"`` and ``"emelvar"`` of :mod:`mowl.representation.library.ballfam_parts` build,
wrapped in the interface of the earlier modules. The direct, hand-written modules are kept
for comparison in :mod:`mowl.baselines.nn.el.emelpp`.
"""

from mowl.nn.el.elem.module import ELEmModule
from mowl.representation.methods import METHODS

__all__ = ["EmELPPModule", "EmELVarModule"]


class EmELPPModule(ELEmModule):
    """Implementation of EmEL++ from [mondal2021]_.

    The geometry is that of ELEmbeddings ([kulmanov2019]_): classes are :math:`n`-balls
    (centre and non-negative radius, stored as a signed number and used through its absolute
    value) and roles are translation vectors, so the embedding tables and their
    initialisation are those of :class:`ELEmModule <mowl.nn.ELEmModule>`. The losses
    are the ones printed in the paper (Eq. 1-10) and differ from ELEmbeddings in three ways:

    * every loss carries the normalisation terms :math:`|\\lVert x \\rVert - 1|` of the
      embedding vectors it involves (the objective's ``unit_norm``), so the module has no
      separate regularisation loss (:meth:`regularization_loss` is zero);
    * the intersection loss has a fourth term requiring the radius of :math:`E` to be at
      least the smaller radius of :math:`C` and :math:`D`;
    * there are losses for role inclusions (Eq. 8) and role chains (Eq. 9), read by the
      translation relation model.

    The role inclusion loss is symmetric in the two roles, so
    :math:`R \\sqsubseteq S` also enforces :math:`S \\sqsubseteq R`; see the catalogue card.

    Individuals, if present, are treated as nominal classes (paper, Section 4), i.e. as balls
    with the same class losses.

    :param nb_ont_classes: Number of ontology classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (``0`` or ``None`` for none).
    :param embed_dim: Embedding dimension.
    :param margin: Margin :math:`\\gamma` of the paper.
    """

    #: Eq. 10 is the only negative loss of the paper (``C ⋢ ∃R.D``); the assertion
    #: ``R(a, b)`` is the same normal form with nominals.
    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})

    role_axiom_capable = True

    #: The unit-norm penalties are part of every axiom loss (Eq. 1-10), so the penalty is
    #: always per axiom (see :attr:`ELEmModule.norm_penalty`).
    norm_penalty = "per_axiom"

    #: The builder of the method this module is the combination of.
    _method = "emelpp"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=0, embed_dim=50, margin=0.1):
        self._build(nb_ont_classes, nb_rels, nb_inds, embed_dim, margin)

    def _build(self, nb_ont_classes, nb_rels, nb_inds, embed_dim, margin, **options):
        geometry = METHODS[self._method](
            nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=margin,
            gci0_shift=self._subclass_shift, **options)
        self._assemble(geometry, nb_ont_classes, nb_rels, nb_inds or 0, embed_dim, 1)

    def _rad(self, idx):
        """The non-negative radius of the classes ``idx`` (the class model's)."""
        return self.class_model(idx).radius

    def enable_all_negative_losses(self, epsilon=None):
        """EmEL++ has one negative loss (Eq. 10: :math:`C \\not\\sqsubseteq \\exists R.D`), and
        so there are no negative losses of the other normal forms to enable. (The earlier
        module accepted the call and ignored it: the flag changed nothing.)"""
        raise NotImplementedError(
            "EmEL++ and EmEL(var) define the negative loss of C not-subclass-of "
            "some R.D only (Eq. 10); the negative losses of DELE for the other normal forms "
            "belong to ELEmbeddings (ELEmModule) and RegD.")

    def _sigma(self, rel_idx):
        """Variance of the roles; ``None`` for EmEL++."""
        return None

    # -- options, as settings of the parts --------------------------------------------------

    @property
    def intersection_radius_term(self):
        """Whether the loss of :math:`C \\sqcap D \\sqsubseteq E` contains the term on the
        smaller radius (Eq. 2 of the paper); the measure's ``radius_term``. The EmEL++
        baseline of the Box\\ :sup:`2`\\ EL code leaves it out; the training regime
        :class:`~mowl.training.Box2ELBaselineLosses` sets this to ``False``."""
        return self.gci1_radius_term

    @intersection_radius_term.setter
    def intersection_radius_term(self, value):
        self.gci1_radius_term = value

    @property
    def radius_form(self):
        """How the stored radius is turned into a non-negative radius: ``"abs"`` (default) or
        ``"relu"`` (:math:`\\max(0, r)`, the EmEL++ Keras code); the class model's
        ``radius_form``. Set by the training regime
        :class:`~mowl.training.EmELppCodeObjective`."""
        return self.class_model.radius_form

    @radius_form.setter
    def radius_form(self, value):
        if value not in ("abs", "relu", "softplus"):
            raise ValueError("radius_form must be 'abs', 'relu' or 'softplus'.")
        self.class_model.radius_form = value

    @property
    def role_direction(self):
        """How the direction term of the role axiom losses is computed: ``"row"`` (default,
        the cosine of every pair, Eq. 8-9) or ``"batch"`` (the EmEL++ Keras code normalises
        along the batch axis, which gives one number per batch); the relation model's
        ``direction``. Set by the training regime
        :class:`~mowl.training.EmELppCodeObjective`."""
        return self.relation_model.direction

    @role_direction.setter
    def role_direction(self, value):
        self.relation_model.direction = value


class EmELVarModule(EmELPPModule):
    """Implementation of EmEL(var) from [mohapatra2021]_.

    EmEL(var) is EmEL++ with one extra learned scalar :math:`\\sigma(r) \\geq 0` per role
    (the "variance" of the translation): the relation model ``"translation with tolerance"``.
    It enters the losses of :math:`C \\sqsubseteq \\exists R.D` and
    :math:`\\exists R.C \\sqsubseteq D` (Eq. 7 and 8 of the paper): it is subtracted inside the
    hinge, so the translated ball :math:`C + R` only has to lie within :math:`\\sigma(R)` of
    :math:`D`, and added to the loss to keep it finite. All other losses, including the
    negative loss of Eq. 10 and the role axiom losses, are those of EmEL++ (the paper: "the
    loss function for other normal forms remain the same"). The parameter is stored as a
    signed number and used through its absolute value (the paper does not say how
    non-negativity is enforced).

    .. note::
        As published, the term :math:`+\\sigma(r)` has weight 1 in every axiom, which exactly
        cancels the gain of the hinge: for any :math:`\\sigma \\geq 0`,
        :math:`\\max(0, x - \\sigma) + \\sigma \\geq \\max(0, x)`, so the total loss is never
        below that of EmEL++. The optional ``sigma_reg_weight`` (**not from the paper**,
        default 1 = the paper) scales that term so that the variance can pay off.

    :param sigma_init: Initial value of :math:`\\sigma` (the paper does not give one).
    :param sigma_reg_weight: Weight of the :math:`+\\sigma(r)` term. The paper's value is 1.
    """

    _method = "emelvar"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=0, embed_dim=50, margin=0.1,
                 sigma_init=0.1, sigma_reg_weight=1.0):
        self._build(nb_ont_classes, nb_rels, nb_inds, embed_dim, margin,
                    sigma_init=sigma_init, sigma_reg_weight=sigma_reg_weight)
        #: The tolerance table of the relation model (the learned :math:`\\sigma`).
        self.rel_sigma = self.relation_model.tolerances

    def _sigma(self, rel_idx):
        return self.relation_model.tolerance(rel_idx)

    @property
    def sigma_reg_weight(self):
        """Weight of the :math:`+\\sigma(r)` term: the objective's ``tolerance_weight``."""
        return self.objective.tolerance_weight

    @sigma_reg_weight.setter
    def sigma_reg_weight(self, value):
        self.objective.tolerance_weight = float(value)
