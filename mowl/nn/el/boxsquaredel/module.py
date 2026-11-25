from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.multibox_methods import box2el
from mowl.representation.library.multibox_modules import ReleasedTables, released_names
from mowl.representation.objectives import FORMS
from mowl.training.regularization import NormPenalty
from mowl.nn.el.elmodule import ELModule  # noqa: F401  (importable from here in 2.1.0)


@released_names(
    class_center="class_model.centres", class_offset="class_model.offsets",
    bump_classes="class_model.bumps",
    head_center="relation_model.head_centres", head_offset="relation_model.head_offsets",
    tail_center="relation_model.tail_centres", tail_offset="relation_model.tail_offsets",
    ind_center="individual_model.centres", ind_offset="individual_model.offsets",
    bump_individuals="individual_model.bumps")
class BoxSquaredELModule(ReleasedTables, ComposedGeometry):
    """
    Implementation of Box :math:`^2` EL from [jackermeier2023]_.

    The module is the combination of parts that Box2EL is
    (:func:`~mowl.representation.library.multibox_methods.box2el`): boxes with a bump vector
    (:class:`~mowl.representation.library.multibox_parts.BumpedBoxClasses`), relations as a
    head box and a tail box
    (:class:`~mowl.representation.library.multibox_parts.HeadTailRelations`), the box measure
    :class:`~mowl.representation.Bounds`, the objective of Box2EL (squared positive losses,
    the :math:`(\\delta - \\mu)^2` negatives, a slack margin) and the norm penalty on the bump
    vectors. It is a :class:`~mowl.representation.ComposedGeometry`; the tables of the parts
    keep the names they had (``class_center``, ``head_offset``, ``bump_classes``, ...) and
    models saved under those names load.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``. See also :meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional

    Every per-axiom loss is a column of shape ``(B, 1)`` (:class:`~mowl.nn.el.per_axiom_column.PerAxiomColumn`;
    set ``per_axiom_column = False`` for the earlier flat ``(B,)``).

    ``gci3_bot`` (:math:`\\exists r.C \\sqsubseteq \\bot`) shrinks the head box of :math:`r`, while its
    DELE negative (``all_negative_losses=True``, Eq. 17 of DELE) uses the offset of the filler
    :math:`C`. Both are kept as published; they do not describe the same constraint.
    """

    #: Which of the losses of the DELE reference code combines the scores of this module
    #: (:class:`~mowl.training.DELECodeLoss`).
    dele_code_form = "box-with-bumps"

    #: Normal forms that get a negative loss with ``all_negative_losses=True``.
    DELE_NEG_GCIS = frozenset({"gci0", "gci1", "gci2", "gci3",
                               "gci0_bot", "gci1_bot", "gci3_bot"})

    margin_attribute = "gamma"

    #: Options of the objective that the reading of the DELE reference code
    #: (:class:`~mowl.training.DELECodeLosses`) leaves alone: the negative and disjointness
    #: losses of this module take the margin as the reference code of Box2EL does.
    reference_reading_keeps = ("negative_margin",)

    #: The order in which the tables were initialised (a seeded module has its numbers).
    released_order = (
        "class_model.centres", "class_model.offsets", "relation_model.head_centres",
        "relation_model.head_offsets", "relation_model.tail_centres",
        "relation_model.tail_offsets", "class_model.bumps", "individual_model.bumps",
        "individual_model.centres", "individual_model.offsets")
    #: ``per_axiom_column = False``: the flat ``(B,)`` of the formulas, but for the role forms.
    flat_forms = FORMS

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, gamma=0, delta=2,
                 reg_factor=0.05, all_negative_losses=False, epsilon=0.01):
        self.initialise_in_released_order(lambda: box2el(
            nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=gamma,
            delta=delta, reg_factor=reg_factor, all_negatives=all_negative_losses,
            epsilon=epsilon, into=self))
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.strict_negatives = True

    # -- the numbers of the method, which are those of the parts -------------------------
    @property
    def gamma(self):
        """The margin :math:`\\gamma` (the margin of the objective)."""
        return self.objective.margin

    @gamma.setter
    def gamma(self, value):
        self.objective.margin = float(value)

    @property
    def delta(self):
        """The :math:`\\delta` of the negative loss (of the objective)."""
        return self.objective.delta

    @delta.setter
    def delta(self, value):
        self.objective.delta = float(value)

    @property
    def epsilon(self):
        """The :math:`\\varepsilon` of the DELE negative losses for bottom forms."""
        return self.objective.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.objective.epsilon = float(value)

    @property
    def reg_factor(self):
        """The weight of the norm penalty on the bump vectors."""
        return self._bump_penalty().weight

    @reg_factor.setter
    def reg_factor(self, value):
        self._bump_penalty().weight = value

    def _bump_penalty(self):
        return next(r for r in self.regularizers if isinstance(r, NormPenalty))

    @property
    def all_negative_losses(self):
        """Whether the DELE negative losses of all normal forms are on."""
        return self.DELE_NEG_GCIS <= self.neg_capable_gcis

    def enable_all_negative_losses(self, epsilon=None):
        """Switch on the DELE negative losses of all normal forms (see the class \
parameter ``all_negative_losses``) and extend :attr:`neg_capable_gcis` accordingly.

        :param epsilon: New value of :math:`\\varepsilon`, or ``None`` to keep the current one.
        :type epsilon: float, optional
        """
        if epsilon is not None:
            self.epsilon = float(epsilon)
        forms = frozenset(self.DELE_NEG_GCIS | {"object_property_assertion"})
        self.objective.neg_forms = forms
        self.objective.negatives = "forms"
        self.neg_capable_gcis = forms

    def init_embeddings(self, num_entities, embed_dim, min=-1, max=1):
        """A new table ``(num_entities, embed_dim)`` drawn uniformly from ``[min, max]``,
        every row scaled to length one.

        .. deprecated:: 2.2.0
           The tables are made by the class and relation models
           (:class:`~mowl.representation.library.multibox_parts.BumpedBoxClasses`,
           :class:`~mowl.representation.library.multibox_parts.HeadTailRelations`)."""
        import warnings
        import torch.nn as nn
        from mowl.representation.library.multibox_parts import _unit_rows
        warnings.warn("BoxSquaredELModule.init_embeddings is deprecated: the tables are made "
                      "by the parts (BumpedBoxClasses, HeadTailRelations).",
                      DeprecationWarning, stacklevel=2)
        return _unit_rows(nn.Embedding(num_entities, embed_dim), low=min, high=max)

    def _dele_neg(self, neg):
        return neg and self.all_negative_losses

    def get_loss_function(self, gci_name):
        # a loss installed on the instance (a training regime that switches the losses to
        # those of a reference code, mowl.training.DELECodeLosses) is the one that is called
        patched = self.__dict__.get(f"{gci_name}_loss")
        if patched is not None:
            return patched
        return super().get_loss_function(gci_name)
