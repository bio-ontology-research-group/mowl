from mowl.nn.el.part_built import _PartBuiltModule
from mowl.representation.class_models import BoxClasses
from mowl.representation.objectives import FORMS, Objective


class ELBEModule(_PartBuiltModule):
    """Implementation of ELBE from [peng2020]_.

    The module is a :class:`~mowl.representation.ComposedGeometry` made of the parts that the
    builder ``elbe`` of :mod:`mowl.representation.methods` names: boxes (a centre and the
    absolute value of an offset) for classes and individuals, a translation per relation,
    bounds per coordinate that are separated in every coordinate as the measure, and the
    reading of the axiom forms of ELBE as the objective. ``module.parts`` says which. The
    earlier hand-written module, which the tests compare this one with, is
    ``mowl.baselines.nn.el.elbe.module.ELBEModule``; checkpoints of that module load here.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``, which leaves the module exactly as before. See also \
:meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional
    :param gci3_form: Loss of :math:`\\exists r.C \\sqsubseteq D`. ``"overlap"`` (default, \
as in the ELBE and Box2EL codes): :math:`\\lVert\\max(0, |c - r - d| - o_C - o_D + m)\\rVert`, \
zero as soon as the translated box of :math:`C` overlaps the box of :math:`D`. \
``"inclusion"`` (the corrected loss of the TransBox code, ``model/Elbe.py``): \
:math:`\\lVert\\max(0, |c - r - d| + o_C - o_D + m)\\rVert`.
    :type gci3_form: str, optional
    :param init: ``"separate"`` (default): centres and offsets are each drawn from \
:math:`U(-1, 1)^n` and scaled to unit norm. ``"joint"`` (the Box2EL and TransBox codes): \
centre and offset of a class form one row of :math:`U(-1, 1)^{2n}` scaled to unit norm.
    :type init: str, optional
    """

    #: Which of the losses of the DELE reference code combines the scores of this module
    #: (:class:`~mowl.training.DELECodeLoss`).
    dele_code_form = "box"

    #: What the joint initialisation of the reference code normalises together
    #: (:class:`~mowl.training.JointNormalisedInitialisation`).
    joint_init = "box"
    joint_init_roles = "rel_embed"

    #: Normal forms that get a negative loss with ``all_negative_losses=True``.
    DELE_NEG_GCIS = frozenset({"gci0", "gci1", "gci2", "gci3",
                               "gci0_bot", "gci1_bot", "gci3_bot"})

    #: With ``per_axiom_column = False`` these two return ``(B,)`` as the hand-written module did.
    flat_forms = frozenset({"gci0_bot", "gci3_bot"})

    state_dict_aliases = {"class_embed.weight": "class_model.centres.weight",
                          "class_offset.weight": "class_model.offsets.weight",
                          "rel_embed.weight": "relation_model.vectors.weight",
                          "ind_embed.weight": "individual_model.centres.weight",
                          "ind_offset.weight": "individual_model.offsets.weight"}

    #: ELBE has no role axioms (a translation reads them, a combination such as
    #: ``ComposedEL(dataset, classes="box", relations="translation")`` trains on them): the
    #: module is the published method, as the released one was.
    role_axiom_capable = False
    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 all_negative_losses=False, epsilon=0.01, gci3_form="overlap",
                 init="separate"):
        from mowl.representation.methods import elbe
        if gci3_form not in ("overlap", "inclusion"):
            raise ValueError("gci3_form must be 'overlap' or 'inclusion'.")
        if init not in ("separate", "joint"):
            raise ValueError("init must be 'separate' or 'joint'.")
        built = elbe(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                     margin=margin, epsilon=epsilon, gci3=gci3_form, init=init)
        # the earlier module had a (possibly empty) table of individuals whenever it was
        # told their number
        individuals = BoxClasses(nb_inds, embed_dim) if nb_inds == 0 else None
        super().__init__(built, individuals)
        self.supported_forms = frozenset(FORMS)
        self.init = init
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.all_negative_losses = False
        if all_negative_losses:
            self.enable_all_negative_losses()

    # -- the names of the earlier module -----------------------------------------------------
    @property
    def class_offset(self):
        """The table of the offsets (half-widths, before the absolute value) of the classes."""
        return self.class_model.offsets

    @property
    def ind_offset(self):
        """The table of the offsets of the individuals, or ``None``."""
        return None if self.individual_model is None else self.individual_model.offsets

    @property
    def epsilon(self):
        """The :math:`\\varepsilon` of the DELE negative losses of the bottom forms."""
        return self.objective.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.objective.epsilon = float(value)

    @property
    def gci3_form(self):
        """``"overlap"`` or ``"inclusion"``: the reading of :math:`\\exists r.C \\sqsubseteq D`."""
        return self.objective.gci3_reading

    @gci3_form.setter
    def gci3_form(self, value):
        if value not in ("overlap", "inclusion"):
            raise ValueError("gci3_form must be 'overlap' or 'inclusion'.")
        self.objective.gci3_reading = value

    def enable_all_negative_losses(self, epsilon=None):
        """Switch on the DELE negative losses of all normal forms (see the class \
parameter ``all_negative_losses``) and extend :attr:`neg_capable_gcis` accordingly.

        :param epsilon: New value of :math:`\\varepsilon`, or ``None`` to keep the current one.
        :type epsilon: float, optional
        """
        if epsilon is not None:
            self.epsilon = epsilon
        self.all_negative_losses = True
        forms = frozenset(Objective.elbe().neg_forms | self.DELE_NEG_GCIS)
        self.objective.neg_forms = forms
        self.neg_capable_gcis = forms

    def role_inclusion_loss(self, gci, neg=False):
        raise NotImplementedError(self._role_axiom_error("role_inclusion"))

    def role_chain_loss(self, gci, neg=False):
        raise NotImplementedError(self._role_axiom_error("role_chain"))

    def _dele_neg(self, neg):
        return neg and self.all_negative_losses

    def gci2_score(self, data):
        """The loss of :math:`C \\sqsubseteq \\exists r.D`, which is also its score."""
        return self.cost("gci2", data)


__all__ = ["ELBEModule"]
