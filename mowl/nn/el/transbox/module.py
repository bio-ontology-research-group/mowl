from mowl.nn.el.part_built import _PartBuiltModule


class TransBoxModule(_PartBuiltModule):
    """Implementation of TransBox from [yang2025]_.

    TransBox is an EL++-closed ontology embedding method: atomic concepts
    and roles are embedded as axis-aligned boxes (a center and a
    non-negative offset in R^n) and individuals as points. Complex EL++
    concepts are embedded by composing the atomic boxes (intersection for
    conjunction, box translation for existential restrictions, box addition
    for role composition).

    The module is a :class:`~mowl.representation.ComposedGeometry` made of the parts that the
    builder ``transbox`` of :mod:`mowl.representation.library.boxrel_methods` names (boxes for
    classes, a box of translations per relation, bounds per coordinate that leave out the
    coordinates in which an intersection is empty, the TransBox objective, two penalties);
    ``module.parts`` says which. The earlier hand-written module, which the tests compare this
    one with, is ``mowl.baselines.nn.el.transbox.module.TransBoxModule``; checkpoints of that
    module load here.

    .. note::
        `Original implementation: <https://github.com/HuiYang1997/TransBox>`_

    :param formulation: ``"mowl"`` (default): the losses of mOWL. ``"reference"``: the losses
        of the authors' code (:mod:`mowl.nn.el.transbox.code_losses`), with which the published
        numbers were obtained, and its initialisation (centre and offset of a box drawn
        together from :math:`U(-1, 1)^{2n}` and scaled to unit norm, instead of each to
        unit norm). These losses are meant to be squared per axiom, with negatives
        pushed to a target, by the training regime
        :class:`~mowl.training.TransBoxCodeTraining`.
    :param strict_negatives: If ``True`` (default), calling the module with ``neg=True``
        for a normal form without a negative loss (everything but ``gci2``) raises
        :class:`NotImplementedError`. ``False`` is the earlier behaviour: the positive loss
        is returned silently. (The rule and the option are those of every
        :class:`~mowl.nn.ELModule`.)
    :param reference_bottom_losses: With ``formulation="reference"``: ``"code"`` (default)
        scores ``C ⊑ ⊥`` and ``∃R.C ⊑ ⊥`` as the code scores ``C ⊑ D`` and ``∃R.C ⊑ D``
        with the bottom class as :math:`D` (including its centre-norm terms); ``"mowl"`` is
        the earlier behaviour, the formulas of the ``"mowl"`` formulation (no centre-norm
        term). The code has no counterpart for assertions, which keep the ``"mowl"``
        formulas.
    """

    FORMULATIONS = ("mowl", "reference")

    #: With ``per_axiom_column = False`` every loss is ``(B,)``, as the hand-written module
    #: returned them.
    flat_forms = frozenset({"gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                            "class_assertion", "object_property_assertion", "role_inclusion",
                            "role_chain"})

    state_dict_aliases = {
        "class_center_embedding.weight": "class_model.centres.weight",
        "class_offset_embedding.weight": "class_model.offsets.weight",
        "relation_center_embedding.weight": "relation_model.vectors.weight",
        "relation_offset_embedding.weight": "relation_model.offsets.weight",
        "ind_embedding.weight": "individual_model.centres.weight"}

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 margin=0.1, use_enhancement=True, reg_factor=0.1, formulation="mowl",
                 strict_negatives=True, reference_bottom_losses="code"):
        from mowl.representation.library.boxrel_methods import transbox
        if reference_bottom_losses not in ("code", "mowl"):
            raise ValueError("reference_bottom_losses must be 'code' or 'mowl'.")
        if formulation not in self.FORMULATIONS:
            raise ValueError(f"formulation must be one of {self.FORMULATIONS}.")
        built = transbox(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                         margin=margin, enhanced=use_enhancement, reg_factor=reg_factor,
                         formulation=formulation,
                         reference_bottom_losses=reference_bottom_losses,
                         strict_negatives=strict_negatives)
        super().__init__(built)
        self.strict_negatives = strict_negatives
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self._reg_factor = reg_factor

    # -- the names of the earlier module -----------------------------------------------------
    @property
    def class_center_embedding(self):
        return self.class_model.centres

    @property
    def class_offset_embedding(self):
        return self.class_model.offsets

    @property
    def relation_center_embedding(self):
        return self.relation_model.vectors

    @property
    def relation_offset_embedding(self):
        return self.relation_model.offsets

    @property
    def ind_embedding(self):
        return None if self.individual_model is None else self.individual_model.centres

    @property
    def formulation(self):
        return self.objective.formulation

    @property
    def use_enhancement(self):
        """Whether :math:`C \\sqsubseteq \\exists r.D` is trained against the semantic
        enhancement :math:`\\exists r^{all}.D` (Section 4.3 of the paper)."""
        return self.objective.enhanced

    @use_enhancement.setter
    def use_enhancement(self, value):
        self.objective.enhanced = bool(value)

    @property
    def reference_bottom_losses(self):
        return self.objective.reference_bottom_losses

    @property
    def reg_factor(self):
        """Weight of the two penalties (the centre norms of the classes, the norm of the role
        offsets)."""
        return self._reg_factor

    @reg_factor.setter
    def reg_factor(self, value):
        self._reg_factor = value
        for regularizer in self.regularizers:
            regularizer.weight = value

    def class_center(self, idx):
        return self.class_model.centres(idx)

    def class_offset(self, idx):
        return self.class_model.offsets(idx).abs()

    def relation_center(self, idx):
        return self.relation_model.vectors(idx)

    def relation_offset(self, idx):
        return self.relation_model.offsets(idx).abs()


__all__ = ["TransBoxModule"]
