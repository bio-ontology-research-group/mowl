
from mowl.nn.el.part_built import _PartBuiltModule
from mowl.nn.el.elmodule import ELModule  # noqa: F401  (importable from here in 2.1.0)


class BoxELModule(_PartBuiltModule):
    """Implementation of BoxEL from [xiong2022]_.

    The module is a :class:`~mowl.representation.ComposedGeometry` made of the parts that the
    builder ``boxel`` of :mod:`mowl.representation.library.boxrel_methods` names: boxes stored
    as a lower corner and a log-width for classes and individuals, an affine map per relation,
    the smoothed volume ratio as the measure, the objective of BoxEL, and the penalty that
    keeps the boxes in the unit cube. ``module.parts`` says which. The earlier hand-written
    module, which the tests compare this one with, is
    ``mowl.baselines.nn.el.boxel.module.BoxELModule``; checkpoints of that module load here.

    .. note::
        `Original implementation: <https://github.com/Box-EL/BoxEL>`_

    :param min_bounds, delta_bounds, relation_bounds, scaling_bounds: Ranges of the uniform
        initialisation of the lower corners, the log-widths, the shifts and the scale factors.
    :param temperature: :math:`\\beta` of the softplus volume.
    :param positive_scale: Use :math:`|s| + \\epsilon` as the scale of a role (default) so
        that no role inverts a box; ``False`` is the signed scale of the original code.
    :param role_chain_composition: ``"in_order"`` (default) or ``"as_released"``, see
        :class:`~mowl.representation.library.boxrel_relations.Affine`.
    :param gci3_preimage: The box of :math:`\\exists r.C` in the loss of
        :math:`\\exists r.C \\sqsubseteq D`. ``"corners"`` (default): both corners of the box
        of :math:`C` are mapped back by the role. ``"as_released"``: the box of mOWL 2.1.0,
        whose mapped upper corner was taken as a log-width (a defect; the original BoxEL
        code maps both corners). ``preimage_reading`` of
        :class:`~mowl.representation.library.boxrel_relations.Affine`.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = True
    #: BoxEL has no margin of inclusion: a ratio of volumes cannot express room to spare.
    margin_semantics = None

    #: With ``per_axiom_column = False`` every loss but the two role axioms is ``(B,)``, as the
    #: hand-written module returned them.
    flat_forms = frozenset({"gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                            "class_assertion", "object_property_assertion"})

    released_state_dict_names = True

    state_dict_aliases = {"min_embedding.weight": "class_model.mins.weight",
                          "delta_embedding.weight": "class_model.deltas.weight",
                          "relation_embedding.weight": "relation_model.vectors.weight",
                          "scaling_embedding.weight": "relation_model.scales.weight",
                          "ind_embedding.weight": "individual_model.mins.weight",
                          "ind_delta_embedding.weight": "individual_model.deltas.weight"}

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 min_bounds=(1e-4, 0.2), delta_bounds=(-0.1, 0), relation_bounds=(-0.1, 0.1),
                 scaling_bounds=(0.9, 1.1), temperature=1.0, positive_scale=True,
                 role_chain_composition="in_order", gci3_preimage="corners", *,
                 _builder=None, **options):
        if _builder is None:
            from mowl.representation.library.boxrel_methods import boxel as _builder
        built = _builder(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                         temperature=temperature,
                         scale="positive" if positive_scale else "signed",
                         role_chain_composition=role_chain_composition,
                         min_bounds=tuple(min_bounds), delta_bounds=tuple(delta_bounds),
                         relation_bounds=tuple(relation_bounds),
                         scaling_bounds=tuple(scaling_bounds), **options)
        super().__init__(built)
        self.gci3_preimage = gci3_preimage
        self.margin_semantics = None
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim

    # -- the names of the earlier module -----------------------------------------------------
    def init_entity_embedding(self, num_entities, embed_dim, bounds):
        """A new table ``(num_entities, embed_dim)`` drawn uniformly from ``bounds``.

        .. deprecated:: 2.2.0
           The tables are made by the class and relation models
           (:class:`~mowl.representation.library.boxrel_boxes.MinDeltaBoxes`,
           :class:`~mowl.representation.library.boxrel_relations.Affine`)."""
        import warnings
        from mowl.representation.library.boxrel_boxes import _uniform
        warnings.warn("BoxELModule.init_entity_embedding is deprecated: the tables are made "
                      "by the parts (MinDeltaBoxes, Affine).", DeprecationWarning, stacklevel=2)
        return _uniform(num_entities, embed_dim, bounds[0], bounds[1])

    @property
    def margin(self):
        raise AttributeError("BoxELModule has no margin: the volume ratio of BoxEL cannot "
                             "express room to spare (the objective's margin is 0).")

    @property
    def min_embedding(self):
        return self.class_model.mins

    @property
    def delta_embedding(self):
        return self.class_model.deltas

    @property
    def relation_embedding(self):
        return self.relation_model.vectors

    @property
    def scaling_embedding(self):
        return self.relation_model.scales

    @property
    def ind_embedding(self):
        return None if self.individual_model is None else self.individual_model.mins

    @property
    def ind_delta_embedding(self):
        return None if self.individual_model is None else self.individual_model.deltas

    @property
    def temperature(self):
        """:math:`\\beta` of the softplus volume (a schedule may change it)."""
        return self.measure.temperature

    @temperature.setter
    def temperature(self, value):
        self.measure.temperature = value

    @property
    def positive_scale(self):
        """Whether a role's scale factor is read as :math:`|s| + \\epsilon` (``True``) or as
        the signed :math:`s + \\epsilon` of the original code."""
        return self.relation_model.scale == "positive"

    @positive_scale.setter
    def positive_scale(self, value):
        self.relation_model.scale = "positive" if value else "signed"

    @property
    def gci3_preimage(self):
        """``"corners"`` or ``"as_released"``: the box of :math:`\\exists r.C` (the
        ``preimage_reading`` of the relation model)."""
        return self.relation_model.preimage_reading

    @gci3_preimage.setter
    def gci3_preimage(self, value):
        if value not in self.relation_model.PREIMAGE_READINGS:
            raise ValueError(f"gci3_preimage must be one of "
                             f"{self.relation_model.PREIMAGE_READINGS}.")
        self.relation_model.preimage_reading = value

    @property
    def role_chain_composition(self):
        return self.relation_model.role_chain_composition

    @role_chain_composition.setter
    def role_chain_composition(self, value):
        if value not in self.relation_model.COMPOSITIONS:
            raise ValueError(f"role_chain_composition must be one of "
                             f"{self.relation_model.COMPOSITIONS}.")
        self.relation_model.role_chain_composition = value


__all__ = ["BoxELModule"]
