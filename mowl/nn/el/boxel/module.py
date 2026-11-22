import torch as th

from mowl.nn.el.part_built import PartBuiltModule


class BoxELModule(PartBuiltModule):
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
    """

    #: Which regulariser is used: ``"global"`` (default), the mean over all class boxes of
    #: the excess of the upper corners over 1 and of negative lower corners, once per step
    #: (the penalty part of the geometry); or ``"box2el_code"``, the regulariser of the BoxEL
    #: baseline of the Box\ :sup:`2`\ EL code (the mean excess of the upper corners over 1 plus
    #: the excess of the norm of the lower corners, of every box of every batch, including
    #: negatives, intersections and translated boxes). Set by the training regime
    #: :class:`~mowl.training.Box2ELBaselineLosses`.
    regulariser = "global"

    state_dict_aliases = {"min_embedding.weight": "class_model.mins.weight",
                          "delta_embedding.weight": "class_model.deltas.weight",
                          "relation_embedding.weight": "relation_model.vectors.weight",
                          "scaling_embedding.weight": "relation_model.scales.weight",
                          "ind_embedding.weight": "individual_model.mins.weight",
                          "ind_delta_embedding.weight": "individual_model.deltas.weight"}

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 min_bounds=(1e-4, 0.2), delta_bounds=(-0.1, 0), relation_bounds=(-0.1, 0.1),
                 scaling_bounds=(0.9, 1.1), temperature=1.0, positive_scale=True,
                 role_chain_composition="in_order", *, _builder=None, **options):
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
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self._pending_regulariser = 0

    # -- the names of the earlier module -----------------------------------------------------
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
    def role_chain_composition(self):
        return self.relation_model.role_chain_composition

    @role_chain_composition.setter
    def role_chain_composition(self, value):
        if value not in self.relation_model.COMPOSITIONS:
            raise ValueError(f"role_chain_composition must be one of "
                             f"{self.relation_model.COMPOSITIONS}.")
        self.relation_model.role_chain_composition = value

    # -- the regulariser of the Box2EL code ---------------------------------------------------
    def _axiom_boxes(self, gci_name, data):
        """The boxes that the Box2EL code regularises for a batch of one normal form: the
        classes of the axiom, the intersection of :math:`C \\sqcap D` and the translated box
        of :math:`C`."""
        def box(col):
            return self.classes(data[:, col])

        if gci_name in ("gci0", "gci1_bot"):
            return [box(0), box(1)]
        if gci_name == "gci1":
            c, d = box(0), box(1)
            return [c.intersect(d)[0], c, d, box(2)]
        if gci_name == "gci2":
            c, d = box(0), box(2)
            return [self.relation_model.image(data[:, 1], c), c, d]
        if gci_name == "gci3":
            c, d = box(1), box(2)
            return [self.relation_model.preimage(data[:, 0], c), c, d]
        return []

    @staticmethod
    def _side_regulariser(box):
        """The regulariser of the BoxEL baseline of the Box2EL code for one batch of boxes:
        the mean excess of the upper corners over 1 plus the excess of the Frobenius norm of
        the batch's lower corners over 1 (one scalar per batch, not per box)."""
        return (th.mean(th.relu(box.upper - 1 + 1e-8))
                + th.relu(th.linalg.norm(box.lower) - 1))

    def forward(self, gci, gci_name, neg=False):
        if self.regulariser == "box2el_code" and self.training and th.is_grad_enabled():
            for box in self._axiom_boxes(gci_name, gci):
                self._pending_regulariser = (self._pending_regulariser
                                             + self._side_regulariser(box))
        return super().forward(gci, gci_name, neg=neg)

    def regularization_loss(self):
        if self.regulariser == "box2el_code":
            # The terms recorded for the batches of this step (see forward).
            out = self._pending_regulariser
            self._pending_regulariser = 0
            if isinstance(out, int):
                return th.zeros((), device=self.device)
            return out
        return super().regularization_loss()


__all__ = ["BoxELModule"]
