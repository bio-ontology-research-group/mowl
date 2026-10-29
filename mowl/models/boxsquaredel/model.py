from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import BoxSquaredELModule


class BoxSquaredEL(EmbeddingELModel):
    """
    Implementation based on [jackermeier2023]_.

    Parameters
    ----------
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
        Rows are matched by IRI against :attr:`class_index_dict`; rows of the
        auxiliary classes introduced by normalisation keep their random
        initialisation. Defaults leave behaviour unchanged.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
        Defaults leave behaviour unchanged.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 init_boxes=None,
                 anchor_weight=0.0
                 ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)

        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.num_negs = num_negs
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = BoxSquaredELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict)

    def get_regularization_loss(self):
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

