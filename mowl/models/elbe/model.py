from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELBEModule
import torch as th
from deprecated.sphinx import deprecated


class ELBE(EmbeddingELModel):
    """
    Implementation based on [peng2020]_.

    This model uses MSE loss to train the embeddings, where positive samples
    should have scores close to 0 and negative samples should have scores close to 1.

    Parameters
    ----------
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
        Defaults leave behaviour unchanged.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
        Defaults leave behaviour unchanged.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
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
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self._loaded = False
        self.extended = False
        self._mse_criterion = th.nn.MSELoss()
        self.init_module()

    def init_module(self):
        self.module = ELBEModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(self.module)

    def get_regularization_loss(self):
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for ELBE.

        Positive samples should have scores close to 0.
        Negative samples should have scores close to 1.
        """
        pos_mean = th.mean(pos_scores)
        loss = self._mse_criterion(pos_mean, th.zeros_like(pos_mean, requires_grad=False))

        if neg_scores is not None:
            neg_mean = th.mean(neg_scores)
            loss += self._mse_criterion(neg_mean, th.ones_like(neg_mean, requires_grad=False))

        return loss



