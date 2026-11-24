import mowl.nn.el.boxel.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BoxELModule(ELModule):
    """Implementation of BoxEL from [xiong2022]_.

    .. note::
        `Original implementation: <https://github.com/Box-EL/BoxEL>`_

    """

    neg_capable_gcis = frozenset({"gci2"})
    entity_embedding_names = (
        ("min_embedding", "delta_embedding"),
        ("relation_embedding", "scaling_embedding"),
        ("ind_embedding", "ind_delta_embedding"),
    )

    #: Which regulariser is used: ``"global"`` (default), the mean over all class boxes of
    #: the excess of the upper corners over 1 and of negative lower corners, once per step;
    #: or ``"box2el_code"``, the regulariser of the BoxEL baseline of the
    #: Box\ :sup:`2`\ EL code (:func:`~mowl.nn.el.boxel.losses.box2el_code_side_regularizer`
    #: of every box of every batch, including negatives, intersections and translated
    #: boxes). Set by the training regime :class:`~mowl.training.Box2ELBaselineLosses`.
    regulariser = "global"

    def _record_regulariser(self, gci_name, data):
        if self.regulariser != "box2el_code" or not (self.training and th.is_grad_enabled()):
            return
        boxes = L.axiom_boxes(gci_name, data, self.min_embedding, self.delta_embedding,
                              self.relation_embedding, self.scaling_embedding)
        for box in boxes:
            self._pending_regulariser = (self._pending_regulariser
                                         + L.box2el_code_side_regularizer(box))

    def forward(self, gci, gci_name, neg=False):
        self._record_regulariser(gci_name, gci)
        return super().forward(gci, gci_name, neg=neg)
    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 min_bounds=[1e-4, 0.2], delta_bounds=[-0.1, 0],
                 relation_bounds=[-0.1, 0.1],
                 scaling_bounds=[0.9,1.1], temperature=1.0):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.temperature = temperature
        self._pending_regulariser = 0
        
        self.min_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, min_bounds)
        self.delta_embedding = self.init_entity_embedding(nb_ont_classes, embed_dim, delta_bounds)
        self.relation_embedding = self.init_entity_embedding(nb_rels, embed_dim, relation_bounds)
        self.scaling_embedding = self.init_entity_embedding(nb_rels, embed_dim, scaling_bounds)

        if self.nb_inds is not None:
            self.ind_embedding = self.init_entity_embedding(nb_inds, embed_dim, min_bounds)
            self.ind_delta_embedding = self.init_entity_embedding(nb_inds, embed_dim, delta_bounds)
        else:
            self.ind_embedding = None
            self.ind_delta_embedding = None
        

    def init_entity_embedding(self, num_entities, embed_dim, bounds):
        embed = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embed.weight, bounds[0], bounds[1])
        return embed


    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.min_embedding, self.delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_embedding, self.ind_delta_embedding, self.min_embedding, self.delta_embedding, self.temperature, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(data, self.ind_embedding, self.ind_delta_embedding, self.relation_embedding, self.scaling_embedding, self.temperature, neg=neg)

    def regularization_loss(self):
        if self.regulariser == "box2el_code":
            # The terms recorded for the batches of this step (see _record_regulariser).
            out = self._pending_regulariser
            self._pending_regulariser = 0
            if isinstance(out, int):
                return th.zeros((), device=self.min_embedding.weight.device)
            return out
        return L.regularization_loss(self.min_embedding, self.delta_embedding)
