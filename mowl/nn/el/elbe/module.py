import mowl.nn.el.elbe.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn

class ELBEModule(ELModule):
    """Implementation of ELBE from [peng2020]_.
    """
    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)

        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_offset.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_offset.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_offset.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.rel_embed.weight.data, axis=1).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        if self.nb_inds is not None:
            self.ind_embed = nn.Embedding(self.nb_inds, embed_dim)
            nn.init.uniform_(self.ind_embed.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.ind_embed.weight.data, axis=1).reshape(-1, 1)
            self.ind_embed.weight.data /= weight_data_normalized

            self.ind_offset = nn.Embedding(self.nb_inds, embed_dim)
            nn.init.uniform_(self.ind_offset.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.ind_offset.weight.data, axis=1).reshape(-1, 1)
            self.ind_offset.weight.data /= weight_data_normalized
        else:
            self.ind_embed = None
            self.ind_offset = None
        
        self.margin = margin

    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_offset, neg=neg)
    
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_embed, self.class_offset, self.rel_embed,
                           self.margin, neg=neg)

    def gci2_score(self, data):
        return L.gci2_score(data, self.class_embed, self.class_offset, self.rel_embed,
                            self.margin)
    
    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_embed, self.class_offset, self.rel_embed,
                           self.margin, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_offset, neg=neg)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_embed, self.ind_offset, self.class_embed, self.class_offset, self.margin, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(data, self.ind_embed, self.ind_offset, self.rel_embed, self.margin, neg=neg)
