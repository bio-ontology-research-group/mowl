import mowl.nn.el.elem.losses as L
from mowl.nn import ELModule
from torch import nn
import torch as th
from deprecated.sphinx import versionchanged

@versionchanged(version="0.4.0", reason="The class ELEmModule receives an optional parameter nb_inds")
class ELEmModule(ELModule):
    """
    Implementation of ELEmbeddings from [kulmanov2019]_.

    """
    
    
    def __init__(self, nb_ont_classes, nb_rels, nb_inds, embed_dim=50, margin=0.1, reg_norm=1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        if self.nb_inds == 0:
            self.nb_inds = None

        self.reg_norm = reg_norm
        self.embed_dim = embed_dim

        
        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)

        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)

        weight_data_normalized = th.linalg.norm(self.class_rad.weight.data, axis=1).reshape(-1, 1)
        self.class_rad.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)

        weight_data_normalized = th.linalg.norm(self.rel_embed.weight.data, axis=1).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        if self.nb_inds is not None:
            self.ind_embed = nn.Embedding(self.nb_inds, embed_dim)
            nn.init.uniform_(self.ind_embed.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.ind_embed.weight.data, axis=1).reshape(-1, 1)
            self.ind_embed.weight.data /= weight_data_normalized

            self.ind_rad = nn.Embedding(self.nb_inds, 1)
            nn.init.uniform_(self.ind_rad.weight, a=-1, b=1)
            
        else:
            self.ind_embed = None
            self.ind_rad = None
            
        self.margin = margin

    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_embed, self.class_rad, self.margin,
                           neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_rad)
        
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_embed, self.class_rad, self.margin,
                           neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_rad, self.margin,
                               neg=neg)

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return L.gci2_loss(data, self.class_embed, self.class_rad, self.rel_embed,
                           self.margin, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_embed, self.class_rad, self.rel_embed,
                           self.margin, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_rad)


    def gci2_score(self, data):
        return L.gci2_score(data, self.class_embed, self.class_rad, self.rel_embed, self.margin)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_embed, self.ind_rad, self.class_embed, self.class_rad, self.margin, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(data, self.ind_embed, self.ind_rad, self.rel_embed, self.margin, neg=neg)

    
    def regularization_loss(self):
        return L.regularization_loss(self.class_embed, self.ind_embed, self.reg_norm)
