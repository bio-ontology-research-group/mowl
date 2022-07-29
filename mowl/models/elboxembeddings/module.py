import mowl.models.elboxembeddings.losses as L
from mowl.nn.elmodule import ELModule
import torch as th
import torch.nn as nn

class ELBoxModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.embed_dim = embed_dim
        
        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        self.class_embed.weight.data /= th.linalg.norm(self.class_embed.weight.data,axis=1).reshape(-1,1)

        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_offset.weight, a=-1, b=1)
        self.class_offset.weight.data /= th.linalg.norm(self.class_offset.weight.data,axis=1).reshape(-1,1)

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        self.rel_embed.weight.data /= th.linalg.norm(self.rel_embed.weight.data,axis=1).reshape(-1,1)
        
        self.margin = margin

    def gci0_loss(self, data, neg = False):
        return L.gci0_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci1_loss(self, data, neg = False):
        return L.gci1_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci1_bot_loss(self, data, neg = False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci2_loss(self, data, neg = False):
        return L.gci2_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.margin, neg = neg)
    def gci3_loss(self, data, neg = False):
        return L.gci3_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.margin, neg = neg)
