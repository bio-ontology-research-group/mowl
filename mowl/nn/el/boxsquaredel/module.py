import mowl.nn.el.boxsquaredel.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn


class BoxSquaredELModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, gamma=0, delta = 2, reg_factor = 0.05):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.embed_dim = embed_dim

        self.class_center = self.init_embeddings(nb_ont_classes, embed_dim)
        self.class_offset = self.init_embeddings(nb_ont_classes, embed_dim)

        self.head_center = self.init_embeddings(nb_rels, embed_dim)
        self.head_offset = self.init_embeddings(nb_rels, embed_dim)
        self.tail_center = self.init_embeddings(nb_rels, embed_dim)
        self.tail_offset = self.init_embeddings(nb_rels, embed_dim)

        self.bump = self.init_embeddings(nb_ont_classes, embed_dim)

        self.gamma = gamma
        self.delta = delta
        self.reg_factor = reg_factor

    def init_embeddings(self, num_entities, embed_dim, min=-1, max=1):
        embeddings = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        embeddings.weight.data /= th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings
        
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_offset)
    
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_center, self.class_offset, self.head_center,
                            self.head_offset, self.tail_center, self.tail_offset, self.bump,
                           self.gamma, self.delta, self.reg_factor, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_center, self.class_offset, self.head_center,
                            self.head_offset, self.tail_center, self.tail_offset, self.bump,
                            self.gamma, self.reg_factor, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.head_offset)
