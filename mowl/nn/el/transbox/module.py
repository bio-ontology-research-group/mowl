import mowl.nn.el.transbox.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn


class TransBoxModule(ELModule):
    """Implementation of TransBox from [yang2025]_.

    TransBox is an EL++-closed ontology embedding method: atomic concepts
    and roles are embedded as axis-aligned boxes (a center and a
    non-negative offset in R^n) and individuals as points. Complex EL++
    concepts are embedded by composing the atomic boxes (intersection for
    conjunction, box translation for existential restrictions, box addition
    for role composition).

    .. note::
        `Original implementation: <https://github.com/HuiYang1997/TransBox>`_

    """

    neg_capable_gcis = frozenset({"gci2"})

    #: TransBox implements the EL++ role axiom losses, so a model built with
    #: ``load_role_axioms=True`` can train on role inclusions and chains.
    role_axiom_capable = True

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50,
                 margin=0.1, use_enhancement=True, reg_factor=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.margin = margin
        # If True, GCIs of the form C ⊑ ∃R.D are trained against the
        # semantic enhancement C ⊑ ∃Rall.D (see the paper, Section 4.3).
        self.use_enhancement = use_enhancement
        self.reg_factor = reg_factor

        self.class_center_embedding = self._init_embedding(nb_ont_classes, embed_dim)
        self.class_offset_embedding = self._init_embedding(nb_ont_classes, embed_dim)
        self.relation_center_embedding = self._init_embedding(nb_rels, embed_dim)
        self.relation_offset_embedding = self._init_embedding(nb_rels, embed_dim)

        if self.nb_inds is not None:
            self.ind_embedding = self._init_embedding(nb_inds, embed_dim)
        else:
            self.ind_embedding = None

    @staticmethod
    def _init_embedding(num_entities, embed_dim):
        """Uniform(-1, 1) initialization, then row-normalized to unit norm."""
        embed = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embed.weight, a=-1, b=1)
        norms = th.linalg.norm(embed.weight, dim=1).reshape(-1, 1)
        embed.weight.data.div_(th.clamp_min(norms, 1e-8))
        return embed

    def class_center(self, idx):
        return self.class_center_embedding(idx)

    def class_offset(self, idx):
        return self.class_offset_embedding(idx).abs()

    def relation_center(self, idx):
        return self.relation_center_embedding(idx)

    def relation_offset(self, idx):
        return self.relation_offset_embedding(idx).abs()

    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_center, self.class_offset, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_center, self.class_offset, self.margin, neg=neg)

    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_center, self.class_offset, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_center, self.class_offset, self.margin, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_center, self.class_offset,
                           self.relation_center, self.relation_offset,
                           self.margin, enhanced=self.use_enhancement, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_center, self.class_offset,
                           self.relation_center, self.relation_offset, self.margin, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_center, self.class_offset,
                               self.relation_center, self.relation_offset, self.margin, neg=neg)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_embedding, self.class_center,
                                      self.class_offset, self.margin, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(
            data, self.ind_embedding, self.relation_center, self.relation_offset,
            self.margin, neg=neg)

    def role_inclusion_loss(self, data, neg=False):
        return L.role_inclusion_loss(data, self.relation_center, self.relation_offset,
                                     self.margin, neg=neg)

    def role_chain_loss(self, data, neg=False):
        return L.role_chain_loss(data, self.relation_center, self.relation_offset,
                                 self.margin, neg=neg)

    def regularization_loss(self):
        return L.regularization_loss(self.class_center_embedding,
                                     self.relation_offset_embedding, self.reg_factor)
