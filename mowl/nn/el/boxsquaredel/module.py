import mowl.nn.el.boxsquaredel.losses as L
from mowl.nn import ELModule
from mowl.nn.el.per_axiom_column import PerAxiomColumn
import torch as th
import torch.nn as nn


class BoxSquaredELModule(PerAxiomColumn, ELModule):
    """
    Implementation of Box :math:`^2` EL from [jackermeier2023]_.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``, which leaves the module exactly as before. See also \
:meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional

    Every per-axiom loss is a column of shape ``(B, 1)`` (:class:`~mowl.nn.el.per_axiom_column.PerAxiomColumn`;
    set ``per_axiom_column = False`` for the earlier flat ``(B,)``).

    ``gci3_bot`` (:math:`\\exists r.C \\sqsubseteq \\bot`) shrinks the head box of :math:`r`, while its
    DELE negative (``all_negative_losses=True``, Eq. 17 of DELE) uses the offset of the filler
    :math:`C`. Both are kept as published; they do not describe the same constraint.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})

    #: Normal forms that get a negative loss with ``all_negative_losses=True``.
    DELE_NEG_GCIS = frozenset({"gci0", "gci1", "gci2", "gci3",
                               "gci0_bot", "gci1_bot", "gci3_bot"})

    def enable_all_negative_losses(self, epsilon=None):
        """Switch on the DELE negative losses of all normal forms (see the class \
parameter ``all_negative_losses``) and extend :attr:`neg_capable_gcis` accordingly.

        :param epsilon: New value of :math:`\\varepsilon`, or ``None`` to keep the current one.
        :type epsilon: float, optional
        """
        if epsilon is not None:
            self.epsilon = float(epsilon)
        self.all_negative_losses = True
        self.neg_capable_gcis = frozenset(type(self).neg_capable_gcis | self.DELE_NEG_GCIS)

    def _dele_neg(self, neg):
        return neg and self.all_negative_losses
    
    margin_semantics = "slack"
    margin_attribute = "gamma"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, gamma=0, delta=2,
                 reg_factor=0.05, all_negative_losses=False, epsilon=0.01):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds

        self.embed_dim = embed_dim

        self.class_center = self.init_embeddings(nb_ont_classes, embed_dim)
        self.class_offset = self.init_embeddings(nb_ont_classes, embed_dim)

        self.head_center = self.init_embeddings(nb_rels, embed_dim)
        self.head_offset = self.init_embeddings(nb_rels, embed_dim)
        self.tail_center = self.init_embeddings(nb_rels, embed_dim)
        self.tail_offset = self.init_embeddings(nb_rels, embed_dim)

        self.bump_classes = self.init_embeddings(nb_ont_classes, embed_dim)

        if self.nb_inds is not None and self.nb_inds > 0:
            self.bump_individuals = self.init_embeddings(nb_inds, embed_dim)
            self.ind_center = self.init_embeddings(nb_inds, embed_dim)
            self.ind_offset = self.init_embeddings(nb_inds, embed_dim)
            
        else:
            self.bump_individuals = None
            self.ind_center = None
            self.ind_offset = None
            
        self.gamma = gamma
        self.delta = delta
        self.reg_factor = reg_factor

        self.epsilon = float(epsilon)
        self.all_negative_losses = False
        if all_negative_losses:
            self.enable_all_negative_losses()

    def init_embeddings(self, num_entities, embed_dim, min=-1, max=1):
        embeddings = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        embeddings.weight.data /= th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings
        
    def gci0_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci0_loss_neg(data, self.class_center, self.class_offset, self.gamma)
        return L.gci0_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci0_bot_loss_neg(data, self.class_offset, self.epsilon)
        return L.gci0_bot_loss(data, self.class_offset)
    
    def gci1_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci1_loss_neg(data, self.class_center, self.class_offset, self.gamma)
        return L.gci1_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci1_bot_loss_neg(data, self.class_center, self.class_offset,
                                       self.epsilon)
        return L.gci1_bot_loss(data, self.class_center, self.class_offset, self.gamma, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_center, self.class_offset, self.head_center,
                            self.head_offset, self.tail_center, self.tail_offset, self.bump_classes,
                           self.gamma, self.delta, neg=neg)

    def gci3_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci3_loss_neg(data, self.class_center, self.class_offset,
                                   self.head_center, self.head_offset, self.bump_classes,
                                   self.gamma, self.delta)
        return L.gci3_loss(data, self.class_center, self.class_offset, self.head_center,
                            self.head_offset, self.tail_center, self.tail_offset, self.bump_classes,
                            self.gamma, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci3_bot_loss_neg(data, self.class_offset, self.epsilon)
        return L.gci3_bot_loss(data, self.head_offset)


    #: The module implements the role inclusion and role chain losses of the paper, so
    #: :class:`~mowl.models.BoxSquaredEL` accepts ``load_role_axioms=True``.
    role_axiom_capable = True

    def role_inclusion_loss(self, data, neg=False):
        return L.role_inclusion_loss(data, self.head_center, self.head_offset,
                                     self.tail_center, self.tail_offset, self.gamma)

    def role_chain_loss(self, data, neg=False):
        return L.role_chain_loss(data, self.head_center, self.head_offset,
                                 self.tail_center, self.tail_offset, self.gamma)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_center is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_center, self.ind_offset, self.class_center, self.class_offset, self.gamma, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_center is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(data, self.ind_center, self.ind_offset, self.head_center, self.head_offset, self.tail_center, self.tail_offset, self.bump_individuals, self.gamma, self.delta, neg=neg)

    def regularization_loss(self):
        loss = L.reg_loss(self.bump_classes, self.reg_factor)
        if self.bump_individuals is not None:
            loss += L.reg_loss(self.bump_individuals, self.reg_factor)
        return loss
