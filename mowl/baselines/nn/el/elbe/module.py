import mowl.baselines.nn.el.elbe.losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn

class ELBEModule(ELModule):
    """Implementation of ELBE from [peng2020]_.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``, which leaves the module exactly as before. See also \
:meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional
    :param gci3_form: Loss of :math:`\\exists r.C \\sqsubseteq D`. ``"overlap"`` (default, \
as in the ELBE and Box2EL codes): :math:`\\lVert\\max(0, |c - r - d| - o_C - o_D + m)\\rVert`, \
zero as soon as the translated box of :math:`C` overlaps the box of :math:`D`. \
``"inclusion"`` (the corrected loss of the TransBox code, ``model/Elbe.py``): \
:math:`\\lVert\\max(0, |c - r - d| + o_C - o_D + m)\\rVert`.
    :type gci3_form: str, optional
    :param init: ``"separate"`` (default): centres and offsets are each drawn from \
:math:`U(-1, 1)^n` and scaled to unit norm. ``"joint"`` (the Box2EL and TransBox codes): \
centre and offset of a class form one row of :math:`U(-1, 1)^{2n}` scaled to unit norm.
    :type init: str, optional
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

    margin_semantics = "strict"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 all_negative_losses=False, epsilon=0.01, gci3_form="overlap",
                 init="separate"):
        super().__init__()
        if gci3_form not in ("overlap", "inclusion"):
            raise ValueError("gci3_form must be 'overlap' or 'inclusion'.")
        if init not in ("separate", "joint"):
            raise ValueError("init must be 'separate' or 'joint'.")
        self.gci3_form = gci3_form
        self.init = init
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

        if init == "joint":
            weight = th.empty(self.nb_ont_classes, 2 * embed_dim)
            nn.init.uniform_(weight, a=-1, b=1)
            weight = weight / th.linalg.norm(weight, dim=1, keepdim=True)
            with th.no_grad():
                self.class_embed.weight.copy_(weight[:, :embed_dim])
                self.class_offset.weight.copy_(weight[:, embed_dim:])

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

        self.epsilon = float(epsilon)
        self.all_negative_losses = False
        if all_negative_losses:
            self.enable_all_negative_losses()

    def gci0_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci0_loss_neg(data, self.class_embed, self.class_offset, self.margin)
        return L.gci0_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci0_bot_loss_neg(data, self.class_offset, self.epsilon)
        return L.gci0_bot_loss(data, self.class_offset, neg=neg)
    
    def gci1_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci1_loss_neg(data, self.class_embed, self.class_offset, self.margin)
        return L.gci1_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci1_bot_loss_neg(data, self.class_embed, self.class_offset,
                                       self.epsilon)
        return L.gci1_bot_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_embed, self.class_offset, self.rel_embed,
                           self.margin, neg=neg)

    def gci2_score(self, data):
        return L.gci2_score(data, self.class_embed, self.class_offset, self.rel_embed,
                            self.margin)
    
    def gci3_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci3_loss_neg(data, self.class_embed, self.class_offset, self.rel_embed,
                                   self.margin)
        if self.gci3_form == "inclusion" and not neg:
            return L.gci3_inclusion_loss(data, self.class_embed, self.class_offset,
                                         self.rel_embed, self.margin)
        return L.gci3_loss(data, self.class_embed, self.class_offset, self.rel_embed,
                           self.margin, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci3_bot_loss_neg(data, self.class_offset, self.epsilon)
        return L.gci3_bot_loss(data, self.class_offset, neg=neg)

    def class_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.class_assertion_loss(data, self.ind_embed, self.ind_offset, self.class_embed, self.class_offset, self.margin, neg=neg)

    def object_property_assertion_loss(self, data, neg=False):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this loss function.")
        return L.object_property_assertion_loss(data, self.ind_embed, self.ind_offset, self.rel_embed, self.margin, neg=neg)
