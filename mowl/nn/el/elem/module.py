import mowl.nn.el.elem.losses as L
from mowl.nn import ELModule
from torch import nn
import torch as th
from deprecated.sphinx import versionchanged

@versionchanged(version="0.4.0", reason="The class ELEmModule receives an optional parameter nb_inds")
class ELEmModule(ELModule):
    """
    Implementation of ELEmbeddings from [kulmanov2019]_.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``, which leaves the module exactly as before. See also \
:meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional
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

    #: Form of the negative loss of :math:`C \not\sqsubseteq \exists R.D`: ``"hinge"``
    #: (default) :math:`\max(0, r_C + r_D - \lVert c + R - d \rVert + \gamma)`, or
    #: ``"linear"``, the same expression without the hinge, as in the ELEm and EmEL++
    #: baselines of the Box\ :sup:`2`\ EL code. Set by the training regime
    #: :class:`~mowl.training.UnboundedNegativeLoss`; the default leaves the module unchanged.
    negative_loss_form = "hinge"

    #: Where the unit-norm penalty :math:`|\lVert c \rVert - 1|` of the class centres is
    #: applied: ``"global"`` (default) once per step, averaged over all classes
    #: (:meth:`regularization_loss`), or ``"per_axiom"``, added to the loss of every axiom
    #: for every class centre it contains (including negatives), with
    #: :meth:`regularization_loss` zero, as in the original ELEmbeddings code and the
    #: baselines of the Box\ :sup:`2`\ EL code. Set by the training regime
    #: :class:`~mowl.training.PerAxiomNormPenalty`.
    norm_penalty = "global"

    def _penalised(self, loss, data, columns):
        """``loss`` plus, with ``norm_penalty="per_axiom"``, the unit-norm penalty of the
        class centres in ``columns`` of ``data``."""
        if self.norm_penalty != "per_axiom":
            return loss
        for col in columns:
            centre = self.class_embed(data[:, col])
            loss = loss + th.abs(th.linalg.norm(centre, dim=1, keepdim=True) - self.reg_norm)
        return loss

    def __init__(self, nb_ont_classes, nb_rels, nb_inds, embed_dim=50, margin=0.1, reg_norm=1,
                 all_negative_losses=False, epsilon=0.01):
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

        self.epsilon = float(epsilon)
        self.all_negative_losses = False
        if all_negative_losses:
            self.enable_all_negative_losses()

    def gci0_loss(self, data, neg=False):
        shift = getattr(self, "subclass_relation", None)
        if self._dele_neg(neg):
            loss = L.gci0_loss_neg(data, self.class_embed, self.class_rad, self.margin)
        elif shift is not None:
            # C ⊑ D as a translation by a learned "SubClassOf" vector (set by the training
            # regime EmELppCodeObjective, as in the ELEmbeddings/EmEL++ Keras code).
            c = self.class_embed(data[:, 0]) + shift
            d = self.class_embed(data[:, 1])
            rc = th.abs(self.class_rad(data[:, 0]))
            rd = th.abs(self.class_rad(data[:, 1]))
            loss = L.ball_inclusion_score(c, rc, d, rd, self.margin)
        else:
            loss = L.gci0_loss(data, self.class_embed, self.class_rad, self.margin, neg=neg)
        return self._penalised(loss, data, (0, 1))

    def gci0_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci0_bot_loss_neg(data, self.class_rad, self.epsilon)
        return L.gci0_bot_loss(data, self.class_rad)
        
    def gci1_loss(self, data, neg=False):
        if self._dele_neg(neg):
            loss = L.gci1_loss_neg(data, self.class_embed, self.class_rad, self.margin)
        else:
            loss = L.gci1_loss(data, self.class_embed, self.class_rad, self.margin, neg=neg)
        return self._penalised(loss, data, (0, 1, 2))

    def gci1_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            loss = L.gci1_bot_loss_neg(data, self.class_embed, self.class_rad, self.margin)
        else:
            loss = L.gci1_bot_loss(data, self.class_embed, self.class_rad, self.margin,
                                   neg=neg)
        return self._penalised(loss, data, (0, 1))

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        loss = L.gci2_loss(data, self.class_embed, self.class_rad, self.rel_embed,
                           self.margin, neg=neg,
                           linear_neg=self.negative_loss_form == "linear")
        return self._penalised(loss, data, (0, 2))

    def gci3_loss(self, data, neg=False):
        if self._dele_neg(neg):
            loss = L.gci3_loss_neg(data, self.class_embed, self.class_rad, self.rel_embed,
                                   self.margin)
        else:
            loss = L.gci3_loss(data, self.class_embed, self.class_rad, self.rel_embed,
                               self.margin, neg=neg)
        return self._penalised(loss, data, (1, 2))

    def gci3_bot_loss(self, data, neg=False):
        if self._dele_neg(neg):
            return L.gci3_bot_loss_neg(data, self.class_rad, self.epsilon)
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
        if self.norm_penalty == "per_axiom":
            return th.zeros((), device=self.class_embed.weight.device)
        return L.regularization_loss(self.class_embed, self.ind_embed, self.reg_norm)
