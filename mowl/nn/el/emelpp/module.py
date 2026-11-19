import torch as th
import torch.nn as nn

import mowl.nn.el.emelpp.losses as L
from mowl.nn.el.elem.module import ELEmModule


class EmELPPModule(ELEmModule):
    """Implementation of EmEL++ from [mondal2021]_.

    The geometry is that of ELEmbeddings ([kulmanov2019]_): classes are :math:`n`-balls
    (centre and non-negative radius, stored as a signed number and used through its absolute
    value) and roles are translation vectors, so the embedding tables, their initialisation
    and the scoring are inherited from :class:`ELEmModule <mowl.nn.ELEmModule>`. The losses
    are the ones printed in the paper (Eq. 1-10) and differ from ELEmbeddings in three ways:

    * every loss carries the normalisation terms :math:`|\\lVert x \\rVert - 1|` of the
      embedding vectors it involves, so the module has no separate regularisation loss
      (:meth:`regularization_loss` is zero);
    * the intersection loss has a fourth term requiring the radius of :math:`E` to be at
      least the smaller radius of :math:`C` and :math:`D`;
    * there are losses for role inclusions (Eq. 8) and role chains (Eq. 9).

    The role inclusion loss is symmetric in the two roles, so
    :math:`R \\sqsubseteq S` also enforces :math:`S \\sqsubseteq R`; see the catalogue card.

    Individuals, if present, are treated as nominal classes (paper, Section 4), i.e. as balls
    with the same class losses.

    :param nb_ont_classes: Number of ontology classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (``0`` or ``None`` for none).
    :param embed_dim: Embedding dimension.
    :param margin: Margin :math:`\\gamma` of the paper.
    """

    #: Eq. 10 is the only negative loss of the paper (``C ⋢ ∃R.D``); the assertion
    #: ``R(a, b)`` is the same normal form with nominals.
    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})

    role_axiom_capable = True

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=0, embed_dim=50, margin=0.1):
        super().__init__(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                         margin=margin)

    # ----------------------------------------------------------------- helpers

    def _rad(self, idx):
        return th.abs(self.class_rad(idx))

    def _sigma(self, rel_idx):
        """Variance of the roles; ``None`` for EmEL++."""
        return None

    def _gci2(self, c, rc, rel_idx, d, rd, neg):
        rel = self.rel_embed(rel_idx)
        if neg:
            return L.gci2_neg_loss(c, rc, rel, d, rd, self.margin)
        return self._gci2_pos(c, rc, rel, rel_idx, d, rd)

    def _gci2_pos(self, c, rc, rel, rel_idx, d, rd):
        return L.gci2_loss(c, rc, rel, d, rd, self.margin, sigma=self._sigma(rel_idx),
                           sigma_reg_weight=getattr(self, "sigma_reg_weight", 1.0))

    # ------------------------------------------------------------- concept GCIs

    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(self.class_embed(data[:, 0]), self._rad(data[:, 0]),
                           self.class_embed(data[:, 1]), self._rad(data[:, 1]), self.margin)

    def gci0_bot_loss(self, data, neg=False):
        return L.bot_loss(self._rad(data[:, 0]))

    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(self.class_embed(data[:, 0]), self._rad(data[:, 0]),
                           self.class_embed(data[:, 1]), self._rad(data[:, 1]),
                           self.class_embed(data[:, 2]), self._rad(data[:, 2]), self.margin)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(self.class_embed(data[:, 0]), self._rad(data[:, 0]),
                               self.class_embed(data[:, 1]), self._rad(data[:, 1]),
                               self.margin)

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return self._gci2(self.class_embed(data[:, 0]), self._rad(data[:, 0]), data[:, 1],
                          self.class_embed(data[:, 2]), self._rad(data[:, 2]), neg)

    def gci3_loss(self, data, neg=False):
        rel_idx = data[:, 0]
        return L.gci3_loss(self.rel_embed(rel_idx), self.class_embed(data[:, 1]),
                           self._rad(data[:, 1]), self.class_embed(data[:, 2]),
                           self._rad(data[:, 2]), self.margin, sigma=self._sigma(rel_idx),
                           sigma_reg_weight=getattr(self, "sigma_reg_weight", 1.0))

    def gci3_bot_loss(self, data, neg=False):
        return L.bot_loss(self._rad(data[:, 1]))

    def gci2_score(self, data):
        return self.gci2_loss(data)

    # --------------------------------------------------------------- individuals

    def _check_individuals(self):
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function.")

    def class_assertion_loss(self, data, neg=False):
        self._check_individuals()
        return L.gci0_loss(self.ind_embed(data[:, 0]), th.abs(self.ind_rad(data[:, 0])),
                           self.class_embed(data[:, 1]), self._rad(data[:, 1]), self.margin)

    def object_property_assertion_loss(self, data, neg=False):
        self._check_individuals()
        return self._gci2(self.ind_embed(data[:, 0]), th.abs(self.ind_rad(data[:, 0])),
                          data[:, 1], self.ind_embed(data[:, 2]),
                          th.abs(self.ind_rad(data[:, 2])), neg)

    # ---------------------------------------------------------------- role axioms

    def role_inclusion_loss(self, data, neg=False):
        return L.role_inclusion_loss(self.rel_embed(data[:, 0]), self.rel_embed(data[:, 1]),
                                     self.margin)

    def role_chain_loss(self, data, neg=False):
        return L.role_chain_loss(self.rel_embed(data[:, 0]), self.rel_embed(data[:, 1]),
                                 self.rel_embed(data[:, 2]), self.margin)

    def regularization_loss(self):
        # The normalisation terms are part of each axiom loss (Eq. 1-10).
        return th.zeros((), device=self.class_embed.weight.device)


class EmELVarModule(EmELPPModule):
    """Implementation of EmEL(var) from [mohapatra2021]_.

    EmEL(var) is EmEL++ with one extra learned scalar :math:`\\sigma(r) \\geq 0` per role
    (the "variance" of the translation). It enters the losses of
    :math:`C \\sqsubseteq \\exists R.D` and :math:`\\exists R.C \\sqsubseteq D` (Eq. 7 and 8
    of the paper): it is subtracted inside the hinge, so the translated ball :math:`C + R`
    only has to lie within :math:`\\sigma(R)` of :math:`D`, and added to the loss to keep it
    finite. All other losses, including the negative loss of Eq. 10 and the role axiom losses,
    are those of EmEL++ (the paper: "the loss function for other normal forms remain the
    same"). The parameter is stored as a signed number and used through its absolute value
    (the paper does not say how non-negativity is enforced).

    .. note::
        As published, the term :math:`+\\sigma(r)` has weight 1 in every axiom, which exactly
        cancels the gain of the hinge: for any :math:`\\sigma \\geq 0`,
        :math:`\\max(0, x - \\sigma) + \\sigma \\geq \\max(0, x)`, so the total loss is never
        below that of EmEL++. The optional ``sigma_reg_weight`` (**not from the paper**,
        default 1 = the paper) scales that term so that the variance can pay off.

    :param sigma_init: Initial value of :math:`\\sigma` (the paper does not give one).
    :param sigma_reg_weight: Weight of the :math:`+\\sigma(r)` term. The paper's value is 1.
    """

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=0, embed_dim=50, margin=0.1,
                 sigma_init=0.1, sigma_reg_weight=1.0):
        super().__init__(nb_ont_classes, nb_rels, nb_inds, embed_dim=embed_dim, margin=margin)
        self.sigma_reg_weight = sigma_reg_weight
        self.rel_sigma = nn.Embedding(nb_rels, 1)
        nn.init.constant_(self.rel_sigma.weight, sigma_init)

    def _sigma(self, rel_idx):
        return th.abs(self.rel_sigma(rel_idx))
