"""Variants of two ELEmbeddings losses that the DELE reference code uses (the regime
:class:`~mowl.training.DELECodeLosses` switches an ELEm module to them), written on the tables
of a module so that they do not depend on how the module is built."""
import torch as th


def gci3_loss_reference(data, class_embed, class_rad, rel_embed, margin, neg=False):
    r"""The GCI3 (:math:`\exists r.C \sqsubseteq D`) loss of the DELE reference code:
    zero iff the ball of C translated by ``-r`` lies inside the ball of D (up to
    ``margin``): the radius of the filler is *added*, ``relu(‖c − r − d‖ + rc − rd −
    margin)``, where ELEmbeddings (Kulmanov et al.) subtracts it."""
    if neg:
        return gci3_loss_neg(data, class_embed, class_rad, rel_embed, margin)
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    return th.relu(euc + rc - rd - margin)


def gci3_loss_neg(data, class_embed, class_rad, rel_embed, margin):
    """Eq. 2 of DELE, :math:`\\exists r.A \\not\\sqsubseteq B`: zero iff the ball of A
    translated by ``-r`` and the ball of B are at least ``margin`` apart."""
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    dst = th.linalg.norm(c - r - d, dim=1, keepdim=True)
    return th.relu(rc + rd - dst + margin)
