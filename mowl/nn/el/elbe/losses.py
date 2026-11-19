import torch as th
import numpy as np

def box_inclusion_score(sub_center, sub_offset, super_center, super_offset, margin):
    euc = th.abs(sub_center - super_center)
    dst = th.reshape(th.linalg.norm(th.relu(euc + sub_offset - super_offset + margin), axis=1), [-1, 1])
    return dst

def class_assertion_loss(data, ind_embed, ind_offset, class_embed, class_offset, margin, neg=False):
    i = ind_embed(data[:, 0])
    off_i = th.abs(ind_offset(data[:, 0]))
    c = class_embed(data[:, 1])
    off_c = th.abs(class_offset(data[:, 1]))
    return box_inclusion_score(i, off_i, c, off_c, margin)
                

def object_property_assertion_loss(data, ind_embed, ind_off_set, rel_embed, margin, neg=False):
    if neg:
        fn = gci2_score_neg
    else:
        fn = gci2_score

    ind_1 = ind_embed(data[:, 0])
    rel = rel_embed(data[:, 1])
    ind_2 = ind_embed(data[:, 2])

    off_i1 = th.abs(ind_off_set(data[:, 0]))
    off_i2 = th.abs(ind_off_set(data[:, 2]))
        
    return fn(ind_1, off_i1, ind_2, off_i2, rel, margin)
         
def gci0_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    return box_inclusion_score(c, off_c, d, off_d, margin)
                
def gci0_bot_loss(data, class_offset, neg=False):
    off_c = th.abs(class_offset(data[:, 0]))
    loss = th.linalg.norm(off_c, axis=1)
    return loss

def gci1_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))

    startAll = th.maximum(c - off_c, d - off_d)
    endAll = th.minimum(c + off_c, d + off_d)

    new_offset = th.abs(startAll - endAll) / 2

    cen1 = (startAll + endAll) / 2
    euc = th.abs(cen1 - e)

    # Both terms are reshaped to a column. Adding a column to a flat vector would broadcast
    # to an (n, n) matrix instead of summing the two scores of each sample.
    dst = th.reshape(th.linalg.norm(th.relu(euc + new_offset - off_e + margin), axis=1),
                     [-1, 1]) + th.reshape(th.linalg.norm(th.relu(startAll - endAll), axis=1),
                                           [-1, 1])
    return dst


def gci1_bot_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))

    euc = th.abs(c - d)
    dst = th.reshape(th.linalg.norm(th.relu(-euc + off_c + off_d + margin), axis=1), [-1, 1])
    return dst

def gci2_score(c, off_c, d, off_d, rel, margin):
    euc = th.abs(c + rel - d)
    dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + margin), axis=1), [-1, 1])
    return dst

def gci2_score_neg(c, off_c, d, off_d, rel, margin):
    euc = th.abs(c + rel - d)
    dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d - margin), axis=1), [-1, 1])
    return dst
    

def gci2_loss(data, class_embed, class_offset, rel_embed, margin, neg=False):
    if neg:
        fn = gci2_score_neg
    else:
        fn = gci2_score

    c = class_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 2]))
    return fn(c, off_c, d, off_d, r, margin)
                            
def gci3_loss(data, class_embed, class_offset, rel_embed, margin, neg=False):
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))

    euc = th.abs(c - r - d)
    dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d + margin), axis=1), [-1, 1])
    return dst



def gci3_bot_loss(data, class_offset, neg=False):
    off_c = th.abs(class_offset(data[:, 1]))
    loss = th.linalg.norm(off_c, axis=1)
    return loss


# ---------------------------------------------------------------------------------------
# Negative losses of DELE (Mashkova et al., "DELE: Deductive EL++ Embeddings for Knowledge
# Base Completion", Section 4.2, Eqs. 7-12). Used only when the module was built with
# ``all_negative_losses=True`` (see ELBEModule).
# ---------------------------------------------------------------------------------------

def _intersection(c, off_c, d, off_d):
    """Centre and offset of the intersection box as in ELBE's GCI1 loss: the offset is
    ``|upper - lower| / 2``, which is also positive when the boxes do not intersect."""
    lower = th.maximum(c - off_c, d - off_d)
    upper = th.minimum(c + off_c, d + off_d)
    return (lower + upper) / 2, th.abs(upper - lower) / 2


def _disjointness(c, off_c, d, off_d, margin):
    """``‖relu(-|c - d| + o_c + o_d + margin)‖``: zero iff the boxes are separated by at
    least ``margin`` in every dimension."""
    return th.linalg.norm(th.relu(-th.abs(c - d) + off_c + off_d + margin), dim=1,
                          keepdim=True)


def gci0_loss_neg(data, class_embed, class_offset, margin):
    """Eq. 7, :math:`A \\not\\sqsubseteq B`."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    return _disjointness(c, off_c, d, off_d, margin)


def gci1_loss_neg(data, class_embed, class_offset, margin):
    """Eq. 8, :math:`A \\sqcap B \\not\\sqsubseteq E`: Eq. 7 between the intersection box of
    A and B and the box of E."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))
    cen, off = _intersection(c, off_c, d, off_d)
    return _disjointness(cen, off, e, off_e, margin)


def gci1_bot_loss_neg(data, class_embed, class_offset, epsilon):
    """Eq. 9, :math:`A \\sqcap B \\not\\sqsubseteq \\bot`: zero iff the offset of the
    intersection box has norm at least ``epsilon``."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    _, off = _intersection(c, off_c, d, off_d)
    return th.relu(epsilon - th.linalg.norm(off, dim=1, keepdim=True))


def gci3_loss_neg(data, class_embed, class_offset, rel_embed, margin):
    """Eq. 10, :math:`\\exists r.A \\not\\sqsubseteq B`: Eq. 7 between the box of A
    translated by ``-r`` and the box of B."""
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))
    return _disjointness(c - r, off_c, d, off_d, margin)


def gci0_bot_loss_neg(data, class_offset, epsilon):
    """Eq. 11, :math:`A \\not\\sqsubseteq \\bot`: zero iff ``‖o_A‖ >= epsilon``."""
    off_c = th.abs(class_offset(data[:, 0]))
    return th.relu(epsilon - th.linalg.norm(off_c, dim=1, keepdim=True))


def gci3_bot_loss_neg(data, class_offset, epsilon):
    """Eq. 12, :math:`\\exists r.A \\not\\sqsubseteq \\bot`: zero iff ``‖o_A‖ >= epsilon``."""
    off_c = th.abs(class_offset(data[:, 1]))
    return th.relu(epsilon - th.linalg.norm(off_c, dim=1, keepdim=True))
