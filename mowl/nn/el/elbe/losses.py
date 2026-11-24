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



def gci3_inclusion_loss(data, class_embed, class_offset, rel_embed, margin):
    """:math:`\\exists r.C \\sqsubseteq D` as the inclusion of the box of :math:`C`,
    translated by :math:`-r`, in the box of :math:`D` (``nf4Loss`` of ``model/Elbe.py`` in
    the TransBox code)."""
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))
    euc = th.abs(c - r - d)
    return th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + margin), axis=1), [-1, 1])


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
    cen, off, _, _ = _intersection_corners(c, off_c, d, off_d)
    return cen, off


def _intersection_corners(c, off_c, d, off_d):
    """As :func:`_intersection`, also returning the lower and upper corners (the
    intersection is non-empty iff ``lower <= upper``)."""
    lower = th.maximum(c - off_c, d - off_d)
    upper = th.minimum(c + off_c, d + off_d)
    return (lower + upper) / 2, th.abs(upper - lower) / 2, lower, upper


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
    A and B and the box of E, plus (as in the reference code, which adds a term the
    paper's Eq. 8 omits) the distance of the intersection box from being non-empty."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))
    cen, off, lower, upper = _intersection_corners(c, off_c, d, off_d)
    return (_disjointness(cen, off, e, off_e, margin)
            + th.linalg.norm(th.relu(lower - upper), dim=1, keepdim=True))


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


# ---------------------------------------------------------------------------------------
# The loss variants that the reference implementation of DELE
# (bio-ontology-research-group/DELE, losses/elbe_losses.py) trains ELBE with. They differ
# from the ELBE code of Peng et al. 2022 (and therefore from the losses above, which are
# faithful to it) in four points, none of them mentioned in the DELE paper:
#
# * the margin of the inclusion losses is a *slack* (``- margin``) instead of a strict
#   gap (``+ margin``) -- as in ELEmbeddings and Box2EL;
# * the GCI3 loss is corrected: ``relu(|c - r - d| + o_c - o_d - margin)`` (the
#   inclusion of the translated box of C in the box of D) instead of the original
#   ``relu(|c - r - d| - o_c - o_d + margin)``; see also Mashkova & Hoehndorf 2026,
#   Appendix C;
# * the GCI2 negative loss penalises overlap of the translated C-box and the D-box
#   (``relu(-|c + r - d| + o_c + o_d + margin)``) instead of their separation;
# * the GCI1 negative loss adds the distance of the intersection box from being
#   non-empty (see gci1_loss_neg above).
#
# The regime ``dele-code-losses`` switches a module to these variants; the default losses
# above are unchanged. Evaluation through the module (module(data, "gci0")) uses the
# switched loss, as the reference's eval_method does.
# ---------------------------------------------------------------------------------------

def gci0_loss_dele(data, class_embed, class_offset, margin):
    r"""GCI0 (:math:`C \sqsubseteq D`) loss of the DELE reference code (slack margin)."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    euc = th.abs(c - d)
    return th.linalg.norm(th.relu(euc + off_c - off_d - margin), dim=1).reshape(-1, 1)


def gci1_loss_dele(data, class_embed, class_offset, margin):
    r"""GCI1 (:math:`C \sqcap D \sqsubseteq E`) loss of the DELE reference code (slack
    margin)."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))
    cen, off, lower, upper = _intersection_corners(c, off_c, d, off_d)
    euc = th.abs(cen - e)
    return (th.linalg.norm(th.relu(euc + off - off_e - margin), dim=1).reshape(-1, 1)
            + th.linalg.norm(th.relu(lower - upper), dim=1).reshape(-1, 1))


def gci2_score_dele(c, off_c, d, off_d, rel, margin):
    r"""GCI2 (:math:`C \sqsubseteq \exists r.D`) score of the DELE reference code."""
    euc = th.abs(c + rel - d)
    return th.linalg.norm(th.relu(euc + off_c - off_d - margin), dim=1).reshape(-1, 1)


def gci2_score_neg_dele(c, off_c, d, off_d, rel, margin):
    """GCI2 negative score of the DELE reference code: zero iff the translated C-box and
    the D-box are separated by ``margin`` in every dimension."""
    euc = th.abs(c + rel - d)
    return th.linalg.norm(th.relu(-euc + off_c + off_d + margin), dim=1).reshape(-1, 1)


def gci2_loss_dele(data, class_embed, class_offset, rel_embed, margin, neg=False):
    if neg:
        fn = gci2_score_neg_dele
    else:
        fn = gci2_score_dele
    c = class_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 2]))
    return fn(c, off_c, d, off_d, r, margin)


def gci3_loss_dele(data, class_embed, class_offset, rel_embed, margin):
    r"""GCI3 (:math:`\exists r.C \sqsubseteq D`) loss of the DELE reference code (corrected
    inclusion, slack margin)."""
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))
    euc = th.abs(c - r - d)
    return th.linalg.norm(th.relu(euc + off_c - off_d - margin), dim=1).reshape(-1, 1)
