import torch as th
import numpy as np


def ball_inclusion_score(sub_center, sub_rad, super_center, super_rad, margin):
    dist = th.linalg.norm(sub_center - super_center, dim=1, keepdim=True) + sub_rad - super_rad
    score = th.relu(dist - margin)
    return score

def class_assertion_loss(data, ind_embed, ind_rad, class_embed, class_rad, margin, neg=False):
    i = ind_embed(data[:, 0])
    ri = th.abs(ind_rad(data[:, 0]))
    c = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 1]))

    return ball_inclusion_score(i, ri, c, rc, margin)
                
def object_property_assertion_loss(data, ind_embed, ind_rad, rel_embed, margin, neg=False):
    if neg:
        fn = gci2_score_neg
    else:
        fn = gci2_score

    ind_1 = ind_embed(data[:, 0])
    rel = rel_embed(data[:, 1])
    ind_2 = ind_embed(data[:, 2])

    rad_i1 = th.abs(ind_rad(data[:, 0]))
    rad_i2 = th.abs(ind_rad(data[:, 2]))

    return fn(ind_1, rad_i1, ind_2, rad_i2, rel, margin)
    
def gci0_loss(data, class_embed, class_rad, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    return ball_inclusion_score(c, rc, d, rd, margin)
            
def gci0_bot_loss(data, class_rad, neg=False):
    rc = class_rad(data[:, 0])
    return rc


def gci1_loss(data, class_embed, class_rad, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd

    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
    dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
    loss = (th.relu(dst - sr - margin) + th.relu(dst2 - rc - margin) + th.relu(dst3 - rd - margin))

    return loss 


def gci1_bot_loss(data, class_embed, class_rad, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd
    dst = th.reshape(th.linalg.norm(d - c, axis=1), [-1, 1])
    return th.relu(sr - dst + margin) 


def gci2_score(c, rad_c, d, rad_d, rel, margin):
    # C subClassOf R some D
                        
    dst = th.linalg.norm(c + rel - d, dim=1, keepdim=True)
    score = th.relu(dst + rad_c - rad_d - margin) + 10e-6
    return score

def gci2_score_neg(c, rad_c, d, rad_d, rel, margin):
    dst = th.linalg.norm(c + rel - d, dim=1, keepdim=True)
    loss = th.relu(rad_c + rad_d - dst + margin)
    return loss 


def gci2_score_neg_linear(c, rad_c, d, rad_d, rel, margin):
    """The negative loss of :func:`gci2_score_neg` without the hinge (ELEm baseline of the
    Box2EL code, ``model/Elem.py``, ``neg_loss``): it keeps decreasing as the balls move
    apart and as their radii shrink."""
    dst = th.linalg.norm(c + rel - d, dim=1, keepdim=True)
    return rad_c + rad_d - dst + margin


def gci2_loss(data, class_embed, class_rad, rel_embed, margin, neg=False, linear_neg=False):

    if neg:
        fn = gci2_score_neg_linear if linear_neg else gci2_score_neg
    else:
        fn = gci2_score

    c = class_embed(data[:, 0])
    rel = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))

    score = fn(c, rc, d, rd, rel, margin)
    return score

def gci3_loss(data, class_embed, class_rad, rel_embed, margin, neg=False):
    # R some C subClassOf D
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))

    euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    loss = th.relu(euc - rc - rd - margin)
    return loss 

def gci3_bot_loss(data, class_rad, neg=False):
    rc = class_rad(data[:, 1])
    return rc


def regularization_loss(class_embed, ind_embed = None, reg_norm = 1):
    reg = th.abs(th.linalg.norm(class_embed.weight, axis=1) - reg_norm).mean()
    if ind_embed is not None:
        reg += th.abs(th.linalg.norm(ind_embed.weight, axis=1) - reg_norm).mean()
    return reg


# ---------------------------------------------------------------------------------------
# Variant of the positive GCI3 loss used by the reference implementation of DELE
# (bio-ontology-research-group/DELE, losses/elembeddings_losses.py): the radius of the
# filler is *added*, ``relu(‖c − r − d‖ + rc − rd − margin)``, the inclusion of the
# translated ball of C in the ball of D. mOWL keeps the original ELEmbeddings loss
# ``relu(euc − rc − rd − margin)`` above (faithful to Kulmanov et al.), which subtracts
# rc; the DELE code silently corrects it (see also Mashkova & Hoehndorf 2026,
# Appendix C). Switching is done by the regime ``dele-code-losses``.
# ---------------------------------------------------------------------------------------

def gci3_loss_reference(data, class_embed, class_rad, rel_embed, margin, neg=False):
    r"""The GCI3 (:math:`\exists r.C \sqsubseteq D`) loss of the DELE reference code:
    zero iff the ball of C translated by ``-r`` lies inside the ball of D (up to
    ``margin``)."""
    if neg:
        return gci3_loss_neg(data, class_embed, class_rad, rel_embed, margin)
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    return th.relu(euc + rc - rd - margin)


# ---------------------------------------------------------------------------------------
# Negative losses of DELE (Mashkova et al., "DELE: Deductive EL++ Embeddings for Knowledge
# Base Completion", Section 4.1, Eqs. 1-6). They are used only when the module was built
# with ``all_negative_losses=True`` (see ELEmModule). The regularisation terms
# |‖f(x)‖ - 1| of Eqs. 1-4 are not repeated here: ELEmModule.regularization_loss already
# applies them to every class centre.
# ---------------------------------------------------------------------------------------

def gci0_loss_neg(data, class_embed, class_rad, margin):
    """Eq. 1, :math:`A \\not\\sqsubseteq B`: zero iff the balls of A and B are at least
    ``margin`` apart, i.e. ``‖c_A - c_B‖ >= r_A + r_B + margin``."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    dst = th.linalg.norm(c - d, dim=1, keepdim=True)
    return th.relu(rc + rd - dst + margin)


def gci1_loss_neg(data, class_embed, class_rad, margin):
    """Eq. 3, :math:`A \\sqcap B \\not\\sqsubseteq E`: zero iff the balls of A and B overlap
    (``‖c_A - c_B‖ <= r_A + r_B + margin``) and the centre of E lies outside both balls by at
    least ``margin``."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    dst = th.linalg.norm(c - d, dim=1, keepdim=True)
    dst_ce = th.linalg.norm(c - e, dim=1, keepdim=True)
    dst_de = th.linalg.norm(d - e, dim=1, keepdim=True)
    loss = th.relu(dst - rc - rd - margin) + th.relu(rc - dst_ce + margin)
    return loss + th.relu(rd - dst_de + margin)


def gci3_loss_neg(data, class_embed, class_rad, rel_embed, margin):
    """Eq. 2, :math:`\\exists r.A \\not\\sqsubseteq B`: zero iff the ball of A translated by
    ``-r`` and the ball of B are at least ``margin`` apart."""
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    dst = th.linalg.norm(c - r - d, dim=1, keepdim=True)
    return th.relu(rc + rd - dst + margin)


def gci0_bot_loss_neg(data, class_rad, epsilon):
    """Eq. 5, :math:`A \\not\\sqsubseteq \\bot`: zero iff the radius of A is at least
    ``epsilon``."""
    return th.relu(epsilon - th.abs(class_rad(data[:, 0])))


def gci1_bot_loss_neg(data, class_embed, class_rad, margin):
    """Eq. 4, :math:`A \\sqcap B \\not\\sqsubseteq \\bot`: zero iff the balls of A and B
    overlap, ``‖c_A - c_B‖ <= r_A + r_B + margin``."""
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    dst = th.linalg.norm(c - d, dim=1, keepdim=True)
    return th.relu(dst - rc - rd - margin)


def gci3_bot_loss_neg(data, class_rad, epsilon):
    """Eq. 6, :math:`\\exists r.A \\not\\sqsubseteq \\bot`: zero iff the radius of A is at
    least ``epsilon``."""
    return th.relu(epsilon - th.abs(class_rad(data[:, 1])))
