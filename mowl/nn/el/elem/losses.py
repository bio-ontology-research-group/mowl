import torch as th
import numpy as np


def class_assertion_loss(data, class_embed, class_rad, ind_embed, margin, neg=False):
    c = class_embed(data[:, 0])
    rc = th.abs(class_rad(data[:, 0]))
    i = ind_embed(data[:, 1])

    dist = th.linalg.norm(c - i, dim=1, keepdim=True) - rc
    loss = th.relu(dist - margin)
    return loss


def object_property_assertion_loss(data, rel_embed, ind_embed, margin, neg=False):
    # C subClassOf R some D
    subj = ind_embed(data[:, 0])
    rel = rel_embed(data[:, 1])
    obj = ind_embed(data[:, 2])
                        
    dst = th.linalg.norm(subj + rel - obj, dim=1, keepdim=True)
    score = th.relu(dst  - margin) + 10e-6
    return score

def gci0_loss(data, class_embed, class_rad, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
    loss = th.relu(dist - margin)
    return loss 

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
    rc = class_rad(data[:, 0])
    rd = class_rad(data[:, 1])

    sr = rc + rd
    dst = th.reshape(th.linalg.norm(d - c, axis=1), [-1, 1])
    return th.relu(sr - dst + margin) 


def gci2_score(data, class_embed, class_rad, rel_embed, margin):
    # C subClassOf R some D
    c = class_embed(data[:, 0])
    rE = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r

    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    score = th.relu(dst + rc - rd - margin) + 10e-6
    return score
    
def gci2_loss(data, class_embed, class_rad, rel_embed, margin, neg=False):

    if neg:
        return gci2_loss_neg(data, class_embed, class_rad, rel_embed, margin)

    else:
        score = gci2_score(data, class_embed, class_rad, rel_embed, margin)
        return score 


def gci2_loss_neg(data, class_embed, class_rad, rel_embed, margin):
    # C subClassOf R some D
    c = class_embed(data[:, 0])
    rE = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r

    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = th.relu(rc + rd - dst + margin)
    return loss 


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


def regularization_loss(class_embed, reg_factor):
    res = th.abs(th.linalg.norm(class_embed.weight, axis=1) - reg_factor).mean()
    # res = th.reshape(res, [-1, 1])
    return res

