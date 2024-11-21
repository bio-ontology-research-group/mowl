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


def gci2_loss(data, class_embed, class_rad, rel_embed, margin, neg=False):

    if neg:
        fn = gci2_score_neg
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

