import torch as th
import numpy as np

def gci0_loss(data, class_embed, class_rad, class_reg, margin, neg = False):
    c = class_embed(data[:,0])
    d = class_embed(data[:,1])
    rc = th.abs(class_rad(data[:,0]))
    rd = th.abs(class_rad(data[:,1]))
    dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
    loss = th.relu(dist - margin)
    return loss + class_reg(c) + class_reg(d)

def gci1_loss(data, class_embed, class_rad, class_reg, margin, neg = False):
    c = class_embed(data[:,0])
    d = class_embed(data[:,1])
    e = class_embed(data[:,2])
    rc = th.abs(class_rad(data[:,0]))
    rd = th.abs(class_rad(data[:,1]))
    re = th.abs(class_rad(data[:,2]))
    
    sr = rc + rd
    
    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
    dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
    loss = (th.relu(dst - sr - margin)
            + th.relu(dst2 - rc - margin)
            + th.relu(dst3 - rd - margin))
    
    return loss + class_reg(c) + class_reg(d) + class_reg(e)

def gci1_bot_loss(data, class_embed, class_rad, class_reg, margin, neg = False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = class_rad(data[:, 0])
    rd = class_rad(data[:, 1])
    
    sr = rc + rd
    dst = th.reshape(th.linalg.norm(d - c, axis=1), [-1, 1])
    return th.relu(sr - dst + margin) + class_reg(c) + class_reg(d)


def gci2_loss(data, class_embed, class_rad, rel_embed, class_reg, margin, neg = False):

    if neg:
        return gci2_loss_neg(data, class_embed, class_rad, rel_embed, class_reg, margin)

    else:
        # C subClassOf R some D
        c = class_embed(data[:,0])
        rE = rel_embed(data[:,1])
        d = class_embed(data[:,2])
    
        rc = th.abs(class_rad(data[:,0]))
        rd = th.abs(class_rad(data[:,2]))
        # c should intersect with d + r
    
        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = th.relu(dst + rc - rd  - margin)
        return loss + class_reg(c) + class_reg(d)


def gci2_loss_neg(data, class_embed, class_rad, rel_embed, class_reg, margin):
    # C subClassOf R some D
    c = class_embed(data[:,0])
    rE = rel_embed(data[:,1])
    d = class_embed(data[:,2])
    
    rc = th.abs(class_rad(data[:,0]))
    rd = th.abs(class_rad(data[:,2]))
    # c should intersect with d + r
    
    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = th.relu(rc + rd - dst  + margin)
    return loss + class_reg(c) + class_reg(d)

def gci3_loss(data, class_embed, class_rad, rel_embed, class_reg, margin, neg = False):
    # R some C subClassOf D
    rE = rel_embed(data[:,0])
    c = class_embed(data[:,1])
    d = class_embed(data[:,2])
    rc = th.abs(class_rad(data[:,1]))
    rd = th.abs(class_rad(data[:,2]))

    euc = th.linalg.norm(c - rE  - d, dim=1, keepdim=True)
    loss = th.relu(euc - rc - rd - margin)
    return loss + class_reg(c) + class_reg(d)

