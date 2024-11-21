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

    dst = th.reshape(th.linalg.norm(th.relu(euc + new_offset - off_e + margin), axis=1),
                     [-1, 1]) + th.linalg.norm(th.relu(startAll - endAll), axis=1)
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
