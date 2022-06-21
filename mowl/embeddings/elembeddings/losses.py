import torch as th

def class_dist(data, class_norm, class_embed, class_rad):
        c = class_norm(class_embed(data[:, 0]))
        d = class_norm(class_embed(data[:, 1]))
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
def gci0_loss(data, class_norm, class_embed, class_rad, margin):
    
    pos_dist = class_dist(data, class_norm, class_embed, class_rad)
    loss = th.relu(pos_dist - margin)
    return loss

def gci1_loss(data, class_norm, class_embed, class_rad, margin):
    c = class_norm(class_embed(data[:, 0]))
    d = class_norm(class_embed(data[:, 1]))
    e = class_norm(class_embed(data[:, 2]))
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))
    re = th.abs(class_rad(data[:, 2]))
    
    sr = rc + rd
    dst = th.linalg.norm(c - d, dim=1, keepdim=True)
    dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
    dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
    loss = (th.relu(dst - sr - margin)
            + th.relu(dst2 - rc - margin)
            + th.relu(dst3 - rd - margin))
    
    return loss

def gci3_loss(data, class_norm, class_embed, class_rad, rel_embed, margin):
    # R some C subClassOf D
    n = data.shape[0]
    rE = rel_embed(data[:, 0])
    c = class_norm(class_embed(data[:, 1]))
    d = class_norm(class_embed(data[:, 2]))
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    
    rSomeC = c + rE
    euc = th.linalg.norm(c - rE  - d, dim=1, keepdim=True)
    loss = th.relu(euc - rc - rd - margin)
    return loss


def gci2_loss(data, class_norm, class_embed, class_rad, rel_embed, margin):
    # C subClassOf R some D
    n = data.shape[0]
    c = class_norm(class_embed(data[:, 0]))
    rE = rel_embed(data[:, 1])
    d = class_norm(class_embed(data[:, 2]))
    
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r
    
    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = th.relu(dst + rc - rd  - margin)
    return loss


def gci2_loss_neg(data, class_norm, class_embed, class_rad, rel_embed, margin):
    # C subClassOf R some D
    n = data.shape[0]
    c = class_norm(class_embed(data[:, 0]))
    rE = rel_embed(data[:, 1])
    d = class_norm(class_embed(data[:, 2]))
    
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r
    
    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = th.relu(rc + rd - dst  + margin)
    return loss

