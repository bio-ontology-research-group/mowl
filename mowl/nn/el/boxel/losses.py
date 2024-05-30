import torch as th
import torch.nn.functional as F

eps = 1e-8

class Box:
    def __init__(self, min_embed, delta_embed=None, max_embed=None):
        if delta_embed is None and max_embed is None:
            raise ValueError("Either delta_embed or max_embed must be provided")
        if delta_embed is not None and max_embed is not None:
            raise ValueError("Only one of delta_embed or max_embed must be provided")

        self.min_embed = min_embed
        
        if delta_embed is not None:
            self.delta_embed = th.exp(delta_embed)
            self.max_embed = min_embed + self.delta_embed
        
        
            
        if max_embed is not None:
            self.max_embed = max_embed
            self.delta_embed = max_embed - min_embed

    def l2_side_regularizer(self, log_scale: bool=True):
         min_x = self.min_embed
         delta_x = self.delta_embed
         if not log_scale:
             return th.mean(delta_x ** 2)
         else:
             return th.mean(F.relu(min_x + delta_x - 1 + eps )) +  th.mean(F.relu(-min_x - eps))


def volumes(boxes, temperature):
    return F.softplus(boxes.delta_embed, beta=temperature).prod(1)

def intersection(boxes1, boxes2):
    intersections_min = th.max(boxes1.min_embed, boxes2.min_embed)
    intersections_max = th.min(boxes1.max_embed, boxes2.max_embed)
    intersection_box = Box(intersections_min, max_embed=intersections_max)
    return intersection_box

def inclusion_loss(boxes_c, boxes_d, temperature):
    log_intersection = th.log(th.clamp(volumes(intersection(boxes_c, boxes_d), temperature), 1e-10, 1e4))
    log_box_c = th.log(th.clamp(volumes(boxes_c, temperature), 1e-10, 1e4))
    return 1-th.exp(log_intersection-log_box_c)

def regularization_loss(min_embed, delta_embed, log_scale=True):
    min_embed = min_embed.weight
    delta_embed = delta_embed.weight
    boxes = Box(min_embed, delta_embed=delta_embed)
    return boxes.l2_side_regularizer(log_scale=log_scale)


def gci0_loss(data, min_embed, delta_embed, temperature, neg=False):
    c_min = min_embed(data[:, 0])
    d_min = min_embed(data[:, 1])

    c_delta = delta_embed(data[:, 0])
    d_delta = delta_embed(data[:, 1])

    boxes_c = Box(c_min, delta_embed=c_delta)
    boxes_d = Box(d_min, delta_embed=d_delta)
    
    return inclusion_loss(boxes_c, boxes_d, temperature)

def gci0_bot_loss(*args, **kwargs):
    return gci0_loss(*args, **kwargs)

def gci1_loss(data, min_embed, delta_embed, temperature, neg=False):
    c_min = min_embed(data[:, 0])
    d_min = min_embed(data[:, 1])
    e_min = min_embed(data[:, 2])

    c_delta = delta_embed(data[:, 0])
    d_delta = delta_embed(data[:, 1])
    e_delta = delta_embed(data[:, 2])

    boxes_c = Box(c_min, delta_embed=c_delta)
    boxes_d = Box(d_min, delta_embed=d_delta)
    boxes_e = Box(e_min, delta_embed=e_delta)
    
    inter_box = intersection(boxes_c, boxes_d)
    return inclusion_loss(inter_box, boxes_e, temperature)



def gci1_bot_loss(data, min_embed, delta_embed, temperature, neg=False):
    c_min = min_embed(data[:, 0])
    d_min = min_embed(data[:, 1])

    c_delta = delta_embed(data[:, 0])
    d_delta = delta_embed(data[:, 1])

    boxes_c = Box(c_min, delta_embed=c_delta)
    boxes_d = Box(d_min, delta_embed=d_delta)    
    
    log_intersection = th.log(th.clamp(volumes(intersection(boxes_c, boxes_d), temperature), 1e-10, 1e4))
    log_boxes_c = th.log(th.clamp(volumes(boxes_c, temperature), 1e-10, 1e4))
    log_boxes_d = th.log(th.clamp(volumes(boxes_d, temperature), 1e-10, 1e4))
    union = log_boxes_c + log_boxes_d
    return th.exp(log_intersection-union)



def gci2_loss(data, min_embed, delta_embed, relation_embed, scaling_embed, temperature, neg=False):
    c_min = min_embed(data[:, 0])
    d_min = min_embed(data[:, 2])

    c_delta = delta_embed(data[:, 0])
    d_delta = delta_embed(data[:, 2])

    relation = relation_embed(data[:, 1])
    scaling = scaling_embed(data[:, 1])

    boxes_c = Box(c_min, delta_embed=c_delta)
    boxes_d = Box(d_min, delta_embed=d_delta)
    
    trans_c_min = boxes_c.min_embed*(scaling + eps) + relation
    trans_c_max = boxes_c.max_embed*(scaling + eps) + relation
    trans_boxes = Box(trans_c_min, max_embed=trans_c_max)

    if neg:
        loss = 1 - inclusion_loss(trans_boxes, boxes_d, temperature)
    else:
        loss = inclusion_loss(trans_boxes, boxes_d, temperature)

    return loss

def gci3_loss(data, min_embed, delta_embed, relation_embed, scaling_embed, temperature, neg=False):
    c_min = min_embed(data[:, 1])
    d_min = min_embed(data[:, 2])

    c_delta = delta_embed(data[:, 1])
    d_delta = delta_embed(data[:, 2])

    relation = relation_embed(data[:, 0])
    scaling = scaling_embed(data[:, 0])

    boxes_c = Box(c_min, delta_embed=c_delta)
    boxes_d = Box(d_min, delta_embed=d_delta)
    
    trans_c_min = (boxes_c.min_embed - relation)/(scaling + eps)
    trans_c_max = (boxes_c.max_embed - relation)/(scaling + eps)
    trans_boxes = Box(trans_c_min, trans_c_max)
    return inclusion_loss(trans_boxes, boxes_d, temperature)

def gci3_bot_loss(*args, **kwargs):
    return gci3_loss(*args, **kwargs)

def role_inclusion_loss(data, relation_embed, scaling_embed):
    r_translation = relation_embed(data[:, 0])
    s_translation = relation_embed(data[:, 1])

    r_scaling = scaling_embed(data[:, 0])
    s_scaling = scaling_embed(data[:, 1])
    
    loss_1 = th.norm(r_translation-s_translation, p=2, dim=1,keepdim=True)
    loss_2 = th.norm(F.relu(r_scaling/(s_scaling +eps) -1), p=2, dim=1,keepdim=True)
    return loss_1+loss_2

def role_chain_loss(data, relation_embed, scaling_embed):
    r_translation = relation_embed(data[:, 0])
    s_translation = relation_embed(data[:, 1])
    t_translation = relation_embed(data[:, 2])

    r_scaling = scaling_embed(data[:, 0])
    s_scaling = scaling_embed(data[:, 1])
    t_scaling = scaling_embed(data[:, 2])
    
    loss_1 = th.norm(r_scaling*r_translation + s_translation - t_translation, p=2, dim=1,keepdim=True)
    loss_2 = th.norm(F.relu(r_scaling*s_scaling/(t_scaling +eps) -1), p=2, dim=1,keepdim=True)
    return loss_1+loss_2


