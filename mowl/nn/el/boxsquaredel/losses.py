import torch as th
import numpy as np

def box_distance(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    dist = th.abs(center_a - center_b) - offset_a - offset_b 
    return dist

def box_intersection(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    
    lower = th.maximum(center_a - offset_a, center_b - offset_b)
    upper = th.minimum(center_a + offset_a, center_b + offset_b)
    centers = (lower + upper) / 2
    offsets = th.abs(upper - lower) / 2
    intersection = (centers, offsets)
    return intersection, lower, upper

def inclusion_score(box_a, box_b, gamma):
    dist_a_b = box_distance(box_a, box_b)
    _, offset_a = box_a
    score = th.linalg.norm(th.relu(dist_a_b + 2*offset_a - gamma), dim=1)
    return score

def class_assertion_loss(data, ind_center, ind_offset, class_center, class_offset, gamma, neg = False):
    center_i = ind_center(data[:, 0])
    offset_i = th.abs(ind_offset(data[:, 0]))
    center_c = class_center(data[:, 1])
    offset_c = th.abs(class_offset(data[:, 1]))
    
    box_c = (center_c, offset_c)
    box_d = (center_i, offset_i)
    score = inclusion_score(box_d, box_c, gamma)
    return score

def object_property_assertion_loss(data, ind_center, ind_offset, head_center, head_offset, tail_center, tail_offset, bump_individuals, gamma, delta, neg=False):
    if neg:
        fn = gci2_score_neg
    else:
        fn = gci2_score

    center_i1 = ind_center(data[:, 0])
    offset_i1 = th.abs(ind_offset(data[:, 0]))
    bump_i1 = bump_individuals(data[:, 0])
    center_i2 = ind_center(data[:, 2])
    offset_i2 = th.abs(ind_offset(data[:, 2]))
    bump_i2 = bump_individuals(data[:, 2])

    center_head = head_center(data[:, 1])
    offset_head = th.abs(head_offset(data[:, 1]))
    center_tail = tail_center(data[:, 1])
    offset_tail = th.abs(tail_offset(data[:, 1]))

    return fn(center_i1, offset_i1, bump_i1, center_i2, offset_i2, bump_i2, center_head, offset_head, center_tail, offset_tail, gamma, delta)

def gci0_score(data, class_center, class_offset, gamma):
    center_c = class_center(data[:, 0])
    offset_c = th.abs(class_offset(data[:, 0]))
    center_d = class_center(data[:, 1])
    offset_d = th.abs(class_offset(data[:, 1]))
    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)
    score = inclusion_score(box_c, box_d, gamma)
    return score

def gci0_loss(data, class_center, class_offset, gamma, neg=False):
    score = gci0_score(data, class_center, class_offset, gamma)
    loss = score.square()
    return loss

def gci0_bot_score(data, class_offset):
    offset_c = th.abs(class_offset(data[:, 0]))
    score = th.linalg.norm(offset_c, dim=1)
    return score

def gci0_bot_loss(data, class_offset):
    score = gci0_bot_score(data, class_offset)
    loss = score.square()
    return loss

def gci1_score(data, class_center, class_offset, gamma):
    center_c = class_center(data[:, 0])
    center_d = class_center(data[:, 1])
    center_e = class_center(data[:, 2])
    offset_c = th.abs(class_offset(data[:, 0]))
    offset_d = th.abs(class_offset(data[:, 1]))
    offset_e = th.abs(class_offset(data[:, 2]))

    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)
    box_e = (center_e, offset_e)

    intersection, lower, upper = box_intersection(box_c, box_d)
    box_incl_score = inclusion_score(intersection, box_e, gamma)

    additional_score = th.linalg.norm(th.relu(lower - upper), dim=1)
    score = box_incl_score + additional_score
    return score

def gci1_loss(data, class_center, class_offset, gamma, neg=False):
    score = gci1_score(data, class_center, class_offset, gamma)
    loss = score.square()
    return loss

def gci1_bot_score(data, class_center, class_offset, gamma):

    center_c = class_center(data[:, 0])
    center_d = class_center(data[:, 1])

    offset_c = th.abs(class_offset(data[:, 0]))
    offset_d = th.abs(class_offset(data[:, 1]))

    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)

    box_dist = box_distance(box_c, box_d)
    score = th.linalg.norm(th.relu(-box_dist - gamma), dim=1)
    return score

def gci1_bot_loss(data, class_center, class_offset, gamma, neg=False):
    score = gci1_bot_score(data, class_center, class_offset, gamma)
    loss = score.square()
    return loss

def gci2_score(center_c, offset_c, bump_c,
               center_d, offset_d, bump_d,
               center_head, offset_head,
               center_tail, offset_tail,
               gamma, delta):
    
    box_c = (center_c, offset_c)
    box_head = (center_head, offset_head)
    box_tail = (center_tail, offset_tail)
    box_d = (center_d, offset_d)

    bumped_c = (center_c + bump_d, offset_c)
    bumped_d = (center_d + bump_c, offset_d)

    inclussion_1 = inclusion_score(bumped_c, box_head, gamma)
    inclussion_2 = inclusion_score(bumped_d, box_tail, gamma)

    score = (inclussion_1 + inclussion_2)/2
    return score


def gci2_score_neg(center_c, offset_c, bump_c,
                   center_d, offset_d, bump_d,
                   center_head, offset_head,
                   center_tail, offset_tail,
                   gamma, delta):

    def minimal_distance(box_a, box_b, gamma):
        dist = box_distance(box_a, box_b)
        min_dist = th.linalg.norm(th.relu(dist + gamma), dim=1)
        return min_dist
    
    box_c = (center_c, offset_c)
    box_head = (center_head, offset_head)
    box_tail = (center_tail, offset_tail)
    box_d = (center_d, offset_d)

    bumped_c = (center_c + bump_d, offset_c)
    bumped_d = (center_d + bump_c, offset_d)
    
    first_part = (delta - minimal_distance(bumped_c, box_head, gamma)).square()
    second_part = (delta - minimal_distance(bumped_d, box_tail, gamma)).square()

    loss = first_part + second_part
    return loss

def gci2_loss(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma, delta, neg=False):
    if neg:
        fn = gci2_score_neg
    else:
        fn = gci2_score

    center_c = class_center(data[:, 0])
    offset_c = th.abs(class_offset(data[:, 0]))
    bump_c = bump(data[:, 0])
    center_d = class_center(data[:, 2])
    offset_d = th.abs(class_offset(data[:, 2]))
    bump_d = bump(data[:, 2])

    center_head = head_center(data[:, 1])
    offset_head = th.abs(head_offset(data[:, 1]))
    center_tail = tail_center(data[:, 1])
    offset_tail = th.abs(tail_offset(data[:, 1]))

    return fn(center_c, offset_c, bump_c, center_d, offset_d, bump_d, center_head, offset_head, center_tail, offset_tail, gamma, delta)

def gci3_score(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma):
    center_d = class_center(data[:, 2])
    offset_d = th.abs(class_offset(data[:, 2]))

    center_head = head_center(data[:, 0])
    offset_head = th.abs(head_offset(data[:, 0]))

    bump_c = bump(data[:, 1])

    bumped_head = (center_head - bump_c, offset_head)
    box_d = (center_d, offset_d)
    score = inclusion_score(bumped_head, box_d, gamma)
    return score

def gci3_loss(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma, neg=False):
    score = gci3_score(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma)
    loss = score.square()
    return loss

def gci3_bot_score(data, head_offset):

    offset_head = th.abs(head_offset(data[:, 0]))
    score = th.linalg.norm(offset_head, dim=1)
    return score

def gci3_bot_loss(data, head_offset):
    score = gci3_bot_score(data, head_offset)
    loss = score.square()
    return loss


def reg_loss(bump, reg_factor):
    reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).mean()
    return reg_loss



