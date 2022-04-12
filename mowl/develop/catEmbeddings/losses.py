import torch as th
import torch.nn as nn
import numpy as np
import random
from functools import reduce

def norm(tensor, dim = None):
    return th.linalg.norm(tensor, dim = dim)

def assert_shapes(shapes):
    assert len(set(shapes)) == 1

def exponential_loss(objects, exp_morphisms, emb_up):
    up2down, up2exp, down2exp, up2ant, down2ant, up2cons, down2cons = exp_morphisms
    antecedent, consequent = objects

    if up is None:
        up = emb_up(th.cat([antecedent, consequent], dim = 1))
#    exponential = th.relu(consequent - antecedent)

    exponential = consequent/(antecedent + 1e-10)
    exponential = exponential.where(exponential > 1, th.tensor(1.0).to(up.device))

    down = (exponential + antecedent)/2

    estim_down_from_up = up2down(up)

    estim_exp_from_up = up2exp(up)

    estim_exp_fromdown = down2exp(down)
    estim_exp_from_down_chained = down2exp(estim_down_from_up)

    estim_ant_from_up = up2ant(up)

    estim_ant_from_down = down2ant(down)
    estim_ant_from_down_chained = down2ant(estim_down_from_up)

    estim_cons_from_up = up2cons(up)
    estim_cons_from_down = down2cons(down)
    estim_cons_from_down_chained = down2cons(estim_down_from_up)


    loss1 = norm(estim_exp_from_up - exponential, dim = 1)
    loss2 = norm(estim_ant_from_up - antecedent, dim =1)
    loss3 = norm(estim_exp_fromdown - exponential, dim = 1)
    loss4 = norm(estim_ant_from_down - antecedent, dim = 1)
    loss5 = norm(estim_down_from_up - exponential, dim = 1)
    loss6 = norm(estim_cons_from_down - consequent, dim = 1)
    loss7 = norm(estim_cons_from_up - consequent, dim = 1)


    path_loss1 = norm(estim_exp_from_down_chained - exponential, dim = 1)
    path_loss2 = norm(estim_ant_from_down_chained - antecedent, dim = 1)
    path_loss3 = norm(estim_cons_from_down_chained - consequent, dim = 1)


    losses = [loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2, path_loss3]
    shapes = map(lambda x: x.shape, losses)
    assert_shapes(shapes)

    return sum(losses)




def product_loss(objects, morphisms, emb_big_prod = None, big_prod = None):

    left, right = objects
    if big_prod is None:
        big_prod = emb_big_prod(th.cat([left, right], dim=1))

    prod = (left + right)/2

    big2prod, big2left, big2right, prod2left, prod2right = morphisms

    estim_left_from_big = big2left(big_prod)
    estim_right_from_big = big2right(big_prod)
    estim_prod_from_big = big2prod(big_prod)

    estim_left_from_prod = prod2left(prod)
    estim_right_from_prod = prod2right(prod)

    estim_left_from_prod_chained = prod2left(estim_prod_from_big)
    estim_right_from_prod_chained = prod2right(estim_prod_from_big)


    loss1 = norm(estim_left_from_big - left, dim = 1)
    loss2 = norm(estim_left_from_prod - left, dim = 1)
    loss3 = norm(estim_right_from_big - right, dim = 1)
    loss4 = norm(estim_right_from_prod - right, dim = 1)
    loss5 = norm(estim_prod_from_big - prod, dim = 1)
    path_loss1 = norm(estim_left_from_prod_chained - left, dim = 1)
    path_loss2 = norm(estim_right_from_prod_chained - right, dim = 1)

    losses = [loss1, loss2, loss3, loss4, loss5, path_loss1, path_loss2]
    shapes = map(lambda x: x.shape, losses)
    assert_shapes(shapes)

    return sum(losses)


def pullback_loss(objects, morphisms, embed_pullback):
    left, right, end = objects
    left_from_pullback, end_from_left, right_from_pullback, end_from_right = morphisms

    pullback = embed_pullback(left, right)

    estim_left = left_from_pullback(pullback)
    estim_end_from_left = end_from_left(left)
    estim_right = right_from_pullback(pullback)
    estim_end_from_right = end_from_right(right)

    loss1 = norm(estim_left-left, dim=1)
    loss2 = norm(estim_end_from_left - end, dim=1)
    loss3 = norm(estim_right-right, dim=1)
    loss4 = norm(estim_end_from_right - end, dim=1)

    losses = [loss1, loss2, loss3, loss4]

    
    shapes = map(lambda x: x.shape, losses)
    assert_shapes(shapes)

    return sum(losses)
    
    


def existential_loss(objects, pullback_morphisms, nets):
    
    rel, filler = objects
    gen_slicer, gen_extra, gen_right, gen_pullback, slicer = nets

    slicer = gen_slicer(rel, filler)
    extra = gen_extra(rel, filler)
    left_pullback = (slicer+extra)/2
    right_pullback = gen_right(rel, filler)

    
    loss_pullback = pullback_loss((left_pullback, right_pullback, extra), pullback_morphisms, gen_pullback)


    sliced_rel = slicer(th.cat[rel, slicer, extra, right_pullback])
    sliced_filler = slicer(th.cat[filler, slicer, extra, right_pullback])

    return (sliced_rel + sliced_filler)/2, loss_pullback

