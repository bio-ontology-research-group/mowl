import torch as th
import torch.nn as nn
import numpy as np
import random
from torch.linalg import norm


def exponential_loss(objects, morphisms, emb_up, full = True, up = None):
    up2down, up2exp, down2exp, up2ant, down2ant, up2cons, down2cons, fc = morphisms
    antecedent, consequent = objects
        
    if up is None:
        up = emb_up(th.cat([antecedent, consequent], dim = 1))
    exponential = consequent/(antecedent + 1e-6)
    exponential = exponential.where(exponential > 1, th.tensor(1.0).to(up.device))
        
    down = (exponential + antecedent)/2
    
    if full:

        estim_downFromUp = up2down(up)
        
        estim_expFromUp = up2exp(up)

        estim_expFromdown = down2exp(down)
        estim_expFromDownChained = down2exp(estim_downFromUp)
            
        estim_antFromUp = up2ant(up)

        estim_antFromDown = down2ant(down)
        estim_antFromDownChained = down2ant(estim_downFromUp)
            
        estim_consFromUp = up2cons(up)
        estim_consFromDown = down2cons(down)
        estim_consFromDownChained = down2cons(estim_downFromUp)
            
        loss1 = norm(estim_expFromUp - exponential, dim = 1)
        loss2 = norm(estim_antFromUp - antecedent, dim =1) 
        loss3 = norm(estim_expFromdown - exponential, dim = 1) 
        loss4 = norm(estim_antFromDown - antecedent, dim = 1) 
        loss5 = norm(estim_downFromUp - down, dim = 1) 
        loss6 = norm(estim_consFromDown - consequent, dim = 1) 
        loss7 = norm(estim_consFromUp - consequent, dim = 1) 

            
        path_loss1 = norm(estim_expFromDownChained - exponential, dim = 1)
        path_loss2 = norm(estim_antFromDownChained - antecedent, dim = 1)
        path_loss3 = norm(estim_consFromDownChained - consequent, dim = 1)

            
        assert loss1.shape == loss2.shape
        assert loss2.shape == loss3.shape
        assert loss3.shape == loss4.shape
        assert loss4.shape == loss5.shape
        assert loss5.shape == loss6.shape
        assert loss6.shape == loss7.shape
        assert loss7.shape == path_loss1.shape
        assert path_loss1.shape == path_loss2.shape
        assert path_loss2.shape == path_loss3.shape
            
        loss = loss5 + loss6+ loss7 + path_loss3 + loss1 +loss2 +loss3 + loss4 + path_loss1 + path_loss2
    else:
        
        estimCons = fc(antecedent)
        loss = norm(estimCons, consequent)
        loss = th.mean(loss, dim=1)

    return loss


def product_loss(objects, morphisms, emb_big_prod, big_prod = None):

    left, right = objects
    if big_prod is None:
        big_prod = emb_big_prod(th.cat([left, right], dim=1))
    
    prod = (left + right)/2
    
    big2prod, big2left, big2right, prod2left, prod2right = morphisms

    estim_leftFromBig = big2left(big_prod)
    estim_rightFromBig = big2right(big_prod)
    estim_prodFromBig = big2prod(big_prod)

    estim_leftFromProd = prod2left(prod)
    estim_rightFromProd = prod2right(prod)

    estim_leftFromProdChained = prod2left(estim_prodFromBig)
    estim_rightFromProdChained = prod2right(estim_prodFromBig)

    
    loss1 = norm(estim_leftFromBig - left, dim = 1)
    loss2 = norm(estim_leftFromProd - left, dim = 1)
    loss3 = norm(estim_rightFromBig - right, dim = 1)
    loss4 = norm(estim_rightFromProd - right, dim = 1)
    loss5 = norm(estim_prodFromBig - prod, dim = 1)
    path_loss1 = norm(estim_leftFromProdChained - left, dim = 1)
    path_loss2 = norm(estim_rightFromProdChained - right, dim = 1)


    assert loss1.shape == loss2.shape, f"{loss1.shape}, {loss2.shape}"
    assert loss2.shape == loss3.shape, f"{loss2.shape}, {loss3.shape}"
    assert loss3.shape == loss4.shape, f"{loss2.shape}, {loss3.shape}, {loss4.shape}"
    assert loss4.shape == path_loss1.shape, f"{loss4.shape}, {path_loss1.shape}"
    assert path_loss1.shape == path_loss2.shape, f"{path_loss1.shape}, {path_loss2.shape}"
    assert loss5.shape == path_loss1.shape
    loss = loss1 + loss2 + loss3 + loss4 + path_loss1 + path_loss2 + loss5
    
    return loss

def nf1_loss(objects, morphisms, embed_nets, neg = False, margin = 0):
    embed_objects, embed_exp_up = embed_nets
    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])

    if neg == True:
        neg_loss = exponential_loss((consequents, antecedents), morphisms, embed_exp_up)
        return neg_loss #th.relu(margin - neg_loss)
        # positive_loss = exponential_loss((antecedents, consequents), morphisms, embed_exp_up)
        # positive_loss = 1 - 2*(th.sigmoid(positive_loss) - 0.5)
        # negative_loss = exponential_loss((consequents, antecedents), morphisms, embed_exp_up)
        # negative_loss = 1 - 2*(th.sigmoid(negative_loss) - 0.5)
        # return th.hstack([positive_loss, negative_loss])
    else:
        return exponential_loss((antecedents, consequents), morphisms, embed_exp_up)
    #    return th.relu(positive_loss - negative_loss + 1)

def nf2_loss(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False):
    embed_objects, embed_exp_up, embed_big_prod = embed_nets

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    #    exp_loss = exponential_loss((prod, consequents), exp_morphisms, embed_exp_up)

    if neg:
        antecedents = [antecedents_left, antecedents_right]
        random.shuffle(antecedents)
        prod_loss = product_loss((antecedents[0], consequents), prod_morphisms, embed_big_prod)

        prod = (antecedents[0] + consequents)/2
        exp_loss = exponential_loss((prod, antecedents[1]), exp_morphisms, embed_exp_up)

    else:
        prod_loss = product_loss((antecedents_left, antecedents_right), prod_morphisms, embed_big_prod)
        prod = (antecedents_left + antecedents_right)/2
        exp_loss = exponential_loss((prod, consequents), exp_morphisms, embed_exp_up)

    return prod_loss + exp_loss
#    return prod_loss + exp_loss


def nf3_loss(objects, prod_morphisms, exp_morphisms, embed_nets):
    embed_objects, embed_rels, embed_fst, embed_up, embed_big_prod = embed_nets

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])
    
    chosen_vars = embed_fst(th.cat([antecedents, relations, consequents], dim = 1))
    prod = (chosen_vars + antecedents)/2

    prod_loss = product_loss((chosen_vars, antecedents), prod_morphisms, embed_big_prod, big_prod = prod)

    
    exp_loss = exponential_loss((prod, consequents), exp_morphisms, embed_up)

    return prod_loss + exp_loss

def nf4_loss(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False, num_objects = None, margin = 0):
    embed_objects, embed_rels, embed_snd, embed_up, embed_big_prod = embed_nets
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    chosen_vars = embed_snd(th.cat([consequents, relations, antecedents], dim = 1))

    prod = (antecedents + consequents)/2    
    prod_loss = product_loss((chosen_vars, consequents), prod_morphisms, embed_big_prod)


    # if neg:
    #     positive_loss = exponential_loss((antecedents, prod), exp_morphisms, embed_up)
    #     positive_loss = 1 - 2*(th.sigmoid(positive_loss) - 0.5)
    #     negs = th.tensor(np.random.choice(num_objects, len(objects))).to("cuda")
    #     embed_negs = embed_objects(negs)
    #     negative_loss = exponential_loss((antecedents, embed_negs), exp_morphisms, embed_up)
    #     negative_loss = 1 - 2*(th.sigmoid(negative_loss) - 0.5)
    #     return th.hstack([positive_loss, negative_loss])


    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to("cuda")
        embed_negs = embed_objects(negs)
        chosen_vars = embed_snd(th.cat([embed_negs, relations, antecedents], dim=1))
        prod = (antecedents + embed_negs)/2
        prod_loss = product_loss((chosen_vars, embed_negs), prod_morphisms, embed_big_prod) 
        loss = exponential_loss((antecedents, embed_negs), exp_morphisms, embed_up)
#        loss = th.relu(margin - negative_loss)

    else:
        loss = exponential_loss((antecedents, prod), exp_morphisms, embed_up)
    assert prod_loss.shape == loss.shape

    return prod_loss + loss

