import torch as th
import torch.nn as nn
import numpy as np
import random
import mowl.develop.catEmbeddings.losses as L


def nf1_loss(objects, exponential_net,  embed_objects, neg = False,  num_objects = None, device = "cpu"):

    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        #neg_loss_1 = exponential_net(consequents, antecedents)
        #neg_loss_2 = exponential_net(antecedents, embed_negs)
        neg_loss_1 = 0
        for layer in exponential_net:
            neg_loss_1 += layer(consequents, antecedents)

        neg_loss_2 = 0
        for layer in exponential_net:
            neg_loss_2 += layer(antecedents, embed_negs)
            
        neg_loss = sum([neg_loss_1, neg_loss_2])/2
        return neg_loss
                                        
    else:
        #loss = exponential_net(antecedents, consequents)
        loss = 0
        for layer in exponential_net:
            loss += layer(antecedents, consequents)
        return loss

    
def nf2_loss(objects, exp_net, prod_net, embed_objects, neg = False):


    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    

    if neg:
        antecedents = [antecedents_left, antecedents_right]
        random.shuffle(antecedents)
        prod, prod_loss = prod_net(antecedents[0], consequents)
        exp_loss = exp_net(prod, antecedents[1], prod_net)

    else:
        prod_loss = prod_net(antecedents_left, antecedents_right)
        prod = antecedents_left + antecedents_right
        exp_loss = L.exponential_loss(prod, consequents, prod_net)

    return prod_loss + exp_loss



def nf4_loss(objects, variable_getter, exp_net, prod_net, slicing_net, embed_objects, embed_rels, neg = False):



    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    variable = variable_getter(consequents)
    
    if neg:
        prod_loss = 0
        exp_loss = 0
    else:
        prod = relations + antecedents
        prod_loss = prod_net(relations, antecedents)

        sliced_object = slicing_net(variable, prod)
        exp_loss = exp_net(sliced_object, consequents)

    return prod_loss + exp_loss

def nf3_loss(objects, variable_getter,  exp_net, prod_net, slicing_net, embed_objects, embed_rels, neg = False, num_objects = None, device = "cpu"):
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    variable = variable_getter(antecedents)

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        consequents  = embed_objects(negs)


    prod = relations + consequents
    prod_loss = prod_net(relations, antecedents)

    sliced_object = slicing_net(antecedents, prod) #changed variable -> antecedents
    #sliced_object = prod
    #exp_loss = exp_net(antecedents, sliced_object)
    exp_loss = 0
    for layer in exp_net:
        exp_loss += layer(antecedents, sliced_object)
    return exp_loss + prod_loss

def nf4_loss_old(objects, exp_net, prod_net, pullback_net, ex_net, embed_objects, neg = False, num_objects = None):
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to("cuda")
        embed_negs = embed_objects(negs)

        ex_obj, ex_loss = ex_net(relations, consequents, prod_net, pullback_net)

        _, exp_loss = exp_net(antecedents, ex_obj, prod_net)

    else:
        ex_obj, ex_loss = ex_net(relations, consequents, prod_net, pullback_net)

        _, exp_loss = exp_net(antecedents, ex_obj, prod_net)


    return exp_loss + ex_loss









############################


def nf1_losss(objects, morphisms, prod_morphisms, embed_nets, neg = False, margin = 0, num_objects = None):
    embed_objects, embed_exp_up = embed_nets
    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to("cuda")
        embed_negs = embed_objects(negs)
        neg_loss_1 = L.exponential_loss((consequents, antecedents), morphisms, embed_exp_up)
        neg_loss_2 = L.exponential_loss((antecedents, embed_negs), morphisms, embed_exp_up)
        
        neg_loss = random.choice([neg_loss_1, neg_loss_2])
        return neg_loss
                                        
    else:
        return L.exponential_loss((antecedents, consequents), morphisms, embed_exp_up)

    
def nf2_losss(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False):
    embed_objects, embed_exp_up, embed_big_prod = embed_nets

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    

    if neg:
        antecedents = [antecedents_left, antecedents_right]
        random.shuffle(antecedents)
        prod_loss = L.product_loss((antecedents[0], consequents), prod_morphisms, embed_big_prod)

        prod = (antecedents[0] + consequents)/2
        exp_loss = L.exponential_loss((prod, antecedents[1]), exp_morphisms, embed_exp_up)

    else:
        prod_loss = L.product_loss((antecedents_left, antecedents_right), prod_morphisms, embed_big_prod)
        prod = (antecedents_left + antecedents_right)/2
        exp_loss = L.exponential_loss((prod, consequents), exp_morphisms, embed_exp_up)

    return prod_loss + exp_loss



def nf3_losss(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False):
    embed_objects, embed_rels, embed_fst, embed_up, embed_big_prod = embed_nets

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])
    
    chosen_vars = embed_fst(th.cat([relations, antecedents, consequents], dim = 1))
    prod = (chosen_vars + antecedents)/2

    if neg:
        prod_loss = 0
        exp_loss = 0
    else:
        prod_loss = L.product_loss((chosen_vars, antecedents), prod_morphisms, embed_big_prod)
        exp_loss = L.exponential_loss((prod, consequents), exp_morphisms,embed_up)

    return prod_loss + exp_loss


def nf4_losss_old(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False, num_objects = None):
    embed_objects, embed_rels, embed_snd, embed_up, embed_big_prod = embed_nets
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to("cuda")
        embed_negs = embed_objects(negs)
        chosen_vars = embed_snd(th.cat([antecedents, relations, embed_negs], dim=1))
        prod = (chosen_vars + embed_negs)/2
#        prod_loss = L.product_loss((chosen_vars, embed_negs), prod_morphisms, embed_big_prod) 
        exp_loss = L.exponential_loss((antecedents, prod), exp_morphisms, embed_up)

    else:
        chosen_vars = embed_snd(th.cat([antecedents, relations, consequents], dim = 1))
        prod = (chosen_vars + consequents)/2    
#        prod_loss = L.product_loss((chosen_vars, consequents), prod_morphisms, embed_big_prod)
        exp_loss = L.exponential_loss((antecedents, prod), exp_morphisms, embed_up)

    return exp_loss #+ prod_loss



def nf4_losss(objects, product_morphisms, pullback_morphisms, exp_morphisms, embed_nets, ex_nets, neg = False, num_objects = None):
    embed_objects, embed_rels, embed_snd, embed_up = embed_nets
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to("cuda")
        embed_negs = embed_objects(negs)
        existential_object, ex_loss = L.existential_loss((relations, embed_negs), product_morphisms, pullback_morphisms, ex_nets)
        exp_loss = L.exponential_loss((antecedents, existential_object), exp_morphisms, embed_up)

    else:
        existential_object, ex_loss = L.existential_loss((relations, consequents), product_morphisms,  pullback_morphisms, ex_nets) 
        
        exp_loss = L.exponential_loss((antecedents, existential_object), exp_morphisms, embed_up)

    return exp_loss + ex_loss

