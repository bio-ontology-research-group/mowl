import torch as th
import torch.nn as nn
import numpy as np
import random


def gci1_loss(objects, exponential_net,  embed_objects, neg = False,  num_objects = None, device = "cpu"):

    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])
    
    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        neg_loss_1 = 0
        #neg_loss_1 = exponential_net(consequents, antecedents)
        neg_loss_2 = exponential_net(antecedents, embed_negs)
        
        return (neg_loss_1 + neg_loss_2)
                                        
    else:
        loss = exponential_net(antecedents, consequents)
        return loss

def gci2_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, neg = False, num_objects = None, device = "cpu"):
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        consequents  = embed_objects(negs)

    sliced_prod, prod_loss = slicing_net(relations, consequents, antecedents)
    exp_loss = exp_net(antecedents, sliced_prod)
    return exp_loss + prod_loss

def gci3_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, neg = False, num_objects = None, device = "cpu"):

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents = embed_negs

    sliced_prod, prod_loss = slicing_net(relations, antecedents, consequents)
    exp_loss = exp_net(sliced_prod, consequents)

    return prod_loss + exp_loss


    
def gci4_loss(objects, exp_net, prod_net, embed_objects, neg = False, num_objects = None, device = "cpu"):

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    prod, prod_loss = prod_net(antecedents_left, antecedents_right)
    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents_neg = embed_negs
        exp_loss_neg = exp_net(prod, consequents_neg)
        neg_loss_1 = prod_loss + exp_loss_neg
        
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        prod_neg_1, prod_loss_1 = prod_net(antecedents_left, embed_negs)
        exp_loss = exp_net(prod_neg_1, consequents)
        neg_loss_2 = prod_loss_1 + exp_loss

        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        prod_neg_2, prod_loss_2 = prod_net(embed_negs, antecedents_right)
        exp_loss = exp_net(prod_neg_2, consequents)
        neg_loss_3 = prod_loss_2 + exp_loss


        return neg_loss_1
#        return (neg_loss_2 + neg_loss_3)/2
        return (neg_loss_1 + neg_loss_2 + neg_loss_3)/3
        
    else:
        exp_loss = exp_net(prod, consequents)
        return prod_loss + exp_loss

    
#    prod = (antecedents_left + antecedents_right)/2
        
    

#    print(f"prod: {th.mean(prod_loss)} \t exp: {th.mean(exp_loss)}")
#    return prod_loss + exp_loss

def gci5_loss(objects, exp_net, prod_net, embed_objects, neg = False, num_objects = None, device = "cpu"):

    antecedents = embed_objects(objects[:, 0])
    consequents_left = embed_objects(objects[:, 1])
    consequents_right = embed_objects(objects[:, 2])

    prod, prod_loss = prod_net(consequents_left, consequents_right)
    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        antecedents_neg = embed_negs
        exp_loss_neg = exp_net(antecedents_neg, prod)
        neg_loss_1 = prod_loss + exp_loss_neg
        
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        prod_neg_1, prod_loss_1 = prod_net(consequents_left, embed_negs)
        exp_loss = exp_net(antecedents, prod_neg_1)
        neg_loss_2 = prod_loss_1 + exp_loss

        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        prod_neg_2, prod_loss_2 = prod_net(embed_negs, consequents_right)
        exp_loss = exp_net(antecedents, prod_neg_2)
        neg_loss_3 = prod_loss_2 + exp_loss


        return neg_loss_1
#        return (neg_loss_2 + neg_loss_3)/2
        return (neg_loss_1 + neg_loss_2 + neg_loss_3)/3
        
    else:
        exp_loss = exp_net(antecedents, prod)
        return prod_loss + exp_loss

    
#    prod = (consequents_left + consequents_right)/2
        
    

#    print(f"prod: {th.mean(prod_loss)} \t exp: {th.mean(exp_loss)}")
#    return prod_loss + exp_loss


def gci6_loss(objects, exp_net, slicing_net, prod_net, embed_objects, embed_rels, neg = False, num_objects = None, device = "cpu"):

    antecedents_left = embed_objects(objects[:,0])
    relations = embed_rels(objects[:, 1])
    fillers = embed_objects(objects[:, 2])
    consequents = embed_objects(objects[:, 3])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents = embed_negs

    sliced_prod, sliced_prod_loss = slicing_net(relations, fillers, antecedents_left, consequents)
    prod, prod_loss = prod_net(antecedents_left, sliced_prod)
    exp_loss = exp_net(prod, consequents)
    
    return prod_loss + sliced_prod_loss + exp_loss

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
