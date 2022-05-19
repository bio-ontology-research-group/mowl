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

#    variable = variable_getter(antecedents)

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        consequents  = embed_objects(negs)


    prod = relations + consequents
    prod_loss = prod_net(relations, antecedents)

    sliced_object = slicing_net(antecedents, prod) #changed variable -> antecedents
#    sliced_object = prod
    #exp_loss = exp_net(antecedents, sliced_object)y

    exp_loss = 0
    for layer in exp_net:
        exp_loss += layer(antecedents, sliced_object)
    return exp_loss + prod_loss

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
