import torch as th
import torch.nn as nn
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self, reduction = "mean", eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction = reduction)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = th.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def exponential_loss(objects, morphisms, emb_up, full = True, up = None):
    up2down, up2exp, down2exp, up2ant, down2ant, up2cons, down2cons, fc = morphisms
    rmseNR = RMSELoss(reduction = "none")
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
            
        loss1 = rmseNR(estim_expFromUp, exponential)
        loss1 = th.mean(loss1, dim=1)
        loss2 = rmseNR(estim_antFromUp, antecedent) 
        loss2 = th.mean(loss2, dim=1)


            
        loss3 = rmseNR(estim_expFromdown, exponential) 
        loss3 = th.mean(loss3, dim=1)
        loss4 = rmseNR(estim_antFromDown, antecedent) 
        loss4 = th.mean(loss4, dim=1)
            
        loss5 = rmseNR(estim_downFromUp, down) 
        loss5 = th.mean(loss5, dim=1)
        loss6 = rmseNR(estim_consFromDown, consequent) 
        loss6 = th.mean(loss6, dim=1)
        loss7 = rmseNR(estim_consFromUp, consequent) 
        loss7 = th.mean(loss7, dim=1)

            
        path_loss1 = rmseNR(estim_expFromDownChained, exponential)
        path_loss1 = th.mean(path_loss1, dim=1)
            
        path_loss2 = rmseNR(estim_antFromDownChained, antecedent)
        path_loss2 = th.mean(path_loss2, dim=1)
            
        path_loss3 = rmseNR(estim_consFromDownChained, consequent)
        path_loss3 = th.mean(path_loss3, dim =1)
            
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
        loss = rmseNR(estimCons, consequent)
        loss = th.mean(loss, dim=1)

    return loss


def product_loss(objects, morphisms, emb_big_prod):

    rmseNR = RMSELoss(reduction = "none")
    
    left, right = objects
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

    
    loss1 = rmseNR(estim_leftFromBig, left)
    loss1 = th.mean(loss1, dim = 1)
    
    loss2 = rmseNR(estim_leftFromProd, left)
    loss2 = th.mean(loss2, dim = 1)

    loss3 = rmseNR(estim_rightFromBig, right)
    loss3 = th.mean(loss3, dim = 1)
    
    loss4 = rmseNR(estim_rightFromProd, right)
    loss4 = th.mean(loss4, dim = 1)

    path_loss1 = rmseNR(estim_leftFromProdChained, left)
    path_loss1 = th.mean(path_loss1, dim = 1)

    path_loss2 = rmseNR(estim_rightFromProdChained, right)
    path_loss2 = th.mean(path_loss2, dim = 1)


    assert loss1.shape == loss2.shape, f"{loss1.shape}, {loss2.shape}"
    assert loss2.shape == loss3.shape, f"{loss2.shape}, {loss3.shape}"
    assert loss3.shape == loss4.shape, f"{loss2.shape}, {loss3.shape}, {loss4.shape}"
    assert loss4.shape == path_loss1.shape, f"{loss4.shape}, {path_loss1.shape}"
    assert path_loss1.shape == path_loss2.shape, f"{path_loss1.shape}, {path_loss2.shape}"

    loss = loss1 + loss2 + loss3 + loss4 + path_loss1 + path_loss2
    
    return loss

# class NF1(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, objects, morphisms, embed_nets):
#         embed_objects, embed_exp_up = embed_nets
#         antecedents = embed_objects(objects[:, 0])
#         consequents = embed_objects(objects[:, 1])
#         return exponential_loss((antecedents, consequents), morphisms, embed_exp_up)

def nf1_loss(objects, morphisms, embed_nets, neg = False):
    embed_objects, embed_exp_up = embed_nets
    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])

    if neg == True:
        positive_loss = exponential_loss((antecedents, consequents), morphisms, embed_exp_up)
        positive_loss = 1 - 2*(th.sigmoid(positive_loss) - 0.5)
        negative_loss = exponential_loss((consequents, antecedents), morphisms, embed_exp_up)
        negative_loss = 1 - 2*(th.sigmoid(negative_loss) - 0.5)
        return th.hstack([positive_loss, negative_loss])
    else:
        return exponential_loss((antecedents, consequents), morphisms, embed_exp_up)
    #    return th.relu(positive_loss - negative_loss + 1)

def nf2_loss(objects, prod_morphisms, exp_morphisms, embed_nets):
    embed_objects, embed_exp_up, embed_big_prod = embed_nets

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

#    prod_loss = product_loss((antecedents_left, antecedents_right), prod_morphisms, embed_big_prod)

    prod = (antecedents_left + antecedents_right)/2
#    exp_loss = exponential_loss((prod, consequents), exp_morphisms, embed_exp_up)
    exp_loss = exponential_loss((antecedents_right, consequents), exp_morphisms, embed_exp_up, up = prod)

    return exp_loss
#    return prod_loss + exp_loss


def nf3_loss(objects, prod_morphisms, exp_morphisms, embed_nets):
    embed_objects, embed_rels, embed_fst, embed_up, embed_big_prod = embed_nets

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])
    
    chosen_vars = embed_fst(th.cat([consequents, antecedents], dim = 1))

    prod_loss = product_loss((chosen_vars, antecedents), prod_morphisms, embed_big_prod)

    prod = (chosen_vars + antecedents)/2
    
    loss = exponential_loss((prod, consequents), exp_morphisms, embed_up)

    return loss

def nf4_loss(objects, prod_morphisms, exp_morphisms, embed_nets, neg = False, num_objects = None):
    embed_objects, embed_rels, embed_snd, embed_up, embed_big_prod = embed_nets
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    chosen_vars = embed_snd(th.cat([antecedents, consequents], dim = 1))

    prod_loss = product_loss((chosen_vars, consequents), prod_morphisms, embed_big_prod)

    prod = (chosen_vars + consequents)/2
    

    if neg:
        positive_loss = exponential_loss((antecedents, existentials), exp_morphisms, embed_up)
        positive_loss = 1 - 2*(th.sigmoid(positive_loss) - 0.5)
        negs = th.tensor(np.random.choice(num_objects, len(objects))).to("cuda")
        embed_negs = embed_objects(negs)
        negative_loss = exponential_loss((antecedents, embed_negs), exp_morphisms, embed_up)
        negative_loss = 1 - 2*(th.sigmoid(negative_loss) - 0.5)
        return th.hstack([positive_loss, negative_loss])
    
    else:
        loss = exponential_loss((antecedents, prod), exp_morphisms, embed_up)
        return loss
