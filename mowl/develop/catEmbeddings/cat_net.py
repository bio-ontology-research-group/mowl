import torch as th
import torch.nn as nn

import random
#ACT = nn.Sigmoid()
ACT = nn.Identity()
#ACT = nn.Tanh()

def norm(a,b, dim = 1):
    n = a.shape[1]
    sqe = (a-b)**2
    mse = th.sum(sqe, dim =1)/n 
    return mse


def rand_tensor(shape, device):
    x = th.rand(shape).to(device)
    x = (x*2) - 1
    return x

def assert_shapes(shapes):
    assert len(set(shapes)) == 1



class ObjectGenerator(nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()

        self.transform = nn.Sequential(

            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),

            nn.LayerNorm(embedding_size),
            ACT
        )

    def forward(self, left,right):
        mean = (left+right)/2

        x = th.cat([left,right], dim = 1)
        x = self.transform(x)
        #x = x+mean
        return x

class Product(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, coproduct_net, dropout = 0):
        super().__init__()
        
        self.coproduct_net = coproduct_net
 
        self.prod = ObjectGenerator(embedding_size, dropout)

        self.up = ObjectGenerator(embedding_size, dropout)


        self.pi1 = entailment_net
        self.pi2 = entailment_net
        self.m   = entailment_net
        self.p1  = entailment_net
        self.p2  = entailment_net

        self.ent = entailment_net

    def forward(self, left, right, up=None):

        product = self.prod(left, right)
        up = self.up(left, right)

        
        loss = 0

        #Diagram losses
        loss += self.p1(up, left)
        loss += self.m(up, product)
        loss += self.p2(up, right)
        loss += self.pi1(product, left)
        loss += self.pi2(product, right)

        #Distributivity over conjunction: C and (D or E) entails (C and D) or (C and E)
#        extra_obj = rand_tensor(product.shape, product.device)

#        right_or_extra, roe_loss = self.coproduct_net(right, extra_obj)

        #antecedent = self.prod(th.cat([left, right_or_extra], dim = 1))
#        antecedent = self.prod(left, right_or_extra)
        
        #left_and_extra = self.prod(th.cat([left, extra_obj], dim = 1))
#        left_and_extra = self.prod(left, extra_obj)

#        consequent, cons_loss = self.coproduct_net(product, left_and_extra)

#        loss += roe_loss
#        loss += cons_loss
        
#        loss += self.ent(antecedent, consequent)

        # (A and B) entails (A or B)
        coprod, coprod_loss = self.coproduct_net(left, right)
        loss += self.ent(product, coprod)


#        chosen = random.choice([left, right])
#        other_product = self.prod(th.cat([chosen, extra_obj], dim = 1))

 #       coproduct, coproduct_loss = self.coproduct_net(product, other_product)
 #       loss += coproduct_loss
#        loss += self.ent(up, coproduct)
        return product, loss



    
class Coproduct(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, dropout = 0):
        super().__init__()
        
        self.coprod = ObjectGenerator(embedding_size, dropout)

        self.down = ObjectGenerator(embedding_size, dropout)

        
        self.iota1 = entailment_net
        self.iota2 = entailment_net
        self.m   = entailment_net
        self.i1  = entailment_net
        self.i2  = entailment_net
        

    def forward(self, left, right, up=None):

        coproduct = self.coprod(left, right)# self.coprod(th.cat([left, right], dim = 1))
        down = self.down(left, right) #self.down(th.cat([left, right], dim = 1))
        #        product = (left + right)/2


        #down = (coproduct + rand_tensor(coproduct.shape, coproduct.device))/2
#        if up is None:
#            up = self.embed_up(th.cat([left, right], dim = 1))


        loss = 0
        loss += self.i1(left, down)
        loss += self.m(coproduct, down)
        loss += self.i2(right, down)
        loss += self.iota1(left, coproduct)
        loss += self.iota2(right, coproduct)

        return coproduct, loss

class MorphismBlock(nn.Module):

    def __init__(self, embedding_size, dropout):
        super().__init__()
        
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.bn = nn.LayerNorm(embedding_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        skip = x
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + skip
    
class EntailmentHomSet(nn.Module):
    def __init__(self, embedding_size, hom_set_size = 1, depth = 1, dropout = 0):
        super().__init__()

        self.hom_set = nn.ModuleList()
        
        for i in range(hom_set_size):
            morphism = nn.Sequential()

            for j in range(depth-1):
                morphism.append(MorphismBlock(embedding_size, dropout))

            morphism.append(nn.Linear(embedding_size, embedding_size))
            morphism.append(nn.LayerNorm(embedding_size))

            morphism.append(ACT)

            self.hom_set.append(morphism)
            self.hom_set.append(nn.Identity())
        
    def forward(self, antecedent, consequent):

        best_loss = float("inf")
        losses = []

        for morphism in self.hom_set:
            residual = morphism(antecedent)
            estim_cons =  residual  + antecedent
            loss = norm(estim_cons, consequent)

            losses.append(loss)
#            mean_loss = th.mean(loss)
            
#            if mean_loss < best_loss:
#                chosen_loss = loss
#                best_loss = mean_loss

#        losses = losses.transpose(0,1)
        losses = th.vstack(losses).transpose(0,1)

        losses = th.min(losses, dim = 1)
        losses_values = losses.values
        losses_indices = losses.indices
        #print(type(losses_indices))
        #print(th.unique(losses_indices, sorted = True, return_counts = True))
        return losses_values


#        return loss/len(self.hom_set)


class Existential(nn.Module):
    
    def __init__(self, embedding_size, prod_net, dropout = 0):
        super().__init__()

        self.prod_net = prod_net

        self.bn1 = nn.LayerNorm(2*embedding_size)
        self.bn2 = nn.LayerNorm(embedding_size)
        self.bn3 = nn.LayerNorm(embedding_size)
        
        self.slicing_filler = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),

            nn.LayerNorm(embedding_size),


            ACT
        )

        self.slicing_relation = nn.Sequential(
            nn.Linear(3*embedding_size, 2*embedding_size),
            nn.LayerNorm(2*embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(2*embedding_size, embedding_size),

            nn.LayerNorm(embedding_size),


            ACT
            
        )
        
    def forward(self, outer, relation, filler):

        x = th.cat([outer, filler, relation], dim =1)
        sliced_relation = self.slicing_relation(x)
        x = th.cat([outer, filler], dim =1)
        sliced_filler = self.slicing_filler(x)

        prod, prod_loss = self.prod_net(sliced_relation, sliced_filler)
 
        return prod, prod_loss

        
