import torch as th
import torch.nn as nn
import random
#ACT = nn.Sigmoid()
ACT = nn.Identity()

def norm(a,b, dim = 1):
#    normA = th.linalg.norm(a, dim = dim)
#    normB = th.linalg.norm(b, dim = dim)
#    tensor = normA - normB
#    tensor = a-b
#    tensor = th.relu(tensor)
#    tensor = th.linalg.norm(tensor, dim = dim)
#    return tensor
#    return th.relu(tensor)

    n = a.shape[1]
    sqe = (a-b)**2
    rmse = th.sqrt(th.sum(sqe, dim =1))/n 
    return rmse
#    return th.linalg.norm(a-b, dim = dim)

def norm_(a, b):

    x = th.sum(a * b, dim=1, keepdims=True)
    return 1- th.sigmoid(x)


def rand_tensor(shape, device):
    x = th.rand(shape).to(device)
    x = (x*2) - 1
    return x

def assert_shapes(shapes):
    assert len(set(shapes)) == 1


class Product(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, coproduct_net, dropout = 0):
        super().__init__()
        

        
    
        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            ACT
        )

        self.coproduct_net = coproduct_net
        
        self.prod = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            ACT
        )

        self.pi1 = entailment_net
        self.pi2 = entailment_net
        self.m   = entailment_net
        self.p1  = entailment_net
        self.p2  = entailment_net

        self.ent = entailment_net
                                        
        

    def forward(self, left, right, up=None):

        product = self.prod(th.cat([left, right], dim = 1))


        up = (product + rand_tensor(product.shape, product.device))/2


        
        loss = 0

        #Diagram losses
        loss += self.p1(up, left)
        loss += self.m(up, product)
        loss += self.p2(up, right)
        loss += self.pi1(product, left)
        loss += self.pi2(product, right)

        #Distributivity over conjunction: C and (D or E) entails (C and D) or (C and E)
        extra_obj = rand_tensor(product.shape, product.device)

        right_or_extra, roe_loss = self.coproduct_net(right, extra_obj)
        antecedent = self.prod(th.cat([left, right_or_extra], dim = 1))
        left_and_extra = self.prod(th.cat([left, extra_obj], dim = 1))
        consequent, cons_loss = self.coproduct_net(product, left_and_extra)

        loss += roe_loss
        loss += cons_loss
        
        loss += self.ent(antecedent, consequent)
        
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
        

        self.bn = nn.LayerNorm(embedding_size)

        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            ACT
        )

        self.coprod = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            ACT
        )
        
        self.iota1 = entailment_net
        self.iota2 = entailment_net
        self.m   = entailment_net
        self.i1  = entailment_net
        self.i2  = entailment_net
        

    def forward(self, left, right, up=None):

        coproduct = self.coprod(th.cat([left, right], dim = 1))
#        product = (left + right)/2

        down = (coproduct + rand_tensor(coproduct.shape, coproduct.device))/2
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
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
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
        
        
    def forward(self, antecedent, consequent):

        loss = 0
        for morphism in self.hom_set:
            estim_cons = morphism(antecedent)
        
            loss += norm(estim_cons, consequent)
        
        return loss/len(self.hom_set)

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

        
