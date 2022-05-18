import torch as th
import torch.nn as nn
from mowl.develop.catEmbeddings.functor import Functor

ACT = nn.Sigmoid()


def norm(a,b, dim = 1):
    tensor = a-b
    return th.relu(a-b)
#    return th.linalg.norm(tensor, dim = dim)

def norm_(a, b):

    x = th.sum(a * b, dim=1, keepdims=True)
    return 1- th.sigmoid(x)


    

def assert_shapes(shapes):
    assert len(set(shapes)) == 1

class Product(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size):
        super().__init__()
        
    
        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            ACT
        )

        
        
        self.pi1 = EntailmentMorphism(embedding_size)
        self.pi2 = EntailmentMorphism(embedding_size)
        self.m   = EntailmentMorphism(embedding_size)
        self.p1  = EntailmentMorphism(embedding_size)
        self.p2  = EntailmentMorphism(embedding_size)
        

    def forward(self, left, right, up=None):
        product = left + right

        up = product + th.rand(product.shape).to(product.device)
#        if up is None:
#            up = self.embed_up(th.cat([left, right], dim = 1))


        loss = 0
        loss += self.p1(up, left)
        loss += self.m(up, product)
        loss += self.p2(up, right)
        loss += self.pi1(product, left)
        loss += self.pi2(product, right)

        return loss


class EntailmentMorphism(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        
        self.entails = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, antecedent, consequent):
        
        estim_cons = self.entails(antecedent)
        estim_cons = self.dropout(estim_cons)
        loss1 = norm(estim_cons, consequent)
        losses = [loss1]
    
        return sum(losses)

class Existential(nn.Module):
    
    def __init__(self, embedding_size):
        super().__init__()
        
        self.slicing = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            ACT
        )

        self.transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )
        
    def forward(self, variable, internal):

        var = self.transform(variable)
        return var + internal
        x = th.cat([variable, internal], dim =1)
        return self.slicing(x)

        
