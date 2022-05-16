import torch as th
import torch.nn as nn
from mowl.develop.catEmbeddings.functor import Functor

def norm(tensor, dim = None):
    return th.linalg.norm(tensor, dim = dim)

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
            nn.Sigmoid()
        )
        
        self.pi1 = nn.EntailmentMorphism(embedding_size)
        self.pi2 = nn.EntailmentMorphism(embedding_size)
        self.m   = nn.EntailmentMorphism(embedding_size)
        self.p1  = nn.EntailmentMorphism(embedding_size)
        self.p2  = nn.EntailmentMorphism(embedding_size)
        

    def forward(self, left, right, up=None):
        product = left + right

        if up is None:
            up = self.embed_up(th.cat([left, right], dim = 1))


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
  

    def forward(self, antecedent, consequent):
        
        estim_cons = self.entails(antecedent)

        loss1 = norm(estim_cons - consequent, dim = 1)
        losses = [loss1]
    
        return sum(losses)

class Existential(nn.Module):
    
    def __init__(self, embedding_size):
        super().__init__()
        
        self.slicing = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )
        
    def forward(self, internal, variable):
        
        return self.slicing(variable, internal)

        
