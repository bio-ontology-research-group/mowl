import torch as th
import torch.nn as nn

ACT = nn.Sigmoid()


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


    

def assert_shapes(shapes):
    assert len(set(shapes)) == 1

class Product(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, hom_set_size = 1):
        super().__init__()
        

        self.bn = nn.BatchNorm1d(embedding_size)
        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            #self.bn,
            ACT,
            nn.Dropout(0.3)
#            nn.ReLU(),
#            nn.Linear(embedding_size, embedding_size),
#            ACT
        )

        self.prod = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            #self.bn,
            ACT,
            nn.Dropout(0.3)
#            nn.ReLU(),
#            nn.Linear(embedding_size, embedding_size),
#            ACT
        )
        
        self.pi1 = entailment_net
        self.pi2 = entailment_net
        self.m   = entailment_net
        self.p1  = entailment_net
        self.p2  = entailment_net
        
        # self.pi1 = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.pi2 = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.m   = EntailmentHomSet(embedding_size, hom_set_size = 1)
        # self.p1  = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.p2  = EntailmentHomSet(embedding_size, hom_set_size = 5)
        

    def forward(self, left, right, up=None):

        product = self.prod(th.cat([left, right], dim = 1))
#        product = (left + right)/2

        up = (product + th.rand(product.shape).to(product.device))/2
#        if up is None:
#            up = self.embed_up(th.cat([left, right], dim = 1))


        loss = 0
        loss += self.p1(up, left)
        loss += self.m(up, product)
        loss += self.p2(up, right)
        loss += self.pi1(product, left)
        loss += self.pi2(product, right)

        return product, loss


class EntailmentHomSet(nn.Module):
    def __init__(self, embedding_size, hom_set_size = 1):
        super().__init__()

        

        
        self.hom_set = nn.ModuleList()

        for i in range(hom_set_size):
            morphism = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(embedding_size,embedding_size),
                ACT,
                nn.Dropout(0.3)
                #nn.ReLU()

            )
            self.hom_set.append(morphism)
        
        
    def forward(self, antecedent, consequent):

        loss = 0
        for morphism in self.hom_set:
            estim_cons = morphism(antecedent)
        
            loss += norm(estim_cons, consequent)
        
        return loss

class Existential(nn.Module):
    
    def __init__(self, embedding_size, prod_net):
        super().__init__()

        self.prod_net = prod_net

        self.bn1 = nn.BatchNorm1d(2*embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
        self.slicing_filler = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            #self.bn3,
            nn.ReLU()
#            ACT
        )

        self.slicing_relation = nn.Sequential(
            nn.Linear(3*embedding_size, 2*embedding_size),
            #self.bn1,
            nn.ReLU(),
            nn.Linear(2*embedding_size, embedding_size),
            #self.bn2,
            ACT,
            nn.Dropout(0.3)
#            nn.ReLU()
        )
        
    def forward(self, outer, relation, filler):

        x = th.cat([outer, filler, relation], dim =1)
        sliced_relation = self.slicing_relation(x)
        x = th.cat([outer, filler], dim =1)
        sliced_filler = self.slicing_filler(x)

#        prod = (sliced_relation + sliced_filler)/2

        prod, prod_loss = self.prod_net(sliced_relation, sliced_filler)
 
        return prod, prod_loss

        
