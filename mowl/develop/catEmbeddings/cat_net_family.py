import torch as th
import torch.nn as nn
import random
ACT = nn.Identity()
#ACT = nn.Identity()
#ACT = nn.Sigmoid()
#th.autograd.set_detect_anomaly(True)
def norm_(a,b, dim = 1):
#    normA = th.linalg.norm(a, dim = dim)
#    normB = th.linalg.norm(b, dim = dim)
#    tensor = normA - normB
#    tensor = a-b
#    tensor = th.relu(tensor)
#    tensor = th.linalg.norm(tensor, dim = dim)
#    return tensor
#    return th.relu(tensor)

#    hom_a = a[:, -1].unsqueeze(1)
#    print(a.shape, hom_a.shape)
#    a = a/hom_a
#    hom_b = b[:, -1].unsqueeze(1)
#    b = b/hom_b

#    a = a[:,:-1]
#    b = b[:, :-1]
    
    n = a.shape[1]
    sqe = (a-b)**2
    rmse = th.sqrt(th.sum(sqe, dim =1))/n 
    return rmse
#    return th.linalg.norm(a-b, dim = dim)

def norm(a, b):

    x = th.sum(a * b, dim=1, keepdims=True)
    sim = 1- th.sigmoid(x)
    return sim*sim


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

        self.coproduct_net = coproduct_net
        
        self.prod_ = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
#            nn.LayerNorm(embedding_size),
            ACT
        )

        self.prod = lambda x: (x[:,:embedding_size]+x[:,embedding_size:])/2


        self.up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
 #           nn.LayerNorm(embedding_size),
            ACT
        )

        self.pi1 = entailment_net
        self.pi2 = entailment_net
        self.m   = entailment_net
        self.p1  = entailment_net
        self.p2  = entailment_net

        self.ent = entailment_net
        # self.pi1 = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.pi2 = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.m   = EntailmentHomSet(embedding_size, hom_set_size = 1)
        # self.p1  = EntailmentHomSet(embedding_size, hom_set_size = 5)
        # self.p2  = EntailmentHomSet(embedding_size, hom_set_size = 5)
        

    def forward(self, left, right, up=None):

        product = self.prod(th.cat([left, right], dim = 1))
#        product = (left + right)/2

        up = self.up(th.cat([left, right], dim =1))# (product + rand_tensor(product.shape, product.device))/2
#        if up is None:
#            up = self.embed_up(th.cat([left, right], dim = 1))

        
        loss = 0

        #Diagram losses
#        loss += self.p1(up, left)
#        loss += self.m(up, product)
#        loss += self.p2(up, right)
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

        self.coprod_ = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
  #          nn.LayerNorm(embedding_size),            
            ACT
        )

        self.coprod = lambda x: (x[:,:embedding_size]+x[:,embedding_size:])


        self.down = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
   #         nn.LayerNorm(embedding_size),
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

        down = self.down(th.cat([left, right], dim =1)) #(coproduct + rand_tensor(coproduct.shape, coproduct.device))/2
#        if up is None:
#            up = self.embed_up(th.cat([left, right], dim = 1))


        loss = 0
 #       loss += self.i1(left, down)
 #       loss += self.m(coproduct, down)
 #       loss += self.i2(right, down)
        loss += self.iota1(left, coproduct)
        loss += self.iota2(right, coproduct)

        return coproduct, loss



    
class ProjectiveTranslationMorphismBlock(nn.Module):

    def __init__(self, embedding_size, dropout):
        super().__init__()

                                         
        self.fc = ProjectiveLinear(embedding_size, embedding_size, bias = False)
        
        self.bn = nn.LayerNorm(embedding_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
 #       print(self.mask.shape, self.fc.weight.shape)
    def forward(self, x):
        
 #       self.fc.weigth *= self.mask
 #       x = th.cat([x, self.hom_coord], dim = 1) #self.proj_mask(x)
        
        self.fc.weigth = self.mask
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)

    
class MorphismBlock(nn.Module):

    def __init__(self, embedding_size, dropout):
        super().__init__()

        self.mask = th.ones((embedding_size+1, embedding_size+1))
#        self.mask[:,-1] = 0
#        self.mask[-1,:] = 0
#        self.mask[-1,-1] = 1

#        self.hom_coord = th.ones((embedding_size, 1))

#        self.proj_mask = Projective(embedding_size)
        self.fc = ProjectiveLinear(embedding_size, embedding_size, bias = False)
        self.bn = nn.LayerNorm(embedding_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
 #       print(self.mask.shape, self.fc.weight.shape)
    def forward(self, x):
        
 #       self.fc.weigth *= self.mask
 #       x = th.cat([x, self.hom_coord], dim = 1) #self.proj_mask(x)
        x = self.fc(x)
      #  x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ProjectiveLinear(nn.Module):

    def __init__(self, x_dim, y_dim , bias = True):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.bias = bias
        self.fc = nn.Linear(x_dim, y_dim, bias = bias)
        x = nn.Parameter(th.rand(1))
        y = nn.Parameter(th.rand(1))
        self.x = x
        self.y = y
         

    def forward(self, x):
        fc = nn.Linear(self.x_dim, self.y_dim, bias = self.bias).to(self.x.device)
        mask = th.eye(self.x_dim).to(self.x.device)
        mask[0,-1] = self.x
        mask[1, -1] = self.y
        fc.weigth = mask
        x = fc(x)
#        self.fc.weigth = self.mask
#        x = self.fc(x)
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
            #morphism.append(nn.LayerNorm(embedding_size))
            morphism.append(ACT)

            self.hom_set.append(morphism)
            
    def forward(self, antecedent, consequent):

        best_loss = float("inf")
        for morphism in self.hom_set:
            estim_cons = morphism(antecedent)
            loss = norm(estim_cons, consequent)
            mean_loss = th.mean(loss)

            if mean_loss < best_loss:
                chosen_loss = loss
                best_loss = mean_loss
        return chosen_loss
    
    def forward_(self, antecedent, consequent):

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
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),

    #        nn.LayerNorm(embedding_size),

            ACT
        )

        self.slicing_relation = nn.Sequential(
            nn.Linear(3*embedding_size, 2*embedding_size),
            nn.LayerNorm(2*embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(2*embedding_size, embedding_size),

     #       nn.LayerNorm(embedding_size),

            ACT
            
        )
        
    def forward(self, outer, relation, filler):

        x = th.cat([outer, filler, relation], dim =1)
        sliced_relation = self.slicing_relation(x)
        x = th.cat([outer, filler], dim =1)
        sliced_filler = self.slicing_filler(x)

#        prod = (sliced_relation + sliced_filler)/2

        prod, prod_loss = self.prod_net(sliced_relation, sliced_filler)
 
        return prod, prod_loss

        
