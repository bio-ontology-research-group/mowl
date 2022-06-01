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
        
        self.pi1 = nn.Linear(embedding_size, embedding_size)
        self.pi2 = nn.Linear(embedding_size, embedding_size)
        self.m   = nn.Linear(embedding_size, embedding_size)
        self.p1  = nn.Linear(embedding_size, embedding_size)
        self.p2  = nn.Linear(embedding_size, embedding_size)
        

    def forward(self, left, right, up=None):
        product = left + right

        if up is None:
            up = self.embed_up(th.cat([left, right], dim = 1))
        left_from_down = self.pi1(product)
        right_from_down = self.pi2(product)

        left_from_up = self.p1(up)
        right_from_up = self.p2(up)
        
        down_from_up = self.m(up)

        left_from_down_chained = self.pi1(down_from_up)
        right_from_down_chained = self.pi2(down_from_up)

        loss1 = norm(left_from_up - left, dim = 1)
        loss2 = norm(left_from_down - left, dim = 1)
        loss3 = norm(right_from_up - right, dim = 1)
        loss4 = norm(right_from_down - right, dim = 1)
        loss5 = norm(down_from_up - product, dim = 1)
        path_loss1 = norm(left_from_down_chained - left, dim = 1)
        path_loss2 = norm(right_from_down_chained - right, dim = 1)
        
        losses = [loss1, loss2, loss3, loss4, loss5, path_loss1, path_loss2]
        shapes = map(lambda x: x.shape, losses)
        assert_shapes(shapes)

        return product, sum(losses)

            

class Exponential(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.entailment = nn.Linear(embedding_size, embedding_size)
        
        self.g        = nn.Linear(embedding_size, embedding_size)
        self.eval_    = nn.Linear(embedding_size, embedding_size)
        self.lambda_g = nn.Linear(embedding_size, embedding_size)

    def forward(self, antecedent, consequent, product_model, up=None):
        
        estim_cons = self.entailment(antecedent)

        loss1 = norm(estim_cons - consequent, dim = 1)
        losses = [loss1]
    
        return None, sum(losses)

    def forward_(self, antecedent, consequent, product_model, up=None):
        
        extra_loss = 0
        if up is None:
            up = self.embed_up(th.cat([antecedent, consequent], dim = 1))
        else:
            estim_up = self.embed_up(th.cat([antecedent, consequent], dim =1))
            extra_loss = norm(estim_up - up, dim = 1)

        exponential = consequent/(antecedent + 1e-10)
        exponential = exponential.where(exponential > 1, th.tensor(1.0).to(up.device))

        down, product_loss = product_model(exponential, antecedent, up = up+antecedent)

        exp_from_up = self.lambda_g(up)

        cons_from_up = self.g(up+antecedent)
        cons_from_down = self.eval_(down)
        cons_from_down_chained = self.eval_(exp_from_up+antecedent)


        loss1 = norm(exp_from_up - exponential, dim = 1)
        loss2 = norm(cons_from_up - consequent, dim = 1)
        loss3 = norm(cons_from_down - consequent, dim = 1)

        path_loss1 = norm(cons_from_down_chained - consequent, dim = 1)


        losses = [loss1, loss2, loss3, path_loss1]
        shapes = map(lambda x: x.shape, losses)
        assert_shapes(shapes)

        return exponential, sum(losses) + product_loss + extra_loss


class EntailmentMorphism(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.entails = nn.Linear(embedding_size, embedding_size)
  

    def forward(self, antecedent, consequent):
        
        estim_cons = self.entails(antecedent)

        loss1 = norm(estim_cons - consequent, dim = 1)
        losses = [loss1]
    
        return sum(losses)

    def forward_(self, antecedent, consequent, product_model, up=None):
        
        extra_loss = 0
        if up is None:
            up = self.embed_up(th.cat([antecedent, consequent], dim = 1))
        else:
            estim_up = self.embed_up(th.cat([antecedent, consequent], dim =1))
            extra_loss = norm(estim_up - up, dim = 1)

        exponential = consequent/(antecedent + 1e-10)
        exponential = exponential.where(exponential > 1, th.tensor(1.0).to(up.device))

        down, product_loss = product_model(exponential, antecedent, up = up+antecedent)

        exp_from_up = self.lambda_g(up)

        cons_from_up = self.g(up+antecedent)
        cons_from_down = self.eval_(down)
        cons_from_down_chained = self.eval_(exp_from_up+antecedent)


        loss1 = norm(exp_from_up - exponential, dim = 1)
        loss2 = norm(cons_from_up - consequent, dim = 1)
        loss3 = norm(cons_from_down - consequent, dim = 1)

        path_loss1 = norm(cons_from_down_chained - consequent, dim = 1)


        losses = [loss1, loss2, loss3, path_loss1]
        shapes = map(lambda x: x.shape, losses)
        assert_shapes(shapes)

        return exponential, sum(losses) + product_loss + extra_loss



class Pullback(nn.Module):
    """Representation of the categorical diagram of the pullback. 
    """

    def __init__(self, embedding_size):
        super().__init__()
        
    
        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.embed_center = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.embed_end = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )
        
        self.pi1 = nn.Linear(embedding_size, embedding_size)
        self.pi2 = nn.Linear(embedding_size, embedding_size)
        self.m   = nn.Linear(embedding_size, embedding_size)
        self.p1  = nn.Linear(embedding_size, embedding_size)
        self.p2  = nn.Linear(embedding_size, embedding_size)
        
        self.f   = nn.Linear(embedding_size, embedding_size)
        self.g   = nn.Linear(embedding_size, embedding_size)

    def forward(self, left, right, center = None, end = None, up=None):
        
        
        extra_loss = 0
        if up is None:
            up = self.embed_up(th.cat([left, right], dim = 1))
        else:
            estim_up = self.embed_up(th.cat([left, right], dim = 1))
            extra_loss += norm(estim_up - up, dim = 1)

        if center is None:
            center = self.embed_center(th.cat([left, right], dim = 1))
        else:
            estim_center = self.embed_center(th.cat([left, right], dim = 1))
            extra_loss += norm(estim_center - center, dim = 1)

        if end is None:
            end = self.embed_end(th.cat([left, right], dim = 1))
        else:
            estim_end = self.embed_end(th.cat([left, right], dim = 1))
            extra_loss += norm(estim_end - end, dim = 1)


        left_from_center = self.pi1(center)
        right_from_center = self.pi2(center)

        left_from_up = self.p1(up)
        right_from_up = self.p2(up)
        
        center_from_up = self.m(up)

        left_from_center_chained = self.pi1(center_from_up)
        right_from_center_chained = self.pi2(center_from_up)

        loss1 = norm(left_from_up - left, dim = 1)
        loss2 = norm(left_from_center - left, dim = 1)
        loss3 = norm(right_from_up - right, dim = 1)
        loss4 = norm(right_from_center - right, dim = 1)
        loss5 = norm(center_from_up - center, dim = 1)
        path_loss1 = norm(left_from_center_chained - left, dim = 1)
        path_loss2 = norm(right_from_center_chained - right, dim = 1)
        


        end_from_left = self.f(left)
        end_from_right = self.g(right)

        loss6 = norm(end_from_left - end, dim=1)
        loss7 = norm(end_from_right - end, dim=1)

    


        losses = [loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2]
        shapes = map(lambda x: x.shape, losses)
        assert_shapes(shapes)

        return None, sum(losses)



class Existential(nn.Module):
    
    def __init__(self, embedding_size):
        super().__init__()
        
        self.gen_k = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.gen_a = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.gen_x = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.slicer_x_a = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.slicer_a = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )


        self.induced_functor = Functor(embedding_size)
        self.dependent_sum = Functor(embedding_size)

        self.morphism_left = nn.Linear(embedding_size, embedding_size)
        self.morphism_right = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, relation, filler, product_net, pullback_net):
        
        k = self.gen_k(th.cat([relation, filler], dim = 1))
        a = self.gen_a(th.cat([relation, filler], dim = 1))
        x = self.gen_x(th.cat([relation, filler], dim = 1))
        
        x_prod_a = x + a

        _, pullback_loss = pullback_net(x_prod_a, k, end = a)
        product, product_loss = product_net(relation, filler)

        sliced_product_x_a = self.slicer_x_a(th.cat([product, x_prod_a], dim=1))

        sliced_product_a = self.slicer_a(th.cat([product, a], dim=1))

        left = self.morphism_left(product)
        existential_object, _ = self.dependent_sum(obj = product)
        
        right = self.morphism_right(existential_object)
        
        left_, _ = self.induced_functor(obj = right)

        loss = norm(left - left_, dim = 1)

        return existential_object, pullback_loss + product_loss + loss


        
