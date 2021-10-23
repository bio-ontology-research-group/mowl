import click as ck
import torch as th
import torch.nn as nn
from torch import optim
from torch.linalg import matrix_norm
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
from math import floor
import logging


from mowl.model import Model
from mowl.graph.taxonomy.model import TaxonomyParser

logging.basicConfig(level=logging.DEBUG)

class CatEmbeddings(Model):
    def __init__(self, dataset, file_params = None, seed = 0):
        super().__init__(dataset)
        self.file_params = file_params

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    

    def train(self):

        #Loading subclass axioms
        parser = TaxonomyParser(self.dataset, subclass=True, relations=False)
        edges = parser.parseOWL()
        logging.info("Parsing ontology finished")

        objects = set()

        for edge in edges:
            src = str(edge.src())
            dst = str(edge.dst())

            objects |= {("", "", src), ("", "", dst), ("up", src, dst), ("down", src, dst), ("exp", src, dst)}

        objects_idx = {obj: i for i, obj in enumerate(objects)}

        #Sanity check
        srcs = [str(e.src()) for e in edges]
        dsts = [str(e.dst()) for e in edges]
        classes = list(set(srcs).union(set(dsts)))


        batch_size = 64
        logging.info("Number of objects: %d %d", len(objects), len(classes)+ 3*len(edges))

        train_set, val_set = train_test_split(edges, train_size = 0.8, shuffle = True)
        logging.debug("Data splitting done")

        train_set = CatDataset(train_set, objects_idx)
        logging.debug("Train dataset created")
        val_set = CatDataset(val_set, objects_idx)
        logging.debug("Val dataset created")
        train_data_loader = DataLoader(train_set, batch_size = batch_size)
        logging.debug("Train dataloader created")
        val_data_loader = DataLoader(val_set, batch_size = batch_size)
        logging.debug("Val dataloader created")

        
        model = CatModel(len(classes), len(edges), 2048)
        paramss = sum(p.numel() for p in model.parameters())
        trainParamss = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(paramss, trainParamss)
        logging.debug("Model created")
        optimizer = optim.Adam(model.parameters(), lr = 0.000001)


        criterion = lambda x: th.sum(x)
        for epoch in range(128):

            epoch_loss = 0

            model.train()

            with ck.progressbar(train_data_loader) as bar:
                for i, (batch_objects, batch_morphisms) in enumerate(bar):
                    res = model(batch_objects, batch_morphisms)
                    loss = criterion(res)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                epoch_loss /= (i+1)
            
            print(f'Epoch {epoch}: Loss - {epoch_loss}') #, \tVal loss - {val_loss}, \tAUC - {roc_auc}')

        
            
class CatModel(nn.Module):

    def __init__(self, num_objects, num_axioms, embedding_size):
        super(CatModel, self).__init__()

        num_obj = num_objects + 3*num_axioms

        dim0 = embedding_size
        dim1 = floor(dim0/2)
        dim2 = floor(dim1/2)
        self.net_object = nn.Sequential(
            nn.Embedding(num_obj, embedding_size),
            nn.Linear(dim0, dim1),
            nn.Linear(dim1, dim2)
        )

        self.proj_up_exp = nn.Linear(dim2, dim2)
        self.proj_up_ant = nn.Linear(dim2, dim2)
        self.proj_down_exp = nn.Linear(dim2, dim2)
        self.proj_down_ant = nn.Linear(dim2, dim2)
        self.prod_up_down = nn.Linear(dim2, dim2)
        self.prod_up_cons = nn.Linear(dim2, dim2)
        self.prod_down_cons = nn.Linear(dim2, dim2)
        
    def compute_loss(self, objects, morphisms):

        objects = map(lambda x: x.unsqueeze(1), objects)
        prod_up, prod_down, exponential, antecedent, consequent = map(self.net_object, objects)



        loss1 = abs(exponential - self.proj_up_exp(prod_up))
        loss2 = abs(antecedent - self.proj_up_ant(prod_up))
        
        loss3 = abs(exponential - self.proj_down_exp(prod_down))
        loss4 = abs(antecedent - self.proj_down_ant(prod_down))

        loss5 = abs(prod_down - self.prod_up_down(prod_up))
        loss6 = abs(consequent - self.prod_down_cons(prod_down))
        loss7 = abs(consequent - self.prod_up_cons(prod_up))

        path_loss1 = matrix_norm(self.proj_up_exp.weight - self.prod_up_down.weight@self.proj_down_exp.weight)
        path_loss2 = matrix_norm(self.proj_up_ant.weight - self.prod_up_down.weight@self.proj_down_ant.weight)
        path_loss3 = matrix_norm(self.prod_up_cons.weight - self.prod_up_down.weight@self.prod_down_cons.weight)

        return sum([loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2, path_loss3])

        
        
    def forward(self, objects, morphisms):      
        return self.compute_loss(objects, morphisms)
        

        
        


    

    
class CatDataset(IterableDataset):
    def __init__(self, edges, object_dict):
        self.edges = edges
        self.object_dict = object_dict

        
    def get_data(self):

        for edge in self.edges:
            antecedent = str(edge.src())
            consequent = str(edge.dst())

            morphisms = tuple(range(7))

            prod_up = self.object_dict[("up", antecedent, consequent)]
            prod_down = self.object_dict[("down", antecedent, consequent)]
            exponential = self.object_dict[("exp", antecedent, consequent)]
            antecedent = self.object_dict[("", "", antecedent)]
            consequent = self.object_dict[("", "", consequent)]

            objects = (prod_up, prod_down, exponential, antecedent, consequent)

            yield objects, morphisms
        

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.edges)
