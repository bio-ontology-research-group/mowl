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
import pickle as pkl

from mowl.model import Model
from mowl.graph.taxonomy.model import TaxonomyParser

logging.basicConfig(level=logging.DEBUG)

class CatEmbeddings(Model):
    def __init__(self, dataset, batch_size, file_params = None, seed = 0):
        super().__init__(dataset)
        self.file_params = file_params
        self.batch_size = batch_size

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    

    def train(self):

        #Loading subclass axioms
        # parser = TaxonomyParser(self.dataset, subclass=True, relations=False)
        # edges = parser.parseOWL()
        # logging.info("Parsing ontology finished")

        # objects = set()

        # for edge in edges:
        #     src = str(edge.src())
        #     dst = str(edge.dst())

        #     objects |= {("", "", src), ("", "", dst), ("up", src, dst), ("down", src, dst), ("exp", src, dst)}

        # objects_idx = {obj: i for i, obj in enumerate(objects)}

        # #Sanity check
        # srcs = [str(e.src()) for e in edges]
        # dsts = [str(e.dst()) for e in edges]
        # classes = list(set(srcs).union(set(dsts)))


        # batch_size = 64
        # logging.info("Number of objects: %d %d", len(objects), len(classes)+ 3*len(edges))

        # train_set, val_set = train_test_split(edges, train_size = 0.8, shuffle = True)
        # logging.debug("Data splitting done")

        # train_set = CatDataset(train_set, objects_idx)
        # logging.debug("Train dataset created")
        # val_set = CatDataset(val_set, objects_idx)
        # logging.debug("Val dataset created")
        # train_data_loader = DataLoader(train_set, batch_size = batch_size)
        # logging.debug("Train dataloader created")
        # val_data_loader = DataLoader(val_set, batch_size = batch_size)
        # logging.debug("Val dataloader created")

        train_data_loader, val_data_loader, num_classes, num_edges = self.load_data()
        
        model = CatModel(num_classes, num_edges, 2048)
        paramss = sum(p.numel() for p in model.parameters())
        trainParamss = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(paramss, trainParamss)
        logging.debug("Model created")
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)


        criterion = lambda x: th.mean(x)
        for epoch in range(128):

            epoch_loss = 0

            model.train()

            with ck.progressbar(train_data_loader) as bar:
                for i, batch_objects in enumerate(bar):
                    res = model(batch_objects)
                    loss = criterion(res)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                epoch_loss /= (i+1)


            # model.eval()
            # val_loss = 0
            # preds = []
            # labels = []
            # with th.no_grad():
            #     optimizer.zero_grad()
            #     with ck.progressbar(val_set_batches) as bar:
            #         for iter, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):
            #             feats1 = annots[feats1].view(-1,1)
            #             feats2 = annots[feats2].view(-1,1)
                        
            #             logits = model(batch_g.to(device), feats1.to(device), feats2.to(device))
            #             lbls = batch_labels.unsqueeze(1).to(device)
            #             loss = loss_func(logits, lbls)
            #             val_loss += loss.detach().item()
            #             labels = np.append(labels, lbls.cpu())
            #             preds = np.append(preds, logits.cpu())
            #         val_loss /= (iter+1)

            # roc_auc = self.compute_roc(labels, preds)
            # if not tuning:
            #     if roc_auc > best_roc_auc:
            #         best_roc_auc = roc_auc
            #         th.save(model.state_dict(), self.file_params["output_model"])
            #     print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')

            print(f'Epoch {epoch}: Loss - {epoch_loss}') #, \tVal loss - {val_loss}, \tAUC - {roc_auc}')


    def load_data(self):
        parser = TaxonomyParser(self.dataset, subclass=True, relations = False)


        train_set_path = "data/train_set.pkl"
        val_set_path = "data/val_set.pkl"
        
        try:
            infile_train = open(train_set_path, 'rb')
            train_set = pkl.load(infile_train)
            infile_val = open(val_set_path, 'rb')
            val_set = pkl.load(infile_val)
        except:
            train_set = set(parser.parseOWL(data = "train"))
            val_set = set(parser.parseOWL(data = "val"))
            val_set = val_set - train_set

            train_set = [(str(e.src()), str(e.rel()), str(e.dst())) for e in train_set]
            val_set = [(str(e.src()), str(e.rel()), str(e.dst())) for e in val_set]
            
            outfile_train = open(train_set_path, 'wb')
            pkl.dump(train_set, outfile_train)
            outfile_val = open(val_set_path, 'wb')
            pkl.dump(val_set, outfile_val)
        

        train_loader, objects_idx, num_classes, num_edges  = self.getDataLoader(train_set)
        val_loader, _, _, _ = self.getDataLoader(val_set, objects_idx, mode = "val")

        return train_loader, val_loader, num_classes, num_edges

    def getDataLoader(self, edges, objects_idx=None, mode = "train"):
        objects = set()

        for edge in edges:
            src = str(edge[0])
            dst = str(edge[2])

            if mode == "train":
                objects |= {("", "", src), ("", "", dst), ("up", src, dst), ("down", src, dst), ("exp", src, dst)}
            elif mode in ["val", "train"]:
                objects |= {("", "", src), ("", "", dst)}

        if objects_idx is None:
            objects_idx = {obj: i for i, obj in enumerate(objects)}

        #Sanity check
        srcs = [e[0] for e in edges]
        dsts = [e[2] for e in edges]
        classes = list(set(srcs).union(set(dsts)))
        if mode == "train":
            assert  len(objects) == len(classes)+ 3*len(edges)
        
        data_set = CatDataset(edges, objects_idx, mode)

        data_loader = DataLoader(data_set, batch_size = self.batch_size)
        return data_loader, objects_idx, len(classes), len(edges)
            
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
        
    def compute_loss_train(self, objects):

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

        dot = th.sum(antecedent * consequent, dim=1, keepdims=True)

        return sum([loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2, path_loss3])/10 + th.sigmoid(dot)

    def compute_loss_test(self, objects):

        objects = map(lambda x: x.unsqueeze(1), objects)
        antecedent, consequent = map(self.net_object, objects)

        dot = th.sum(antecedent * consequent, dim=1, keepdims=True)

        return th.sigmoid(dot)

        
        
    def forward(self, objects):
        if self.training:
            return self.compute_loss_train(objects)
        else:
            return self.compute_loss_test(objects)

        
        


    

    
class CatDataset(IterableDataset):
    def __init__(self, edges, object_dict, mode="train"):
        self.edges = edges
        self.object_dict = object_dict
        self.mode = mode
        
    def get_data(self):

        for edge in self.edges:
            antecedent = edge[0]
            consequent = edge[2]

#            morphisms = tuple(range(7))

            antecedent_ = self.object_dict[("", "", antecedent)]
            consequent_ = self.object_dict[("", "", consequent)]

            if self.mode == "train":
                prod_up = self.object_dict[("up", antecedent, consequent)]
                prod_down = self.object_dict[("down", antecedent, consequent)]
                exponential = self.object_dict[("exp", antecedent, consequent)]
    
                objects = (prod_up, prod_down, exponential, antecedent_, consequent_)

            elif self.mode in ["test", "val"]:
                objects = (antecedent_, consequent_)
                
            yield objects #, morphisms
        

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.edges)



