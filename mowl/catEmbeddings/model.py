import click as ck
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.linalg import matrix_norm
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
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

        train_data_loader, val_data_loader, num_classes, num_edges = self.load_data()
        
        model = CatModel(num_classes, num_edges, 2048)
        paramss = sum(p.numel() for p in model.parameters())
        logging.debug("Model created")
        optimizer = optim.Adam(model.parameters(), lr = 0.000001)


        criterion = lambda x: th.mean(x)
        criterion2 = nn.BCELoss()
        for epoch in range(128):

            epoch_loss = 0

            model.train()

            with ck.progressbar(train_data_loader) as bar:
                for i, (batch_objects_pos, batch_objects_neg, batch_labels_pos, batch_labels_neg) in enumerate(bar):
                    cat_loss, logits_pos, logits_neg = model(batch_objects_pos, batch_objects_neg)
                    cat_loss = criterion(cat_loss)
                    logits = th.cat((logits_pos, logits_neg), 0)
                    lbls = th.cat((batch_labels_pos, batch_labels_neg), 0)

                    classif_loss = criterion2(logits.squeeze(), lbls.float())
                    loss = classif_loss# cat_loss + classif_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                epoch_loss /= (i+1)


            model.eval()
            val_loss = 0
            preds = []
            labels = []
            best_roc_auc = 0
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(val_data_loader) as bar:
                    for i, (batch_objects_pos, batch_objects_neg, batch_labels_pos, batch_labels_neg) in enumerate(bar):
                        logits = model(batch_objects_pos, batch_objects_neg)

                        lbls = th.cat((batch_labels_pos, batch_labels_neg), 0)
                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                        loss = criterion2(lbls.float(), logits.squeeze())
                        val_loss += loss.detach().item()

                    val_loss /= (i+1)

            roc_auc = compute_roc(labels, preds)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
#                th.save(model.state_dict(), self.file_params["output_model"])
            print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')
#            print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal AUC - {roc_auc}')

 #           print(f'Epoch {epoch}: Loss - {epoch_loss}') #, \tVal loss - {val_loss}, \tAUC - {roc_auc}')


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


            train_set = [((str(e.src()), str(e.rel()), str(e.dst())), 1) for e in train_set]
            val_set = [((str(e.src()), str(e.rel()), str(e.dst())),1) for e in val_set]

            neg_train, neg_val = generate_negatives(train_set, val_set)

            train_set = train_set + neg_train
            val_set = val_set + neg_val

            random.shuffle(train_set)
            random.shuffle(val_set
            
            outfile_train = open(train_set_path, 'wb')
            pkl.dump(train_set, outfile_train)
            outfile_val = open(val_set_path, 'wb')
            pkl.dump(val_set, outfile_val)
        


            
        train_loader, objects_idx, num_classes, num_edges  = self.getDataLoader(train_set)
        val_loader, _, _, _ = self.getDataLoader(val_set, objects_idx, mode = "val")

        return train_loader, val_loader, num_classes, num_edges

    def getDataLoader(self, edges, objects_idx=None, mode = "train"):
        objects = {("", "", "owl#Thing")}

        for edge in edges:
            src = str(edge[0])
            dst = str(edge[2])

            if mode == "train":
                objects |= {("", "", src), ("", "", dst), ("up", src, dst), ("down", src, dst), ("exp", src, dst)}
                objects |= {("up", dst, src), ("down", dst, src), ("exp", dst, src)}
                
            elif mode in ["val", "test"]:
                objects |= {("", "", src), ("", "", dst)}

        if objects_idx is None:
            objects_idx = {obj: i for i, obj in enumerate(objects)}

        #Sanity check
        srcs = [e[0] for e in edges]
        dsts = [e[2] for e in edges]
        classes = list(set(srcs).union(set(dsts)))
        if mode == "train":
            print(len(objects), len(classes) + 3*len(edges) +1)
            assert  len(objects) == len(classes)+ 6*len(edges) + 1
        
        data_set = CatDataset(edges, objects_idx, mode)

        data_loader = DataLoader(data_set, batch_size = self.batch_size)
        return data_loader, objects_idx, len(classes), len(edges)
            
class CatModel(nn.Module):

    def __init__(self, num_objects, num_axioms, embedding_size):
        super(CatModel, self).__init__()

        num_obj = num_objects + 6*num_axioms + 1

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
        self.sim = nn.Linear(dim2, dim2)

        self.classif = nn.Linear(dim2,1)
        
    def compute_loss_train(self, objects):

        objects = map(lambda x: x, objects)
        prod_up, prod_down, exponential, antecedent, consequent = map(self.net_object, objects)

#         loss1 = abs(exponential - self.proj_up_exp(prod_up))
#         loss2 = abs(antecedent - self.proj_up_ant(prod_up))
        
#         loss3 = abs(exponential - self.proj_down_exp(prod_down))
#         loss4 = abs(antecedent - self.proj_down_ant(prod_down))

#         loss5 = abs(prod_down - self.prod_up_down(prod_up))
#         loss6 = abs(consequent - self.prod_down_cons(prod_down))
#         loss7 = abs(consequent - self.prod_up_cons(prod_up))

# #        loss8 = abs(consequent - self.sim(antecedent))
#         path_loss1 = matrix_norm(self.proj_up_exp.weight - self.prod_up_down.weight@self.proj_down_exp.weight)
#         path_loss2 = matrix_norm(self.proj_up_ant.weight - self.prod_up_down.weight@self.proj_down_ant.weight)
#         path_loss3 = matrix_norm(self.prod_up_cons.weight - self.prod_up_down.weight@self.prod_down_cons.weight)

        sim_loss  = abs(consequent - self.sim(antecedent))

        logit = th.sigmoid(self.classif(sim_loss))

        return th.empty(1), logit
       # return sum([loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2, path_loss3, sim_loss])/11, logit

    def compute_loss_test(self, objects):

        objects = map(lambda x: x, objects)
        antecedent, consequent = map(self.net_object, objects)

        sim_loss  = abs(consequent - self.sim(antecedent))
        logit = th.sigmoid(self.classif(sim_loss))
        
        return logit

        
        
    def forward(self, objects_pos, objects_neg):
        if self.training:
            loss_pos, logit_pos = self.compute_loss_train(objects_pos)
            loss_neg, logit_neg = self.compute_loss_train(objects_neg)
            return F.relu(loss_pos - loss_neg), logit_pos, logit_neg
        else:
            loss_pos = self.compute_loss_test(objects_pos)
            loss_neg = self.compute_loss_test(objects_neg)
            logits = th.cat((loss_pos, loss_neg), 0)
            return logits

        
        


    

    
class CatDataset(IterableDataset):
    def __init__(self, edges, negatives, object_dict, mode="train"):
        self.edges = edges
        self.object_dict = object_dict
        self.mode = mode
        
    def get_data(self):

        for edge in self.edges:
            antecedent = edge[0]
            consequent = edge[2]


            positive = self.generate_objects(antecedent, consequent)
            negative1 = self.generate_objects(consequent, antecedent)
            sampled = random.choice(negatives[antecedent])
            negative2 = self.generate_objects(antecedent, sampled)
            yield positive, negative1, negative2
        

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.edges)

    def generate_objects(self, antecedent, consequent):
 
        antecedent_ = self.object_dict[("", "", antecedent)]
        consequent_ = self.object_dict[("", "", consequent)]

        if self.mode == "train":
            prod_up = self.object_dict[("up", antecedent, consequent)]
            prod_down = self.object_dict[("down", antecedent, consequent)]
            exponential = self.object_dict[("exp", antecedent, consequent)]

            return prod_up, prod_down, exponential, antecedent_, consequent_

        elif self.mode in ["val", "test"]:
            return antecedent_, consequent_
        else:
            ValueError()
                           
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc
