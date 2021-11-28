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
import time
from itertools import chain

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

        train_data_loader, val_data_loader, num_classes, num_edges, num_objects = self.load_data()
        
        model = CatModel(num_classes, num_edges, num_objects, 2048)
        paramss = sum(p.numel() for p in model.parameters())
        logging.debug("Model created")
        optimizer = optim.Adam(model.parameters(), lr = 0.000000001, weight_decay=0)


        criterion = lambda x: th.mean(x)
        criterion2 = nn.BCELoss()
        for epoch in range(50):

            epoch_loss = 0

            model.train()
            print("trrroutosuntahesuhta: ", len(train_data_loader))
            with ck.progressbar(train_data_loader) as bar:
                for i, (batch_objects_pos, batch_objects_neg1, batch_objects_neg2) in enumerate(bar):
                    cat_loss, logits = model(batch_objects_pos, batch_objects_neg1, batch_objects_neg2)
                    cat_loss = criterion(cat_loss)
                    batch_size = len(batch_objects_pos[0])

                    lbls = th.cat([th.ones(batch_size), th.zeros(batch_size)], 0)
                    classif_loss = criterion2(logits.squeeze(), lbls.float())
                    print(cat_loss.detach(), classif_loss.detach())
                    loss = cat_loss + classif_loss
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
                    for i, (batch_objects_pos, batch_objects_neg1, batch_objects_neg2) in enumerate(bar):
                        logits = model(batch_objects_pos, batch_objects_neg1, batch_objects_neg2)

                        batch_size = len(batch_objects_pos[0])
                        lbls = th.cat([th.ones(batch_size), th.zeros(batch_size)], 0)

                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                        loss = criterion2(logits.squeeze(), lbls)
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

        train_set_path = "data/train_set_go_slim.pkl"
        val_set_path = "data/val_set_go_slim.pkl"
        
        try:
            logging.debug("In try")
            infile_train = open(train_set_path, 'rb')
            train_set = pkl.load(infile_train)
            infile_val = open(val_set_path, 'rb')
            val_set = pkl.load(infile_val)
        except:
            logging.debug("In except")
            parser = TaxonomyParser(self.dataset, subclass=True, relations = False)

            train_set = set(parser.parseOWL(data = "train"))
            val_set = set(parser.parseOWL(data = "val"))
            val_set = val_set - train_set


            train_set = [(str(e.src()), str(e.rel()), str(e.dst())) for e in train_set]
            val_set = [(str(e.src()), str(e.rel()), str(e.dst())) for e in val_set]


            random.shuffle(train_set)
            random.shuffle(val_set)

            outfile_train = open(train_set_path, 'wb')
            pkl.dump(train_set, outfile_train)
            outfile_val = open(val_set_path, 'wb')
            pkl.dump(val_set, outfile_val)


        random.shuffle(train_set)
        random.shuffle(val_set)
        print("Total traininig data size: ", len(train_set))
        print("Total validation data size: ", len(val_set))

#        train_set = train_set[:30000]
#        val_set = val_set[:40000]
        logging.debug("Train and val sets loaded")
        neg_train, neg_val = generate_negatives(train_set, val_set)

        logging.debug("Negatives generated")
        
        train_loader, objects_idx, num_classes, num_edges  = self.getDataLoader(train_set, neg_train)
        val_loader, _, _, _ = self.getDataLoader(val_set, neg_val, objects_idx, mode = "val")

        logging.debug("Finished loading data")
        return train_loader, val_loader, num_classes, num_edges, len(objects_idx)

    def getDataLoader(self, edges, negatives, objects_idx=None, mode = "train"):
        objects = {("", "", "owl#Thing")}
        
        for edge in edges:
            src = str(edge[0])
            dst = str(edge[2])

            if mode == "train":
                objects |= {("", "", src), ("", "", dst), ("up", src, dst), ("down", src, dst), ("exp", src, dst)}
                objects |= {("up", dst, src), ("down", dst, src), ("exp", dst, src)}

            elif mode in ["val", "test"]:
                objects |= {("", "", src), ("", "", dst)}

        for s, neg in negatives.items():

            if mode == "train":
                src_neg = str(neg[0])
                dst_neg = str(neg[2])

                if ("up", src_neg, dst_neg) in objects:
                    print("Already: ", ("up", src_neg,dst_neg))

                objects |= {("", "", src_neg), ("", "", dst_neg), ("up", src_neg, dst_neg), ("down", src_neg, dst_neg), ("exp", src_neg, dst_neg)}

                
            elif mode in ["val", "test"]:
                
                src = str(neg[0])
                dst = str(neg[2])
                objects |= {("", "", src), ("", "", dst)}

                
#        if objects_idx is None:
        objects_idx = {obj: i for i, obj in enumerate(objects)}

        #Sanity check
        srcs = [e[0] for e in edges]
        dsts = [e[2] for e in edges]
        classes = list(set(srcs).union(set(dsts)))
        if mode == "train":
            print(len(objects), len(classes) + 6*len(edges) + 3*len(negatives) +1)
            assert  len(objects) == len(classes)+ 6*len(edges) + 3*len(negatives) + 1
        
        data_set = CatDataset(edges, negatives, objects_idx, mode)
        print("len data_set: ", len(data_set))
        data_loader = DataLoader(data_set, batch_size = self.batch_size, drop_last=True)
        print("len_dataloeadet: ", len(data_loader))
        return data_loader, objects_idx, len(classes), len(edges)


def generate_negatives(train_set, val_set):

    neg_train_file = "data/neg_train_go_slim.pkl2"
    neg_val_file = "data/neg_val_go_slim.pkl2"

    srcs_train = {s for (s, _, _) in train_set}
    r = train_set[0][1]
    entities = [[s,d] for (s, _, d) in train_set]
    entities = set(chain(*entities))

    train_set_inversed = [(d,r,s) for (s,r,d) in train_set]
    val_set_inversed = [(d,r,s) for (s,r,d) in val_set]
    
    try:
        logging.debug("Try to load training negatives")
        infile_neg_train = open(neg_train_file, 'rb')
        train_neg = pkl.load(infile_neg_train)
        logging.debug("training set loaded")
    except:
        logging.debug("Could not load training negatives, generating them...")

        banned_set = set(val_set) | set(val_set_inversed) | set(train_set) | set(train_set_inversed)
        
        train_neg = {}
        for s in list(srcs_train):
            start = time.time()
            triple_in_val = True
            train_entities = list(entities)[:]
            while triple_in_val:
                newD = random.choice(train_entities)
                if newD != s and (s,r,newD) not in banned_set | set(train_neg.values()):
                    train_neg[s] = (s,r,newD)
                    triple_in_val = False
                else:
                    train_entities.remove(newD)
            end = time.time()

#        outfile_train = open(neg_train_file, 'wb')
#        pkl.dump(train_neg, outfile_train)
        
        logging.debug("Traning negatives generated")

    srcs_val = {s for s,_,_ in val_set}
    
    try:
        logging.debug("Try to load validation negatives")
        infile_neg_val = open(neg_val_file, 'rb')
        val_neg = pkl.load(infile_neg_val)
        logging.debug("validation set loaded")

    except:
        logging.debug("Could not load validation negatives, generating them...")
        val_neg = {}
        train_neg_values = set(train_neg.values())

        banned_set = set(train_set) | set(train_set_inversed) | set(val_set) | set(val_set_inversed) | train_neg_values
        for s in list(srcs_val):
            start = time.time()
            triple_in_train = True
            val_entities = list(entities)[:]
            while triple_in_train:
                newD = random.choice(val_entities)
                if newD != s and (s,r,newD) not in banned_set | set(val_neg.values()):
                    val_neg[s] = (s,r,newD)
                    triple_in_train = False
                else:
                    val_entities.remove(newD)
            end = time.time()
            
#        outfile_val = open(neg_val_file, 'wb')
#        pkl.dump(val_neg, outfile_val)

        logging.debug("Validation negatives generated")


    train_neg = {k:v for k,v in train_neg.items() if k in srcs_train}
    print(len(train_neg), len(srcs_train))
    assert len(train_neg) == len(srcs_train)

    val_neg = {k:v for k,v in val_neg.items() if k in srcs_val}
    print("VALLLLL: ", len(srcs_val), len(val_neg))
    return train_neg, val_neg

class CatModel(nn.Module):

    def __init__(self, num_classses, num_axioms, num_objects, embedding_size):
        super(CatModel, self).__init__()

        num_obj = num_objects 

        dim0 = embedding_size
        dim1 = floor(dim0/2)
        dim2 = floor(dim1/2)
        self.net_object = nn.Sequential(
            nn.Embedding(num_obj, embedding_size),
            nn.Linear(dim0, dim1),
            nn.Linear(dim1, dim2)
        )

        
        self.proj_up_exp = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.proj_up_ant = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.proj_down_exp = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.proj_down_ant = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.prod_up_down = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.prod_up_cons = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.prod_down_cons = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))
        self.sim = nn.Sequential(nn.Linear(dim2, dim2),
                                         nn.ReLU(),
                                         nn.Linear(dim2, floor(dim2)))

        self.classif = nn.Linear(dim2,1)
        self.dropout = nn.Dropout()
        
    def compute_loss_train(self, objects):

        prod_up, prod_down, exponential, antecedent, consequent = map(self.net_object, objects)

        full = True
        if full:
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

        sim_loss  = abs(consequent - self.sim(antecedent))

        logit = th.sigmoid(self.classif(sim_loss))

        if not full:
            return th.empty(1), logit
        else:
            return sum([loss1, loss2, loss3, loss4, loss5, loss6, loss7, path_loss1, path_loss2, path_loss3, sim_loss])/11, logit

    def compute_loss_test(self, objects):

        antecedent, consequent = map(self.net_object, objects)

        sim_loss  = abs(consequent - self.sim(antecedent))
        logit = th.sigmoid(self.classif(sim_loss))
        
        return logit

        
        
    def forward(self, positive, negative1, negative2):
        if self.training:
            loss_pos, logit_pos = self.compute_loss_train(positive)
            neg = random.choice([negative1, negative2])
            loss_neg, logit_neg = self.compute_loss_train(neg)
            logits = th.cat([logit_neg, logit_pos], 0)

            cat_loss = F.relu(loss_pos - loss_neg) 
            return 15*cat_loss, logits
        else:
            loss_pos = self.compute_loss_test(positive)
            neg = random.choice([negative1, negative2])
            loss_neg = self.compute_loss_test(neg)
            logits = th.cat([loss_neg, loss_pos], 0)
            return logits

        
        


    

    
class CatDataset(IterableDataset):
    def __init__(self, edges, negatives, object_dict, mode="train"):
        self.edges = edges
        self.object_dict = object_dict
        self.mode = mode
        self.negatives = negatives
        self.srcs = list({s for s,_,_ in edges})
    def get_data(self):

        for edge in self.edges:
            antecedent = edge[0]
            consequent = edge[2]


            positive = self.generate_objects(antecedent, consequent)
            negative1 = self.generate_objects(consequent, antecedent)
            sampled = self.negatives[antecedent][2]
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
