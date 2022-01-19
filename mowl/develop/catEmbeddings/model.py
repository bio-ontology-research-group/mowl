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
import math

from mowl.model import Model
from mowl.graph.taxonomy.model import TaxonomyParser
from mowl.graph.edge import Edge

logging.basicConfig(level=logging.DEBUG)

class CatEmbeddings(Model):
    def __init__(self, dataset, batch_size, file_params = None, seed = 0):
        super().__init__(dataset)
        self.file_params = file_params
        self.batch_size = 32 # batch_size

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    

    def train(self):

        train_data_loader, val_data_loader, num_classes, num_edges, num_objects = self.load_data()
        
        model = CatModel(num_classes, num_edges, num_objects,1024)
        paramss = sum(p.numel() for p in model.parameters())
        logging.info("Number of parameters: %d", paramss)
        logging.debug("Model created")
        lr = 1e-1
        optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay=1e-3)


        criterion = lambda x: x
        criterion2 = nn.BCELoss()
        best_roc_auc = 0
        best_roc_auc_inv = 1
        for epoch in range(128):

            epoch_loss = 0
            train_cat_loss = 0
            model.train()

            with ck.progressbar(train_data_loader) as bar:
                for i, (batch_objects_pos, batch_objects_neg1, batch_objects_neg2) in enumerate(bar):
                    cat_loss, logits = model(batch_objects_pos, batch_objects_neg1, batch_objects_neg2)
                    cat_loss = criterion(cat_loss)
                    batch_size = len(batch_objects_pos[0])

                    lbls = th.cat([th.ones(batch_size), th.zeros(batch_size), th.zeros(batch_size)], 0)
                    classif_loss = criterion2(logits.squeeze(), lbls)
                    #                    print(cat_loss.detach(), classif_loss.detach())
                    loss = classif_loss + cat_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()
                    train_cat_loss += cat_loss.detach().item()
                epoch_loss /= (i+1)
                train_cat_loss /= (i+1)
                
            model.eval()
            val_loss = 0
            val_loss_pred = 0
            val_loss_cat = 0
            preds = []
            labels = []
        
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(val_data_loader) as bar:
                    for i, (batch_objects_pos, batch_objects_neg1, batch_objects_neg2) in enumerate(bar):
                        cat_loss, logits = model(batch_objects_pos, batch_objects_neg1, batch_objects_neg2)
                        cat_loss = criterion(cat_loss)
                        batch_size = len(batch_objects_pos[0])
                        lbls = th.cat([th.ones(batch_size), th.zeros(batch_size), th.zeros(batch_size)], 0)
                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                        loss = criterion2(logits.squeeze(), lbls) + cat_loss
                        val_loss += loss.detach().item()
                        val_loss_cat += cat_loss
                    val_loss /= (i+1)
                    val_loss_cat /= (i+1)

            roc_auc = compute_roc(labels, preds)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_epoch = epoch
            if roc_auc < best_roc_auc_inv:
                best_roc_auc_inv = roc_auc
                best_epoch_inv = epoch
                #                th.save(model.state_dict(), self.file_params["output_model"])

            print(f'Epoch {epoch}: Loss - {epoch_loss:.6}, \t TCatLoss - {train_cat_loss:.6}, \tVal loss - {val_loss:.6}, \tCat loss - {float(val_loss_cat):.6}, \tAUC - {roc_auc:.6}')
#            print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal AUC - {roc_auc}')

 #           print(f'Epoch {epoch}: Loss - {epoch_loss}') #, \tVal loss - {val_loss}, \tAUC - {roc_auc}')


    def load_data(self):

        train_set_path = "data/train_set_go_slim.pkl"
        val_set_path = "data/val_set_go_slim.pkl"
        
        try:
            logging.debug("In try")
            infile_train = open(train_set_path, 'rb')
            edges_train_set = pkl.load(infile_train)
            infile_val = open(val_set_path, 'rb')
            edges_val_set = pkl.load(infile_val)
        except:
            logging.debug("In except")

            train_parser = TaxonomyParser(self.dataset.ontology, bidirectional_taxonomy = False)
            edges_train_set = train_parser.parse()

            val_parser = TaxonomyParser(self.dataset.validation, bidirectional_taxonomy = False)
            edges_val_set = val_parser.parse()
            edges_val_set = list(set(edges_val_set) - set(edges_train_set))

            outfile_train = open(train_set_path, 'wb')
            pkl.dump(edges_train_set, outfile_train)
            outfile_val = open(val_set_path, 'wb')
            pkl.dump(edges_val_set, outfile_val)

        train_set = list( map(lambda x: x.astuple(), edges_train_set))
        val_set = list(map(lambda x: x.astuple(), edges_val_set))

        _, relations = Edge.getEntitiesAndRelations(edges_train_set)
        logging.info("Relations are: %s", str(relations))

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
        objects = {"owl#Thing"}
        
        for edge in edges:
            src = str(edge[0])
            dst = str(edge[2])

            objects |= {src, dst}

        # for s, neg in negatives.items():

        #     if mode == "train":
        #         src_neg = str(neg[0])
        #         dst_neg = str(neg[2])

        #         if ("up", src_neg, dst_neg) in objects:
        #             print("Already: ", ("up", src_neg,dst_neg))

        #         objects |= {("", "", src_neg), ("", "", dst_neg), ("up", src_neg, dst_neg)}

                
        #     elif mode in ["val", "test"]:
                
        #         src = str(neg[0])
        #         dst = str(neg[2])
        #         objects |= {("", "", src), ("", "", dst)}

                
#        if objects_idx is None:
        objects_idx = {obj: i for i, obj in enumerate(objects)}

        #Sanity check
        srcs = [e[0] for e in edges]
        dsts = [e[2] for e in edges]
        classes = list(set(srcs).union(set(dsts)))
        if mode == "train":
            print(len(objects), len(classes) + 2*len(edges) + len(negatives) +1)
            assert  len(objects) == len(classes) + 1
        
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

        outfile_train = open(neg_train_file, 'wb')
        pkl.dump(train_neg, outfile_train)
        
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
            
        outfile_val = open(neg_val_file, 'wb')
        pkl.dump(val_neg, outfile_val)

        logging.debug("Validation negatives generated")


    train_neg = {k:v for k,v in train_neg.items() if k in srcs_train}
    print(len(train_neg), len(srcs_train))
    assert len(train_neg) == len(srcs_train)

    val_neg = {k:v for k,v in val_neg.items() if k in srcs_val}
    return train_neg, val_neg


class RMSELoss(nn.Module):
    def __init__(self, reduction = "mean", eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction = reduction)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = th.sqrt(self.mse(yhat,y) + self.eps)
        return loss
        
class CatModel(nn.Module):

    def __init__(self, num_classses, num_axioms, num_objects, embedding_size):
        super(CatModel, self).__init__()

        num_obj = num_objects 

        self.dropout = nn.Dropout(0.4)

        self.embed = nn.Embedding(num_obj, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -0, 1)
        
        self.net_object = nn.Sequential(
            self.embed,
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.emb_exp = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )
        
        self.emb_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
#            nn.Linear(1024, dim2)
        )

        self.emb_down = nn.Sequential(
            nn.Linear(3*embedding_size, 2*embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(2*embedding_size, embedding_size),
            nn.Sigmoid()
        )

        self.up2exp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )
        
        self.up2ant = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )

        self.down2exp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )

        self.down2ant = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )
        
        self.up2down = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )
        self.up2cons = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )
        
        self.down2cons = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )
        
    def compute_loss(self, objects):

        rmse = RMSELoss()
        rmseNR = RMSELoss(reduction = "none")
        antecedent, consequent = map(self.net_object, objects)
        
        
        up = self.emb_up(th.cat([antecedent, consequent], dim = 1))
#        exponential = consequent/(antecedent + 1e-6)
#        exponential = exponential.where(exponential > 1, th.tensor(1.0))
        
        exponential = self.emb_exp(th.cat([antecedent, consequent], dim = 1))
#        down = exponential + antecedent
        down = self.emb_down(th.cat([antecedent, consequent, exponential], dim =1))
        full = True
        
        if full:

            estim_downFromUp = self.up2down(up)

            estim_expFromUp = self.up2exp(up)

            estim_expFromdown = self.down2exp(down)
            estim_expFromDownChained = self.down2exp(estim_downFromUp)
            
            estim_antFromUp = self.up2ant(up)

            estim_antFromDown = self.down2ant(down)
            estim_antFromDownChained = self.down2ant(estim_downFromUp)
            
            estim_consFromUp = self.up2cons(up)
            estim_consFromDown = self.down2cons(down)
            estim_consFromDownChained = self.down2cons(estim_downFromUp)
            
#            estim_consFromAnt = self.sim(antecedent)

            margin = 1
            loss1 = rmseNR(estim_expFromUp, exponential) #th.sum(th.relu(estim_expFromUp - exponential + margin))
            loss1 = th.mean(loss1, dim=1)
            loss2 = rmseNR(estim_antFromUp, antecedent) # th.sum(th.relu(estim_antFromUp - antecedent + margin))
            loss2 = th.mean(loss2, dim=1)
            
            loss3 = rmseNR(estim_expFromdown, exponential) #th.sum(th.relu(exponential - estim_expFromdown + margin))
            loss3 = th.mean(loss3, dim=1)
            loss4 = rmseNR(estim_antFromDown, antecedent) #th.sum(th.relu(antecedent - estim_antFromDown + margin))
            loss4 = th.mean(loss4, dim=1)
            
            loss5 = rmseNR(estim_downFromUp, down) #th.sum(th.relu(down - estim_downFromUp + margin))
            loss5 = th.mean(loss5, dim=1)
            loss6 = rmseNR(estim_consFromDown, consequent) #th.sum(th.relu(consequent - estim_consFromDown + margin))
            loss6 = th.mean(loss6, dim=1)
            loss7 = rmseNR(estim_consFromUp, consequent) #th.sum(th.relu(consequent - estim_consFromUp + margin))
            loss7 = th.mean(loss7, dim=1)

            #            loss8 = th.sum(consequent * estim_cons_ant, dim=1, keepdims = True)

            

            path_loss1 = rmseNR(estim_expFromDownChained, exponential)
            path_loss1 = th.mean(path_loss1, dim=1)
            
            path_loss2 = rmseNR(estim_antFromDownChained, antecedent)
            path_loss2 = th.mean(path_loss2, dim=1)
            
            path_loss3 = rmseNR(estim_consFromDownChained, consequent)
            path_loss3 = th.mean(path_loss3, dim =1)
#            path_loss4 = th.sum(estim_consequent_down * self.sim(estim_antecedent_down), dim=1, keepdims = True)
            #sim_loss  = self.similarity(th.cat([estim_antecedent_up, estim_antecedent_down, estim_consequent_up, estim_consequent_down, estim_exponential_up, estim_exponential_down, estim_prod_down], dim=1))  #abs(consequent - self.sim(antecedent))
            #sim_loss  = abs(consequent - self.sim(antecedent)) + abs(estim_consequent_down - self.sim(estim_antecedent_down)) + abs(estim_consequent_up - self.sim(estim_antecedent_up))
#            sim_loss  = th.sum(estim_consequent_down * estim_antecedent_down * estim_exponential_down, dim = 1, keepdims = True)
#            sim_loss1  = rmseNR(estim_consFromDown, estim_antFromDown + estim_expFromdown)
#            sim_loss2  = rmseNR(estim_consFromUp, estim_antFromUp + estim_expFromUp)
            sim_loss1  = rmseNR(estim_consFromDown, antecedent + exponential)
            sim_loss1 = th.mean(sim_loss1, dim=1)
            sim_loss2  = rmseNR(estim_consFromUp, antecedent + exponential)
            sim_loss2  = th.mean(sim_loss2, dim=1)
            sim_loss3  = rmseNR(consequent, antecedent + exponential)
            sim_loss3 = th.mean(sim_loss3, dim=1)
            
            sim_loss = sim_loss1 + sim_loss2 + sim_loss3

            assert loss1.shape == loss2.shape
            assert loss2.shape == loss3.shape
            assert loss3.shape == loss4.shape
            assert loss4.shape == loss5.shape
            assert loss5.shape == sim_loss.shape
            assert loss6.shape == sim_loss.shape
            assert loss7.shape == sim_loss.shape
            assert path_loss3.shape == sim_loss.shape
            assert path_loss1.shape == path_loss2.shape
            assert path_loss2.shape == path_loss3.shape
            sim_loss = loss5 + loss6+ loss7 + path_loss3 + sim_loss + loss1 +loss2 +loss3 + loss4 + path_loss1 + path_loss2
        else:
            sim_loss  = th.abs(consequent - self.sim(antecedent))

            #            sim_loss  = self.similarity_simple(th.cat([antecedent, consequent], dim=1))  #abs(consequent - s            

#        logit = 
        #logit = th.sigmoid(sim_loss) #
        logit = 1 - 2*(th.sigmoid(sim_loss) - 0.5)

        if not full:
            return th.zeros(antecedent.shape), logit
        else:
            losses = [loss5, loss6, loss7]
#            losses = [loss1, loss2, loss3, loss4, loss5,  loss6, loss7, 10*th.sum(sim_loss), path_loss1, path_loss2, path_loss3]
            return th.sum(th.stack(losses))/len(losses), logit
 #           return 0, logit
 
        
        
    def forward(self, positive, negative1, negative2):
        loss_pos, logit_pos = self.compute_loss(positive)
        if False and self.training:
            neg = negative1
        else:
#            neg = negative1
            neg1 = negative1 # random.choice([negative1, negative2])
            neg2 = negative2 #random.choice([negative1, negative2])
   
        loss_neg1, logit_neg1 = self.compute_loss(neg1)
        loss_neg2, logit_neg2 = self.compute_loss(neg2)
        logits = th.cat([logit_pos, logit_neg1, logit_neg2], 0)
        margin = 0
        cat_loss = th.tensor(0.0, requires_grad = True) # th.relu(2*loss_pos -loss_neg1- loss_neg2 + margin)

        #print(cat_loss.shape)
        return cat_loss, logits
        
        


    

    
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
 
        antecedent_ = self.object_dict[antecedent]
        consequent_ = self.object_dict[consequent]

        return antecedent_, consequent_
                            
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc
