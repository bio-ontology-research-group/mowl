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
import mowl.develop.catEmbeddings.losses as L
logging.basicConfig(level=logging.DEBUG)

class CatEmbeddings():
    def __init__(self, data_root, batch_size, embedding_size, lr = 0.001,  file_params = None, seed = 0, device = "cpu"):
#        super().__init__(dataset)
        self.data_root = data_root
        self.file_params = file_params
        self.lr = lr
        self.batch_size =  batch_size

        self.embedding_size = embedding_size
        self.device = device

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


    def train(self):

        train_data_loader, test_data_loader, num_classes = self.load_data()
        
        model = CatModel(num_classes, self.embedding_size).to(self.device)
        paramss = sum(p.numel() for p in model.parameters())
        logging.info("Number of parameters: %d", paramss)
        logging.debug("Model created")

#        if self.go_slim:
 #           lr = 5e-2 #go_slim
  #      else:
        lr = self.lr # 1e-0 go
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=0)
        criterion = nn.BCELoss()
        for epoch in range(256):

            epoch_loss = 0
            train_cat_loss = 0
            model.train()

            with ck.progressbar(train_data_loader) as bar:
                for i, (samples, lbls) in enumerate(bar):
                    samples = list(map(lambda x: x.to(self.device), samples))
                    logits = model(samples)
                    lbls = lbls.float().to(self.device)

                    loss = criterion(logits, lbls)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()
                epoch_loss /= (i+1)
                
            model.eval()
            val_loss = 0
            val_loss_pred = 0
            preds = []
            labels = []
        
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(test_data_loader) as bar:
                    for i, (samples, lbls) in enumerate(bar):
                        samples = list(map(lambda x: x.to(self.device), samples))
 
                        logits = model(samples) 
                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                        lbls = lbls.float().to(self.device)
                        loss = criterion(logits.squeeze(), lbls)
                        val_loss += loss.detach().item()
                    val_loss /= (i+1)

            roc_auc = compute_roc(labels, preds)

            print(f'Epoch {epoch}: Loss - {epoch_loss:.6}, \tVal loss - {val_loss:.6}, \tAUC - {roc_auc:.6}')


    def load_data(self):
        train_file = self.data_root + "train_data.pkl"
        
        test_file = self.data_root + "test_data.pkl"

        with open(train_file, "rb") as f:
            train_data = list(pkl.load(f))

        with open(test_file, "rb") as f:
            test_data = list(pkl.load(f))

        random.shuffle(train_data)
        objects = {"owl#Thing"}
        objects |= {s for (s,_),_ in train_data}
        objects |= {t for (_,t),_ in train_data}

        objects = list(objects)
        objects_idx = {v:k for k,v in enumerate(objects)}

        train_set = CatDataset(train_data, objects_idx)
        test_set  = CatDataset(test_data, objects_idx)

        train_dl = DataLoader(train_set, self.batch_size, drop_last = False)
        test_dl = DataLoader(test_set, self.batch_size, drop_last = False)

        return train_dl, test_dl, len(objects)
        
    # def load_data_old(self):

    #     if self.go_slim:
    #         train_set_path = "data/train_set_go_slim.pkl"
    #         val_set_path = "data/val_set_go_slim.pkl"
    #     else:
    #         train_set_path = "data/train_set_go.pkl"
    #         val_set_path = "data/val_set_go.pkl"
        
    #     try:
    #         logging.debug("In try")
    #         infile_train = open(train_set_path, 'rb')
    #         edges_train_set = pkl.load(infile_train)
    #         infile_val = open(val_set_path, 'rb')
    #         edges_val_set = pkl.load(infile_val)
    #     except:
    #         logging.debug("In except")

    #         train_parser = TaxonomyParser(self.dataset.ontology, bidirectional_taxonomy = False)
    #         edges_train_set = train_parser.parse()

    #         val_parser = TaxonomyParser(self.dataset.validation, bidirectional_taxonomy = False)
    #         edges_val_set = val_parser.parse()
    #         edges_val_set = list(set(edges_val_set) - set(edges_train_set))

    #         outfile_train = open(train_set_path, 'wb')
    #         pkl.dump(edges_train_set, outfile_train)
    #         outfile_val = open(val_set_path, 'wb')
    #         pkl.dump(edges_val_set, outfile_val)

    #     train_set = list( map(lambda x: x.astuple(), edges_train_set))
    #     val_set = list(map(lambda x: x.astuple(), edges_val_set))

    #     entitiesTrain, relations = Edge.getEntitiesAndRelations(edges_train_set)
    #     entitiesVal, _ = Edge.getEntitiesAndRelations(edges_val_set)

    #     allEntities = list(set(entitiesTrain) | set(entitiesVal) | {"owlThing"})
    #     objects_idx = {o: i for i, o in enumerate(allEntities)}
    #     num_classes = len(list(objects_idx.keys()))
        
    #     logging.info("Relations are: %s", str(relations))

    #     random.shuffle(train_set)
    #     random.shuffle(val_set)
    #     print("Total traininig data size: ", len(train_set))
    #     print("Total validation data size: ", len(val_set))

    #     if not self.go_slim:
    #         train_set = train_set[:60000]
    #         #val_set = val_set[:100000]
    #     logging.debug("Train and val sets loaded")
    #     neg_train, neg_val = generate_negatives(train_set, val_set, self.go_slim)

    #     logging.debug("Negatives generated")
        
    #     train_loader, num_edges  = self.getDataLoader(train_set, neg_train, objects_idx)
    #     val_loader, _ = self.getDataLoader(val_set, neg_val, objects_idx, mode = "val")

    #     logging.debug("Finished loading data")
    #     return train_loader, val_loader, num_classes, num_edges, len(objects_idx)

    # def getDataLoader(self, edges, negatives, objects_idx, mode = "train"):

    #     data_set = CatDataset(edges, negatives, objects_idx, mode)
    #     print("len data_set: ", len(data_set))
    #     data_loader = DataLoader(data_set, batch_size = self.batch_size, drop_last=False)
    #     print("len_dataloeadet: ", len(data_loader))
    #     return data_loader, len(edges)


# def generate_negatives(train_set, val_set, go_slim = True):

#     if go_slim:
#         neg_train_file = "data/neg_train_go_slim.pkl2"
#         neg_val_file = "data/neg_val_go_slim.pkl2"
#     else:
#         neg_train_file = "data/neg_train_go.pkl2"
#         neg_val_file = "data/neg_val_go.pkl2"

#     srcs_train = {s for (s, _, _) in train_set}
#     r = train_set[0][1]
#     entities = [[s,d] for (s, _, d) in train_set]
#     entities = set(chain(*entities))

#     train_set_inversed = [(d,r,s) for (s,r,d) in train_set]
#     val_set_inversed = [(d,r,s) for (s,r,d) in val_set]
    
#     try:
#         logging.debug("Try to load training negatives")
#         infile_neg_train = open(neg_train_file, 'rb')
#         train_neg = pkl.load(infile_neg_train)
#         logging.debug("training set loaded")
#     except:
#         logging.debug("Could not load training negatives, generating them...")

#         banned_set = set(val_set) | set(val_set_inversed) | set(train_set) | set(train_set_inversed)
        
#         train_neg = {}
#         for s in list(srcs_train):
#             start = time.time()
#             triple_in_val = True
#             train_entities = list(entities)[:]
#             while triple_in_val:
#                 newD = random.choice(train_entities)
#                 if newD != s and (s,r,newD) not in banned_set | set(train_neg.values()):
#                     train_neg[s] = (s,r,newD)
#                     triple_in_val = False
#                 else:
#                     train_entities.remove(newD)
#             end = time.time()

#         outfile_train = open(neg_train_file, 'wb')
#         pkl.dump(train_neg, outfile_train)
        
#         logging.debug("Traning negatives generated")

#     srcs_val = {s for s,_,_ in val_set}
    
#     try:
#         logging.debug("Try to load validation negatives")
#         infile_neg_val = open(neg_val_file, 'rb')
#         val_neg = pkl.load(infile_neg_val)
#         logging.debug("validation set loaded")

#     except:
#         logging.debug("Could not load validation negatives, generating them...")
#         val_neg = {}
#         train_neg_values = set(train_neg.values())

#         banned_set = set(train_set) | set(train_set_inversed) | set(val_set) | set(val_set_inversed) | train_neg_values
#         for s in list(srcs_val):
#             start = time.time()
#             triple_in_train = True
#             val_entities = list(entities)[:]
#             while triple_in_train:
#                 newD = random.choice(val_entities)
#                 if newD != s and (s,r,newD) not in banned_set | set(val_neg.values()):
#                     val_neg[s] = (s,r,newD)
#                     triple_in_train = False
#                 else:
#                     val_entities.remove(newD)
#             end = time.time()
            
#         outfile_val = open(neg_val_file, 'wb')
#         pkl.dump(val_neg, outfile_val)

#         logging.debug("Validation negatives generated")


#     train_neg = {k:v for k,v in train_neg.items() if k in srcs_train}
#     print(len(train_neg), len(srcs_train))
#     assert len(train_neg) == len(srcs_train)

#     val_neg = {k:v for k,v in val_neg.items() if k in srcs_val}
#     return train_neg, val_neg


class RMSELoss(nn.Module):
    def __init__(self, reduction = "mean", eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction = reduction)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = th.sqrt(self.mse(yhat,y) + self.eps)
        return loss
        
class CatModel(nn.Module):

    def __init__(self, num_classes, embedding_size):
        super(CatModel, self).__init__()

        num_obj = num_classes
        self.embedding_size  = embedding_size
        self.dropout = nn.Dropout(0)

        self.embed = nn.Embedding(num_obj, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -k, k)
        
        self.net_object = nn.Sequential(
            self.embed,
#            self.dropout,
#            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

   #     self.norm = nn.BatchNorm()

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
        )

  #       self.emb_down = nn.Sequential(
  #           nn.Linear(3*embedding_size, 2*embedding_size),
  #           self.dropout,
  # #          nn.ReLU(),
  #           nn.Linear(2*embedding_size, embedding_size),
  #           nn.Sigmoid()
  #       )

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

        self.cons2exp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.dropout
        )


        self.exponential_morphisms = (self.up2down, self.up2exp, self.down2exp, self.up2ant, self.down2ant, self.up2cons, self.down2cons)

                #Morphisms for the product
        self.big2left = self.create_morphism()
        self.big2right = self.create_morphism()
        self.prod2left = self.create_morphism()
        self.prod2right = self.create_morphism()
        self.big2prod = self.create_morphism()

        self.product_morphisms = (self.big2prod, self.big2left, self.big2right, self.prod2left, self.prod2right)

#     def compute_loss2(self, objects):

#         rmse = RMSELoss()
#         rmseNR = RMSELoss(reduction = "none")
#         antecedent, consequent = map(self.net_object, objects)
        
        
#         up = self.emb_up(th.cat([antecedent, consequent], dim = 1))
#         exponential = consequent/(antecedent + 1e-6)
#         exponential = exponential.where(exponential > 1, th.tensor(1.0))
        
# #        exponential = self.emb_exp(th.cat([antecedent, consequent], dim = 1))
#         down = (exponential + antecedent)/2
# #        down = self.emb_down(th.cat([antecedent, consequent, exponential], dim =1))
#         full = True
        
#         if full:

#             estim_downFromUp = self.up2down(up)

#             estim_expFromUp = self.up2exp(up)

#             estim_expFromdown = self.down2exp(down)
#             estim_expFromDownChained = self.down2exp(estim_downFromUp)
            
#             estim_antFromUp = self.up2ant(up)

#             estim_antFromDown = self.down2ant(down)
#             estim_antFromDownChained = self.down2ant(estim_downFromUp)
            
#             estim_consFromUp = self.up2cons(up)
#             estim_consFromDown = self.down2cons(down)
#             estim_consFromDownChained = self.down2cons(estim_downFromUp)
            
#             loss1 = rmseNR(estim_expFromUp, exponential)
#             loss1 = th.mean(loss1, dim=1)
#             loss2 = rmseNR(estim_antFromUp, antecedent) 
#             loss2 = th.mean(loss2, dim=1)


            
#             loss3 = rmseNR(estim_expFromdown, exponential) 
#             loss3 = th.mean(loss3, dim=1)
#             loss4 = rmseNR(estim_antFromDown, antecedent) 
#             loss4 = th.mean(loss4, dim=1)
            
#             loss5 = rmseNR(estim_downFromUp, down) 
#             loss5 = th.mean(loss5, dim=1)
#             loss6 = rmseNR(estim_consFromDown, consequent) 
#             loss6 = th.mean(loss6, dim=1)
#             loss7 = rmseNR(estim_consFromUp, consequent) 
#             loss7 = th.mean(loss7, dim=1)

#             #Using the negation form
#             exp_sum = self.cons2exp(consequent)
#             loss8 = rmseNR(exp_sum, exponential)
#             loss8 = th.mean(loss8, dim=1)
            
#             path_loss1 = rmseNR(estim_expFromDownChained, exponential)
#             path_loss1 = th.mean(path_loss1, dim=1)
            
#             path_loss2 = rmseNR(estim_antFromDownChained, antecedent)
#             path_loss2 = th.mean(path_loss2, dim=1)
            
#             path_loss3 = rmseNR(estim_consFromDownChained, consequent)
#             path_loss3 = th.mean(path_loss3, dim =1)

#             path_loss4 = rmseNR(self.cons2exp(estim_consFromDown), exponential)
#             path_loss4 = th.mean(path_loss4, dim=1)
            
#             assert loss1.shape == loss2.shape
#             assert loss2.shape == loss3.shape
#             assert loss3.shape == loss4.shape
#             assert loss4.shape == loss5.shape
#             assert loss5.shape == loss6.shape
#             assert loss6.shape == loss7.shape
#             assert loss7.shape == path_loss1.shape
#             assert path_loss1.shape == path_loss2.shape
#             assert path_loss2.shape == path_loss3.shape
#             assert path_loss3.shape == path_loss4.shape
#             assert path_loss4.shape == loss8.shape
            
#             sim_loss = loss5 + loss6+ loss7 + path_loss3 + loss1 +loss2 +loss3 + loss4 + path_loss1 + path_loss2 + loss8 + path_loss4
#         else:

#             estimCons = self.fc(antecedent)
#             sim_loss = rmseNR(estimCons, consequent)
#             sim_loss = th.mean(sim_loss, dim=1)

#         logit = 1 - 2*(th.sigmoid(sim_loss) - 0.5)

#         return logit
 
        
    def create_morphism(self):
#        fc = nn.Sequential(
#            nn.Linear(self.embedding_size, self.embedding_size),
  #          self.dropout
 #       )
  #      return fc
        return nn.Linear(self.embedding_size, self.embedding_size)

        
    def forward(self, samples):
        samples = th.vstack(samples).transpose(0,1)

        loss = L.nf1_loss(samples, self.exponential_morphisms, self.product_morphisms,(self.net_object, self.emb_up)) # self.compute_loss(positive)
#        loss_pos = self.norm(loss_pos)

        min_ = th.min(loss)
        max_ = th.max(loss)


        loss = (loss-min_)/(max_ - min_)
#        loss_pos = loss_pos/self.embedding_size
        logits = 1 - 2*(th.sigmoid(loss) - 0.5)

      

        
        return logits

#     def forward(self, positive, negative1, negative2):
        


#         positive = th.vstack(positive).transpose(0,1)

#         negative1 = th.vstack(negative1).transpose(0,1)
#         negative2 = th.vstack(negative2).transpose(0,1)

#         loss_pos = L.nf1_loss(positive, self.exponential_morphisms, self.product_morphisms,(self.net_object, self.emb_up)) # self.compute_loss(positive)
# #        loss_pos = self.norm(loss_pos)

#         min_pos = th.min(loss_pos)
#         max_pos = th.max(loss_pos)

#         loss_neg1 = L.nf1_loss(negative1, self.exponential_morphisms, self.product_morphisms, (self.net_object, self.emb_up)) #self.compute_loss(negative1)
#  #       loss_neg1 = self.norm(loss_neg1)

#         min_neg1 = th.min(loss_neg1)
#         max_neg1 = th.max(loss_neg1)

#         loss_neg2 = L.nf1_loss(negative2, self.exponential_morphisms, self.product_morphisms, (self.net_object, self.emb_up)) #self.compute_loss(negative2)
  

#         min_neg2 = th.min(loss_neg2)
#         max_neg2 = th.max(loss_neg2)
  
#         min_ = min(min_pos, min_neg1, min_neg2)
#         max_ = max(max_pos, max_neg1, max_neg2)
        

#         loss_pos = (loss_pos-min_)/(max_ - min_)
# #        loss_pos = loss_pos/self.embedding_size
#         logit_pos = 1 - 2*(th.sigmoid(loss_pos) - 0.5)

#         loss_neg1 = (loss_neg1-min_)/(max_ - min_)
#         logit_neg1 = 1 - 2*(th.sigmoid(loss_neg1) - 0.5)
        
#         loss_neg2 = (loss_neg2-min_)/(max_ - min_)
#         logit_neg2 = 1 - 2*(th.sigmoid(loss_neg2) - 0.5)
      

        
#         return logit_pos, logit_neg1, logit_neg2

    
class CatDataset(IterableDataset):
    def __init__(self, data, object_dict):
        self.data = data
        self.object_dict = object_dict
        
    def get_data(self):

        for sample, label  in self.data:

            sample = self.generate_objects(*sample)
            yield sample, label
        

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.data)

    def generate_objects(self, antecedent, consequent):
 
        antecedent_ = self.object_dict[antecedent]
        consequent_ = self.object_dict[consequent]

        return antecedent_, consequent_


class CatDataset2(IterableDataset):
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
