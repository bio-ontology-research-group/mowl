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
import mowl.develop.catEmbeddings.lossesInt as L
from mowl.develop.catEmbeddings.cat_net_int import Product, EntailmentHomSet, Existential, Coproduct, norm
import os
from mowl.model import Model
from mowl.reasoning.normalize import ELNormalizer
from mowl.projection.taxonomy.model import TaxonomyProjector
from mowl.projection.edge import Edge
from org.semanticweb.owlapi.util import SimpleShortFormProvider
from scipy.stats import rankdata
from mowl.develop.catEmbeddings.evaluate_interactions import evalGCI2Loss, print_metrics
from mowl.develop.catEmbeddings.evaluate import CatEmbeddingsIntersectionEvaluator
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
ACT = nn.Identity()
#ACT = nn.Tanh()

class CatEmbeddings(Model):
    def __init__(
            self, 
            dataset, 
            batch_size, 
            embedding_size,
            lr,
            epochs,
            num_points_eval,
            milestones,
            dropout = 0,
            decay = 0,
            gamma = None,
            eval_intersection = True,
            size_hom_set = 1,
            depth = 1,
            margin = 0,
            seed = -1,
            early_stopping = 10,
            species = "yeast",
            device = "cpu"
    ):
        super().__init__(dataset)

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.epochs = epochs
        self.num_points_eval = num_points_eval
        self.milestones = milestones
        self.dropout = dropout
        self.decay = decay
        self.gamma = gamma
        self.eval_intersection = eval_intersection
        self.size_hom_set = size_hom_set
        self.depth = depth
        self.margin = margin
        self.early_stopping = early_stopping
        self.device = device
        self.dataset = dataset
            
        self.species = species
        milestones_str = "_".join(str(m) for m in milestones)
        self.data_root = f"data/models/{species}/"
        self.file_name = f"bs{self.batch_size}_emb{self.embedding_size}_lr{lr}_epochs{epochs}_eval{num_points_eval}_mlstns_{milestones_str}_drop_{self.dropout}_decay_{self.decay}_gamma_{self.gamma}_evalintersection_{self.eval_intersection}_margin{self.margin}.th"
        self.model_filepath = self.data_root + self.file_name
        self.predictions_file = f"data/predictions/{self.species}/" + self.file_name
        self.labels_file = f"data/labels/{self.species}/" + self.file_name
        print(f"model will be saved in {self.model_filepath}")

        self._loaded = False

        if seed>=0:
            seed_everything(seed)

        self.load_data()
        

        self.model = None
        ### For eval ppi
        

        self.num_classes = len(self.classes_index_dict)
        self.num_rels = len(self.relations)

        self.create_dataloaders(device = self.device)
        self.model = CatModel(self.num_classes, self.num_rels, self.size_hom_set, self.embedding_size, dropout = self.dropout, depth = self.depth)
        #self.ppi_evaluator = CatEmbeddingsPPIEvaluator(self.model.gci2_loss, self.dataset.ontology, self.classes_index_dict, self.relations, proteins, device = self.device)
        self.intersection_evaluator = CatEmbeddingsIntersectionEvaluator(norm, self.classes_index_dict, self.model.prod_net, self.model.entailment_net,  device = self.device)

    def train(self):

        device = self.device
        
        th.save(self.model.state_dict(), self.model_filepath)

        paramss = sum(p.numel() for p in self.model.parameters())

        logging.info("Number of parameters: %d", paramss)
        logging.debug("Model created")
        
        self.model = self.model.to(device)

 
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=self.decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.milestones, gamma = self.gamma) #only nf4#

        best_mean_rank = float("inf")
        best_val_loss = float("inf")
        best_train_loss = float("inf")

#        nf1, nf2, nf3, nf4 = self.train_nfs
        stop_value = self.early_stopping
        train_early_stopping = stop_value
        

                                                                
        forward_function = self.forward_step
        train_data = self.train_dl
        

        for epoch in range(self.epochs):

            batch = self.batch_size
            self.model.train()
            self.model = self.model.to(device)

            
            train_loss = forward_function(
                train_data, 
                self.margin,
                train = True)
    
            self.model.eval()
            
            if best_train_loss < train_loss:
                train_early_stopping -= 1
            else:
                print("saving model")
                th.save(self.model.state_dict(), self.model_filepath)

                best_train_loss = train_loss
                train_early_stopping = stop_value

            
            if self.eval_intersection and (epoch) % 10 == 0:
                metrics, fmetrics = self.evaluate_intersection_valid()
                mean_rank = metrics["mean_rank"]
                if best_mean_rank >= mean_rank:
                    best_mean_rank = mean_rank
                    print("saving model")
                    th.save(self.model.state_dict(), self.model_filepath)
                    
            print(f'Epoch {epoch}: Loss - {train_loss:.6}')

            
            self.scheduler.step()

            if train_early_stopping == 0:
                print(f"Stop training (early stopping): {train_early_stopping}: {best_train_loss}, {valid_early_stopping}, {best_val_loss}")
                break
        
        logging.info("Finished training. Generating predictions")
#        self.run_and_save_predictions(self.model)


    def get_embeddings(self, load_best_model = True):
        self.load_data()
        if load_best_model:
            print('Load the best model', self.model_filepath)
            self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

#        ent_embeds = {k:v for k,v in zip(self.classes_index_dict.keys(), self.model.embed.weight.cpu().detach().numpy())}
        ent_embeds = {k: self.model.net_object(th.tensor([v], device = self.device)).squeeze().cpu().detach().numpy() for k,v in self.classes_index_dict.items()}
#        ent_embeds = {k: v/v[-1] for k, v in ent_embeds.items()}
 #       rel_embeds = {k:v for k,v in zip(self.relations.keys(), self.model.embed_rel.weight.cpu().detach().numpy())}
        rel_embeds = {k: self.model.net_rel(th.tensor([v], device = self.device) ).squeeze().cpu().detach().numpy() for k,v in self.relations.items()}
        return ent_embeds, rel_embeds

        
    def forward_nf(self, nf, idx, margin, train = True):
        
        nf_loss = 0.0
        
        for i, batch_nf in enumerate(nf):
            pos_loss = self.model(batch_nf, idx)

            if train:
                neg_loss = self.model(batch_nf, idx, neg = True)

                assert pos_loss.shape == neg_loss.shape, f"{pos_loss.shape}, {neg_loss.shape}"
            
                loss = pos_loss - neg_loss + margin
#                print(th.mean(pos_loss), th.mean(neg_loss))
#                print(th.mean(loss))
                loss = - th.mean(F.logsigmoid(-loss))
#                print(th.mean(loss))
#                print("here")
                step_loss  = loss
                nf_loss += loss.detach().item()

                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()

            else:
                nf_loss += th.mean(pos_loss).detach().item()
                
            nf_loss /= (i+1)

        return nf_loss

    def forward_step(self, 
                     dataloaders, 
                     margin,
                     train = True
                 ):

        if train:
            nb_nf0, nb_nf1, nb_nf2, nb_nf3 = tuple(map(len, self.train_nfs))
        else:
            nb_nf0, nb_nf1, nb_nf2, nb_nf3 = tuple(map(len, self.valid_nfs))
        
        total = nb_nf0 + nb_nf1 + nb_nf2 + nb_nf3

        data_nf0, data_nf1, data_nf2, data_nf3 = dataloaders
       
        nf0_loss = self.forward_nf(data_nf0, 0, margin, train = train)
        nf1_loss = self.forward_nf(data_nf1, 1, margin, train = train)
        nf2_loss = self.forward_nf(data_nf2, 2, margin, train = train)
        nf3_loss = self.forward_nf(data_nf3, 3, margin, train = train)

        #nf0_loss *= nb_nf0/total
        #nf1_loss *= nb_nf1/total
        nf2_loss *= nb_nf2/total
        nf3_loss *= nb_nf3/total

        print(f"nf0: {nf0_loss:.6}")
        print(f"nf1: {nf1_loss:.6}")
        print(f"nf2: {nf2_loss:.6}")
        print(f"nf3: {nf3_loss:.6}")

        return nf0_loss + nf1_loss + nf2_loss + nf3_loss
        

    def evaluate(self):
#        self.load_data()
        self.model = CatModel(len(self.classes_index_dict), len(self.relations), self.size_hom_set, 1024, dropout = self.dropout, depth = self.depth).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        test_loss = self.forward_step(self.test_dl, self.margin, train=False) #model(self.test_nfs).detach().item()
        print('Test Loss:', test_loss)
 


    def evaluate_intersection_valid(self):

        embeddings,_ = self.get_embeddings(load_best_model = False)
        self.intersection_evaluator(self.gci1_train, embeddings)
        self.intersection_evaluator.print_metrics()

        return self.intersection_evaluator.metrics, self.intersection_evaluator.fmetrics
#        return results


        
    def load_data(self):
        if self._loaded:
            return
        
        normalizer = ELNormalizer()
        self.training_axioms = normalizer.normalize(self.dataset.ontology)

        classes = set()
        relations = set()

        for axioms_dict in [self.training_axioms]:
            for axiom in axioms_dict["gci0"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci0_bot"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)
                
            for axiom in axioms_dict["gci1"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci1_bot"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)
                
            for axiom in axioms_dict["gci2"]:
                classes.add(axiom.subclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

            for axiom in axioms_dict["gci3"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

            for axiom in axioms_dict["gci3_bot"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.obj_property)

        classes = list(classes)
        classes.sort()
        relations = list(relations)
        relations.sort()
        
        self.classes_index_dict = {v: k  for k, v in enumerate(classes)}
        print(f"Number of classes: {len(self.classes_index_dict)}" )
        self.relations = {v: k for k, v in enumerate(relations)}

        training_nfs = self.load_normal_forms(self.training_axioms, self.classes_index_dict, self.relations)
        self.gci1_train = training_nfs[1][:5]
        
        
        self.train_nfs = self.nfs_to_tensors(training_nfs, self.device)

        self._loaded = True
        
    def load_normal_forms(self, axioms_dict, classes_dict, relations_dict):
        gci0 = []
        gci1 = []
        gci2 = []
        gci3 = []

        for axiom in axioms_dict["gci0"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            gci0.append((cl1, cl2))

        for axiom in axioms_dict["gci0_bot"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            gci0.append((cl1, cl2))

        for axiom in axioms_dict["gci1"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            gci1.append((cl1, cl2, cl3))
            gci0.append((cl3,cl1))
            gci0.append((cl3,cl2))
            
        for axiom in axioms_dict["gci1_bot"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            gci1.append((cl1, cl2, cl3))
            gci0.append((cl3,cl1))
            gci0.append((cl3,cl2))
            
        for axiom in axioms_dict["gci2"]:
            cl1 = classes_dict[axiom.subclass]
            rel = relations_dict[axiom.obj_property]
            cl2 = classes_dict[axiom.filler]
            gci2.append((cl1, rel, cl2))
            
            
        for axiom in axioms_dict["gci3"]:
            rel = relations_dict[axiom.obj_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            gci3.append((rel, cl1, cl2))

        for axiom in axioms_dict["gci3_bot"]:
            rel = relations_dict[axiom.obj_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            gci3.append((rel, cl1, cl2))

        return gci0, gci1, gci2, gci3

    def nfs_to_tensors(self, nfs, device, train = True):
        if train:
            nf1, nf2, nf3, nf4 = nfs
            nf1 = th.LongTensor(nf1).to(device)
            nf2 = th.LongTensor(nf2).to(device)
            nf3 = th.LongTensor(nf3).to(device)
            nf4 = th.LongTensor(nf4).to(device)
        else:
            nf1 = th.empty((1,1)).to(device)
            nf2 = th.empty((1,1)).to(device)
            nf4 = th.empty((1,1)).to(device)
            nf3 = th.LongTensor(nfs).to(device)

        nfs = nf1, nf2, nf3, nf4
        nb_data_points = tuple(map(len, nfs))
        print(f"Number of data points: {nb_data_points}")
        return nfs


    def create_dataloaders(self, device):
        train_nfs = tuple(map(lambda x: x.to(device), self.train_nfs))


        train_ds = map(lambda x: NFDataset(x), train_nfs)
        self.train_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), train_ds))


        



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
class CatModel(nn.Module):

    def __init__(self, num_objects, num_rels, size_hom_set, embedding_size, dropout = 0, depth = 1):
        super(CatModel, self).__init__()

        self.embedding_size = embedding_size
        self.num_obj = num_objects 
        self.size_hom_set = size_hom_set
        self.dropout = dropout
        self.act = ACT
        self.depth = depth

        self.embed = nn.Embedding(self.num_obj, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -1, 1)

        self.embed_rel = nn.Embedding(num_rels, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed_rel.weight, -1, 1)

        
        self.entailment_net = EntailmentHomSet(self.embedding_size, hom_set_size =  self.size_hom_set,  depth = self.depth, dropout = self.dropout)
        self.coprod_net = Coproduct(self.embedding_size, self.entailment_net, dropout = self.dropout)
        self.prod_net = Product(self.embedding_size, self.entailment_net, self.coprod_net, dropout = self.dropout)
        self.ex_net = Existential(self.embedding_size, self.prod_net, dropout = self.dropout)

        self.dummy_param = nn.Parameter(th.empty(0))
        
        
        # Embedding network for the ontology ojects
        self.net_object = nn.Sequential(
            self.embed,
            nn.Linear(embedding_size, embedding_size),
            ACT,

        )

        # Embedding network for the ontology relations
        self.net_rel = nn.Sequential(
            self.embed_rel,
            nn.Linear(embedding_size, embedding_size),
            ACT

        )
        
        
    def gci0_loss(self, data, neg = False):
        device = self.dummy_param.device
        return L.gci0_loss(data, self.entailment_net, self.net_object, neg = neg, num_objects = self.num_obj, device = device)

    def gci1_loss(self, data, neg = False):
        device = self.dummy_param.device
        return L.gci1_loss(data, self.entailment_net, self.prod_net, self.net_object, neg = neg, num_objects = self.num_obj, device = device)

    def gci3_loss(self, data, neg = False):
        device = self.dummy_param.device
        return L.gci3_loss(data, self.entailment_net, self.ex_net, self.net_object, self.net_rel, neg = neg, num_objects = self.num_obj, device = device)

    def gci2_loss(self, data, neg = False):
        device = self.dummy_param.device
        return L.gci2_loss(data, self.entailment_net, self.ex_net, self.net_object, self.net_rel, neg=neg, num_objects = self.num_obj, device = device)

    
    def forward(self, data, idx, neg = False):
        
        loss = 0
        
        if idx == 0:
            loss = self.gci0_loss(data, neg)

        elif idx == 1:
            loss = self.gci1_loss(data, neg)
            
        elif idx == 3:
            loss = self.gci3_loss(data, neg)
            
        elif idx == 2:
            loss = self.gci2_loss(data, neg)
        else:
            raise ValueError("Invalid index")

        return loss

    
class TestModule(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.catModel = model
        

        

    def forward(self, x):
        bs, num_prots, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        x = self.catModel(x, 2)

        x = x.reshape(bs, num_prots)

        return x



class TestDataset(IterableDataset):
    def __init__(self, data, prot_index, prot_dict, r):
        super().__init__()
        self.data = data
        self.prot_dict = prot_dict
        self.len_data = len(data)
        self.predata = np.array([[0, r, x] for x in prot_index])
        

    def get_data(self):
        for c, r, d in self.data:
            c_emb = c.detach().item()
            c = c.detach().item()
            c = self.prot_dict[c]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_emb
            
            tensor = new_array
            yield c, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data
            


class NFDataset(IterableDataset):
    def __init__(self, nf):
        self.nf = nf

    def get_data(self):

        for item in self.nf:
            
            yield item
        

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.nf)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


