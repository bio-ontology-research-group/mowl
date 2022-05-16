import os
import pickle as pkl
from mowl.model import Model
import click as ck
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_curve, auc
import random
import pandas as pd
import numpy as np
import re
import math
import logging
from scipy.stats import rankdata

import torch as th
from torch import nn

class ELEmbeddings():


    def __init__(self, data_root, batch_size, lr = 0.001, embedding_size=50):

        self.data_root = data_root
        self.batch_size = batch_size
        self.lr = lr
        self.embedding_size = embedding_size
        self._loaded = False
        self.device = "cuda"
        
        self.model_filepath = "data/models/elem/model.th"

    def train(self, margin=0, reg_norm=1, epochs=1000):
        train_data_loader, test_data_loader, self.classes = self.load_data()
        self.relations = []
        model = ELModel(len(self.classes), len(self.relations), self.device, embed_dim = self.embedding_size).to(self.device)
        optimizer = th.optim.Adam(model.parameters(), lr=self.lr)
        best_loss = float('inf')
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            with ck.progressbar(train_data_loader) as bar:
                for i, (samples, lbls) in enumerate(bar):
                    samples = list(map(lambda x: x.to(self.device), samples))
                    logits = model(samples)
                    lbls = lbls.float().to(self.device)

                    loss = criterion(logits.squeeze(), lbls)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            train_loss /= (i+1)
            model.eval()

            preds = []
            labels = []

            valid_loss = 0

            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(test_data_loader) as bar:
                    for i, (samples, lbls) in enumerate(bar):
                        samples = list(map(lambda x: x.to(self.device), samples))
                        logits = model(samples)
                        lbls = lbls.float().to(self.device)

                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                        loss = criterion(logits.squeeze(), lbls)
                        valid_loss += loss.detach().item()
                    valid_loss /= (i+1)

            roc_auc = compute_roc(labels, preds)



            valid_loss /= (i+1)
            if best_loss > valid_loss:
                best_loss = valid_loss
                print('Saving the model')
                th.save(model.state_dict(), self.model_filepath)
                
            print(f'Epoch {epoch}: Train loss: {loss.detach().item()} Valid loss: {valid_loss}, \tAUC - {roc_auc:.6}')

    def evaluate(self):
        self.load_data()
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        test_loss = model(self.test_nfs).detach().item()
        print('Test Loss:', test_loss)
        
    def get_embeddings(self):
        self.load_data()
        self.load_data()
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        return self.classes, model.go_embed.weight.detach().numpy()



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

        return train_dl, test_dl, objects

    
class ELModel(nn.Module):

    def __init__(self, nb_gos, nb_rels, device, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.nb_rels = nb_rels
        # ELEmbeddings
        self.embed_dim = embed_dim
        self.go_embed = nn.Embedding(nb_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin




    def forward(self, samples):
        samples = th.vstack(samples).transpose(0,1)

        loss = (self.nf1_loss(samples))

        logits = 1 - 2*(th.sigmoid(loss) - 0.5)

        return logits

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
    def nf1_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.relu(pos_dist - self.margin)
        return loss

    def nf2_loss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))
        
        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = (th.relu(dst - sr - self.margin)
                    + th.relu(dst2 - rc - self.margin)
                    + th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        
        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.relu(euc + rc - rd - self.margin)
        return loss


    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.relu(dst - sr - self.margin)
        return loss



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





def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc
