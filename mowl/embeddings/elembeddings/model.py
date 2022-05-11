import os

from mowl.model import EmbeddingModel
from mowl.reasoning.normalize import ELNormalizer

from de.tudresden.inf.lat.jcel.owlapi.main import JcelReasoner
from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator
from org.semanticweb.owlapi.model import OWLAxiom
from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl
from java.util import HashSet
from org.semanticweb.owlapi.util import SimpleShortFormProvider

import pandas as pd
import numpy as np
import re
import math
import logging
from scipy.stats import rankdata
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
from tqdm import trange, tqdm

import torch as th
from torch import nn

class ELEmbeddings(EmbeddingModel):


    def __init__(self, dataset, embed_dim=50, margin=0, reg_norm=1, learning_rate=0.001, epochs=1000, device = 'cpu'):
        super().__init__(dataset)


        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.model_filepath = "data/model.th"
        self.short_form_provider = SimpleShortFormProvider()
        self._loaded = False
        self._loaded_eval = False

        self.load_eval_data()
    def get_entities_index_dict(self):
        return self.classes_index_dict, self.relations



    def load_eval_data(self):
        if self._loaded_eval:
            return
        eval_projector = projector_factory('taxonomy_rels', taxonomy=False, relations=["http://interacts_with"])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        training_entities,_ = Edge.getEntitiesAndRelations(self._training_set)
        testing_entities,_ = Edge.getEntitiesAndRelations(self._testing_set)

        entities = list(set(training_entities) | set(testing_entities))
        
        self._head_entities = entities
        self._tail_entities = entities
        self._loaded_eval = True
        

    
    def load_data(self):
        if self._loaded:
            return
        
        normalizer = ELNormalizer()
        self.training_axioms = normalizer.normalize(self.dataset.ontology)
        self.validation_axioms = normalizer.normalize(self.dataset.validation)
        self.testing_axioms = normalizer.normalize(self.dataset.testing)

        classes = set()
        relations = set()

        for axioms_dict in [self.training_axioms, self.validation_axioms, self.testing_axioms]:
            for axiom in axioms_dict["gci0"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci1"]:
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

        self.classes_index_dict = {v: k  for k, v in enumerate(list(classes))}
        self.relations = {v: k for k, v in enumerate(list(relations))}

        training_nfs = self.load_normal_forms(self.training_axioms, self.classes_index_dict, self.relations)
        validation_nfs = self.load_normal_forms(self.validation_axioms, self.classes_index_dict, self.relations)
        testing_nfs = self.load_normal_forms(self.testing_axioms, self.classes_index_dict, self.relations)
        
        self.train_nfs = self.nfs_to_tensors(training_nfs, self.device)
        self.valid_nfs = self.nfs_to_tensors(validation_nfs, self.device)
        self.test_nfs = self.nfs_to_tensors(testing_nfs, self.device)
        self._loaded = True
        
    def load_normal_forms(self, axioms_dict, classes_dict, relations_dict):
        nf0 = []
        nf1 = []
        nf2 = []
        nf3 = []

        for axiom in axioms_dict["gci0"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            nf0.append((cl1, cl2))

        for axiom in axioms_dict["gci1"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            nf1.append((cl1, cl2, cl3))
            
        for axiom in axioms_dict["gci2"]:
            cl1 = classes_dict[axiom.subclass]
            rel = relations_dict[axiom.obj_property]
            cl2 = classes_dict[axiom.filler]
            nf2.append((cl1, rel, cl2))
        
        for axiom in axioms_dict["gci3"]:
            rel = relations_dict[axiom.obj_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            nf3.append((rel, cl1, cl2))

        return nf0, nf1, nf2, nf3
        
    def nfs_to_tensors(self, nfs, device):
        nf1, nf2, nf3, nf4 = nfs
        nf1 = th.LongTensor(nf1).to(device)
        nf2 = th.LongTensor(nf2).to(device)
        nf3 = th.LongTensor(nf3).to(device)
        nf4 = th.LongTensor(nf4).to(device)
        nfs = nf1, nf2, nf3, nf4
        return nfs


    def train(self):
        self.load_data()
        self.model = ELModel(len(self.classes_index_dict), len(self.relations), self.device).to(self.device)
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        for epoch in trange(self.epochs):
            self.model.train()
            loss = self.model(self.train_nfs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            valid_loss = self.model(self.valid_nfs).cpu().detach().item()
            if best_loss > valid_loss:
                best_loss = valid_loss
#                print('Saving the model')
                th.save(self.model.state_dict(), self.model_filepath)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Train loss: {loss.cpu().detach().item()} Valid loss: {valid_loss}')

    def eval_method(self, data):
        self.load_data()
        scores = self.model.gci2_loss(data)
        return scores
        
        
            
    def evaluate(self):
        self.load_data()
        self.model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
        test_loss = self.model(self.test_nfs).cpu().detach().item()
        print('Test Loss:', test_loss)
        
    def get_embeddings(self):
        self.load_data()
        self.model = ELModel(len(self.classes_index_dict), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k:v for k,v in zip(self.classes_index_dict.keys(), self.model.go_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k:v for k,v in zip(self.relations.keys(), self.model.rel_embed.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds
 

    def evaluate_ppi(self):
        self.load_data()
        model = ELModel(len(self.classes_index_dict), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        
        proteins = [str(c)[1:-1] for c in self.dataset.get_evaluation_classes() if "4932." in str(c)]
#        proteins = [str(self.short_form_provider.getShortForm(cls)) for cls in proteins]
        
        proteins = [self.classes_index_dict[p_id] for p_id in proteins if p_id in self.classes_index_dict]
        print(len(proteins))
        prot_dict = {v:k for k, v in enumerate(proteins)}
        #print(prot_dict)
        train_ppis = {}
        _, _, nf3, _ = self.train_nfs
        print(self.relations)
        for c, r, d in nf3:
            c, r, d = c.item(), r.item(), d.item()
            if r != self.relations["http://interacts"]:
                continue
            if c not in train_ppis:
                train_ppis[c] = []
            train_ppis[c].append(d)
        _, _, nf3, _ = self.valid_nfs
        for c, r, d in nf3:
            c, r, d = c.item(), r.item(), d.item()
            if r != self.relations["http://interacts"]:
                continue
            if c not in train_ppis:
                train_ppis[c] = []
            train_ppis[c].append(d)

        
        proteins = th.LongTensor(proteins)
        nf1, nf2, nf3, nf4 = self.test_nfs
        mean_rank = 0
        n_nf3 = len(nf3)
        for it in tqdm(nf3):
            c, r, d = it
            c, r, d = c.item(), r.item(), d.item()
            #print(r)
            #            if c not in train_ppis or d not in train_ppis:
            
            #                continue
            if c not in prot_dict:
                print(c, "not in prot dict")
                n_nf3 -= 1
                continue

            if d not in prot_dict:
                print(d, "not in prot dict")
                n_nf3-=1
                continue

            n = len(proteins)
            data = th.zeros(n, 3, dtype=th.long).to(self.device)
            data[:, 0] = c
            data[:, 1] = r
            data[:, 2] = proteins
            scores = model.gci2_loss(data).cpu().detach().cpu().numpy()
            scores = scores.flatten()
            for td in train_ppis[c]:
                if td in prot_dict:
                    scores[prot_dict[td]] = 1000.0
            index = rankdata(scores, method='average')
            rank = index[prot_dict[d]]
            mean_rank += rank

        print("total",n_nf3)
        mean_rank /= n_nf3
        print(mean_rank)
        
            
    
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
        nn.init.uniform_(self.go_embed.weight, -0, 1)
        self.go_rad = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -0, 1)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin

    def forward(self, go_normal_forms):
        nf0, nf1, nf2, nf3 = go_normal_forms
        loss = 0
        if len(nf0) > 1:
            loss += th.mean(self.gci0_loss(nf0))
        if len(nf1) > 1:
            loss += th.mean(self.gci1_loss(nf1))
        if len(nf2) > 1:
            loss += th.mean(self.gci2_loss(nf2))
            loss += th.mean(self.gci2_loss_neg(nf2))
        if len(nf3) > 1:
            loss += th.mean(self.gci3_loss(nf3))
        return loss

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
    def gci0_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.relu(pos_dist - self.margin)
        return loss

    def gci1_loss(self, data):
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

    def gci3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        
        rSomeC = c + rE
        euc = th.linalg.norm(c - rE  - d, dim=1, keepdim=True)
        loss = th.relu(euc - rc - rd - self.margin)
        return loss


    def gci2_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        # c should intersect with d + r
        
        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = th.relu(dst + rc - rd  - self.margin)
        return loss


    def gci2_loss_neg(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        # c should intersect with d + r
        
        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = th.relu(rc + rd - dst  + self.margin)
        return loss
