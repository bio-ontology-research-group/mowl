import os

from mowl.model import Model

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

import torch as th
from torch import nn

class ELEmbeddings(Model):


    def __init__(self, dataset):
        super().__init__(dataset)

        self.training_filepath = os.path.join(
            self.dataset.data_root, self.dataset.dataset_name, 'ontology.nf')
        self.validation_filepath = os.path.join(
            self.dataset.data_root, self.dataset.dataset_name, 'valid.nf')
        self.testing_filepath = os.path.join(
            self.dataset.data_root, self.dataset.dataset_name, 'test.nf')
        self.model_filepath = os.path.join(
            self.dataset.data_root, self.dataset.dataset_name, 'elmodel.th')
        
        self.create_normal_forms(self.dataset.ontology, self.training_filepath)
        self.create_normal_forms(self.dataset.validation, self.validation_filepath)
        self.create_normal_forms(self.dataset.testing, self.testing_filepath)
        self._loaded = False
        self.short_form_provider = SimpleShortFormProvider()
        
    def create_normal_forms(self, ontology, normal_forms_filepath):
        if os.path.exists(normal_forms_filepath):
            return
        jReasoner = JcelReasoner(ontology, False)
        rootOnt = jReasoner.getRootOntology()
        translator = jReasoner.getTranslator()
        axioms = HashSet()
        axioms.addAll(rootOnt.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(
            rootOnt)
        for ont in rootOnt.getImportsClosure():
            axioms.addAll(ont.getAxioms())
            translator.getTranslationRepository().addAxiomEntities(
                ont)
        
        intAxioms = translator.translateSA(axioms)
        
        normalizer = OntologyNormalizer()
        factory = IntegerOntologyObjectFactoryImpl()
        normalizedOntology = normalizer.normalize(intAxioms, factory)
        rTranslator = ReverseAxiomTranslator(translator, self.dataset.ontology)
        renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
        with open(normal_forms_filepath, 'w') as f:
            for ax in normalizedOntology:
                try:
                    axiom = renderer.render(rTranslator.visit(ax))
                    f.write(f'{axiom}\n')
                except Exception as e:
                    print(f'Ignoring {ax}', e)

    def load_normal_forms(self, filepath, classes={}, relations={}):
        nf1 = []
        nf2 = []
        nf3 = []
        nf4 = []
        print(filepath)
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.find('SubClassOf') == -1:
                    continue
                left, right = line.split(' SubClassOf ')
                # C SubClassOf D
                if len(left) == 10 and len(right) == 10:
                    go1, go2 = left, right
                    if go1 not in classes:
                        classes[go1] = len(classes)
                    if go2 not in classes:
                        classes[go2] = len(classes)
                    g1, g2 = classes[go1], classes[go2]
                    nf1.append((g1, g2))
                elif left.find('and') != -1: # C and D SubClassOf E
                    go1, go2 = left.split(' and ')
                    go3 = right
                    if go1 not in classes:
                        classes[go1] = len(classes)
                    if go2 not in classes:
                        classes[go2] = len(classes)
                    if go3 not in classes:
                        classes[go3] = len(classes)
                    
                    nf2.append((classes[go1], classes[go2], classes[go3]))
                elif left.find('some') != -1:  # R some C SubClassOf D
                    rel, go1 = left.split(' some ')
                    go2 = right
                    if go1 not in classes:
                        classes[go1] = len(classes)
                    if go2 not in classes:
                        classes[go2] = len(classes)
                    if rel not in relations:
                        relations[rel] = len(relations)
                    nf3.append((relations[rel], classes[go1], classes[go2]))
                elif right.find('some') != -1: # C SubClassOf R some D
                    go1 = left
                    rel, go2 = right.split(' some ')
                    if go1 not in classes:
                        classes[go1] = len(classes)
                    if go2 not in classes:
                        classes[go2] = len(classes)
                    
                    if rel not in relations:
                        relations[rel] = len(relations)
                    nf4.append((classes[go1], relations[rel], classes[go2]))
        normal_forms = nf1, nf2, nf3, nf4
        return normal_forms, classes, relations

    def nfs_to_tensors(self, nfs, device):
        nf1, nf2, nf3, nf4 = nfs
        nf1 = th.LongTensor(nf1).to(device)
        nf2 = th.LongTensor(nf2).to(device)
        nf3 = th.LongTensor(nf3).to(device)
        nf4 = th.LongTensor(nf4).to(device)
        nfs = nf1, nf2, nf3, nf4
        return nfs

    def load_data(self):
        if self._loaded:
            return
        self.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
        train_nfs, classes, relations = self.load_normal_forms(self.training_filepath)
        valid_nfs, classes, relations = self.load_normal_forms(self.validation_filepath, classes, relations)
        test_nfs, classes, relations = self.load_normal_forms(self.testing_filepath, classes, relations)
        self.classes = classes
        self.class_dict = {v: k for k, v in classes.items()}
        self.relations = relations
        self.train_nfs = self.nfs_to_tensors(train_nfs, self.device)
        self.valid_nfs = self.nfs_to_tensors(valid_nfs, self.device)
        self.test_nfs = self.nfs_to_tensors(test_nfs, self.device)
        self._loaded = True

    def train(self, embed_dim=50, margin=0, reg_norm=1, learning_rate=0.001, epochs=1000):
        self.load_data()
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            loss = model(self.train_nfs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            valid_loss = model(self.valid_nfs).detach().item()
            if best_loss > valid_loss:
                best_loss = valid_loss
                print('Saving the model')
                th.save(model.state_dict(), self.model_filepath)
                
            print(f'Epoch {epoch}: Train loss: {loss.detach().item()} Valid loss: {valid_loss}')

    def evaluate(self):
        self.load_data()
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        test_loss = model(self.test_nfs).detach().item()
        print('Test Loss:', test_loss)

    def evaluate_ppi(self):
        self.load_data()
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        
        proteins = self.dataset.get_evaluation_classes()
        proteins = [str(self.short_form_provider.getShortForm(cls)) for cls in proteins]
        proteins = [self.classes[p_id] for p_id in proteins]
        prot_dict = {v:k for k, v in enumerate(proteins)}
        train_ppis = {}
        _, _, _, nf4 = self.train_nfs
        for c, r, d in nf4:
            c, r, d = c.item(), r.item(), d.item()
            if r != 0:
                continue
            if c not in train_ppis:
                train_ppis[c] = []
            train_ppis[c].append(d)
        _, _, _, nf4 = self.valid_nfs
        for c, r, d in nf4:
            c, r, d = c.item(), r.item(), d.item()
            if r != 0:
                continue
            if c not in train_ppis:
                train_ppis[c] = []
            train_ppis[c].append(d)
        
        proteins = th.LongTensor(proteins)
        nf1, nf2, nf3, nf4 = self.test_nfs
        mean_rank = 0
        for it in nf4:
            c, r, d = it
            c, r, d = c.item(), r.item(), d.item()
            if c not in prot_dict or d not in prot_dict:
                continue
            n = len(proteins)
            data = th.zeros(n, 3, dtype=th.long)
            data[:, 0] = c
            data[:, 1] = r
            data[:, 2] = proteins
            scores = model.nf4_loss(data).detach().cpu().numpy()
            scores = scores.flatten()
            for td in train_ppis[c]:
                if td in prot_dict:
                    scores[prot_dict[td]] = 1000.0
            index = rankdata(scores, method='average')
            rank = index[prot_dict[d]]
            mean_rank += rank
        mean_rank /= len(nf4)
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
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin

    def forward(self, go_normal_forms):
        nf1, nf2, nf3, nf4 = go_normal_forms
        loss = 0
        if len(nf1) > 0:
            loss += th.mean(self.nf1_loss(nf1))
        if len(nf2) > 0:
            loss += th.mean(self.nf2_loss(nf2))
        if len(nf3) > 0:
            loss += th.mean(self.nf3_loss(nf3))
        if len(nf4) > 0:
            loss += th.mean(self.nf4_loss(nf4))
        return loss

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
