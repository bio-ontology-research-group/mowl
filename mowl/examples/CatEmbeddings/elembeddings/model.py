import os

from mowl.model import Model

import pandas as pd
import numpy as np
import re
import math
import logging
from scipy.stats import rankdata

import torch as th
from torch import nn
th.manual_seed(0)
np.random.seed(0)

class ELEmbeddings(Model):


    def __init__(self, dataset):
        super().__init__(dataset)

        self.training_filepath, self.valid_filepath, self.test_filepath = dataset

        self.model_filepath =  'data/models/elem/elmodelfull.th'
        self._loaded = False
#        self.create_normal_forms(self.dataset.ontology, self.training_filepath)
#        self.create_normal_forms(self.dataset.validation, self.validation_filepath)
#        self.create_normal_forms(self.dataset.testing, self.testing_filepath)
#        self._loaded = False
 
        
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
        rTranslator = ReverseAxiomTranslator(translator, ontology)
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

    # def nfs_to_tensors(self, nfs, device):
    #     nf1, nf2, nf3, nf4 = nfs
    #     nf1 = th.LongTensor(nf1).to(device)
    #     nf2 = th.LongTensor(nf2).to(device)
    #     nf3 = th.LongTensor(nf3).to(device)
    #     nf4 = th.LongTensor(nf4).to(device)
    #     nfs = nf1, nf2, nf3, nf4
    #     return nfs

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
            nf3 = th.empty((1,1)).to(device)
            nf4 = th.LongTensor(nfs).to(device)

        nfs = nf1, nf2, nf3, nf4
        nb_data_points = tuple(map(len, nfs))
        print(f"Number of data points: {nb_data_points}")
        return nfs


    def load_data_old(self, train_file, valid_file, test_file, device = 'cpu'):
        if self._loaded:
            return

        if device == 'cuda':
            self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        print(f"In device: {self.device}")

        import sys
        sys.path.append('../../../')
        from mowl.develop.catEmbeddings.load_data import load_data, load_valid_data
        
        nfs, classes, relations = load_data(train_file)
        self.classes = classes
        self.class_dict = {v: k for k, v in classes.items()}
        self.relations = relations
        print(relations)
        
        print(type(nfs['nf1']))
        train_nfs = nfs['nf1'], nfs['nf2'], nfs['nf4'], nfs['nf3']
        self.train_nfs = self.nfs_to_tensors(train_nfs, self.device)

        valid_nfs = load_valid_data(valid_file, classes, relations)
        test_nfs = load_valid_data(test_file, classes, relations)
        print(f"Valid data points: {len(valid_nfs)}. Test data points: {len(test_nfs)}")

        self.valid_nfs = self.nfs_to_tensors(valid_nfs, self.device, train = False)
        self.test_nfs = self.nfs_to_tensors(test_nfs, self.device, train = False)
        self._loaded = True

    def create_dataloaders(self, device):
        train_nfs = tuple(map(lambda x: x.to(device), self.train_nfs))
        valid_nfs = tuple(map(lambda x: x.to(device), self.valid_nfs))
        test_nfs = tuple(map(lambda x: x.to(device), self.test_nfs))

        train_ds = map(lambda x: NFDataset(x), train_nfs)
        self.train_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), train_ds))

        val_ds = map(lambda x: NFDataset(x), valid_nfs)
        self.val_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), val_ds))

        test_ds = map(lambda x: NFDataset(x), test_nfs)
        self.test_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), test_ds))
        


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

    def train(self, embed_dim=50, margin=-0.1, reg_norm=1, learning_rate=0.001, epochs=6000):
        self.load_data_old(self.training_filepath, self.valid_filepath, self.test_filepath, device = 'cuda')

        self.model = ELModel(len(self.classes), len(self.relations), self.device, embed_dim = embed_dim, margin=margin).to(self.device)
        optimizer = th.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay = 0.000002)
        best_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            loss = self.model(self.train_nfs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            valid_loss = self.model(self.valid_nfs).detach().item()
            if best_loss > valid_loss:
                best_loss = valid_loss
                print('Saving the model')
                th.save(self.model.state_dict(), self.model_filepath)
                
            print(f'Epoch {epoch}: Train loss: {loss.detach().item()} Valid loss: {valid_loss}')

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

    def evaluate_ppi(self):
        
        
        model = ELModel(len(self.classes), len(self.relations), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        model.load_state_dict(th.load(self.model_filepath))
        model.eval()
        prots = {}
        prot_names = []
        for k, v in self.classes.items():
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_') and not k.startswith("GO:"):
                prots[k] = v
                prot_names.append(k)
        
        self.prot_index = prots.values()



        proteins = [self.classes[p_id] for p_id in prots]
        prot_dict = {v: k for k, v in enumerate(proteins)}
        # proteins = self.dataset.get_evaluation_classes()
        # proteins = [str(self.short_form_provider.getShortForm(cls)) for cls in proteins]
        # proteins = [self.classes[p_id] for p_id in proteins]
        # prot_dict = {v:k for k, v in enumerate(proteins)}



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
        nf1, nf2, _, nf4 = self.test_nfs
        
        top1 = 0
        top10 = 0
        top100 = 0
        top1000 = 0
        mean_rank = 0
        ftop1 = 0
        ftop10 = 0
        ftop100 = 0
        fmean_rank = 0

        ranks = {}
        franks = {}

        top10 = 0
        top100 = 0
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
            scores = model.nf4_loss(data.to("cuda")).detach().cpu().numpy()
            scores = scores.flatten()
        
            index = rankdata(scores, method='average')
            rank = index[prot_dict[d]]
            
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            if rank <= 1000:
                top1000 += 1
                
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            for td in train_ppis[c]:
                if td in prot_dict:
                    scores[prot_dict[td]] = 1000.0
            index = rankdata(scores, method='average')
            rank = index[prot_dict[d]]

            
            if rank == 1:
                ftop1 += 1
            if rank <= 10:
                ftop10 += 1
            if rank <= 100:
                ftop100 += 1
            fmean_rank += rank


            if rank not in franks:
                franks[rank] = 0
            franks[rank] += 1


        n = len(nf4)

        top1 /= n
        top10 /= n
        top100 /= n
        top1000 /= n
        mean_rank /= n
        ftop1 /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n

        rank_auc = compute_rank_roc(ranks, len(prots))
        frank_auc = compute_rank_roc(franks, len(prots))

        print(f'{top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
        print(f'{ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')


        
            
    
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
        if len(nf1) > 1:
            loss += th.mean(self.nf1_loss(nf1))
        if len(nf2) > 1:
            loss += th.mean(self.nf2_loss(nf2))
        if len(nf3) > 1:
            loss += th.mean(self.nf3_loss(nf3))
        if len(nf4) > 1:
            loss += th.mean(self.nf4_loss(nf4))
            loss += th.mean(self.nf4_neg_loss(nf4))
        return loss

    def class_dist(self, data):

        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist + self.reg(c) + self.reg(d)
        

    def reg (self, x):
        x = th.linalg.norm(x, dim=1)
        x = th.abs(x-1)
        return x

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

        return loss + self.reg(c) + self.reg(d)

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

        return loss + self.reg(c) + self.reg(d)


    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc - rd
        # c should intersect with d + r
        rSomeD = d - rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.relu(dst + sr - self.margin)

        return loss + self.reg(c).unsqueeze(dim =1) + self.reg(d).unsqueeze(dim=1)


    def nf4_neg_loss(self, data):
        # C subClassOf R some D

        n = data.shape[0]
        negs = np.random.choice(self.nb_gos, n, replace = True)
        negs = th.tensor(negs).to(data.device)
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(negs))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = rE - d
        dst = th.linalg.norm(c + rSomeD, dim=1, keepdim=True)
        loss = th.relu(-dst + sr + self.margin)
        return loss



def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc
