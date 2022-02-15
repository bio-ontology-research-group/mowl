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
import mowl.develop.catEmbeddings.losses as L
import os
from mowl.model import Model
from mowl.graph.taxonomy.model import TaxonomyParser
from mowl.graph.edge import Edge
from org.semanticweb.owlapi.util import SimpleShortFormProvider
from scipy.stats import rankdata
from mowl.develop.catEmbeddings.evaluate_interactions import evalNF4Loss
logging.basicConfig(level=logging.DEBUG)


from de.tudresden.inf.lat.jcel.owlapi.main import JcelReasoner
from java.util import HashSet
from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator
from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl


class CatEmbeddings(Model):
    def __init__(self, dataset, batch_size, embedding_size = 1024, file_params = None, seed = 0):
        super().__init__(dataset)
        self.file_params = file_params
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        
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

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.model = None
        ### For eval ppi
        
        self.load_data(device = 'cpu')
        _, _, _, train_nf4 = self.train_nfs
        proteins = {}
        for k, v in self.classes.items():
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
                proteins[k] = v

        self.prot_index = proteins.values()
        self.prot_dict = {v: k for k, v in enumerate(self.prot_index)}

        print("prot dict created")
        self.trlabels = {}
        
        for c,r,d in train_nf4:
            if r != 0:
                continue
            c, r, d = c.detach().item(), r.detach().item(), d.detach().item()

            if c not in self.prot_index or d not in self.prot_index:
                continue
        
            c, d =  self.prot_dict[c], self.prot_dict[d]

            if r not in self.trlabels:
                self.trlabels[r] = np.ones((len(self.prot_index), len(self.prot_index)), dtype=np.int32)
            self.trlabels[r][c, d] = 1000
        print("trlabels created")


        

    def train(self):
        self._loaded = False

        self.load_data(device="cuda")
        device = "cuda"
        num_classes = len(self.classes)
        num_rels = len(self.relations)
        
        self.model = CatModel(num_classes, num_rels, self.embedding_size)
        paramss = sum(p.numel() for p in self.model.parameters())

        logging.info("Number of parameters: %d", paramss)
        logging.debug("Model created")

        lr = 0.1
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr = lr, weight_decay=0)

        best_loss = float("inf")

        nf1, nf2, nf3, nf4 = self.train_nfs
        criterion = nn.BCELoss()
        for epoch in range(128):

            batch = self.batch_size
            self.model.train()
            self.model = self.model.to(device)
            
            rand_index = np.random.choice(len(nf1), size=batch)
            nf1_batch = nf1[rand_index].to(self.device)
            nf1_logits = self.model(nf1_batch, 1, neg=True)
            labels = th.cat([th.ones(batch), th.zeros(batch)], 0).to(self.device)
#            print(labels.shape, nf1_logits.shape)
            nf1_loss = criterion(nf1_logits.squeeze(), labels)      

            rand_index = np.random.choice(len(nf2), size=batch)
            nf2_batch = nf2[rand_index].to(self.device)
            nf2_loss = self.model(nf2_batch, 2)

            rand_index = np.random.choice(len(nf3), size=batch)
            nf3_batch = nf3[rand_index].to(self.device)
            nf3_loss = self.model(nf3_batch, 3)

#            rand_index = np.random.choice(len(nf4), size=batch)
            nf4_batch = nf4[rand_index].to(self.device)
            nf4_loss = self.model(nf4_batch, 4, neg = False)
#            nf4_loss = criterion(nf4_loss, labels)
            
            print(f"nf1: {nf1_loss}")
            print(f"nf2: {nf2_loss}")
            print(f"nf3: {nf3_loss}")
            print(f"nf4: {nf4_loss}")
            train_loss = nf1_loss + nf2_loss + nf3_loss + nf4_loss
            train_loss.backward()
            self.optimizer.step()

            #train_loss = self.forward_step(self.train_dl)

    
            self.model.eval()

            with th.no_grad():
                self.optimizer.zero_grad()
                val_loss  = self.forward_step(self.val_dl, train = False)
                if best_loss > val_loss:
                    best_loss = val_loss
                    th.save(self.model.state_dict(), self.model_filepath)
            top1, top10, top100, mean_rank, ftop1, ftop10, ftop100, fmean_rank = self.evaluate_ppi_valid()
            print(f'Epoch {epoch}: Loss - {train_loss:.6}, \tVal loss - {val_loss:.6}, \tMR - {mean_rank}, \tFMR - {fmean_rank}, \tT100 - {ftop100}')


    def forward_step(self, dataloaders, train=True):
        train_nf1, train_nf2, train_nf3, train_nf4 = dataloaders
        nf1_loss = 0
        nf2_loss = 0
        nf3_loss = 0
        nf4_loss = 0


        with ck.progressbar(train_nf1) as bar:
            i = 0
            for i, batch_nf in enumerate(bar):
                step_loss = self.model(batch_nf, 1)
                if train:
                    self.optimizer.zero_grad()
                    step_loss.backward()
                    self.optimizer.step()

                nf1_loss += step_loss.detach().item()
            nf1_loss /= (i+1)

        with ck.progressbar(train_nf2) as bar:
            i=0
            for i, batch_nf in enumerate(bar):
                step_loss = self.model(batch_nf, 2)
                if train:
                    self.optimizer.zero_grad()
                    step_loss.backward()
                    self.optimizer.step()

                nf2_loss += step_loss.detach().item()
            nf2_loss /= (i+1)

        with ck.progressbar(train_nf3) as bar:
            i=0
            for i, batch_nf in enumerate(bar):
                step_loss = self.model(batch_nf, 3)
                if train:
                    self.optimizer.zero_grad()
                    step_loss.backward()
                    self.optimizer.step()
                            
                nf3_loss += step_loss.detach().item()
            nf3_loss /= (i+1)

        with ck.progressbar(train_nf4) as bar:
            i=0
            for i, batch_nf in enumerate(bar):
                step_loss = self.model(batch_nf, 4)
                if train:
                    self.optimizer.zero_grad()
                    step_loss.backward()
                    self.optimizer.step()

                nf4_loss += step_loss.detach().item()
            nf4_loss /= (i+1)

        print(f"nf1: {nf1_loss}")
        print(f"nf2: {nf2_loss}")
        print(f"nf3: {nf3_loss}")
        print(f"nf4: {nf4_loss}")
        return nf1_loss + nf2_loss + nf3_loss + nf4_loss



    def evaluate(self):
        self.load_data()
        self.model = CatModel(len(self.classes), len(self.relations), 1024).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
        test_loss = self.forward_step(self.test_dl, train=False) #model(self.test_nfs).detach().item()
        print('Test Loss:', test_loss)
 
    def evaluate_ppi_valid(self):
        self.model.eval()
        _, _, _, valid_nf4 = self.valid_nfs
        index = np.random.choice(len(valid_nf4), size = 10)
        valid_nfs = valid_nf4[index]
        mean_rank = evalNF4Loss(self.model, valid_nfs, self.prot_dict, self.prot_index, self.trlabels, len(self.prot_index), device= 'cpu')

        return mean_rank

    def evaluate_ppi(self):
        self.load_data(device = "cuda")
        
        self.model = CatModel(len(self.classes), len(self.relations), self.embedding_size).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
        print(self.device)
        evalNF4Loss(self.model, self.classes, self.relations, self.train_nfs, self.test_nfs, device= self.device)
        
    #########################################
    ### Borrowed code from ELEmbeddings

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

    def load_data(self, device = 'cpu'):
        if self._loaded:
            return
        if device == 'cuda':
            self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        print(f"In device: {self.device}")
        train_nfs, classes, relations = self.load_normal_forms(self.training_filepath)
        valid_nfs, classes, relations = self.load_normal_forms(self.validation_filepath, classes, relations)
        test_nfs, classes, relations = self.load_normal_forms(self.testing_filepath, classes, relations)
        self.classes = classes
        self.class_dict = {v: k for k, v in classes.items()}
        self.relations = relations
        print(relations)
        self.train_nfs = self.nfs_to_tensors(train_nfs, self.device)
        self.valid_nfs = self.nfs_to_tensors(valid_nfs, self.device)
        self.test_nfs = self.nfs_to_tensors(test_nfs, self.device)


        train_ds = map(lambda x: NFDataset(x), self.train_nfs)
        self.train_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), train_ds))

        val_ds = map(lambda x: NFDataset(x), self.valid_nfs)
        self.val_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), val_ds))

        test_ds = map(lambda x: NFDataset(x), self.test_nfs)
        self.test_dl = tuple(map(lambda x: DataLoader(x, batch_size = self.batch_size), test_ds))

        self._loaded = True



class CatModel(nn.Module):

    def __init__(self, num_objects, num_rels, embedding_size):
        super(CatModel, self).__init__()

        self.embedding_size = embedding_size
        self.num_obj = num_objects 

        self.dropout = nn.Dropout(0)

        self.embed = nn.Embedding(self.num_obj, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -0, 1)

        self.embed_rel = nn.Embedding(num_rels, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -0, 1)

        # Embedding network for the ontology ojects
        self.net_object = nn.Sequential(
            self.embed,
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        # Embedding network for the ontology relations
        self.net_rel = nn.Sequential(
            self.embed_rel,
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        # Embedding network for left part of 3rd normal form
        self.embed_ex = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )
        # Embedding network for right part of 2nd normal form
        self.embed_snd = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        # Embedding network for the objects in the exponential diagram
        self.embed_up = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )


        # Morphisms for the exponential diagram
        self.up2exp = self.create_morphism()
        self.up2ant = self.create_morphism()
        self.down2exp = self.create_morphism()
        self.down2ant = self.create_morphism()
        self.up2down = self.create_morphism()
        self.up2cons = self.create_morphism()
        self.down2cons = self.create_morphism()
        self.cons2exp = self.create_morphism()
        self.fc = self.create_morphism()

        self.exponential_morphisms = (self.up2down, self.up2exp, self.down2exp, self.up2ant, self.down2ant, self.up2cons, self.down2cons, self.fc)

        #Product

        # Embedding network for the objects in the exponential diagram
        self.embed_bigger_prod = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )


        #Morphisms for the product
        self.big2left = self.create_morphism()
        self.big2right = self.create_morphism()
        self.prod2left = self.create_morphism()
        self.prod2right = self.create_morphism()
        self.big2prod = self.create_morphism()

        self.product_morphisms = (self.big2prod, self.big2left, self.big2right, self.prod2left, self.prod2right)


    
    def nf4_loss(self, data):
        embed_nets = (self.net_object, self.net_rel, self.embed_snd, self.embed_up, self.embed_bigger_prod)
        return L.nf4_loss(data, self.product_morphisms, self.exponential_morphisms, embed_nets)
    def create_morphism(self):
        fc = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self.dropout
        )
        return fc

        
    def forward(self, normal_form, idx, neg = False):
#        nf1, nf2, nf3, nf4 = normal_forms

        # logging.debug(f"NF1: {len(nf1)}")
        # logging.debug(f"NF2: {len(nf2)}")
        # logging.debug(f"NF3: {len(nf3)}")
        # logging.debug(f"NF4: {len(nf4)}")
        loss = 0
        
        if idx == 1:
            embed_nets = (self.net_object, self.embed_up)
            if neg:
                logits = L.nf1_loss(normal_form, self. exponential_morphisms, embed_nets, neg = True)
                return logits
            else:
                loss += th.mean(L.nf1_loss(normal_form, self. exponential_morphisms, embed_nets, neg = False))
        elif idx == 2:
            embed_nets = (self.net_object, self.embed_up, self.embed_bigger_prod)
            loss += th.mean(L.nf2_loss(normal_form, self.product_morphisms, self.exponential_morphisms, embed_nets))
        elif idx == 3:
            embed_nets = (self.net_object, self.net_rel, self.embed_ex, self.embed_up, self.embed_bigger_prod)
            loss += th.mean(L.nf3_loss(normal_form, self.product_morphisms, self.exponential_morphisms, embed_nets))
        elif idx == 4:
            embed_nets = (self.net_object, self.net_rel, self.embed_ex, self.embed_up, self.embed_bigger_prod)
            if neg == True:
                logits = L.nf4_loss(normal_form, self.exponential_morphisms, embed_nets, neg = True, num_objects = self.num_obj)
                return logits
            else:
                loss += th.mean(L.nf4_loss(normal_form, self.product_morphisms, self.exponential_morphisms, embed_nets))
        else:
            raise ValueError("Invalid index")
        
        return loss

    
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


