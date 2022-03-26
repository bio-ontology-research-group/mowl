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
            eval_ppi = False,
            sampling = False,
            nf1 = False, 
            nf1_neg = False, 
            nf2 = False,
            nf2_neg = False,
            nf3 = False,
            nf3_neg = False,
            nf4 = False,
            nf4_neg = False,
            margin = 0,
            seed = -1,
            early_stopping = 10,
            device = "cuda"
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
        self.eval_ppi = eval_ppi
        self.sampling = sampling
        self.nf1 = nf1
        self.nf1_neg = nf1_neg
        self.nf2 = nf2
        self.nf2_neg = nf2_neg
        self.nf3 = nf3
        self.nf3_neg = nf3_neg
        self.nf4 = nf4
        self.nf4_neg = nf4_neg
        self.margin = margin
        self.early_stopping = early_stopping
        self.device = device
        train_file, valid_file, test_file = dataset
        
        if "yeast" in train_file:
            species = "yeast"
        elif "human" in train_file:
            species = "human"
        else:
            raise ValueError("Species not defined")
            
        self.species = species
        milestones_str = "_".join(str(m) for m in milestones)
        self.data_root = f"data/models/{species}/"
        self.file_name = f"bs{self.batch_size}_emb{self.embedding_size}_lr{lr}_epochs{epochs}_eval{num_points_eval}_mlstns_{milestones_str}_drop_{self.dropout}_decay_{self.decay}_gamma_{self.gamma}_evalppi_{self.eval_ppi}_sampling_{sampling}_nf1{self.nf1}{self.nf1_neg}_nf2{self.nf2}{self.nf2_neg}_nf3{self.nf3}{self.nf3_neg}_nf4{self.nf4}{self.nf4_neg}_margin{self.margin}.th"
        self.model_filepath = self.data_root + self.file_name
        self.predictions_file = f"data/predictions/{self.species}/" + self.file_name
        self.labels_file = f"data/labels/{self.species}/" + self.file_name
        print(f"model will be saved in {self.model_filepath}")

        self._loaded = False

        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.model = None
        ### For eval ppi
        self.load_data_old(train_file, valid_file, test_file, device = 'cuda')

        if self.eval_ppi or True:
            _, _, _, train_nf4 = self.train_nfs
            proteins = {}
            for k, v in self.classes.items():
                if not k.startswith('<http://purl.obolibrary.org/obo/GO_') and not k.startswith("GO:"):
                    proteins[k] = v
            self.prot_index = proteins.values()
            self.prot_dict = {v: k for k, v in enumerate(self.prot_index)}

            print(f"prot dict created. Number of proteins: {len(self.prot_index)}")
            self.trlabels = {}


            print("Generating training labels")
            with ck.progressbar(train_nf4) as bar:
                for c,r,d in bar:
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


 #       self._loaded = False
        device = "cuda"
#        self.load_data_old(device="cuda")
        
        if not self.sampling:
            self.create_dataloaders(device = device)

        self.train_nfs = tuple(map(lambda x: x.to(device), self.train_nfs))
        
        self.num_classes = len(self.classes)
        self.num_rels = len(self.relations)
        
        self.model = CatModel(self.num_classes, self.num_rels, self.embedding_size, dropout = self.dropout)
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
        valid_early_stopping = stop_value

        if self.sampling:
            forward_function = self.forward_step_sampling
            train_data = self.train_nfs
            valid_data = self.valid_nfs
            test_data = self.test_nfs
        else:
            forward_function = self.forward_step
            train_data = self.train_dl
            valid_data = self.val_dl
            test_data = self.test_dl

        for epoch in range(self.epochs):

            batch = self.batch_size
            self.model.train()
            self.model = self.model.to(device)
            
            train_loss = forward_function(
                train_data, 
                self.nf1,
                self.nf1_neg,
                self.nf2, 
                self.nf2_neg,
                self.nf3, 
                self.nf3_neg,
                self.nf4,
                self.nf4_neg,
                self.margin,
                train = True)
    
            self.model.eval()
            

            with th.no_grad():
                self.optimizer.zero_grad()
                val_loss  = forward_function(
                    valid_data,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    self.nf4,
                    False,
                    self.margin,
                    train = False
                    )


            if best_train_loss < train_loss:
                train_early_stopping -= 1
            else:
                best_train_loss = train_loss
                train_early_stopping = stop_value

            if best_val_loss < val_loss:
                valid_early_stopping -= 1
            else:
                best_val_loss = val_loss
                valid_early_stopping = stop_value
                if not self.eval_ppi:
                    th.save(self.model.state_dict(), self.model_filepath)

            if self.eval_ppi:
                top1, top10, top100, top1000, mean_rank, ftop1, ftop10, ftop100, fmean_rank = self.evaluate_ppi_valid()

                if best_mean_rank >= mean_rank:
                    best_mean_rank = mean_rank
                    th.save(self.model.state_dict(), self.model_filepath)
            print(f'Epoch {epoch}: Loss - {train_loss:.6}, \tVal loss - {val_loss:.6}')
            if self.eval_ppi:
                print(f' MR - {mean_rank}, \tT10 - {top10},\tT100 - {top100}, \tT1000 - {top1000}')
                print(f'FMR - {fmean_rank}, \tT10 - {ftop10},\tT100 - {ftop100}')
            print("\n\n")
            self.scheduler.step()

            if train_early_stopping == 0:
                print(f"Stop training (early stopping): {train_early_stopping}: {best_train_loss}, {valid_early_stopping}, {best_val_loss}")
                break
        
        logging.info("Finished training. Generating predictions")
#        self.run_and_save_predictions(self.model)


    def run_and_save_predictions(self, model, samples=None, save = True):
        device = "cuda"

        model.eval()
        print(self.device)

        if samples is None:
            logging.info("No data points specified. Proceeding to compute predictions on test set")
            model.load_state_dict(th.load( self.model_filepath))
            model = model.to(device)
            _, _, _, test_nf4 = self.test_nfs

            eval_data = test_nf4

        else:
            eval_data = samples

        test_model = TestModule(model)

        preds = np.zeros((len(self.prot_index), len(self.prot_index)), dtype=np.float32)
        labels = np.zeros((len(self.prot_dict), len(self.prot_dict)), dtype=np.int32)
        already = set()


 #       logging.info("Filling up labels")
        with ck.progressbar(eval_data) as prog_data:
            for c, r, d in prog_data:
                c_emb = c
                d_emb = d
                c, d = c.detach().item(), d.detach().item()
                c, d = self.prot_dict[c], self.prot_dict[d]

                labels[c, d] = 1
            

        test_dataset = TestDataset(eval_data, self.prot_index, self.prot_dict, 0)
        if self.species == "yeast":
            bs = 16
        else:
            bs = 8
        test_dl = DataLoader(test_dataset, batch_size = bs)

        with ck.progressbar(test_dl) as prog_data:
            for idxs, batch in prog_data:

                res = test_model(batch.to(device))
                res = res.cpu().detach().numpy()
                preds[idxs,:] = res

        

        if save:
            with open(self.predictions_file, "wb") as f:
                pkl.dump(preds, f)

            with open(self.labels_file, "wb") as f:
                pkl.dump(labels, f)

        return preds, labels

        

    def forward_nf(self, nf, idx, margin, pos, neg, train = True):
        nf_pos_loss = 0
        nf_neg_loss = 0
        nf_diff_loss = 0
        if pos:
            i = 0
            for i, batch_nf in enumerate(nf):
                pos_loss = self.model(batch_nf, idx)
                step_loss = th.mean(pos_loss)
                if neg:
                    neg_loss = self.model(batch_nf, idx, neg = True)
                    assert pos_loss.shape == neg_loss.shape, f"{pos_loss.shape}, {neg_loss.shape}"
                    new_margin = margin
 #                   new_margin = margin if th.mean(neg_loss - pos_loss) > margin else 2*th.mean(pos_loss)

                    diff_loss = th.mean(th.relu(pos_loss - neg_loss + new_margin))
                    step_loss += diff_loss
                    nf_neg_loss += th.mean(neg_loss).detach().item()
                    nf_diff_loss += diff_loss.detach().item()

                if train:
                    self.optimizer.zero_grad()
                    step_loss.backward()
                    self.optimizer.step()

                nf_pos_loss += th.mean(pos_loss).detach().item()


            nf_pos_loss /= (i+1)
            nf_neg_loss /= (i+1)
            nf_diff_loss /= (i+1)

        return nf_pos_loss, nf_neg_loss, nf_diff_loss

    def forward_nf_sampling(self, nf, idx, margin, pos, neg, nb_data_points, train = True):
        nf_pos_loss = 0
        nf_neg_loss = 0
        nf_diff_loss = 0
        if pos:
            pos_loss = self.model(nf, idx)
            step_loss = th.mean(pos_loss)
            if neg:
                neg_loss = self.model(nf, idx, neg = True)
                assert pos_loss.shape == neg_loss.shape, f"{pos_loss.shape}, {neg_loss.shape}"
                new_margin = margin
#                new_margin = margin if th.mean(neg_loss - pos_loss) > margin else 2*th.mean(pos_loss)
                diff_loss = th.mean(th.relu(pos_loss - neg_loss + new_margin))
                
                step_loss += diff_loss

                nf_neg_loss += th.mean(neg_loss).detach().item()
                nf_diff_loss += diff_loss.detach().item()

            print(f"step loss {step_loss}   {step_loss*len(nf)/nb_data_points}")

            step_loss = step_loss*len(nf)/nb_data_points
            if train:
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()

            nf_pos_loss += th.mean(pos_loss).detach().item()

        return nf_pos_loss*len(nf)/nb_data_points, nf_neg_loss*len(nf)/nb_data_points, nf_diff_loss*len(nf)/nb_data_points
       


    def forward_step(self, 
                     dataloaders, 
                     nf1,
                     nf1_neg,
                     nf2,
                     nf2_neg,
                     nf3,
                     nf3_neg,
                     nf4,
                     nf4_neg,
                     margin,
                     train = True
                 ):

        if train:
            nb_nf1, nb_nf2, nb_nf3, nb_nf4 = tuple(map(len, self.train_nfs))
        else:
            nb_nf1, nb_nf2, nb_nf3, nb_nf4 = tuple(map(len, self.valid_nfs))
        
        nb_nf1 = nb_nf1 if self.nf1 else 0
        nb_nf2 = nb_nf2 if self.nf2 else 0
        nb_nf3 = nb_nf3 if self.nf3 else 0
        nb_nf4 = nb_nf4 if self.nf4 else 0
        
        total = nb_nf1 + nb_nf2 + nb_nf3 + nb_nf4

        data_nf1, data_nf2, data_nf3, data_nf4 = dataloaders
       
        nf1_pos_loss, nf1_neg_loss, nf1_diff_loss = self.forward_nf(data_nf1, 1, margin, nf1, nf1_neg, train = train)
        nf2_pos_loss, nf2_neg_loss, nf2_diff_loss = self.forward_nf(data_nf2, 2, margin, nf2, nf2_neg, train = train)
        nf3_pos_loss, nf3_neg_loss, nf3_diff_loss = self.forward_nf(data_nf3, 3, margin, nf3, nf3_neg, train = train)
        nf4_pos_loss, nf4_neg_loss, nf4_diff_loss = self.forward_nf(data_nf4, 4, margin, nf4, nf4_neg, train = train)

        nf1_pos_loss *= nb_nf1/total
        nf2_pos_loss *= nb_nf2/total
        nf3_pos_loss *= nb_nf3/total
        nf4_pos_loss *= nb_nf4/total
        nf1_diff_loss *= nb_nf1/total
        nf2_diff_loss *= nb_nf2/total
        nf3_diff_loss *= nb_nf3/total
        nf4_diff_loss *= nb_nf4/total

        print(f"nf1: {nf1_diff_loss}, \tnf1p: {nf1_pos_loss}, \tnf1n: {nf1_neg_loss}")
        print(f"nf2: {nf2_diff_loss}, \tnf2p: {nf2_pos_loss}, \tnf2n: {nf2_neg_loss}")
        print(f"nf3: {nf3_diff_loss}, \tnf3p: {nf3_pos_loss}, \tnf3n: {nf3_neg_loss}")
        print(f"nf4: {nf4_diff_loss}, \tnf4p: {nf4_pos_loss}, \tnf4n: {nf4_neg_loss}")
      
        return  nf1_pos_loss + nf1_diff_loss+ nf2_pos_loss + nf2_diff_loss + nf3_pos_loss + nf3_diff_loss + nf4_pos_loss + nf4_diff_loss


    def forward_step_sampling(self,
                              data,
                              nf1,
                              nf1_neg,
                              nf2,
                              nf2_neg,
                              nf3,
                              nf3_neg,
                              nf4,
                              nf4_neg,
                              margin,
                              train = True
                              ):


        data_nf1, data_nf2, data_nf3, data_nf4 = data
 
        nb_nf1, nb_nf2, nb_nf3, nb_nf4 = tuple(map(len, data))

        if nb_nf1 > self.batch_size:
            
            rand_index = np.random.choice(nb_nf1, size=self.batch_size, replace = False)
            data_nf1 = data_nf1[rand_index].to(self.device)
            nb_nf1 = self.batch_size

        if nb_nf2 > self.batch_size:
            
            rand_index = np.random.choice(nb_nf2, size=self.batch_size, replace = False)
            data_nf2 = data_nf2[rand_index].to(self.device)
            nb_nf2 = self.batch_size

        if nb_nf3 > self.batch_size:
            
            rand_index = np.random.choice(nb_nf3, size=self.batch_size, replace = False)
            data_nf3 = data_nf3[rand_index].to(self.device)
            nb_nf3 = self.batch_size

        if nb_nf4 > self.batch_size:
            
            rand_index = np.random.choice(nb_nf4, size=self.batch_size, replace = False)
            data_nf4 = data_nf4[rand_index].to(self.device)
            nb_nf4 = self.batch_size


        nb_nf1  = nb_nf1 if self.nf1 else 0 
        nb_nf2  = nb_nf2 if self.nf2 else 0 
        nb_nf3  = nb_nf3 if self.nf3 else 0 
        nb_nf4  = nb_nf4 if self.nf4 else 0 
        nb_data_points = nb_nf1 + nb_nf2 + nb_nf3 + nb_nf4

        nf1_pos_loss, nf1_neg_loss, nf1_diff_loss = self.forward_nf_sampling(data_nf1, 1, margin, nf1, nf1_neg, nb_data_points, train = train)
        nf2_pos_loss, nf2_neg_loss, nf2_diff_loss = self.forward_nf_sampling(data_nf2, 2, margin, nf2, nf2_neg, nb_data_points, train = train)
        nf3_pos_loss, nf3_neg_loss, nf3_diff_loss = self.forward_nf_sampling(data_nf3, 3, margin, nf3, nf3_neg, nb_data_points, train = train)
        nf4_pos_loss, nf4_neg_loss, nf4_diff_loss = self.forward_nf_sampling(data_nf4, 4, margin, nf4, nf4_neg, nb_data_points, train = train)
        


        print(f"nf1: {nf1_diff_loss}, \tnf1p: {nf1_pos_loss}, \tnf1n: {nf1_neg_loss}")
        print(f"nf2: {nf2_diff_loss}, \tnf2p: {nf2_pos_loss}, \tnf2n: {nf2_neg_loss}")
        print(f"nf3: {nf3_diff_loss}, \tnf3p: {nf3_pos_loss}, \tnf3n: {nf3_neg_loss}")
        print(f"nf4: {nf4_diff_loss}, \tnf4p: {nf4_pos_loss}, \tnf4n: {nf4_neg_loss}")
      
        return nf1_pos_loss + nf1_diff_loss + nf2_pos_loss + nf2_diff_loss + nf3_pos_loss + nf3_diff_loss + nf4_pos_loss + nf4_diff_loss

        

    def evaluate(self):
#        self.load_data()
        self.model = CatModel(len(self.classes), len(self.relations), 1024).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        test_loss = self.forward_step(self.test_dl, train=False) #model(self.test_nfs).detach().item()
        print('Test Loss:', test_loss)
 
    def evaluate_ppi_valid(self):
        self.model.eval()
        _, _, _, valid_nf4 = self.valid_nfs
        index = np.random.choice(len(valid_nf4), size = self.num_points_eval, replace = False)
#        index = list(range(self.num_points_eval))
        valid_nfs = valid_nf4[index]
        preds, labels = self.run_and_save_predictions(self.model, samples = valid_nfs, save = False)

#        results = evalNF4Loss(self.model, valid_nfs, self.prot_dict, self.prot_index, self.trlabels, len(self.prot_index), device = "cuda")
        results = evalNF4Loss(valid_nfs, self.prot_dict, self.prot_index, self.trlabels, len(self.prot_index), device= 'cuda', preds =preds, labels = labels)

        return results

    def evaluate_ppi(self):

        self.run_and_save_predictions(self.model)
        #self.load_data(device = "cuda")
        #self.device = "cuda"
        #test_model = TestModule((len(self.classes), len(self.relations), self.embedding_size, self.model_filepath)).to(self.device)

        #_, _, _, test_nf4 = self.test_nfs

        #test_nf4 = test_nf4.cpu().detach().numpy()
        
        #test_dataset = TestDataset(test_nf4, self.prot_index, 0)

        #self.create_dataloaders(device = "cuda")
        _, _, _, test_nf4 = self.test_nfs
#        index = np.random.choice(len(test_nf4), size = 2000, replace = False)
#        index = list(range(100))
        test_nfs = test_nf4#[index]
        print(f"Device: {self.device}")

 #       self.model = CatModel(len(self.classes), len(self.relations), self.embedding_size).to(self.device)
 #       print('Load the best model', self.model_filepath)
 #       self.model.load_state_dict(th.load(self.model_filepath))

  #      results = evalNF4Loss(self.model, test_nf4, self.prot_dict, self.prot_index, self.trlabels, len(self.prot_index), device = "cuda", show = True)
        evalNF4Loss(test_nfs, self.prot_dict, self.prot_index, self.trlabels, len(self.prot_index), device= self.device, show = True, preds_file =  self.predictions_file, labels_file = self.labels_file )
        
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

        self._loaded = True

    def load_data_old(self, train_file, valid_file, test_file, device = 'cpu'):
        if self._loaded:
            return

        if device == 'cuda':
            self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        print(f"In device: {self.device}")

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
        


class CatModel(nn.Module):

    def __init__(self, num_objects, num_rels, embedding_size, dropout = 0):
        super(CatModel, self).__init__()

        self.embedding_size = embedding_size
        self.num_obj = num_objects 
        
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Embedding(self.num_obj, embedding_size)
        k = math.sqrt(1 / embedding_size)
        nn.init.uniform_(self.embed.weight, -0, 1)

        self.embed_rel = nn.Embedding(num_rels, embedding_size)
        k = math.sqrt(1 / embedding_size)
#        nn.init.uniform_(self.embed_rel.weight, -k,k)

        # Embedding network for the ontology ojects
        self.net_object = nn.Sequential(
            self.embed,
            nn.Linear(embedding_size, embedding_size),
#            nn.ReLU(),
#            nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
            self.dropout
        )

        # Embedding network for the ontology relations
        self.net_rel = nn.Sequential(
            self.embed_rel,

            nn.Linear(embedding_size, embedding_size),
            
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
            self.dropout
        )

        # Embedding network for left part of 3rd normal form
        self.embed_fst = nn.Sequential(

            nn.Linear(3*embedding_size, 2*embedding_size),
            
            # nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
            self.dropout
        )

        self.embed_snd = nn.Sequential(

            nn.Linear(3*embedding_size, 2*embedding_size),
            
            # nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(2*embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
            self.dropout
        )

        # Embedding network for left part of 3rd normal form
        self.embed_up = nn.Sequential(

            nn.Linear(2*embedding_size, embedding_size),
#            self.dropout,
            
            # nn.Linear(embedding_size, embedding_size),
             nn.ReLU(),
             nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
            self.dropout
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

        self.exponential_morphisms = (self.up2down, self.up2exp, self.down2exp, self.up2ant, self.down2ant, self.up2cons, self.down2cons)

        #Product

        # Embedding network for the objects in the exponential diagram
        self.embed_bigger_prod = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),

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
#        fc = nn.Sequential(
#            nn.Linear(self.embedding_size, self.embedding_size),
  #          self.dropout
 #       )
  #      return fc
        return nn.Linear(self.embedding_size, self.embedding_size)


    # def cos_sim(self, data):
    #     srcs = self.net_object(data[:,0])
    #     dsts = self.net_object(data[:,2])
        
    #     x = th.sum(srcs*dsts, dim = 1, keepdim = True).squeeze()
    #     return th.sigmoid(x)

    def forward(self, normal_form, idx, neg = False, margin = 0):
#        nf1, nf2, nf3, nf4 = normal_forms

        # logging.debug(f"NF1: {len(nf1)}")
        # logging.debug(f"NF2: {len(nf2)}")
        # logging.debug(f"NF3: {len(nf3)}")
        # logging.debug(f"NF4: {len(nf4)}")
        loss = 0
        bs, _ = normal_form.shape
        
        if idx == 1:
            embed_nets = (self.net_object, self.embed_up)
            loss = L.nf1_loss(normal_form, self.exponential_morphisms, self.product_morphisms, embed_nets, neg = neg, num_objects = self.num_obj)

        elif idx == 2:
            embed_nets = (self.net_object, self.embed_up, self.embed_bigger_prod)
            loss = L.nf2_loss(normal_form, self.product_morphisms, self.exponential_morphisms, embed_nets, neg = neg)

        elif idx == 3:
            embed_nets = (self.net_object, self.net_rel, self.embed_fst, self.embed_up, self.embed_bigger_prod)
            loss = L.nf3_loss(normal_form, self.product_morphisms, self.exponential_morphisms, embed_nets, neg = neg)

        elif idx == 4:
            embed_nets = (self.net_object, self.net_rel, self.embed_snd, self.embed_up, self.embed_bigger_prod)
            loss = L.nf4_loss(normal_form, self.product_morphisms,  self.exponential_morphisms, embed_nets, neg = neg, num_objects = self.num_obj)
        else:
            raise ValueError("Invalid index")
            
#        bs_end = loss.shape
        
#        assert bs == bs_end, f"Dimensions mismatch: {bs}, {bs_end}"
        return loss

    
class TestModule(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.catModel = model
        

        

    def forward(self, x):
        bs, num_prots, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        x = self.catModel(x, 4)

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


