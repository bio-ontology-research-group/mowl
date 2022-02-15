from mowl.model import Model
from mowl.graph.util import gen_factory
import pandas as pd
import click as ck
import os
import pickle as pkl
import logging
import dgl
from dgl import nn as dglnn
from dgl.dataloading import GraphDataLoader
import torch as th
import torch.distributed as dist
import numpy as np
from math import floor
from torch import nn
from torch.nn import functional as F
from torch import optim
from .baseRGCN import BaseRGCN
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from torch.utils.data import DataLoader, IterableDataset
#from dgl.nn.pytorch import RelGraphConv
from .rgcnConv import RelGraphConv

from mowl.graph.edge import Edge

import random
from ray import tune

class GNNSimGD(Model):
    def __init__(self,
                 dataset,
                 n_hidden,
                 dropout,
                 learning_rate,
                 num_bases,
                 batch_size,
                 epochs,
                 use_case,
                 graph_generation_method = "taxonomy", #Default: generate graph taxonomy
                 normalize = False,
                 regularization = 0,
                 self_loop = False,
                 min_edges = 0, #Only takes the relation types in which the number of triples is greater than min_edges. If 0 then takes all the relation types
                 seed = -1,
                 file_params = None #Dictionary of data file paths corresponding to the graph generation method (NEEDS REFACTORING)
                 ):
        super().__init__(dataset)
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_bases = None if num_bases < 0 else num_bases
        self.batch_size =  batch_size
        self.epochs = epochs
        self.use_case = use_case
        self.graph_generation_method = graph_generation_method
        self.normalize = normalize
        self.regularization = regularization
        self.self_loop = self_loop
        self.min_edges = min_edges
        self.file_params = file_params
        
        if seed>=0:
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.test_data = None
        self.g = None
        self.gene_annots = None
        self.disease_annots = None
        self.gene_idx = None
        self.disease_idx = None
        self.num_rels = None

    def train(self, checkpoint_dir = None, tuning= False):

        logging.info("loading graph data...")
        g, gene_annots, disease_annots, gene_idx, disease_idx, num_rels = self.load_graph_data()
        logging.info(f"G-D use case: Num nodes: {g.number_of_nodes()}")
        logging.info(f"G-D use case: Num edges: {g.number_of_edges()}")
        train_data_, val_data_, test_data = self.load_gene_disease_data()


        gene_annots = th.FloatTensor(gene_annots)
        disease_annots = th.FloatTensor(disease_annots)

        self.test_data = test_data
        self.g = g
        self.gene_annots = gene_annots
        self.disease_annots = disease_annots
        self.gene_idx = gene_idx
        self.disease_idx = disease_idx
        self.num_rels = num_rels


        train_data = GraphDatasetGD(g, train_data_, gene_annots, disease_annots, gene_idx, disease_idx)
        val_data = GraphDatasetGD(g, val_data_, gene_annots, disease_annots, gene_idx, disease_idx)

        train_dataloader = GraphDataLoader(train_data, batch_size = self.batch_size, drop_last = True)
        val_dataloader = GraphDataLoader(val_data, batch_size = self.batch_size, drop_last = True)
            

        if self.num_bases is None:
            self.num_bases = num_rels
        
        num_nodes = g.number_of_nodes()

        device = "cuda"
        model = PPIModel(num_rels, self.num_bases, num_nodes, self.n_hidden, self.dropout).to(device)
        print(model)
        
        if checkpoint_dir:
            model_state, optimizer_state = th.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        loss_func = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

        early_stopping_limit = 3
        early_stopping = early_stopping_limit
        best_loss = float("inf")
        best_roc_auc = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            model.train()

            with ck.progressbar(train_dataloader) as bar:
            
                for i, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):

                    feats1 = gene_annots[feats1].view(-1,1)
                    feats2 = disease_annots[feats2].view(-1,1)

                    logits = model(batch_g.to(device), feats1.to(device), feats2.to(device)).squeeze()

                    labels = batch_labels.squeeze().to(device)
                    loss = loss_func(logits, labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    i += 1

                epoch_loss /= (i+1)
        
            model.eval()
            val_loss = 0
            preds = []
            labels = []
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(val_dataloader) as bar:
                    for i, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):
                        
                        feats1 = gene_annots[feats1].view(-1,1)
                        feats2 = disease_annots[feats2].view(-1,1)

                        logits = model(batch_g.to(device), feats1.to(device), feats2.to(device)).squeeze()
                        lbls = batch_labels.squeeze().to(device)
                        loss = loss_func(logits, lbls.float())
                        val_loss += loss.detach().item()
                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                    val_loss /= (i+1)

            roc_auc = self.compute_roc(labels, preds)
            if not tuning:
                if best_roc_auc < roc_auc:
                    best_roc_auc = roc_auc
                    th.save(model.state_dict(), self.file_params["output_model"])
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping = early_stopping_limit
                else:
                    early_stopping -= 1
                print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')
                
                if early_stopping == 0:
                    print("Finished training (early stoppnig)")
                    break
            else:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    th.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=(val_loss), auc=roc_auc)
        print("Finished Training")
        

    def evaluate(self, tuning=False, best_checkpoint_dir=None):

        device = "cpu"

        if self.test_data is None:
            _, _, test_data = self.load_gene_disease_data()
        else:
            test_data = self.test_data
        g = self.g
        gene_annots = self.gene_annots
        disease_annots = self.disease_annots
        gene_idx = self.gene_idx
        disease_idx = self.disease_idx
        num_rels = self.num_rels

        test_data = GraphDatasetGD(g, test_data, gene_annots, disease_annots, gene_idx, disease_idx)

        test_dataloader = GraphDataLoader(test_data, batch_size = self.batch_size, drop_last = True)

        num_nodes = g.number_of_nodes()
    
        if self.num_bases is None:
            self.num_bases = num_rels

        loss_func = nn.BCELoss()

        model = PPIModel(num_rels, self.num_bases, num_nodes, self.n_hidden, self.dropout)

        if tuning:
            model_state, optimizer_state = th.load(os.path.join(best_checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
 
        else:
            model.load_state_dict(th.load(self.file_params["output_model"]))
        
        model.to(device)
        model.eval()
        test_loss = 0

        preds = []
        all_labels = []
        with th.no_grad():
            with ck.progressbar(test_dataloader) as bar:
                for iter, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):
                    feats1 = gene_annots[feats1].view(-1,1)
                    feats2 = disease_annots[feats2].view(-1,1)
                    
                    logits = model(batch_g.to(device), feats1.to(device), feats2.to(device)).squeeze()
                    labels = batch_labels.squeeze().to(device)
                    loss = loss_func(logits, labels.float())
                    test_loss += loss.detach().item()
                    preds = np.append(preds, logits.cpu())
                    all_labels = np.append(all_labels, labels.cpu())
                test_loss /= (iter+1)

        roc_auc = self.compute_roc(all_labels, preds)
        print(f'Test loss - {test_loss}, \tAUC - {roc_auc}')

        return test_loss, roc_auc

    def compute_roc(self, labels, preds):
    # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
        roc_auc = auc(fpr, tpr)

        return roc_auc

    def get_batches(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate, drop_last=True, pin_memory = True, num_workers = 4)
    
    def collate(self, samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        labels, prots1, prots2 = map(list, zip(*samples))
        return th.tensor(labels), prots1, prots2 #th.cat(feats1, dim=0), th.cat(feats2, dim=0)


    def load_gene_disease_data(self):
        with open(self.file_params["train_inter_file"], "rb") as f:
            train_data = pkl.load(f)
            train_data = list(train_data.items())
            random.shuffle(train_data)
            train_data = train_data[:40000]

        with open(self.file_params["valid_inter_file"], "rb") as f:
            val_data = pkl.load(f)
            val_data = list(val_data.items())
            random.shuffle(val_data)
            val_data = val_data[:10000]

        with open(self.file_params["test_inter_file"], "rb") as f:
            test_data = pkl.load(f)
            test_data = list(test_data.items())
            random.shuffle(test_data)
            test_data = test_data[:10000]

        return train_data, val_data, test_data



    def load_graph_data(self):

        logging.debug("Creating graph generation method...")
        parser = gen_factory(self.graph_generation_method, self.dataset.ontology)
        edges_path = f"data/edges_{self.use_case}_{self.graph_generation_method}.pkl"
        logging.debug("Created")
        try:
            logging.debug("try")
            infile = open(edges_path, 'rb')
            edges = pkl.load(infile)
        except:
            logging.debug("except")
            logging.debug("Parsing ontology...")
            edges = parser.parse()
            logging.info("Finished parsing ontology")
            edges = [(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
            outfile = open(edges_path, 'wb')
            pkl.dump(edges, outfile)

        terms_file = self.file_params["terms_file"]

        with open(terms_file) as f:
            terms = f.read().splitlines()
            edges = [(s, r, d) for s,r,d in edges if s in terms and  d in terms]
        

        srcs, rels, dsts = tuple(map(list, zip(*edges)))
        

        g = dgl.DGLGraph()
        nodes = list(set(srcs).union(set(dsts)))
        g.add_nodes(len(nodes))
        node_idx = {v: k for k, v in enumerate(nodes)}
        
        if self.self_loop:
            srcs += nodes
            dsts += nodes
            rels += ["id" for _ in range(len(nodes))]

            
        srcs = [node_idx[s] for s in srcs]
        dsts = [node_idx[d] for d in dsts]
        
        edges_per_rel = {}
        for rel in rels:
            if not rel in edges_per_rel:
                edges_per_rel[rel] = 0
            edges_per_rel[rel] += 1

            

        if self.normalize:
            edge_node = {}
            rels_dst = list(zip(rels, dsts))

            for rel, dst in rels_dst:
                if (rel, dst) not in edge_node:
                    edge_node[(rel, dst)] = 0
                edge_node[(rel, dst)] += 1
            
            edge_node = {k: 1/v for k, v in edge_node.items()}
            
            norm = [edge_node[i] for i in rels_dst]


            zipped_data = list(zip(srcs, dsts, rels, norm))
            srcs, dsts, rels, norm = zip(*[(s, d, r, n) for s, d, r, n in zipped_data if edges_per_rel[r] > self.min_edges])

            norm = th.Tensor(norm).view(-1, 1)

            
        else:
            norm = None

            zipped_data = list(zip(srcs, dsts, rels))
            srcs, dsts, rels = zip(*[(s, d, r) for s, d, r in zipped_data if edges_per_rel[r] > self.min_edges])

            
        rels_idx = {v: k for k, v in enumerate(set(rels))}
        rels = [rels_idx[r] for r in rels]
        num_rels = len(rels_idx)
        rels = th.Tensor(rels)

        
        print("Edges in graph:\n", edges_per_rel)
        g.add_edges(srcs, dsts)

        
        g.edata.update({'rel_type': rels})
        if norm != None:
            g.edata.update({'norm': norm})

        
        num_nodes = g.number_of_nodes()
          
        gene_data_file = self.file_params["gene_data_file"]
        disease_data_file = self.file_params["disease_data_file"]

        true_counter = 0
        false_counter = 0
        toprint = True
        with open(gene_data_file, 'r') as f:
            rows = [line.strip('\n').split('\t') for line in f.readlines()]
            genes = set(map(lambda x: x[0], rows))
            genes_idxs = {gene: i for i, gene in enumerate(genes)}
            gene_annots = np.zeros((len(genes), num_nodes), dtype=np.float32)

            for row in rows:
                gene = row[0]
                mgi_id = row[1].split("/")[-1].replace("_", ":")
                if toprint:
                    print("MGI_ID", mgi_id, list(node_idx.items())[0])
                    toprint = False
                if mgi_id in node_idx:
                    true_counter += 1
                    gene_annots[genes_idxs[gene], node_idx[mgi_id]] = 1
                else:
                    false_counter += 1

            print(f"For GENES: TRUE COUNTER {true_counter}. FALSE COUNTER {false_counter}")
        true_counter = 0
        false_counter = 0
        toprint = True
        with open(disease_data_file, 'r') as f:
            rows = [line.strip('\n').split('\t') for line in f.readlines()]
            diseases  = set(map(lambda x: x[0], rows))
            diseases_idxs = {disease: i for i, disease in enumerate(diseases)}
            disease_annots = np.zeros((len(diseases), num_nodes), dtype=np.float32)

            for row in rows:
                disease = row[0]
                hpo_id = row[1].split("/")[-1].replace("_", ":")

                if toprint:
                    print("HPO_ID: ", hpo_id)
                    toprint = False
                if hpo_id in node_idx:
                    true_counter += 1
                    disease_annots[diseases_idxs[disease], node_idx[hpo_id]] = 1
                else:
                    false_counter += 1
            print(f"For DISEASES: TRUE COUNTER {true_counter}. FALSE COUNTER {false_counter}")

        return g, gene_annots, disease_annots, genes_idxs, diseases_idxs, num_rels

    
class RGCN(BaseRGCN):

         def build_hidden_layer(self, idx):
             act = F.relu if idx < self.num_hidden_layers - 1 else None
             return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

         
class PPIModel(nn.Module):

    def __init__(self, num_rels, num_bases, num_nodes, n_hid, dropout):
        super().__init__()

        self.h_dim = 1
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_nodes = num_nodes

        print(f"Num rels: {self.num_rels}")
        print(f"Num bases: {self.num_bases}")


        self.rgcn_layers = nn.ModuleList()

        for i in range(n_hid):
            act = F.relu if i < n_hid - 1 else None
            newLayer = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, activation = act, self_loop = False, dropout = dropout)
            self.rgcn_layers.append(newLayer)

        
        dim1 = self.num_nodes
        dim2 = floor(self.num_nodes/2)

        self.net = nn.Sequential(
            nn.Linear(dim1, 1024)
#            nn.Dropout(),
#            nn.Linear(dim2, 1024),
#            nn.Dropout()
        )

    def forward_each(self, g, features, edge_type, norm):
        initial = features
        x = features 

        for l in self.rgcn_layers:
            x = l(g, x, edge_type, norm)
            x += initial

        x = th.flatten(x).view(-1, self.num_nodes*self.h_dim)
        x = self.net(x)
        return x

    def forward(self, g, feat1, feat2):

        edge_type = g.edata['rel_type'].long()

        norm = None if not 'norm' in g.edata else g.edata['norm']

        x1 = self.forward_each(g, feat1, edge_type, norm)
        x2 = self.forward_each(g, feat2, edge_type, norm)

        x = th.sum(x1 * x2, dim=1, keepdims=True)
#        x = th.cat([x1, x2]).view(-1,2048)
        return th.sigmoid(x)


class GraphDatasetGD(IterableDataset):

    def __init__(self, g, asoc_data, gene_annots, disease_annots, gene_idx, disease_idx):
        self.graph = g
        self.asoc_data = asoc_data
        self.gene_annots = gene_annots
        self.disease_annots = disease_annots
        self.gene_idx = gene_idx
        self.disease_idx = disease_idx

    def get_data(self):
        for i, ((disease, gene), info) in enumerate(self.asoc_data):
            label = info["label"]
            if gene  not in self.gene_idx or disease not in self.disease_idx:
                continue
            
            gene_i,  disease_i = self.gene_idx[gene], self.disease_idx[disease]

            yield (self.graph, label, gene_i, disease_i)

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.asoc_data)



def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']
