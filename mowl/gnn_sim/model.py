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


class GNNSim(Model):
    def __init__(self,
                 dataset,
                 n_hidden,
                 dropout,
                 learning_rate,
                 num_bases,
                 batch_size,
                 epochs,
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

        self.test_df = None
        self.g = None
        self.annots = None
        self.prot_idx = None
        self.num_rels = None
        

    def train(self, checkpoint_dir = None, tuning= False):

        logging.info("Loading graph data...")
        g, annots, prot_idx, num_rels = self.load_graph_data()
        logging.info(f"PPI use case: Num nodes: {g.number_of_nodes()}")
        logging.info(f"PPI use case: Num edges: {g.number_of_edges()}")
        train_df, val_df, test_df = self.load_ppi_data()

        annots = th.FloatTensor(annots)
            
        self.test_df = test_df
        self.g = g
        self.annots = annots
        self.prot_idx = prot_idx
        self.num_rels = num_rels

        train_labels = th.FloatTensor(train_df['labels'].values)
        val_labels = th.FloatTensor(val_df['labels'].values)
    
        train_data = GraphDataset(g, train_df, train_labels, annots, prot_idx)
        val_data = GraphDataset(g, val_df, val_labels, annots, prot_idx)

        train_dataloader = GraphDataLoader(train_data, batch_size = self.batch_size, drop_last = True)
        val_dataloader = GraphDataLoader(val_data, batch_size = self.batch_size, drop_last = True)
    
        if self.num_bases is None:
            self.num_bases = num_rels
        
        num_nodes = g.number_of_nodes()

        device = "cuda"
        model = PPIModel(num_rels, self.num_bases, num_nodes, self.n_hidden, self.dropout).to(device)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

        if checkpoint_dir:
            model_state, optimizer_state = th.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        loss_func = nn.BCELoss()

        early_stopping_limit = 3
        early_stopping = early_stopping_limit
        best_loss = float("inf")
        best_roc_auc = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            model.train()

            with ck.progressbar(train_dataloader) as bar:
                for i, (batch_g, batch_features, batch_labels) in enumerate(bar):

                    logits = model(batch_g.to(device), batch_features.to(device)).squeeze()
                    
                    labels = batch_labels.to(device).squeeze()
        
                    loss = loss_func(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()

                epoch_loss /= (i+1)
        
            model.eval()
            val_loss = 0
            preds = []
            labels = []
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(val_dataloader) as bar:
                    for i, (batch_g, batch_feat, batch_labels) in enumerate(bar):
                        
                        logits = model(batch_g.to(device), batch_feat.to(device)).squeeze()
                        lbls = batch_labels.squeeze().to(device)
                        loss = loss_func(logits, lbls)
                        val_loss += loss.detach().item()
                        labels = np.append(labels, lbls.cpu())
                        preds = np.append(preds, logits.cpu())
                    val_loss /= (i+1)

            roc_auc = self.compute_roc(labels, preds)
            if not tuning:
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    th.save(model.state_dict(), self.file_params["output_model"])
                if best_loss > val_loss:
                    best_loss = val_loss
                    early_stopping = early_stopping_limit
                else:
                    early_stopping -= 1
                print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')
                if early_stopping == 0:
                    print("Finished training (early stopping)")
                    break

            else:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    th.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=(val_loss), auc=roc_auc)
        print("Finished Training")


    def evaluate(self, tuning=False, best_checkpoint_dir=None):

        device = "cpu"

        if self.test_df is None:
            _, _, test_df = self.load_pp_data()
        else:
            test_df = self.test_df
        g = self.g
        annots = self.annots
        prot_idx = self.prot_idx
        num_rels = self.num_rels
        
        test_labels = th.FloatTensor(test_df['labels'].values)
            
        test_data = GraphDataset(g, test_df, test_labels, annots, prot_idx)

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
                for iter, (batch_g, batch_features, batch_labels) in enumerate(bar):
                    logits = model(batch_g.to(device), batch_features.to(device)).squeeze()
                    labels = batch_labels.squeeze().to(device)
                    loss = loss_func(logits, labels)
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

    def load_ppi_data(self):
        train_df = pd.read_pickle(self.file_params["train_inter_file"])
        valid_df = pd.read_pickle(self.file_params["valid_inter_file"])
        test_df = pd.read_pickle(self.file_params["test_inter_file"])
        logging.info("Original ata sizes: Train %d, Val %d, Test %d", len(train_df), len(valid_df), len(test_df))

        index = np.arange(len(train_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_df = train_df.iloc[index]#[:50000]]

        index = np.arange(len(valid_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        valid_df = valid_df.iloc[index]#[:5000]]
        
        index = np.arange(len(test_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        test_df = test_df.iloc[index]#[:5000]]

        logging.info("Used data sizes: Train %d, Val %d, Test %d", len(train_df), len(valid_df), len(test_df))
        return train_df, valid_df, test_df

    def load_graph_data(self):

        parser = gen_factory(self.graph_generation_method, self.dataset.ontology)
        edges_path = f"data/edges_{self.graph_generation_method}.pkl"

        try:
            infile = open(edges_path, 'rb')
            edges = pkl.load(infile)
        except:
            edges = parser.parse()
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
          

        
        data_file = self.file_params["data_file"]

        prot_idx = {}
        with open(data_file, 'r') as f:
            rows = [line.strip('\n').split('\t') for line in f.readlines()]
            
            annotations = np.zeros((num_nodes, len(rows)), dtype=np.float32)
            
            for i, row  in enumerate(rows):
                prot_id = row[0]
                prot_idx[prot_id] = i
                for go_id in row[1:]:
                    if go_id in node_idx:
                        annotations[node_idx[go_id], i] = 1
        return g, annotations, prot_idx, num_rels


    
class RGCN(BaseRGCN):

         def build_hidden_layer(self, idx):
             act = F.relu if idx < self.num_hidden_layers - 1 else None
             return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

         
# class PPIModel(nn.Module):

#     def __init__(self, num_rels, num_bases, num_nodes, n_hid, dropout):
#         super().__init__()

#         self.h_dim = 2
#         self.num_rels = num_rels
#         self.num_bases = None if num_bases < 0 else num_bases
#         self.num_nodes = num_nodes
#         self.n_hid = n_hid
#         print(f"Num rels: {self.num_rels}")
#         print(f"Num bases: {self.num_bases}")

#         self.rgcn_layers = nn.ModuleList()
#         for i in range(n_hid):
#             act = F.relu if i < n_hid - 1 else None
#             newLayer = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, activation = act, self_loop = False, dropout = dropout)
#             self.rgcn_layers.append(newLayer)

        
#         dim1 = self.num_nodes

#         self.mlp = nn.Sequential(
#             nn.Linear(2*dim1, 1)
#         )

#     def forward(self, g, features):

#         features = features.reshape(-1, self.h_dim)
#         edge_type = g.edata['rel_type'].long()
#         norm = None if not 'norm' in g.edata else g.edata['norm']

#         skip = features 
#         x = features
#         for i, l in enumerate(self.rgcn_layers):
#             x = l(g, x, edge_type, norm)
#             if i % 2 == 1:
#                 x += skip
#                 skip = x
            
#         x = th.flatten(x).view(-1, self.num_nodes*self.h_dim)
#         x = self.mlp(x)
#         return th.sigmoid(x)



class PPIModel(nn.Module):

    def __init__(self, num_rels, num_bases, num_nodes, n_hid, dropout):
        super().__init__()

        self.h_dim = 2
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_nodes = num_nodes
        self.n_hid = n_hid
        print(f"Num rels: {self.num_rels}")
        print(f"Num bases: {self.num_bases}")

        self.rgcn_layers = nn.ModuleList()
        for i in range(n_hid):
            act = F.relu if i < n_hid - 1 else None
            newLayer = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, activation = act, self_loop = False, dropout = dropout)
            self.rgcn_layers.append(newLayer)

        
        dim1 = self.num_nodes

        self.mlp = nn.Sequential(
            nn.Linear(2*dim1, 1)
        )

    def forward(self, g, features):

        features = features.reshape(-1, self.h_dim)
        edge_type = g.edata['rel_type'].long()
        norm = None if not 'norm' in g.edata else g.edata['norm']

        skip = features 
        x = features
        for i, l in enumerate(self.rgcn_layers):
            x = l(g, x, edge_type, norm)
            if i % 2 == 1:
                x += skip
                skip = x
            
        x = th.flatten(x).view(-1, self.num_nodes*self.h_dim)
        x = self.mlp(x)
        return th.sigmoid(x)


    

class GraphDataset(IterableDataset):

    def __init__(self, graph, df, labels, annots, prot_idx):
        self.graph = graph
        self.annots = annots
        self.labels = labels
        self.df = df
        self.prot_idx = prot_idx

    def get_data(self):
        for i, row in enumerate(self.df.itertuples()):
            p1, p2 = row.interactions
            label = self.labels[i].view(1, 1)
            if p1 not in self.prot_idx or p2 not in self.prot_idx:
                continue
            pi1, pi2 = self.prot_idx[p1], self.prot_idx[p2]

            feat = self.annots[:, [pi1, pi2]]

            yield (self.graph, feat, label)

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.df)



def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']




class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.out_dim = out_dim
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        
        g.ndata['z'] = z.view(-1, self.out_dim)
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')



class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return th.cat(head_outs, dim=1)
        else:
            # merge using average
            return th.mean(th.stack(head_outs))



class PPIModel2(nn.Module):
    def __init__(self, num_rels, num_bases, num_nodes, n_hid, dropout):
        super(PPIModel, self).__init__()

        self.h_dim = 2
        self.num_nodes = num_nodes


        in_dim = 2
        hidden_dim = 16
        out_dim = 2
        num_heads = 2
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

        dim1 = num_nodes

        self.mlp = nn.Linear(2*dim1, 1)


    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        

        h = th.flatten(h).view(-1, self.num_nodes*self.h_dim)
        h = self.mlp(h)
        
        

        return th.sigmoid(h)

