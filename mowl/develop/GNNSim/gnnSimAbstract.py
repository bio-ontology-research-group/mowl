from mowl.model import Model
from mowl.graph.factory import parser_factory
import click as ck
import numpy as np
import torch as th
import random
import pandas as pd
import logging
from .baseRGCN import BaseRGCN
import pickle as pkl
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

import dgl
from dgl.dataloading import GraphDataLoader
from torch import optim
from torch import nn
# JPype imports
from java.util import HashMap
from java.util import ArrayList
from org.mowl.IC import IC

class AbsGNNSimPPI(Model):
    def __init__(self,
                 dataset,
                 epochs,
                 bs,
                 lr,
                 regularization,
                 parser,
                 normalization,
                 min_edges,
                 self_loop,
                 seed = -1,
                 ppi_model_params = None,
                 data_params = None,
                 file_params = None
                 ):

        super().__init__(dataset)
        self.ppi_model_params = ppi_model_params
        self.file_params = file_params
        self.data_params = data_params
        
        self.lr = lr
        self.parser = parser
        self.bs = bs
        self.epochs = epochs
        self.min_edges = min_edges
        self.regularization = regularization
        self.normalization = normalization
        self.self_loop = self_loop
        self.seed = seed 


        self.train_df = None
        self.test_df = None
        self.g = None
        self.annots = None
        self.prot_idx = None
        self.num_rels = None


    

    def train(self, checkpoint_dir = None, tuning= False):
        
        train_df, val_df, test_df = self.load_ppi_data()
        self.train_df = train_df
        self.test_df = test_df

        logging.info("Loading graph data...")
        g, annots, prot_idx, num_rels = self.load_graph_data()
        logging.info(f"PPI use case: Num nodes: {g.number_of_nodes()}")
        logging.info(f"PPI use case: Num edges: {g.number_of_edges()}")

        annots = th.FloatTensor(annots)

        self.g = g
        self.annots = annots
        self.prot_idx = prot_idx
        self.num_rels = num_rels
        self.ppi_model_params["num-rels"] = num_rels
            
        train_labels = th.FloatTensor(train_df['labels'].values)
        val_labels = th.FloatTensor(val_df['labels'].values)
    
        train_data = self.GraphDatasetPPI(g, train_df, train_labels, annots, prot_idx)
        val_data = self.GraphDatasetPPI(g, val_df, val_labels, annots, prot_idx)

        train_dataloader = GraphDataLoader(train_data, batch_size = self.bs, drop_last = True)
        val_dataloader = GraphDataLoader(val_data, batch_size = self.bs, drop_last = True)

        self.num_bases = num_rels
        
        num_nodes = g.number_of_nodes()

        self.ppi_model_params["num-nodes"] = num_nodes
        model = self.PPIModel(self.ppi_model_params)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.regularization)
        
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
            device = "cuda"
            model = model.to(device)
            epoch_loss = 0
            model.train()

            with ck.progressbar(train_dataloader) as bar:
            
                for i, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):

                    feats1 = annots[feats1].view(-1,1)
                    feats2 = annots[feats2].view(-1,1)
                    
                    logits = model(batch_g.to(device), feats1.to(device), feats2.to(device)).squeeze()

                    labels = batch_labels.squeeze().to(device)
                    loss = loss_func(logits, labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()

                epoch_loss /= (i+1)
        
            device = "cuda"
            model = model.to(device)
            model.eval()

            val_loss = 0
            preds = []
            labels = []
            with th.no_grad():
                optimizer.zero_grad()
                with ck.progressbar(val_dataloader) as bar:
                    for i, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):
                        
                        feats1 = annots[feats1].view(-1,1)
                        feats2 = annots[feats2].view(-1,1)
                        
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
                    print("Finished training (early stopping)")
                    break
            else:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    th.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=(val_loss), auc=roc_auc)
        print("Finished Training")
        

    def evaluate(self, tuning=False, best_checkpoint_dir=None):

        device = "cuda"

        if self.test_df is None:
            _, _, test_df = self.load_ppi_data()
        else:
            test_df = self.test_df
        g = self.g
        annots = self.annots
        prot_idx = self.prot_idx
        num_rels = self.num_rels
        
        test_labels = th.FloatTensor(test_df['labels'].values)
            
        test_data = self.GraphDatasetPPI(g, test_df, test_labels, annots, prot_idx)

        test_dataloader = GraphDataLoader(test_data, batch_size = self.bs, drop_last = True)

        num_nodes = g.number_of_nodes()

        self.num_bases = num_rels

        loss_func = nn.BCELoss()

        model = self.PPIModel(self.ppi_model_params).to(device)
        

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
                    feats1 = annots[feats1].view(-1,1)
                    feats2 = annots[feats2].view(-1,1)
                    
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

    def load_ppi_data(self):

        train_size = self.data_params["train-size"]
        valid_size = self.data_params["val-size"]
        test_size = self.data_params["test-size"]

        train_df = pd.read_pickle(self.file_params["train_inter_file"])
        valid_df = pd.read_pickle(self.file_params["valid_inter_file"])
        test_df = pd.read_pickle(self.file_params["test_inter_file"])
        logging.info("Original ata sizes: Train %d, Val %d, Test %d", len(train_df), len(valid_df), len(test_df))

        index = np.arange(len(train_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        if train_size == -1:
            train_df = train_df.iloc[index]
        else:
            train_df = train_df.iloc[index[:train_size]]

        index = np.arange(len(valid_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)

        if valid_size == -1:
            valid_df = valid_df.iloc[index]
        else:
            valid_df = valid_df.iloc[index[:valid_size]]
        
        index = np.arange(len(test_df))
        np.random.seed(seed=0)
        np.random.shuffle(index)

        if test_size == -1:
            test_df = test_df.iloc[index]
        else:
            test_df = test_df.iloc[index[:test_size]]


        logging.info("Used data sizes: Train %d, Val %d, Test %d", len(train_df), len(valid_df), len(test_df))
        return train_df, valid_df, test_df



    def load_graph_data(self):

        logging.debug("Creating graph generation method...")
        parser = parser_factory(self.parser, self.dataset.ontology, bidirectional_taxonomy = True)
        edges_path = f"data/edges_ppi_{self.parser}.pkl"
        logging.debug("Created")
        

        #For computing IC
        training_prots = set()
        for row in self.train_df.itertuples():
            p1, p2 = row.interactions
            training_prots.add(p1)
            training_prots.add(p2)

        annots_dict = self.getAnnotsDict(training_prots)
        ics = IC.computeIC(self.dataset.ontology, annots_dict)
        ics = {self.format(str(k)): v for k, v in ics.items()}
        logging.info("ICS for GO:0005575: %s", ics["GO:0005575"])


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
             #            removed_classes = {'GO:0005575', 'GO:0110165'}
        #     removed_classes = set()
            terms = list(f.read().splitlines())
            edges = [(s, r, d) for s,r,d in edges if s in terms and  d in terms]# and ics[s] > 0.3 and ics[d]>0.3]
        

        srcs, rels, dsts = tuple(map(list, zip(*edges)))
        

        g = dgl.DGLGraph()
        nodes = list(set(srcs).union(set(dsts)))
        g.add_nodes(len(nodes))
        node_idx = {v: k for k, v in enumerate(nodes)}
        
        if self.self_loop:
            srcs += nodes
            dsts += nodes
            rels += ["id" for _ in range(len(nodes))]

            
        srcs_idx = [node_idx[s] for s in srcs]
        dsts_idx = [node_idx[d] for d in dsts]
        
        edges_per_rel = {}
        for rel in rels:
            if not rel in edges_per_rel:
                edges_per_rel[rel] = 0
            edges_per_rel[rel] += 1




        #Annotations
        logging.info("Processing annotatios")
        data_file = self.file_params["data_file"]



        prot_idx, annotations = self.getAnnotations(g.number_of_nodes, node_idx, ics)


        if self.normalization:
        
            edge_node = {}
            rels_dst = list(zip(rels, dsts))
            rels_dst_idx = list(zip(rels, dsts_idx))
#            print(dsts)

            for rel, dst_idx in rels_dst_idx:
                if (rel, dst_idx) not in edge_node:
                    edge_node[(rel, dst_idx)] = 0
                edge_node[(rel, dst_idx)] += 1
            
            edge_node = {k: 1/v for k, v in edge_node.items()}


            
            norm = [edge_node[i] for i in rels_dst_idx]
            
            norm_ics = [ics[item[1]] for item in rels_dst]
            norm_comp = [x*y for x, y in zip(norm,norm_ics)]

            zipped_data = list(zip(srcs_idx, dsts_idx, rels, norm_comp))
            srcs, dsts, rels, norm = zip(*[(s, d, r, n) for s, d, r, n in zipped_data if edges_per_rel[r] > self.min_edges])

            norm = th.Tensor(norm).view(-1, 1)

            
        else:
            norm = None

            zipped_data = list(zip(srcs_idx, dsts_idx, rels))
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
          

        


        return g, annotations, prot_idx, num_rels



    def getAnnotsDict(self, training_prots):
        data_file = self.file_params["data_file"]
        
        with open(data_file, 'r') as f:
            rows = [line.strip('\n').split('\t') for line in f.readlines()]

            annots_dict = HashMap()

            for i, row  in enumerate(rows):
                prot_id = row[0]

                if prot_id in training_prots:
                    if not prot_id in annots_dict:
                        annots_dict.put(prot_id, ArrayList())
                    
                for go_id in row[1:]:
                    
                    if prot_id in training_prots:
                        prot_annots = annots_dict[prot_id]
                        prot_annots.add(go_id)
                        annots_dict.put(prot_id, prot_annots)
            
        return annots_dict


    def getAnnotations(self, num_nodes, node_idx, ics):
        data_file = self.file_params["data_file"]
        prot_idx = {}
        with open(data_file, 'r') as f:
            rows = [line.strip('\n').split('\t') for line in f.readlines()]

            annotations = np.zeros((len(rows), num_nodes()), dtype=np.float32)

            for i, row  in enumerate(rows):
                prot_id = row[0]
                    
                prot_idx[prot_id] = i
                for go_id in row[1:]:
                    if go_id in node_idx:
                        annotations[i, node_idx[go_id]] = ics[go_id]
        logging.info("Finished processing annotations")
 
        return prot_idx, annotations
       #######



    class RGCN(BaseRGCN):
    
         def build_hidden_layer(self, idx):
             act = F.relu if idx < self.num_hidden_layers - 1 else None
             return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)


    def node_norm_to_edge_norm(g, node_norm):
        g = g.local_var()
        # convert to edge norm
        g.ndata['norm'] = node_norm
        g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        return g.edata['norm']



            
    def format(self, string):
        '''
        Transforms URIs comming from IC computation
        '''
        identifier = string.split('/')[-1].replace("_", ":")
            
        return identifier

    
        
    class PPIModel():
        def __init__(self, num_rels = None, num_bases = None, num_nodes = None, n_hid = None, dropout = 0):
            raise NotImplementedError()


    
    class GraphDatasetPPI():
        def __init__(self):
            raise NotImplementedError()
