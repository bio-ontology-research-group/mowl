from mowl.develop.gnnSim.gnnSimAbstract import AbsGNNSimPPI
 
import torch as th
import math
from math import floor
from torch import nn
from torch import optim
import click as ck
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from .rgcnConv import RelGraphConv

from torch.nn.parallel import DistributedDataParallel
from dgl.dataloading import GraphDataLoader
import logging
class GNNSimPPI(AbsGNNSimPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
         
    class PPIModel(nn.Module):

        def __init__(self, params):
            super().__init__()
            
            self.siamese = params["siamese"]

            if self.siamese:
                self.h_dim = 1
            else:
                self.h_dim = 2

            self.n_hid = params["n-hid"]
            self.num_rels = params["num-rels"]
            self.num_bases = self.num_rels
            self.num_nodes = params["num-nodes"]
            self.dropout = params["dropout"]
            self.residual = params["residual"]

            print(f"Num rels: {self.num_rels}")
            print(f"Num bases: {self.num_bases}")
            
            self.act = nn.ReLU()
            self.bnorm = nn.BatchNorm1d(self.h_dim)

            self.emb = nn.Embedding(self.num_nodes, self.h_dim)
            k = math.sqrt(1 / self.h_dim)
            nn.init.uniform_(self.emb.weight, -k, k)


            self.rgcn_layers = nn.ModuleList()

            for i in range(self.n_hid):
                #act = F.relu if i < self.n_hid - 1 else None
                act  = None
                newLayer = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, activation = act, self_loop = False, dropout = self.dropout)
                self.rgcn_layers.append(newLayer)
            
        
            dim1 = self.num_nodes
            dim2 = floor(self.num_nodes/2)

            if self.siamese:
                self.fc = nn.Linear(self.h_dim*self.num_nodes, 1024)
            else:
                self.fc = nn.Linear(self.h_dim*self.num_nodes, 1)


        def compute_siamese_score(self, x1, x2):
            x = th.sum(x1*x2, dim=1, keepdims = True)
            return th.sigmoid(x)


        def forward_each(self, g, features, edge_type, norm):
        
            if self.residual:
                skip = features
            
            x = features
            for l in self.rgcn_layers:
                x = l(g, x, edge_type, norm)
#                x = self.bnorm(x)
                x = self.act(x)

                if self.residual:
                    x = skip + x
                    skip = x
            
#            x = x.reshape(-1, self.num_nodes, self.h_dim)
#            x = th.sum(x, dim = 1)
            x = x.view(-1, self.h_dim*self.num_nodes)
            x = self.fc(x)
            return x



        def forward(self, g, feat1, feat2):


            # bs = floor(ffeat1.shape[0]/self.num_nodes)

            # idx = th.tensor([i for i in range(self.num_nodes)]).to(device)

            # feat1 = self.emb(idx).repeat(bs, 1)
            # feat2 = self.emb(idx).repeat(bs, 1)
            
            # mask1 = ffeat1.clone().detach().repeat(1, self.h_dim).to(device)
            # mask2 = ffeat2.clone().detach().repeat(1, self.h_dim).to(device)
            
            # assert feat1.shape == mask1.shape, f"{feat1.shape}, {mask1.shape}"
            # assert feat2.shape == mask2.shape, f"{feat2.shape}, {mask2.shape}"
            # feat1 *= mask1
            # feat2 *= mask2

            edge_type = g.edata['rel_type'].long()

            norm = None if not 'norm' in g.edata else g.edata['norm']

            if self.siamese:
                x1 = self.forward_each(g, feat1, edge_type, norm)
                x2 = self.forward_each(g, feat2, edge_type, norm)
                return self.compute_siamese_score(x1, x2)
            else:
                feats = th.cat([feat1, feat2], dim = 1)
                x = self.forward_each(g, feats, edge_type, norm)
                return th.sigmoid(x)


    def init_process_group(self, world_size, rank):
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:12345',
            world_size=world_size,
            rank=rank)



    def init_model(self, seed, device):
        th.manual_seed(seed)
        model = self.PPIModel(self.ppi_model_params).to(device)
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
        
        return model

    def train(self):
        # Data preparation
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
        self.num_bases = num_rels
        self.ppi_model_params["num-rels"] = num_rels

        train_labels = th.FloatTensor(train_df['labels'].values)
        val_labels = th.FloatTensor(val_df['labels'].values)
    
        self.train_data = self.GraphDatasetPPI(g, train_df, train_labels, annots, prot_idx)
        self.val_data = self.GraphDatasetPPI(g, val_df, val_labels, annots, prot_idx)

        seed = 0
        num_gpus = 2
        procs = []
        for rank in range(num_gpus):
            p = mp.Process(target=self.train_single, args = (rank, num_gpus, seed))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            
    def train_single(self, rank, world_size, seed, checkpoint_dir = None, tuning= False):
        self.init_process_group(world_size, rank)
        device = th.device(f"cuda:{rank}")
        th.cuda.set_device(device)


            
        train_dataloader = GraphDataLoader(self.train_data, batch_size = self.bs, drop_last = True, use_ddp =True, shuffle = False)
        val_dataloader = GraphDataLoader(self.val_data, batch_size = self.bs, drop_last = True)


        
        num_nodes = self.g.number_of_nodes()

        self.ppi_model_params["num-nodes"] = num_nodes
        model = self.init_model(seed, device)#self.PPIModel(self.ppi_model_params)
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
 #           device = "cuda"
            model = model.to(device)
            epoch_loss = 0
            model.train()
            train_dataloader.set_epoch(epoch)
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
        
#            device = "cuda"
#            model = model.to(device)
            model.eval()

            # val_loss = 0
            # preds = []
            # labels = []
            # with th.no_grad():
            #     optimizer.zero_grad()
            #     with ck.progressbar(val_dataloader) as bar:
            #         for i, (batch_g, batch_labels, feats1, feats2) in enumerate(bar):
                        
            #             feats1 = annots[feats1].view(-1,1)
            #             feats2 = annots[feats2].view(-1,1)
                        
            #             logits = model(batch_g.to(device), feats1.to(device), feats2.to(device)).squeeze()
            #             lbls = batch_labels.squeeze().to(device)
            #             loss = loss_func(logits, lbls.float())
            #             val_loss += loss.detach().item()
            #             labels = np.append(labels, lbls.cpu())
            #             preds = np.append(preds, logits.cpu())
            #         val_loss /= (i+1)

            # roc_auc = self.compute_roc(labels, preds)
            # if not tuning:
            #     if best_roc_auc < roc_auc:
            #         best_roc_auc = roc_auc
            #         th.save(model.state_dict(), self.file_params["output_model"])
            #     if val_loss < best_loss:
            #         best_loss = val_loss
            #         early_stopping = early_stopping_limit
            #     else:
            #         early_stopping -= 1
            #     print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')
                
            #     if early_stopping == 0:
            #         print("Finished training (early stopping)")
            #         break
            # else:
            #     with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #         path = os.path.join(checkpoint_dir, "checkpoint")
            #         th.save((model.state_dict(), optimizer.state_dict()), path)

            #     tune.report(loss=(val_loss), auc=roc_auc)
        dist.destroy_process_group()
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

    

    class GraphDatasetPPI(Dataset):

        def __init__(self, g, df, labels, annots, prot_idx):
            self.graph = g
            self.annots = annots
            self.labels = labels
            self.df = df
            self.prot_idx = prot_idx

        # def get_data(self):
        #     for i, row in enumerate(self.df.itertuples()):
        #         p1, p2 = row.interactions
        #         label = self.labels[i].view(1, 1)
        #         if p1 not in self.prot_idx or p2 not in self.prot_idx:
        #             continue
        #         pi1, pi2 = self.prot_idx[p1], self.prot_idx[p2]

        #         feat1 = self.annots[:, pi1]
        #         feat2 = self.annots[:, pi2]

        #         yield (self.graph, label, pi1, pi2)

        # def __iter__(self):
        #     return self.get_data()

        def get_item(self, idx):
            row  = self.df.iloc(idx)
            p1, p2 = row.interactions
            label = self.labels[idx].view(1,1)
            
            pi1, pi2 = self.prot_idx[p1], self.prot_idx[p2]

            feat1 = self.annots[:, pi1]
            feat2 = self.annots[:, pi2]

            return (self.graph, label, pi1, pi2)
            

        def __len__(self):
            return len(self.df)


