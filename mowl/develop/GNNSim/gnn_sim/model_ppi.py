from mowl.develop.GNNSim.gnnSimAbstract import AbsGNNSimPPI
 
import torch as th
import math
from math import floor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from .rgcnConv import RelGraphConv


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
            device = "cuda"

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


    

    class GraphDatasetPPI(IterableDataset):

        def __init__(self, g, df, labels, annots, prot_idx):
            self.graph = g
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

                feat1 = self.annots[:, pi1]
                feat2 = self.annots[:, pi2]

                yield (self.graph, label, pi1, pi2)

        def __iter__(self):
            return self.get_data()

        def __len__(self):
            return len(self.df)


