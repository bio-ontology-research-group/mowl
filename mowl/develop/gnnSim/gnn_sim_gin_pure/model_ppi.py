from mowl.develop.GNNSim.gnnSimAbstract import AbsGNNSimPPI


import torch as th
import math
from math import floor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
#from dgl.nn.pytorch import RelGraphConv
from .rgcnConv import RelGraphConv

from dgl.nn import GINConv

class GNNSimPPI(AbsGNNSimPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
         
    class PPIModel(nn.Module):

        def __init__(self, params):
            super().__init__()

            self.siamese = params["siamese"]
            self.residual = params["residual"]
            if self.siamese:
                self.h_dim = 1
            else:
                self.h_dim = 2


            self.n_hid = params["n-hid"]
            self.num_nodes = params["num-nodes"]
            self.act = nn.Sigmoid()


            self.emb = nn.Embedding(self.num_nodes, self.h_dim)
            k = math.sqrt(1 / self.h_dim)
            nn.init.uniform_(self.emb.weight, -k, k)


            self.post = nn.Sequential(
                nn.Linear(20*self.num_nodes, 2048),
                nn.ReLU(),
                nn.Linear(2048,1024),
                nn.Sigmoid()
                
            )
            self.gin_layers = nn.ModuleList()

            self._features = {i: th.empty(0) for i in range(self.n_hid)}


            self.gin = GINConv(nn.Linear(self.h_dim, 20), "mean", learn_eps = True)

            # for i in range(self.n_hid):
            #     newLayer = GINConv(nn.Linear(self.h_dim, 64), "sum", learn_eps = True)
            #     self.gin_layers.append(newLayer)

            # for i in range(self.n_hid):
            #     self.gin_layers[i].register_forward_hook(self._save_inter_feats(i))



            dim1 = self.num_nodes
            dim2 = floor(self.num_nodes/2)

            if self.siamese:
                self.fc = nn.Linear(self.n_hid*self.h_dim, 1024)
            else:
                self.fc = nn.Linear(self.n_hid*self.h_dim, 1)

        def compute_siamese_score(self, x1, x2):
            x = th.sum(x1*x2, dim=1, keepdims = True)
            return th.sigmoid(x)


        def _save_inter_feats(self, layer_id):
            def fn(_, __, output):
                self._features[layer_id] = output
            return fn

        def forward_each(self, g, features, edge_type, norm):
            
            # if self.residual:
            #     skip = features
            x = features
            x = self.gin(g, x, norm).reshape(-1, self.num_nodes, 20)
            x = self.post(x).reshape(-1, self.h_dim)
#             for l in self.gin_layers:
#                 x = l(g, x, norm)
#                 x = self.act(x)
#                 if self.residual:
#                     x = skip + x
#                     skip = x

#             out_feats = th.stack(list(self._features.values()))
#             out_feats = out_feats.reshape(self.n_hid, -1, self.num_nodes, self.h_dim)
#             out_feats = out_feats.permute(1,0,2,3)
# #            print("out feats ", out_feats.shape)
#             x = th.sum(out_feats, dim = 2)
# #            print("x1", x.shape)

#             x = th.flatten(x).view(-1, self.n_hid*self.h_dim)
# #            print("x2 ", x.shape)
#             x = self.fc(x)
#            print("x3 ", x.shape)
            return x.squeeze()

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
