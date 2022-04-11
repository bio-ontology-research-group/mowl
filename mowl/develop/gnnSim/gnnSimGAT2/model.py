from mowl.develop.GNNSim.gnnSimAbstract import AbsGNNSimPPI
from .gatV2Conv import GATv2Conv
import torch.nn as nn
import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

class GNNSimPPI(AbsGNNSimPPI):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    class PPIModel(nn.Module):
        
        def __init__(self, params):
            super().__init__()

            self.num_nodes = params["num-nodes"]

            actStr = params["act"]
            self.n_heads = params["n-heads"]
            self.siamese = params["siamese"]
            if self.siamese:
                self.h_dim = 1
            else:
                self.h_dim = 2

            if actStr == "relu": 
                act = F.relu
            elif actStr == "sigmoid":
                act = F.sigmoid
            else: 
                act = None

            self.gat = GATv2Conv(
                self.h_dim,
                self.h_dim,
                self.n_heads,
                feat_drop = params["feat-drop"],
                attn_drop = params["attn-drop"],
                negative_slope = params["neg-slope"],
                residual = params["residual"],
                activation = act,
                allow_zero_in_degree = False,
                bias = params["bias"],
                share_weights = params["share-weights"]

                )

            if self.siamese:
                self.fc = nn.Linear(self.h_dim*self.num_nodes*self.n_heads, 1024)
            else:
                self.fc = nn.Linear(self.h_dim*self.num_nodes*self.n_heads, 1)

        def compute_siamese_score(self, x1, x2):
            x = th.sum(x1*x2, dim=1, keepdims = True)
            return th.sigmoid(x)


        def forward_each(self, g, feat, get_attention = True):
            x, attns = self.gat(g, feat, get_attention)
            
            x = x.reshape(-1, self.h_dim*self.num_nodes*self.n_heads)
            x = self.fc(x)
            return x, attns

        def forward(self, g, feat1, feat2, get_attention = True):
                
            if self.siamese:
                x1, attn1 = self.forward_each(g, feat1, get_attention)
                x2, attn2 = self.forward_each(g, feat2, get_attention)
                return self.compute_siamese_score(x1, x2)
            else:
                feats = th.cat([feat1, feat2], dim=1)
                x, attns = self.forward_each(g, feats, get_attention)
                x = th.sigmoid(x)
                return x
                



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

