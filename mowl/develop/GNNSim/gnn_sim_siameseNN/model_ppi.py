from mowl.develop.GNNSim.gnnSimAbstract import AbsGNNSimPPI
 
import torch as th
from math import floor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from .rgcnConv import RelGraphConv


class GNNSimPPI(AbsGNNSimPPI):
    def __init__(self,
                 dataset,
                 n_hidden,
                 dropout,
                 learning_rate,
                 batch_size,
                 epochs,
                 parser = "taxonomy", #Default: generate graph taxonomy
                 normalize = False,
                 regularization = 0,
                 self_loop = False,
                 min_edges = 0, #Only takes the relation types in which the number of triples is greater than min_edges. If 0 then takes all the relation types
                 seed = -1,
                 file_params = None #Dictionary of data file paths corresponding to the graph generation method (NEEDS REFACTORING)
                 ):
        super().__init__(
            dataset,
            n_hidden,
            dropout,
            learning_rate,
            batch_size,
            epochs,
            parser,
            normalize,
            regularization,
            self_loop,
            min_edges,
            seed,
            file_params
        )
        
         
    class PPIModel(nn.Module):

        def __init__(self, num_rels = None, num_bases = None, num_nodes = None, n_hid = None, dropout = 0):
            super().__init__()


            print("Creating PPI model...", flush =True)
            self.h_dim = 1
            self.num_rels = num_rels
            self.num_bases = num_rels
            self.num_nodes = num_nodes

            print(f"Num rels: {self.num_rels}")
            print(f"Num bases: {self.num_bases}")


            self.rgcn_layers = nn.ModuleList()
            self.act = F.relu
            self.bnorm = nn.BatchNorm1d(self.h_dim)
            for i in range(n_hid):
                act = F.relu if i < n_hid - 1 else None
                newLayer = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, activation = act, self_loop = False, dropout = dropout)
                self.rgcn_layers.append(newLayer)


            dim1 = self.num_nodes
            dim2 = floor(self.num_nodes/2)

            self.net = nn.Sequential(
                nn.Linear(dim1, 1024),
                nn.Dropout(),
    #            nn.Linear(dim2, 1024),
    #            nn.Dropout()
            )

        def forward_each(self, g, features, edge_type, norm):
            skip = features
            x = features 

            for l in self.rgcn_layers:
                x = l(g, x, edge_type, norm)
                x = self.bnorm(x)
                x = self.act(x)
                x = skip + x
                skip = x
                
    #            x += initial

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

