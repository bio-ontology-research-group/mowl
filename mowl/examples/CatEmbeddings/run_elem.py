import sys
sys.path.insert(0, '')
sys.path.append('../../../')

from elembeddingsnf1 import ELEmbeddings
from mowl.datasets.base import PathDataset


go = False

if go:
    model = ELEmbeddings("data/subsumption_data/go/", 
                         1024, #batch_size
                         lr = 0.001,
                         embedding_size = 100
                     )
else: 
    model = ELEmbeddings("data/subsumption_data/goslim/", 
                         1024, #batch_size,
                         lr = 0.1,
                         embedding_size = 100
                     )

model.train()
