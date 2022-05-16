import sys
sys.path.insert(0, '')
sys.path.append('../../../')

from elembeddings.model import ELEmbeddings
from mowl.datasets.base import PathDataset


yeast  = True

if yeast:
    ds = 'data/data-train/yeast-classes-normalized.owl', 'data/data-valid/4932.protein.links.v10.5.txt', 'data/data-test/4932.protein.links.v10.5.txt'        


model = ELEmbeddings(ds)
model.train()
model.evaluate_ppi()
