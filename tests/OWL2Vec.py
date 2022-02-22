import sys
sys.path.append("..")


from mowl.datasets.ppi_yeast import PPIYeastSlimDataset

from mowl.graph.graph import GraphGenModel
from mowl.embeddings.graph_based.owl2vec.model import OWL2VecStar as O2V
from mowl.graph.edge import Edge
import pickle as pkl
import time

def testCase():

    
    ds = PPIYeastSlimDataset()
    model = O2V(
            ds,
            "embeddings_owl2vec",
            bidirectional_taxonomy = True,
            only_taxonomy = False,
            include_literals = True,
            walking_method = "node2vec",
            p = 10,
            q = 0.1,
            vector_size = 20,
            wv_epochs = 5,
            window = 5,
            workers = 4
        )


    model.train()

    
if __name__ == "__main__":

    testCase()


