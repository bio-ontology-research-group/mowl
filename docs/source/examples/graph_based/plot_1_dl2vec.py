"""
DL2Vec
========

This example corresponds to the paper `Predicting candidate genes from phenotypes, functions and \
anatomical site of expression <https://doi.org/10.1093/bioinformatics/btaa879>`_. 

This work is a graph-based machine-learning method to learn from biomedical ontologies. This \
method works by transforming the ontology into a graph following a set of rules. Random walks \
are generated from the obtained graph and then processed by a Word2Vec model, which generates 
embeddings of the original ontology classes. This algorithm is applied to generate numerical \
representations of genes and diseases based on the background knowledge found in the Gene \
Ontology, which was extended to incorporate phenotypes, functions of the gene products and \
anatomical location of gene expression. The representations of genes and diseases are then \
used to predict candidate genes for a given disease.
"""

# %%
# To show an example of DL2Vec, we need 3 components:
# 
# - The ontology projector
# - The random walks generator
# - The Word2Vec model

import mowl
mowl.init_jvm("10g")

from mowl.datasets.builtin import GDAHumanDataset
from mowl.projection import DL2VecProjector
from mowl.walking import DeepWalk
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec


# %%
# Projecting the ontology
# -----------------------
#
# We project the ontology using the DL2VecProjector class. The rules used to project the 
# ontology can be found at :doc:`/graphs/projection`. The outcome of the projection algorithm
# is an edgelist.

dataset = GDAHumanDataset()

projector = DL2VecProjector(bidirectional_taxonomy=True)
edges = projector.project(dataset.ontology)

# %%
# Generating random walks
# -----------------------
#
# The random walks are generated using the DeepWalk class. This class implements the DeepWalk
# algorithm with a modification consisting of including the edge labels as part of the walks.

walker = DeepWalk(5, # number of walks per node
                  20, # walk length
                  0.1, # restart probability
                  workers=4) # number of threads

walks = walker.walk(edges)


# %%
# Training the Word2Vec model
# ---------------------------
#
# To train the Word2Vec model, we rely on the Gensim library:

walks_file = walker.outfile
sentences = LineSentence(walks_file)
model = Word2Vec(sentences, vector_size=20, window=3, min_count=1, workers=4)
