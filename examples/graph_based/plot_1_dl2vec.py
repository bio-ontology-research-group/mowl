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

import sys
sys.path.append('../../')
import mowl
mowl.init_jvm("10g")

from mowl.datasets.builtin import GDAMouseDataset
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

dataset = GDAMouseDataset()

projector = DL2VecProjector(bidirectional_taxonomy=True)
edges = projector.project(dataset.ontology)

# %%
# Generating random walks
# -----------------------
#
# The random walks are generated using the DeepWalk class. This class implements the DeepWalk
# algorithm with a modification consisting of including the edge labels as part of the walks.

walker = DeepWalk(20, # number of walks per node
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
model = Word2Vec(sentences, vector_size=100, epochs = 20, window=5, min_count=1, workers=4)

# %%
# Evaluating the embeddings
# ------------------------------
#
# We can evaluate the embeddings using the
# :class:`EmbeddingsRankBasedEvaluator <mowl.evaluation.rank_based.EmbeddingsRankBasedEvaluator>`
# class. We need to do some data preparation.

from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
from mowl.evaluation.base import CosineSimilarity
from mowl.projection import TaxonomyWithRelationsProjector
# %%
# We are going to evaluate the plausability of an association gene-disease with a gene against all
# possible diseases and check the rank of the true disease association.

genes, diseases = dataset.evaluation_classes

projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                           relations=["http://is_associated_with"])

evaluation_edges = projector.project(dataset.testing)
filtering_edges = projector.project(dataset.ontology)
assert len(evaluation_edges) > 0

# %%
# The gene-disease associations will be scoredc using cosine similarity. For that reason we use
# the ``CosineSimilarity`` class.

vectors = model.wv
evaluator = EmbeddingsRankBasedEvaluator(
    vectors,
    evaluation_edges,
    CosineSimilarity,
    training_set=filtering_edges,
    head_entities = genes.as_str,
    tail_entities = diseases.as_str,
    device = 'cpu'
)

evaluator.evaluate(show=True)
