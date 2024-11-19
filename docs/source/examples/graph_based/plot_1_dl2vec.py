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

from mowl.datasets.builtin import GDADatasetV2
from mowl.models import RandomWalkPlusW2VModel
from mowl.projection import DL2VecProjector
from mowl.walking import DeepWalk
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

# %%
# Instantiating the dataset and the model

dataset = GDADatasetV2()

print(f"Number of classes: f{len(dataset.classes)}")

model = RandomWalkPlusW2VModel(dataset)
model.set_projector(DL2VecProjector())
model.set_walker(DeepWalk(5, 5, 0.1, workers=4))
model.set_w2v_model(vector_size=5, epochs=2, window=5, min_count=1, workers=4)
model.train()
# %%
# Evaluating the model

from mowl.evaluation import GDAEvaluator
model.set_evaluator(GDAEvaluator)
model.evaluate(dataset.testing)

