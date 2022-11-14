"""
Onto2Vec
===========

This example corresponds to the paper `Onto2Vec: joint vector-based representation of biological \
entities and their ontology-based annotations <https://doi.org/10.1093/bioinformatics/bty259>`_. 

This method is an approach to learn numerical representations (embeddings) of (biomedical) \
ontologies by representing ontology axioms as text sequences and applying an unsupervised \
learning algorithm such as Word2Vec. Onto2Vec uses an ontology reasoner to infer new axioms as 
a preprocessing step. The algorithm is tested on the protein-protein interaction task.
"""

# %%
# For this algorithm, we need three components:
#
# - The reasoner
# - The corpus generator
# - The Word2Vec model


import mowl
mowl.init_jvm("20g")

from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.corpus import extract_and_save_axiom_corpus
from mowl.owlapi import OWLAPIAdapter
from mowl.reasoning import MOWLReasoner

from org.semanticweb.elk.owlapi import ElkReasonerFactory
from java.util import HashSet

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

# %%
# Inferring new axioms
# --------------------
#
# Onto2Vec uses an ontology reasoner to infer new axioms as a preprocessing step. In the original
# paper, the authors used the HermiT reasoner. For this example, we use the ELK reasoner.

dataset = PPIYeastSlimDataset()

reasoner_factory = ElkReasonerFactory()
reasoner = reasoner_factory.createReasoner(dataset.ontology)
mowl_reasoner = MOWLReasoner(reasoner)

# %%
# We wrap the reasoner into the :class:`MOWLReasoner <mowl.reasoning.base.MOWLReasoner>` class \
# in order to use some shortcuts the mOWL
# provides such as:
#
# - inferring subclass axioms
# - inferring equivalent class axioms
# - inferring disjoint axioms (not applicable for this example since we use ELK reasoner)

classes = dataset.ontology.getClassesInSignature()
subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes)
equivalent_class_axioms = mowl_reasoner.infer_equivalent_class_axioms(classes)

# %%
# We can now add the inferred axioms to the ontology.

adapter = OWLAPIAdapter()
manager = adapter.owl_manager

axioms = HashSet()
axioms.addAll(subclass_axioms)
axioms.addAll(equivalent_class_axioms)

manager.addAxioms(dataset.ontology, axioms)


# %%
# Generating the corpus and training the model
# -----------------------------------------------
#
# Now that we have an extended ontology, we can generate the corpus out of it. After that, we
# can train the Word2Vec model.

extract_and_save_axiom_corpus(dataset.ontology, "onto2vec_corpus.txt")

sentences = LineSentence("onto2vec_corpus.txt")
model = Word2Vec(sentences, vector_size=5, window=2, min_count=1, workers=4)
