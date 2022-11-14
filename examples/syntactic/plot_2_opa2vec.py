"""
OPA2Vec
===========

This example corresponds to the paper `OPA2Vec: combining formal and informal content of \
biomedical ontologies to improve similarity-based \
prediction <https://doi.org/10.1093/bioinformatics/bty933>`_. 

This method is an extension of Onto2Vec that apart from formal knowldege (i.e. axioms) \
it also uses informal knowledge such as entity metadata (i.e. synonyms, definitions, etc.)
"""

# %%
# For this algorithm, we need four components:
#
# - The reasoner
# - The corpus generator
# - The annotations generator
# - The Word2Vec model


import mowl
mowl.init_jvm("20g")

from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.corpus import extract_and_save_axiom_corpus, extract_and_save_annotation_corpus
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
# OPA2Vec uses an ontology reasoner to infer new axioms as a preprocessing step. In the original
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

extract_and_save_axiom_corpus(dataset.ontology, "opa2vec_corpus.txt")
extract_and_save_annotation_corpus(dataset.ontology, "opa2vec_corpus.txt", mode="a")

sentences = LineSentence("opa2vec_corpus.txt")
model = Word2Vec(sentences, vector_size=5, window=2, min_count=1, workers=4)
