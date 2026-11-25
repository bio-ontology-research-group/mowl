EIKE: extensional and intensional knowledge embedding
=====================================================

EIKE (Wang et al., 2024) embeds an ontology in two spaces of the same dimension.

* **Extensional space** (what a concept contains): a concept is an axis-aligned ellipsoid
  (centre, semi-axes, scale), an instance is a point, a relation is a vector.
* **Intensional space** (what a concept means): a concept is a vector, initialised from a
  language model applied to its name or description; an instance is the virtual vector
  :math:`M i`, with :math:`M` the identity (EYE) or a learned matrix (MAT).

Scores (lower is better): InstanceOf and SubClassOf triples add an extensional and an
intensional part, :math:`f = f^{ex} + \alpha f^{in}`; relational triples use TransE. The
losses are margin ranking losses against corrupted triples.

What mOWL implements
--------------------

:class:`~mowl.models.eike.model.EIKE` (catalogue key ``eike``) with the module
:class:`~mowl.nn.el.eike.module.EIKEModule`, which is the combination of parts called ``eike``
(ellipsoids and vectors as linked classes, translations, the ellipsoid measure linked to the
cosine order, the margin ranking loss and head-or-tail corruption as negatives; see
:doc:`/representation/combine`). The loss module written by hand is kept in
:mod:`mowl.baselines`. It uses the EL training loop
(:class:`~mowl.base_models.EmbeddingELModel`) with the three triple kinds of the paper taken
from the ontology:

* SubClassOf triples: atomic ``C ⊑ D`` (``gci0``);
* InstanceOf triples: class assertions ``C(a)`` with a named class;
* relational triples: object property assertions ``r(a, b)``.

Axioms with existential or conjunctive parts, bottom and role axioms are not used. The model
scores ``gci0``, ``class_assertion`` and ``object_property_assertion`` (``axiom_scoring =
"module"``), so it runs in the joint benchmarks for subsumption, instance checking and role
assertion on named classes only. Where the printed formulas of the paper are ambiguous (the
scale :math:`r_c`, the printed subclass score, which does not imply containment) see the notes
of the catalogue card.

The language model is **optional**. EIKE needs no download: without ``intensional_vectors``
the intensional concept vectors are random (the paper's UNP variant). To use text, compute
the vectors yourself with any encoder and pass them in:

.. testcode::

   import numpy as np
   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.models import EIKE

   ad = OWLAPIAdapter()
   ont = ad.create_ontology("http://example/eike")
   cat, animal = ad.create_class("http://example#Cat"), ad.create_class("http://example#Animal")
   tom = ad.create_individual("http://example#tom")
   axs = HashSet()
   axs.add(ad.create_subclass_of(cat, animal))
   axs.add(ad.create_class_assertion(cat, tom))
   ad.owl_manager.addAxioms(ont, axs)

   # any callable list[str] -> array works, e.g. SentenceTransformer(...).encode
   encoder = lambda texts: np.random.rand(len(texts), 8)
   model = EIKE(Dataset(ont), embed_dim=8)
   texts = model.class_texts()
   vectors = EIKE.encode_class_texts(texts, encoder)
   model = EIKE(Dataset(ont), embed_dim=8, intensional_vectors=vectors)
   model.train(epochs=3)
   print(len(texts) >= 2)

.. testoutput::
   :options: +ELLIPSIS

   ...True
