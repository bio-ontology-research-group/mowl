FaithEL
=======

FaithEL (Lacerda, Ozaki & Guimarães, RuleML+RR 2024) embeds a normalised :math:`\mathcal{ELH}`
knowledge base so that the TBox is represented *exactly* and only the ABox is learned.
The catalogue key is ``faithel``; guarantees, sources and the tests that check them are on
the card in the :doc:`method catalogue <catalogue>`.

.. note::

   The RuleML+RR paper could not be obtained. :class:`~mowl.models.faithel.model.FaithEL`
   follows the authors' public code (``victorlacerdab/FaithEL_method``) and the TGDK
   article (Lacerda, Ozaki & Guimarães, 2024) on which the code is based. Statements that
   only the paper makes are not checked here.

Geometry
--------

The TBox is the convex interpretation of the canonical model of Definition 26 of the TGDK
article (:class:`~mowl.geometric.faithfulness.StronglyFaithfulModel` with
``regions="convex"``): one coordinate per canonical element, each concept the convex hull of
its elements, each role the convex hull of concatenated element pairs. A TBox axiom of the
forms :math:`A \sqsubseteq B`, :math:`A \sqcap B \sqsubseteq C`,
:math:`A \sqsubseteq \exists r.B`, :math:`\exists r.A \sqsubseteq B` holds in it exactly
when the ontology entails it. This part is fixed and is not trained.

Learned parameters: one vector per individual, a "moving" parameter per concept, and a head
and a tail parameter per role. Moving parameters start at the centroid of the region and are
projected back to a ball of radius ``radius`` around it after every step.

Loss (as in the authors' code, mean squared error)
--------------------------------------------------

For an assertion :math:`C(a)` with individual vector :math:`x_a`, moving parameter
:math:`p_C` and region centroid :math:`c_C`:

.. math::

   \lVert x_a - c_C\rVert^2 + \gamma \lVert p_C - x_a\rVert^2 + \phi \lVert p_C - c_C\rVert^2
   - \psi \lVert x_a - x_{neg}\rVert^2 ,

with :math:`x_{neg}` a uniformly drawn individual. Role assertions use the same terms on the
concatenation :math:`[x_a, x_b]` with the head and tail parameters. The score of an assertion
is the distance to the parameter plus the distance to the centroid (``centroid_score``).

Supported axioms and tasks
--------------------------

* Training: class assertions and object property assertions of the dataset (the TBox is read
  from the ontology; ABox axioms are not part of the fixed geometry).
* Scores for the joint benchmarks (``axiom_scoring = "method"``): ``class_assertion``,
  ``object_property_assertion`` (learned), and ``gci0``-``gci3`` (0 if the axiom holds in the
  fixed TBox geometry, else 1, so rankings of TBox candidates contain ties).
* Not supported: bottom forms, role chains, nominals, complex ABox axioms.

Differences from the paper
--------------------------

* The embedding dimension is the number of canonical elements minus one; it is not a free
  parameter.
* Negatives are drawn uniformly and not filtered against true assertions (as in the code).
* The code hard-codes parts of the canonical model of its example ontology; mOWL computes it
  with the ELK-style completion for any ontology.

Published results and what can be reproduced
--------------------------------------------

The RuleML+RR paper (9 pages, Springer) is behind a login and no preprint, institutional copy
or listing with its text was found (arXiv has the companion theory article, which names an implementation
as future work); which numbers it reports is therefore unknown, and none are recorded. What is
public is the authors' code, which carries no licence. Its notebook trains on one knowledge
base, ``family_ontology.owl`` (10 classes, 7 roles, 469 individuals from Wikidata), with the
entailed class assertions and the role assertions split 70/30 at random, and reports hits at
1, 3, 10, 100 and all for queries ``C(?)`` and ``r(a, ?)`` (a query is a hit if any of its test
answers is among the first k).

mOWL runs that experiment: the dataset
:class:`~mowl.datasets.builtin.faithel_family.FaithELFamilyDataset` (downloaded at run time
from the pinned commit of the authors' repository, not redistributed), the regime
``assertion-query-hits``
(:class:`~mowl.evaluation.assertion_queries.AssertionQueryHitsRegime`), the benchmark
``faithel-family-assertions`` and the recipe of the notebook (learning rate 0.001, phi = gamma
= 1, psi = 0.7, radius 0.6, 200 epochs, batch 512).

.. code-block:: bash

   python -m mowl.benchmarks run faithel-family-assertions --method faithel --recipe

.. note::

   Published results: not reproducible in mOWL because the paper could not be read, and the
   code stores no outputs. The split of mOWL is NumPy's, not PyTorch's ``random_split``, and the
   embedding dimension is that of mOWL's canonical model (99 here), not the notebook's
   hard-coded 602.

Example
-------

.. testcode::

   import mowl
   mowl.init_jvm("2g")
   from mowl.datasets import PathDataset
   from mowl.models import FaithEL
   from mowl.owlapi import OWLAPIAdapter
   from org.semanticweb.owlapi.model import IRI
   import tempfile, pathlib

   adapter = OWLAPIAdapter()
   mgr, df = adapter.owl_manager, adapter.data_factory
   ns = "http://example.org/"
   ont = adapter.create_ontology(ns + "o")
   cat, animal = adapter.create_class(ns + "Cat"), adapter.create_class(ns + "Animal")
   mgr.addAxiom(ont, adapter.create_subclass_of(cat, animal))
   tom = adapter.create_individual(ns + "tom")
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(cat, tom))
   path = pathlib.Path(tempfile.mkdtemp()) / "o.owl"
   mgr.saveOntology(ont, IRI.create(path.as_uri()))

   model = FaithEL(PathDataset(str(path)), learning_rate=0.01, neg_sampling=False)
   model.train(5)
   print(len(model.loss_history))

.. testoutput::

   5
