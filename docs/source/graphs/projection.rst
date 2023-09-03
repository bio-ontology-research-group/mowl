Projecting ontologies
=======================

Ontologies contain adjacency information that can be projected into a graph. There are different ways of generating such graphs:

* :class:`Taxonomy <mowl.projection.taxonomy.model.TaxonomyProjector>`
* :class:`Taxonomy with relations <mowl.projection.taxonomy_rels.model.TaxonomyWithRelationsProjector>`
* :class:`DL2Vec <mowl.projection.dl2vec.model.DL2VecProjector>`
* :class:`OWL2Vec* <mowl.projection.owl2vec_star.model.OWL2VecStarProjector>`

  
Each method follow different projection rules. In the case of ``Taxonomy``, only axioms of the form :math:`C \sqsubseteq D` will be considered (:math:`C,D` are atomic concepts) and each of them will form a graph edge ``(C, subclassOf, D)``. ``Taxonomy with relations`` is an extension of the previous one that also adds axioms of the form :math:`C \sqsubseteq \exists R. D` as edges ``(C, R, D)``. ``DL2Vec`` and ``OWL2Vec*`` contain more complex rules. Let's have a look at them:

DL2Vec
-------
The DL2Vec graph follows the rules described in the paper `Predicting candidate genes from phenotypes, functions, and anatomical site of expression (2020) <https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa879/5922810>`__. The parsing rules are shown in the table below:


+-------------------------------------------------------+---------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Condition 1						| Condition 2										| Triple(s)											|
+=======================================================+=======================================================================================+===============================================================================================+
| :math:`A \sqsubseteq Q R_{0} \ldots Q R_{m} D`	| :math:`D := B_{1} \sqcup \ldots \sqcup B_{n} | B_{1} \sqcap \ldots \sqcap B_{n}`	| :math:`\left\langle A, (R_{0}...R_{m}), B_i \right\rangle` for :math:`i \in 1 \ldots n`	|
+-------------------------------------------------------+											|												|
| :math:`A \equiv Q R_{0} \ldots Q R_{m} D`		|											|												|
+-------------------------------------------------------+---------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| :math:`A \sqsubseteq B`				|											| :math:`\left\langle A, SubClassOf, B \right\rangle`						|
+-------------------------------------------------------+---------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| :math:`A \equiv B`					|											| :math:`\left\langle A, EquivalentTo, B \right\rangle`						|
+-------------------------------------------------------+---------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+



OWL2Vec*
----------


The OWL2Vec* graph follows the rules described in the paper `OWL2Vec*: embedding of OWL ontologies (2021) <https://link.springer.com/article/10.1007%2Fs10994-021-05997-6>`__. The parsing rules are shown in the table below:


+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+
|Axiom of condition 1					|  Axiom or triple(s) of condition 2									| Projected triple(s)							|
+=======================================================+=======================================================================================================+=======================================================================+
|:math:`A \sqsubseteq \square r . D`			| :math:`D \equiv B\left|B_{1} \sqcup \ldots \sqcup B_{n}\right| B_{1} \sqcap \ldots \sqcap B_{n}`	| :math:`\langle A, r, B\rangle`					|
+-------------------------------------------------------+													|									|
|or							|													|									|
+-------------------------------------------------------+													|									|
|:math:`\square r . D \sqsubseteq A`			|													|									|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+
|:math:`\exists r . \top \sqsubseteq A` (domain)	| :math:`\top \sqsubseteq \forall r . B` (range)							| :math:`\langle A, r, B_{i}\rangle` for :math:`i \in 1, \ldots, n`	|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+									|
|:math:`A \sqsubseteq \exists r .\{b\}`			| :math:`B(b)`												|									|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+									|
|:math:`r \sqsubseteq r^{\prime}`			| :math:`\left\langle A, r^{\prime}, B\right\rangle` has been projected					|									|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+									|
|:math:`r^{\prime} \equiv r^{-}`			| :math:`\left\langle B, r^{\prime}, A\right\rangle` has been projected					|									|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+									|
|:math:`s_{1} \circ \ldots \circ s_{n} \sqsubseteq r`	| :math:`\langle A, s_1, C_1\rangle \ldots \langle C_n, s_n, B\rangle` have been projected		|									|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+
|:math:`B \sqsubseteq A`				| :math:`-`												| :math:`\langle B, r d f s: s u b C l a s s O f, A\rangle`		|
|							|													+-----------------------------------------------------------------------+
|							|													| :math:`\left\langle A, rdfs:subClassOf^{-}, B\right\rangle`		|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+
|:math:`A(a)`						| :math:`-`												| :math:`\langle a, r d f: t y p e, A\rangle`				|
|							|													+-----------------------------------------------------------------------+
|							|													| :math:`\left\langle A, r d f: t y p e^{-}, a\right\rangle`		|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+
|:math:`r(a, b)`					| :math:`-`												| :math:`\langle a, r, b\rangle`					|
+-------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------+


To use any projector, we can initialize them in two ways: directly or through a factory method.

Directly:

.. testcode:: python

   from mowl.projection import TaxonomyProjector, TaxonomyWithRelationsProjector, OWL2VecStarProjector, DL2VecProjector

   projector = TaxonomyProjector(bidirectional_taxonomy=True)

   projector = TaxonomyWithRelationsProjector(
		taxonomy = True,
		bidirectional_taxonomy=False,
		relations = ["name of relation 1", "name of relation 2", "..."])

   projector = DL2VecProjector(bidirectional_taxonomy= True)

   projector = OWL2VecStarProjector(
		bidirectional_taxonomy=True,
		include_literals = False,
		only_taxonomy = True)
	

Using a factory method:

.. testcode:: python

   from mowl.projection.factory import projector_factory
   projector = projector_factory("dl2vec", bidirectional_taxonomy = True)


Given any projector, the input for starting the graph generation is an OWLOntology. For example:

.. testcode:: python

   from mowl.datasets.builtin import FamilyDataset

   dataset = FamilyDataset()

   edges = projector.project(dataset.ontology)


The output is stored in the variable ``edges``, which is a list of :class:`Edge <mowl.projection.edge.Edge>` instances.


DL2Vec extension
-------------------

Initially, DL2Vec projection rules are intended to parse TBox axioms. However, for some cases, useful information might be present as ABox axioms of the form :math:`C(a)`, :math:`\exists R.C (a)` and :math:`R(a,b)` where :math:`C` is an atomic concept, :math:`R` is a role and :math:`a, b` are individuals. The extended rules are the following:


+---------------------------+-----------------------------------------------------------+
| Condition		    | Triple     						|
+===========================+===========================================================+
| :math:`C(a)`	            | :math:`\left\langle a, http://type, C \right\rangle`	|
+---------------------------+-----------------------------------------------------------+
| :math:`\exists R.C (a)`   | :math:`\left\langle a, R, C \right\rangle`		|
+---------------------------+-----------------------------------------------------------+
| :math:`R(a,b)`	    | :math:`\left\langle a, R, b \right\rangle`		|
+---------------------------+-----------------------------------------------------------+

To use the extension, use the ``with_individuals`` parameters in the ``project`` method:

.. testcode:: python

   from mowl.projection import DL2VecProjector
   projector = DL2VecProjector(bidirectional_taxonomy= True)
   edges_with_individuals = projector.project(dataset.
   ontology, with_individuals=True)

